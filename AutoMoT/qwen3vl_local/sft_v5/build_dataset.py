"""构建 SFT v5 的 route-level sequence index。

输入是 `keyframe_filter/collection_output/*_result.json` 的标定结果，输出是
`sequence_index.jsonl`。每行对应一条 route，内部包含该 route 的逐帧 RS/EVENT
训练目标和 Q2 随机选项映射。

典型用法（从 AutoMoT/ 目录运行）：

  python qwen3vl_local/sft_v5/build_dataset.py \
    --collection-dir keyframe_filter/collection_output \
    --data-root lead_data \
    --output-dir checkpoints/sft_v5_data

处理链路是：发现 scenario result → route 级完整性过滤 → frame 级坐标/图像/标签解析 →
固定本帧 EVENT 字母映射 → 以 route 为单位切 train/val。由于 memory 是连续状态，一条
route 中任一帧缺证据时整条跳过，不能删除中间帧后继续拼接成伪连续序列。
"""

from __future__ import annotations

import argparse
import json
import lzma
import pathlib
import pickle
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_v5 import DATASET_VERSION  # noqa: E402
from qwen3vl_local.sft_v5.labels import (  # noqa: E402
    collapse_regular_to_re,
    q2_raw_candidates_for_frame,
    resolve_event_target,
    resolve_rs_target,
    scenario_event_candidates_from_result,
    stable_event_option_map,
    weather_to_text,
)

import numpy as np  # noqa: E402

# LEAD 数据按 4 Hz 落盘；这里取连续 4 帧，即每个训练 frame 最多看到约 0.75 秒历史。
RGB_HISTORY_COUNT = 4
RGB_HISTORY_STEP = 1


def _read_json(path: pathlib.Path) -> Any:
    """以 UTF-8 读取一个 collection result JSON。

    单独封装便于以后加入 schema 校验或 streaming reader；当前异常直接向上抛出，避免
    损坏的标注文件被静默当作空 scenario。
    """

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_int(value: Any, default: int = 0) -> int:
    """容错转换为 ``int``；失败时返回调用者指定的哨兵值而不中断整批构建。"""

    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """容错转换为 ``float``，主要用于 XML weather progress 与数值属性。"""

    try:
        return float(value)
    except Exception:
        return default


def _history_rgb_paths(run_dir: pathlib.Path, frame_id: int) -> List[str]:
    """构造按旧到新排列的 4 帧 stitched RGB 路径。

    route 开头历史不足时把负 frame 截到 0，实现与在线/LeadMoT 一致的 left-pad；若
    文件名不是标准四位数字，则尝试用排序后的文件列表兼容。这里只返回路径，存在性
    在 ``_build_frame_row`` 统一检查。
    """

    rgb_dir = run_dir / "rgb"
    rgb_files = sorted(rgb_dir.glob("*.jpg")) if rgb_dir.exists() else []
    # Qwen 看到的是短历史而不是单帧：从旧到新排列。frame_id 不足 3 时复制/回退到
    # frame 0，和训练时“左 padding”语义一致，避免序列开头样本被丢掉。
    frame_ids = [max(frame_id - i * RGB_HISTORY_STEP, 0) for i in range(RGB_HISTORY_COUNT)]
    ordered = list(reversed(frame_ids))
    paths: List[str] = []
    for idx in ordered:
        direct = rgb_dir / f"{idx:04d}.jpg"
        if direct.exists():
            paths.append(str(direct))
        elif 0 <= idx < len(rgb_files):
            paths.append(str(rgb_files[idx]))
        else:
            paths.append(str(direct))
    return paths


def _meta_path_for_frame(run_dir: pathlib.Path, automot_root: pathlib.Path, ann: Mapping[str, Any], frame_id: int) -> pathlib.Path:
    """解析当前帧 meta 路径；旧 result 没写时按 ``metas/%04d.pkl`` 回退。

    标注中的相对路径可能相对 run，也可能相对 AutoMoT 根目录：优先采用实际存在的
    run-relative 路径，否则保留 automot-relative 候选交给后续存在性检查。
    """

    raw = ann.get("meta_path")
    if raw:
        path = pathlib.Path(str(raw))
        if path.is_absolute():
            return path
        run_relative = run_dir / path
        if run_relative.exists():
            return run_relative
        return automot_root / path
    return run_dir / "metas" / f"{frame_id:04d}.pkl"


def _inverse_conversion_2d(point: np.ndarray, translation: np.ndarray, yaw: float) -> np.ndarray:
    """把 world-frame 点转换到当前 ego frame。

    先减去自车 world translation，再旋转 ``-yaw``；输出约定为 ``x_forward, y_left``。
    该公式与 v3/v4/LeadMoT final_goal 完全一致，不能替换成图像坐标或 CARLA 的
    ``y_right`` 约定。
    """

    pt = np.asarray(point, dtype=np.float32).reshape(2)
    tr = np.asarray(translation, dtype=np.float32).reshape(2)
    # world 点平移到以自车为原点，然后用逆朝向旋到自车坐标系。
    delta = pt - tr
    c = float(np.cos(-yaw))
    s = float(np.sin(-yaw))
    return np.asarray([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1]], dtype=np.float32)


def _extract_final_goal_ego_from_meta(meta: Mapping[str, Any]) -> Tuple[float, float]:
    """从 LEAD meta 取 `next_target_points[-1]` 并转 ego frame。

    这条坐标和 v3/v4 的 `EGO_TO_GOAL_XY` 同源，是学生可见的导航输入，不是标签。
    不回退到 `route[-1]` 或 `(0, 0)`，避免给路口左右转/匝道样本塞错方向信号。
    """

    next_points = np.asarray(meta.get("next_target_points", []), dtype=np.float32)
    if next_points.size == 0:
        raise KeyError("meta missing next_target_points")
    next_points = next_points.reshape(-1, next_points.shape[-1])
    if next_points.shape[-1] < 2:
        raise ValueError(f"next_target_points last dim < 2: shape={next_points.shape}")
    if "pos_global" not in meta:
        raise KeyError("meta missing pos_global")
    if "theta" not in meta:
        raise KeyError("meta missing theta")
    pos_xy = np.asarray(meta["pos_global"], dtype=np.float32).reshape(-1)[:2]
    theta = float(np.asarray(meta["theta"], dtype=np.float32).reshape(-1)[0])
    goal = _inverse_conversion_2d(next_points[-1, :2], pos_xy, theta)
    return float(goal[0]), float(goal[1])


def _load_ego_to_goal_xy(meta_path: pathlib.Path) -> Optional[Tuple[float, float]]:
    """读取当前帧 meta 并提取目的地相对坐标。

    LEAD meta 常用 lzma 压缩但历史文件也可能是普通 pickle，因此按压缩/非压缩顺序尝试。
    任一步失败都返回 ``None``，由 route 构建层丢弃整条序列，绝不填 ``UNKNOWN`` 或零点。
    """

    try:
        with lzma.open(meta_path, "rb") as f:
            meta = pickle.load(f)
    except Exception:
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
        except Exception:
            return None
    try:
        return _extract_final_goal_ego_from_meta(meta)
    except Exception:
        return None


def _xml_path_from_route(route: Mapping[str, Any], automot_root: pathlib.Path) -> Optional[pathlib.Path]:
    """从 route result 中取 XML 路径，并把相对路径解析到 AutoMoT 根目录。"""

    raw = route.get("xml_path")
    if not raw:
        return None
    path = pathlib.Path(str(raw))
    if path.is_absolute():
        return path
    return automot_root / path


def _parse_weather_node(node: ET.Element) -> Dict[str, Any]:
    """解析 XML ``<weather .../>`` 属性，数值能转换时保存为浮点。"""

    out: Dict[str, Any] = {}
    for key, value in node.attrib.items():
        try:
            out[key] = float(value)
        except Exception:
            out[key] = value
    return out


def _load_xml_weathers(xml_path: Optional[pathlib.Path]) -> List[Dict[str, Any]]:
    """读取 route XML 中所有 weather 节点；缺失或解析失败返回空列表。

    weather 只用于 privileged teacher 描述，不决定样本标签，因此这里允许空值继续；
    与 RGB/meta/逐帧 annotation 的硬缺失策略不同。
    """

    if xml_path is None or not xml_path.exists():
        return []
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return []
    return [_parse_weather_node(node) for node in root.findall(".//weathers/weather")]


def _weather_for_frame(ann: Mapping[str, Any], xml_weathers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """选择当前帧 weather。

    新标定结果通常已经在 ``evidence.xml_weather`` 里保存了按 route progress 选择后的
    weather；旧结果没有时，在 XML weather 列表里选择进度最接近的一段。若 XML 只有
    一个节点，该规则自然等价于使用第一段。
    """

    evidence = ann.get("evidence") or {}
    weather = evidence.get("xml_weather")
    if isinstance(weather, Mapping) and weather:
        return dict(weather)
    if not xml_weathers:
        return {}
    progress = _safe_float(
        ann.get("route_percentage", ann.get("route_progress", evidence.get("route_percentage", evidence.get("route_progress", 0.0)))),
        0.0,
    )

    def _weather_progress(item: Mapping[str, Any]) -> float:
        """统一读取新旧 XML weather 节点里的 route 进度字段，缺失时按起点处理。"""

        return _safe_float(item.get("route_percentage", item.get("route_progress", 0.0)), 0.0)

    return dict(min(xml_weathers, key=lambda item: abs(_weather_progress(item) - progress)))


def _skip_sets(result: Mapping[str, Any]) -> Tuple[set[str], set[str]]:
    """从 result 顶层提取异常时长与数据缺失 route ID 集合。

    两类集合分开返回，使 summary 能分别统计跳过原因；同时兼容旧字段 ``run_id``。
    ``review_required`` 不属于 skip 集合，它仍正常参与训练。
    """

    abnormal = {
        str(x.get("route_id") or x.get("run_id"))
        for x in result.get("abnormal_duration_skipped", [])
        if x.get("route_id") or x.get("run_id")
    }
    missing = {str(x.get("route_id") or x.get("run_id")) for x in result.get("data_missing_skipped", []) if x.get("route_id") or x.get("run_id")}
    return abnormal, missing


def _build_frame_row(
    *,
    ann: Mapping[str, Any],
    run_dir: pathlib.Path,
    route_id: str,
    scenario_candidates: List[str],
    option_seed: int,
    xml_weathers: List[Dict[str, Any]],
    automot_root: pathlib.Path,
) -> Optional[Dict[str, Any]]:
    """把一帧完整 annotation 转成 v5 sequence index frame。

    返回 ``None`` 表示证据链不完整：frame ID、两类逐帧 annotation、meta 目的地或任一
    RGB 缺失。成功时会同时保存 canonical 目标、原始候选、展示候选和字母映射，后续
    训练/eval 不需要再次读取 collection result 或重新随机候选。
    """

    # 先做所有硬依赖检查，再解析标签；任一失败由 route 层升级为整条 route 跳过。
    frame_id = _safe_int(ann.get("frame_id"), -1)
    if frame_id < 0:
        return None
    if not isinstance(ann.get("frame_rs_annotation") or {}, Mapping):
        return None
    if not isinstance(ann.get("frame_event_annotation") or {}, Mapping):
        return None
    meta_path = _meta_path_for_frame(run_dir, automot_root, ann, frame_id)
    if not meta_path.exists():
        return None
    ego_to_goal_xy = _load_ego_to_goal_xy(meta_path)
    if ego_to_goal_xy is None:
        return None
    history = _history_rgb_paths(run_dir, frame_id)
    if not history or any(not pathlib.Path(path).exists() for path in history):
        return None
    # 标签解析集中在 labels.py：RS 双标签按置信度稳定选一，EVENT regular 折叠为 RE。
    rs_target = resolve_rs_target(ann)
    event_target = resolve_event_target(ann)
    # Q2 候选优先取 frame_event_annotation.allowed_events；只有旧数据缺失时才 fallback。
    # raw_candidates 仍保留 R-E*/U-E*，后面 display_candidates/option_map 才折叠 regular。
    raw_candidates = q2_raw_candidates_for_frame(
        ann,
        scenario_candidates=scenario_candidates,
        rs_label=rs_target.label,
    )
    # option_map 是本帧“字母 -> RE/U-E*”的唯一真相。训练/eval/probe 都保存它，
    # 避免运行时重新随机导致 target 和 prompt 对不上。
    option_map = stable_event_option_map(
        run_id=route_id,
        frame_id=frame_id,
        rs_label=rs_target.label,
        scenario_candidates=scenario_candidates,
        raw_candidates=raw_candidates,
        seed=option_seed,
    )
    display_candidates = collapse_regular_to_re(raw_candidates, rs_target.label)
    weather = _weather_for_frame(ann, xml_weathers)
    regular_event_codes = [code for code in raw_candidates if code.startswith("R-E")]
    if not regular_event_codes:
        # 如果 allowed_events 只有 UE，没有显式 regular code，仍保存 event_target 里的
        # regular_event_codes 供 RE 文案兜底。collapse_regular_to_re 会统一加入一个
        # RE 负类选项，但不伪造原始 R-E* 审计 code。
        regular_event_codes = list(event_target.regular_event_codes)
    # source 保留定位原始证据所需的最小信息，不把整个大 annotation 复制进 jsonl。
    return {
        "frame_id": frame_id,
        "frame_time_s": ann.get("frame_time_s", round(frame_id * 0.25, 3)),
        "rgb_path": history[-1] if history else str(run_dir / "rgb" / f"{frame_id:04d}.jpg"),
        "history_rgb_paths": history,
        "weather": weather,
        "weather_text": weather_to_text(weather),
        "ego_to_goal_xy": [round(float(ego_to_goal_xy[0]), 4), round(float(ego_to_goal_xy[1]), 4)],
        "rs_label": rs_target.label,
        "rs_option": rs_target.option,
        "rs_confidence": rs_target.confidence,
        "rs_secondary": list(rs_target.secondary),
        "rs_candidates": rs_target.candidates,
        "event_labels_raw": list(event_target.raw_events),
        "event_label": event_target.label,
        "event_code": event_target.event_code,
        "abnormal": bool(event_target.abnormal),
        "scenario_event_candidates": list(scenario_candidates),
        "frame_allowed_events_raw": list(raw_candidates),
        "regular_event_codes": regular_event_codes,
        "event_candidate_codes": list(display_candidates),
        "event_option_map": option_map,
        "candidate_mismatch": event_target.label not in set(option_map.values()),
        "review_required": bool((ann.get("frame_rs_annotation") or {}).get("review_required")),
        "source": {
            "meta_path": str(meta_path),
            "annotation_comment": ann.get("annotation_comment") or (ann.get("frame_rs_annotation") or {}).get("comment"),
            "event_comment": (ann.get("frame_event_annotation") or {}).get("comment"),
        },
    }


def _build_route_row(
    *,
    scenario: str,
    route: Mapping[str, Any],
    data_root: pathlib.Path,
    automot_root: pathlib.Path,
    scenario_candidates: List[str],
    option_seed: int,
    max_frames_per_route: int,
) -> Optional[Dict[str, Any]]:
    """把单条 route result 转成一个连续 sequence row。

    v5 的 dataset row 与训练 sampler 单元都是 route。函数只接受 ``status=success`` 且
    XML 存在的 route；可用 ``max_frames_per_route`` 做 smoke 截断，但不能在正式构建中
    删除任意中间坏帧后继续，因为这会改变 memory 的真实前后关系。
    """

    # 只有结构完整的 success route 进入训练。review_required=true 是正常训练样本，
    # 不在这里过滤；真正跳过的只有 noScenarios、异常时长、数据缺失、XML/RGB/meta 缺失。
    if route.get("status") != "success":
        return None
    route_id = str(route.get("route_id", ""))
    if not route_id:
        return None
    run_dir = data_root / scenario / route_id
    xml_path = _xml_path_from_route(route, automot_root)
    if not bool(route.get("xml_available", xml_path is not None)) or xml_path is None or not xml_path.exists():
        return None
    xml_weathers = _load_xml_weathers(xml_path)
    annotations = list(route.get("annotations") or [])
    if max_frames_per_route > 0:
        annotations = annotations[:max_frames_per_route]
    frames: List[Dict[str, Any]] = []
    for ann in annotations:
        frame_row = _build_frame_row(
            ann=ann,
            run_dir=run_dir,
            route_id=route_id,
            scenario_candidates=scenario_candidates,
            option_seed=option_seed,
            xml_weathers=xml_weathers,
            automot_root=automot_root,
        )
        if frame_row is None:
            # v5 的训练单元是整条 route sequence；任一帧证据链缺失都会破坏 memory 轨迹，
            # 所以这里选择丢整条 route，而不是只删中间一帧造成时间跳变。
            return None
        frames.append(frame_row)
    if not frames:
        return None
    return {
        "dataset_version": DATASET_VERSION,
        "scenario": scenario,
        "route_id": route_id,
        "run_dir": str(run_dir),
        "xml_path": str(xml_path) if xml_path else None,
        "xml_available": bool(route.get("xml_available", xml_path is not None)),
        "num_frames": len(frames),
        "frames": frames,
    }


def _split_rows(rows: List[Dict[str, Any]], *, val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按 route 粒度稳定随机切分 train/val。

    同一 route 的所有帧始终位于同一 split，避免相邻画面和 memory 轨迹跨集合泄漏。
    先用 seed 打乱 key，再按原 ``rows`` 顺序输出，兼顾可复现和源文件顺序审计。
    """

    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * max(0.0, min(1.0, val_ratio))))
    val_keys = {(row["scenario"], row["route_id"]) for row in shuffled[:n_val]}
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        if (row["scenario"], row["route_id"]) in val_keys:
            out["split"] = "val"
            val.append(out)
        else:
            out["split"] = "train"
            train.append(out)
    return train, val


def _write_jsonl(path: pathlib.Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """把 route rows 逐行写成 UTF-8 JSONL，并返回实际写入的 route 数量。"""

    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    """执行完整数据构建并返回 summary。

    路径参数既支持绝对路径，也支持相对 AutoMoT 根目录。输出包含 train、val、all 三份
    route-level index 与 summary.json；本函数只写 ``output_dir``，不会修改 collection
    标注和 LEAD 原始数据。
    """

    automot_root = pathlib.Path(args.automot_root).resolve()
    collection_dir = pathlib.Path(args.collection_dir)
    if not collection_dir.is_absolute():
        collection_dir = automot_root / collection_dir
    data_root = pathlib.Path(args.data_root)
    if not data_root.is_absolute():
        data_root = automot_root / data_root
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = automot_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 三类 counter 既用于人工检查样本分布，也能快速发现某个过滤条件意外吃掉全量数据。
    rows: List[Dict[str, Any]] = []
    skip_counter: Counter[str] = Counter()
    rs_counter: Counter[str] = Counter()
    event_counter: Counter[str] = Counter()

    scenario_filter = {s.strip() for s in str(args.scenarios or "").split(",") if s.strip()}
    files = sorted(collection_dir.glob("*_result.json"))
    for path in files:
        scenario = path.name[: -len("_result.json")]
        if scenario == "noScenarios":
            # noScenarios 是用户明确要求排除的收集结果；它可能含可视证据，
            # 但不属于本轮 RS/EVENT OPSD 的训练分布。
            skip_counter["skip_noScenarios_file"] += 1
            continue
        if scenario_filter and scenario not in scenario_filter:
            continue
        result = _read_json(path)
        scenario = str(result.get("scenario") or scenario)
        scenario_candidates = scenario_event_candidates_from_result(result)
        abnormal_skips, missing_skips = _skip_sets(result)
        route_rows = list(result.get("routes") or [])
        if args.max_routes > 0:
            route_rows = route_rows[: args.max_routes]
        for route in route_rows:
            route_id = str(route.get("route_id", ""))
            if route_id in abnormal_skips:
                # lead_video_tools 的异常时长规则在 keyframe_filter 前置产物里已经记录；
                # 这里再次执行，防止长异常 route 混入训练。
                skip_counter["skip_abnormal_duration"] += 1
                continue
            if route_id in missing_skips:
                # 数据结构缺失、RGB/meta/XML 不完整的 route 不训练；这和
                # review_required=true 不同，后者只是需要人工关注但仍有完整证据链。
                skip_counter["skip_data_missing"] += 1
                continue
            row = _build_route_row(
                scenario=scenario,
                route=route,
                data_root=data_root,
                automot_root=automot_root,
                scenario_candidates=scenario_candidates,
                option_seed=int(args.option_seed),
                max_frames_per_route=int(args.max_frames_per_route),
            )
            if row is None:
                skip_counter["skip_empty_or_failed_route"] += 1
                continue
            rows.append(row)
            for frame in row["frames"]:
                rs_counter[str(frame.get("rs_label"))] += 1
                event_counter[str(frame.get("event_label"))] += 1

    # 直到所有 route 构建完成后再切分，保证 val_ratio 以有效 route 为分母。
    train_rows, val_rows = _split_rows(rows, val_ratio=float(args.val_ratio), seed=int(args.seed))
    train_path = output_dir / "train_sequence_index.jsonl"
    val_path = output_dir / "val_sequence_index.jsonl"
    all_path = output_dir / "sequence_index.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    _write_jsonl(all_path, rows)

    summary = {
        "dataset_version": DATASET_VERSION,
        "route_count": len(rows),
        "train_route_count": len(train_rows),
        "val_route_count": len(val_rows),
        "frame_count": sum(len(row.get("frames", [])) for row in rows),
        "skip_distribution": dict(sorted(skip_counter.items())),
        "rs_distribution": dict(sorted(rs_counter.items())),
        "event_distribution": dict(sorted(event_counter.items())),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "all_path": str(all_path),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    """定义并解析数据构建 CLI；``0`` 个数限制均表示不限制，适用于正式全量构建。"""

    p = argparse.ArgumentParser(description="Build SFT v5 RS/EVENT sequence dataset")
    p.add_argument("--automot-root", type=str, default=str(_AUTOMOT_ROOT))
    p.add_argument("--collection-dir", type=str, default="keyframe_filter/collection_output")
    p.add_argument("--data-root", type=str, default="lead_data")
    p.add_argument("--output-dir", type=str, default="checkpoints/sft_v5_data")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--option-seed", type=int, default=20260711)
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--max-frames-per-route", type=int, default=0)
    p.add_argument("--scenarios", type=str, default="", help="comma-separated scenario names for smoke/debug builds")
    return p.parse_args()


def main() -> None:
    """CLI 入口：调用主构建函数，并打印有效 route/frame 规模和 train/val index 路径。"""

    summary = build_dataset(parse_args())
    print(
        "[sft_v5 build] "
        f"routes={summary['route_count']} frames={summary['frame_count']} "
        f"train={summary['train_route_count']} val={summary['val_route_count']}"
    )
    print(f"[sft_v5 build] outputs: {summary['train_path']} | {summary['val_path']}")


if __name__ == "__main__":
    main()
