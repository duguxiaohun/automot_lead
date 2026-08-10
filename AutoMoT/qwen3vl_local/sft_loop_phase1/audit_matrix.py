#!/usr/bin/env python3
"""生成第一轮四问的场景 × RS × EVENT RGB 审计矩阵。

此工具不把 scenario 名、RS 或 EVENT 自动转换成四个训练标签。它从已有逐帧标注中
分层抽取真实 RGB，写出待人工填写的矩阵和可视证据图；这样不会把 ``PedestrianCrossing``
的无行人进入帧，或 ``U-E7`` 的非灯具路权问题，错误地固化为场景级答案。
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw


_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
if str(_AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTOMOT_ROOT))

from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route  # noqa: E402


QUESTION_KEYS = ("highway", "obstacle", "vulnerable", "traffic_light_abnormal")


@dataclass(frozen=True)
class FrameRef:
    """一个可打开的标注帧及其分层键。"""

    scenario: str
    route_id: str
    town: str
    frame_id: int
    rs: str
    event: str
    rgb_path: pathlib.Path


def _iter_routes_stream(path: pathlib.Path, chunk_bytes: int = 1024 * 1024) -> Iterable[Mapping[str, Any]]:
    """流式读取一个 result 的 ``routes``，不把数 GB 的完整 JSON 装进内存。

    ``collection_output`` 的单文件可超过 2 GB；标准 ``json.load`` 会把它展开成远大于
    文件本身的 Python 对象。这里仅定位顶层 ``routes`` 数组，再逐 route 累积一个完整 JSON
    object。JSON 字符串内的花括号与转义双引号会被正确跳过。
    """

    marker = b'"routes"'
    before_routes = bytearray()
    found_routes = False
    object_buffer = bytearray()
    object_depth = 0
    in_string = False
    escaped = False
    routes_done = False

    with path.open("rb") as handle:
        while not routes_done:
            block = handle.read(chunk_bytes)
            if not block:
                break
            if not found_routes:
                before_routes.extend(block)
                marker_index = before_routes.find(marker)
                if marker_index < 0:
                    # marker 的最长跨 chunk 边界长度为 7；其余内容无需保留。
                    del before_routes[:-len(marker)]
                    continue
                array_index = before_routes.find(b"[", marker_index + len(marker))
                if array_index < 0:
                    continue
                block = bytes(before_routes[array_index + 1 :])
                before_routes.clear()
                found_routes = True

            for byte in block:
                if object_depth == 0:
                    if byte == ord("{"):
                        object_buffer = bytearray((byte,))
                        object_depth = 1
                        in_string = False
                        escaped = False
                    elif byte == ord("]"):
                        routes_done = True
                        break
                    continue

                object_buffer.append(byte)
                if in_string:
                    if escaped:
                        escaped = False
                    elif byte == ord("\\"):
                        escaped = True
                    elif byte == ord('"'):
                        in_string = False
                    continue
                if byte == ord('"'):
                    in_string = True
                elif byte == ord("{"):
                    object_depth += 1
                elif byte == ord("}"):
                    object_depth -= 1
                    if object_depth == 0:
                        try:
                            yield json.loads(object_buffer.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise ValueError(f"invalid route object in {path}: {exc}") from exc
                        object_buffer.clear()
    if not found_routes:
        raise ValueError(f"top-level routes array not found: {path}")
    if object_depth != 0:
        raise ValueError(f"unterminated route object in {path}")


def _rgb_path(run_dir: pathlib.Path, frame_id: int) -> pathlib.Path | None:
    """兼容 LEAD 的常见数字帧名，返回确实存在的 stitched RGB。"""

    rgb_dir = run_dir / "rgb"
    for name in (f"{frame_id:04d}.jpg", f"{frame_id:05d}.jpg", f"{frame_id}.jpg"):
        candidate = rgb_dir / name
        if candidate.exists():
            return candidate
    return None


def _town(annotation: Mapping[str, Any], route: Mapping[str, Any]) -> str:
    """优先使用 XML 实际 town；缺失时保留 UNKNOWN，不从场景名猜。"""

    evidence = annotation.get("evidence") or {}
    return str(evidence.get("xml_town") or route.get("xml_town") or "UNKNOWN")


def _event(annotation: Mapping[str, Any]) -> str:
    """取当前 primary EVENT；不把同帧 secondary overlay 混入组键。"""

    event = annotation.get("primary_event") or (annotation.get("frame_event_annotation") or {}).get("label")
    return str(event or "UNKNOWN")


def _rs(annotation: Mapping[str, Any]) -> str:
    """取当前 primary ROAD_STRUCTURE。"""

    value = annotation.get("primary_road_structure") or (annotation.get("frame_rs_annotation") or {}).get("label")
    return str(value or "UNKNOWN")


def _iter_valid_frames(
    collection_dir: pathlib.Path, data_root: pathlib.Path, scenarios: set[str] | None = None
) -> Iterable[FrameRef]:
    """逐 result 读取可见、非异常时长 route 的所有标注帧。"""

    for result_path in sorted(collection_dir.glob("*_result.json")):
        # 文件名是 collector 的固定 ``<scenario>_result.json`` 约定；不读取顶层 JSON
        # 是为了在 noScenarios 这类 2 GB 文件上仍保持固定内存。
        scenario = result_path.stem.removesuffix("_result")
        if scenarios is not None and scenario not in scenarios:
            continue
        for route in _iter_routes_stream(result_path):
            route_id = str(route.get("route_id") or "")
            if not route_id or str(route.get("status")) == "data_missing_skip":
                continue
            run_dir = data_root / scenario / route_id
            if not run_dir.is_dir() or is_abnormal_lead_route(run_dir, scenario)[0]:
                continue
            for annotation in route.get("annotations", []) or []:
                try:
                    frame_id = int(annotation.get("frame_id"))
                except (TypeError, ValueError):
                    continue
                rgb = _rgb_path(run_dir, frame_id)
                if rgb is None:
                    continue
                yield FrameRef(
                    scenario=scenario,
                    route_id=route_id,
                    town=_town(annotation, route),
                    frame_id=frame_id,
                    rs=_rs(annotation),
                    event=_event(annotation),
                    rgb_path=rgb,
                )


def _evenly_spaced(items: Sequence[FrameRef], count: int) -> List[FrameRef]:
    """按 route/frame 排序后等距抽样，避免只看 event span 的某一端。"""

    ordered = sorted(items, key=lambda x: (x.route_id, x.frame_id))
    if len(ordered) <= count:
        return list(ordered)
    positions = [round(i * (len(ordered) - 1) / max(1, count - 1)) for i in range(count)]
    return [ordered[index] for index in dict.fromkeys(positions)]


def _stratified_sample(
    frames: Sequence[FrameRef], routes_per_town: int, frames_per_route: int
) -> List[FrameRef]:
    """每个 town 先抽多条 route，再从每条 route 抽时间分散帧。

    不能把同一条 route 的若干帧伪装成“多个 id”。先在每个 town 的全部合法 route
    间等距选 ``routes_per_town`` 条，再从每条的全帧序列等距取 ``frames_per_route`` 帧。
    """

    by_town: Dict[str, List[FrameRef]] = defaultdict(list)
    for frame in frames:
        by_town[frame.town].append(frame)
    chosen: List[FrameRef] = []
    for town in sorted(by_town):
        by_route: Dict[str, List[FrameRef]] = defaultdict(list)
        for frame in by_town[town]:
            by_route[frame.route_id].append(frame)
        route_refs = [min(route_frames, key=lambda item: item.frame_id) for route_frames in by_route.values()]
        for route_ref in _evenly_spaced(route_refs, routes_per_town):
            chosen.extend(_evenly_spaced(by_route[route_ref.route_id], frames_per_route))
    return chosen


def _sheet(samples: Sequence[FrameRef], out_path: pathlib.Path, group_name: str) -> None:
    """把每个组的真实 stitched RGB 做成可人工逐格检查的联系表。"""

    tile_w, tile_h, caption_h, cols = 384, 128, 42, 3
    rows = max(1, math.ceil(len(samples) / cols))
    canvas = Image.new("RGB", (cols * tile_w, 34 + rows * (tile_h + caption_h)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), f"{group_name} | sampled real RGB; newest frame labels shown below", fill="black")
    for index, item in enumerate(samples):
        x = (index % cols) * tile_w
        y = 34 + (index // cols) * (tile_h + caption_h)
        try:
            image = Image.open(item.rgb_path).convert("RGB")
            image.thumbnail((tile_w, tile_h))
            tile = Image.new("RGB", (tile_w, tile_h), "black")
            tile.paste(image, ((tile_w - image.width) // 2, (tile_h - image.height) // 2))
            canvas.paste(tile, (x, y))
        except OSError:
            draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline="red", width=2)
            draw.text((x + 6, y + 6), "unreadable RGB", fill="red")
        draw.text((x + 4, y + tile_h + 3), f"{item.town} {item.route_id[-25:]}", fill="black")
        draw.text((x + 4, y + tile_h + 20), f"f={item.frame_id}  {item.rs}/{item.event}", fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def build_audit_matrix(
    *,
    collection_dir: pathlib.Path,
    data_root: pathlib.Path,
    output_dir: pathlib.Path,
    samples_per_town: int,
    frames_per_route: int = 3,
    scenarios: set[str] | None = None,
) -> Dict[str, Any]:
    """生成未定标签矩阵及全部 RGB 证据页，返回 manifest。"""

    groups: Dict[Tuple[str, str, str], List[FrameRef]] = defaultdict(list)
    for frame in _iter_valid_frames(collection_dir, data_root, scenarios):
        groups[(frame.scenario, frame.rs, frame.event)].append(frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix: List[Dict[str, Any]] = []
    for index, ((scenario, rs, event), frames) in enumerate(sorted(groups.items()), 1):
        samples = _stratified_sample(frames, samples_per_town, frames_per_route)
        slug = f"{index:03d}_{scenario}__{rs}__{event}".replace("/", "_")
        sheet_path = output_dir / "sheets" / f"{slug}.jpg"
        _sheet(samples, sheet_path, f"{scenario} | {rs} | {event}")
        matrix.append(
            {
                "scenario": scenario,
                "rs": rs,
                "event": event,
                "frame_count": len(frames),
                "towns_seen": sorted({frame.town for frame in frames}),
                "samples": [
                    {
                        "town": frame.town,
                        "route_id": frame.route_id,
                        "frame_id": frame.frame_id,
                        "rgb_path": str(frame.rgb_path),
                    }
                    for frame in samples
                ],
                "answers": {key: None for key in QUESTION_KEYS},
                "review_status": "pending_visual_review",
                "label_source": "must_be_current_frame_RGB_review; rs_event_only_are_strata_not_answers",
                "evidence_sheet": str(sheet_path),
            }
        )
    manifest = {
        "format": "phase1_four_question_audit_v1",
        "questions": list(QUESTION_KEYS),
        "samples_per_town": samples_per_town,
        "frames_per_route": frames_per_route,
        "scenarios_filter": sorted(scenarios) if scenarios is not None else None,
        "group_count": len(matrix),
        "groups": matrix,
    }
    (output_dir / "phase1_four_question_matrix.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def merge_audit_matrices(input_dirs: Sequence[pathlib.Path], output_dir: pathlib.Path) -> Dict[str, Any]:
    """合并分批审计 manifest，保留各批原始 RGB evidence 路径。

    长达 33 GB 的 collection 输出应分批处理。合并只处理很小的 manifest，不复制图片、
    不改原始标注，也拒绝重复的 ``scenario × RS × EVENT`` 组合。
    """

    groups: List[Dict[str, Any]] = []
    seen = set()
    samples_per_town = set()
    for input_dir in input_dirs:
        matrix_path = input_dir / "phase1_four_question_matrix.json"
        payload = json.loads(matrix_path.read_text(encoding="utf-8"))
        if payload.get("format") != "phase1_four_question_audit_v1":
            raise ValueError(f"unsupported phase1 audit manifest: {matrix_path}")
        samples_per_town.add(int(payload.get("samples_per_town", 0)))
        for group in payload.get("groups", []):
            key = (str(group.get("scenario")), str(group.get("rs")), str(group.get("event")))
            if key in seen:
                raise ValueError(f"duplicate audit group while merging: {key}")
            seen.add(key)
            groups.append(group)
    if len(samples_per_town) > 1:
        raise ValueError(f"cannot merge different --samples-per-town values: {sorted(samples_per_town)}")
    groups.sort(key=lambda item: (str(item["scenario"]), str(item["rs"]), str(item["event"])))
    manifest = {
        "format": "phase1_four_question_audit_v1",
        "questions": list(QUESTION_KEYS),
        "samples_per_town": next(iter(samples_per_town), 0),
        "merged_from": [str(path) for path in input_dirs],
        "group_count": len(groups),
        "groups": groups,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase1_four_question_matrix.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数。"""

    parser = argparse.ArgumentParser(description="Build RGB-first four-question audit matrix")
    parser.add_argument("--collection-dir", default=str(_AUTOMOT_ROOT / "keyframe_filter" / "collection_output"))
    parser.add_argument("--data-root", default=str(_AUTOMOT_ROOT / "lead_data"))
    parser.add_argument(
        "--output-dir",
        default=str(_AUTOMOT_ROOT / "keyframe_filter" / "collection_output" / "phase1_four_question_audit"),
    )
    parser.add_argument("--samples-per-town", type=int, default=3)
    parser.add_argument("--frames-per-route", type=int, default=3)
    parser.add_argument(
        "--scenarios",
        default="",
        help="comma-separated scenario subset; run subsets into different output dirs for resumable audits",
    )
    parser.add_argument(
        "--merge-inputs",
        default="",
        help="comma-separated batch output dirs; merge their manifests instead of reading collection_output",
    )
    return parser.parse_args()


def main() -> None:
    """CLI 入口。"""

    args = parse_args()
    if args.merge_inputs:
        input_dirs = [pathlib.Path(part.strip()) for part in str(args.merge_inputs).split(",") if part.strip()]
        if not input_dirs:
            raise ValueError("--merge-inputs did not contain a directory")
        manifest = merge_audit_matrices(input_dirs, pathlib.Path(args.output_dir))
        print(f"phase1 audit merge: groups={manifest['group_count']} output={args.output_dir}")
        return
    if args.samples_per_town <= 0 or args.frames_per_route <= 0:
        raise ValueError("--samples-per-town and --frames-per-route must be positive")
    manifest = build_audit_matrix(
        collection_dir=pathlib.Path(args.collection_dir),
        data_root=pathlib.Path(args.data_root),
        output_dir=pathlib.Path(args.output_dir),
        samples_per_town=args.samples_per_town,
        frames_per_route=args.frames_per_route,
        scenarios={part.strip() for part in str(args.scenarios).split(",") if part.strip()} or None,
    )
    print(f"phase1 audit matrix: groups={manifest['group_count']} output={args.output_dir}")


if __name__ == "__main__":
    main()
