#!/usr/bin/env python3
"""从 LEAD metas/bboxes 客观重建高速切入窗口，生成 ``highway_ue3_rgb_decisions_v2.jsonl``。

2026-10-30 审计发现：v1 决策只覆盖 4 条路线，其中训练正例全部来自 Town13 两条左侧
匝道汇入路线，而测试正例是 Town12 右侧跨虚线切入；另外 64 条训练 HighwayCutIn 路线
从未逐帧标定。逐帧回放 metas 发现每条路线都有 ``cut_in_actors_ids`` 指向的脚本化切入
车，其 ego 坐标系横向位置 ``y``、车身偏航角 ``yaw`` 和 ``lane_id`` 能客观给出切入起止：

- 起点（onset）：切入车已在相邻车道边缘（``|y| < ONSET_MAX_ABS_Y``），车头朝 ego 车道
  偏转（``yaw`` 朝向 ego 方向 ≥ ``ONSET_MIN_YAW`` 度），且最近 4 帧横向距离在缩小；
- 终点（settle）：切入车已居中（``|y| < SETTLE_MAX_ABS_Y``）且车身摆正
  （``|yaw| < SETTLE_MAX_YAW``），之后是普通稳定前车，按提示词口径回到 NO。

本工具只提出窗口并附带客观轨迹；窗口在 2026-10-30 已对 14 条路线逐帧 RGB 抽查确认
（见 ``HIGHWAY_UE3_OBJECTIVE_SPANS_20261030.md``）。它不读取 scenario 名或 R3 自动造正例：
没有切入车轨迹、轨迹不满足起止条件的路线不会生成 YES。测试路线保持 v1 决策逐字不变，
其客观窗口只写入 trace 作为待审记录，不改主评分。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import lzma
import math
import os
import pathlib
import pickle
from typing import Any, Dict, List, Optional, Sequence

from qwen3vl_local.sft_new_loop_phase2.highway_ue3_audit import load_highway_ue3_decisions

SCENARIO = "HighwayCutIn"
FORMAT = "sft_new_loop_phase2_highway_ue3_rgb_decisions_v2"
OBJECTIVE_SOURCE = "lead_metas_bboxes_cut_in_actor_trajectory_20261030"
# 与 build_dataset / 2026-08 采集过滤一致：超过 360 帧的 HighwayCutIn 录制为异常时长。
MAX_FRAMES = 360
MAX_ACTOR_X = 45.0
ONSET_MAX_ABS_Y = 3.25
ONSET_MIN_YAW = 5.0
ONSET_MIN_DY = 0.12
ONSET_LOOKBACK = 4
SETTLE_MAX_ABS_Y = 0.6
SETTLE_MAX_YAW = 4.0
MAX_SPAN_WITHOUT_SETTLE = 20
CONTROL_FRAMES = 8
LANE_WIDTH_FALLBACK = 3.5


def load_pickle(path: pathlib.Path) -> Any:
    """LEAD metas/bboxes 是 xz 压缩 pickle；个别旧文件为裸 pickle。"""

    try:
        with lzma.open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception:
        with path.open("rb") as handle:
            return pickle.load(handle)


def stable_unit(value: str) -> float:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16) / float(16 ** 12)


def route_split(route_id: str, *, seed: int, test_ratio: float, val_ratio: float) -> str:
    """与 ``build_dataset._split`` 完全一致的 route-disjoint 划分。"""

    value = stable_unit(f"{seed}:{SCENARIO}:{route_id}")
    return "test" if value < test_ratio else "val" if value < test_ratio + val_ratio else "train"


def actor_track(route_dir: pathlib.Path) -> List[Optional[Dict[str, Any]]]:
    """逐帧读取 ``cut_in_actors_ids`` 指向车辆在 ego 坐标系中的位置/朝向/车道。"""

    frames = sorted(int(p.stem) for p in (route_dir / "metas").glob("*.pkl") if p.stem.isdigit())
    track: List[Optional[Dict[str, Any]]] = []
    for frame in frames:
        meta = load_pickle(route_dir / "metas" / f"{frame:04d}.pkl")
        ids = list(meta.get("cut_in_actors_ids") or [])
        if not ids:
            track.append(None)
            continue
        boxes = load_pickle(route_dir / "bboxes" / f"{frame:04d}.pkl")
        actor = next((b for b in boxes if b.get("id") == ids[0]), None)
        if actor is None:
            track.append(None)
            continue
        track.append(
            {
                "f": frame,
                "x": float(actor["position"][0]),
                "y": float(actor["position"][1]),
                "yaw": math.degrees(float(actor["yaw"])),
                "lane": actor.get("lane_id"),
                "road": actor.get("road_id"),
                "ego_lane": meta.get("ego_lane_id"),
                "ego_road": meta.get("road_id"),
                "lane_width": float(meta.get("ego_lane_width") or LANE_WIDTH_FALLBACK),
            }
        )
    return track


def detect_span(track: Sequence[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """按客观规则给出 side/onset/settle/lane_change；无切入车或无起点时 onset=None。"""

    pts = [t for t in track if t and 0.0 < t["x"] < MAX_ACTOR_X]
    result: Dict[str, Any] = {"side": None, "onset": None, "settle": None, "lane_change": None, "last_tracked": None}
    if not pts:
        return result
    side = 0
    for t in pts:
        if t["x"] < 30.0 and abs(t["y"]) > 2.0:
            side = 1 if t["y"] > 0 else -1
            break
    if side == 0:
        return result
    result["side"] = "right" if side > 0 else "left"
    result["last_tracked"] = pts[-1]["f"]
    by_frame = {t["f"]: t for t in pts}
    onset = settle = None
    for t in pts:
        ay = abs(t["y"])
        toward = t["yaw"] * (-side)  # 正值表示车头朝 ego 车道偏转
        prev = [by_frame[g] for g in range(t["f"] - ONSET_LOOKBACK, t["f"]) if g in by_frame]
        dy = (abs(prev[0]["y"]) - ay) if prev else 0.0
        if onset is None and ay < ONSET_MAX_ABS_Y and toward >= ONSET_MIN_YAW and dy > ONSET_MIN_DY:
            onset = t["f"]
        if onset is not None and settle is None and t["f"] > onset and ay < SETTLE_MAX_ABS_Y and abs(t["yaw"]) < SETTLE_MAX_YAW:
            settle = t["f"]
            break
    result.update(onset=onset, settle=settle)
    if onset is not None:
        result["lane_change"] = next(
            (t["f"] for t in pts if t["f"] >= onset and t["lane"] == t["ego_lane"] and t["road"] == t["ego_road"]),
            None,
        )
        result["y_at_onset"] = round(by_frame[onset]["y"], 2)
        result["yaw_at_onset"] = round(by_frame[onset]["yaw"], 1)
    return result


def span_end(detection: Dict[str, Any]) -> Optional[int]:
    """settle 缺失（录制在切入完成前结束）时，窗口最多延续 MAX_SPAN_WITHOUT_SETTLE 帧。"""

    onset = detection.get("onset")
    if onset is None:
        return None
    if detection.get("settle") is not None:
        return int(detection["settle"])
    return min(int(detection["last_tracked"]), int(onset) + MAX_SPAN_WITHOUT_SETTLE)


def _row(route_id: str, start: int, end: int, answer: str, split: str, cue: str, note: str, **extra: Any) -> Dict[str, Any]:
    row = {
        "scenario": SCENARIO,
        "route_id": route_id,
        "start_frame": int(start),
        "end_frame": int(end),
        "ue3_answer": answer,
        "expected_split": split,
        "rgb_review": "HIGHWAY_UE3_OBJECTIVE_SPANS_20261030.md (objective cut-in actor trajectory + frame-by-frame RGB spot checks)",
        "visible_cue": cue,
        "boundary_note": note,
        "objective_source": OBJECTIVE_SOURCE,
    }
    row.update(extra)
    return row


def propose(data_root: pathlib.Path, *, seed: int, test_ratio: float, val_ratio: float) -> Dict[str, Any]:
    """扫描全部 HighwayCutIn 路线，返回客观窗口与建议行。"""

    routes = sorted(p for p in (data_root / SCENARIO).iterdir() if p.is_dir())
    trace: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    unlabeled_test: List[Dict[str, Any]] = []
    for route_dir in routes:
        frames = len(list((route_dir / "metas").glob("*.pkl")))
        if frames == 0 or frames > MAX_FRAMES:
            trace[route_dir.name] = {"frames": frames, "skipped": "abnormal_duration_or_missing_metas"}
            continue
        split = route_split(route_dir.name, seed=seed, test_ratio=test_ratio, val_ratio=val_ratio)
        detection = detect_span(actor_track(route_dir))
        end = span_end(detection)
        trace[route_dir.name] = {"frames": frames, "split": split, **detection, "end": end}
        if detection["onset"] is None or end is None:
            continue
        onset = int(detection["onset"])
        side = detection["side"]
        cue = (
            f"scripted cut-in vehicle on the {side} side turns toward ego's lane "
            f"(|y| {abs(detection['y_at_onset']):.1f} m, yaw {abs(detection['yaw_at_onset']):.0f} deg at onset) "
            "and crosses the lane divider into ego's corridor"
        )
        note = (
            f"onset={onset} settle={detection['settle']} lane_change={detection['lane_change']}; "
            "frames before onset show the vehicle parallel in its own lane; after settle it is an ordinary lead"
        )
        if split == "test":
            unlabeled_test.append({"route_id": route_dir.name, "onset": onset, "end": end, **detection})
            continue
        objective = {k: detection[k] for k in ("side", "onset", "settle", "lane_change", "y_at_onset", "yaw_at_onset")}
        rows.append(_row(route_dir.name, onset, end, "YES", split, cue, note, true_rs_override="R3", objective=objective))
        if onset - 1 >= max(0, onset - CONTROL_FRAMES):
            rows.append(_row(route_dir.name, max(0, onset - CONTROL_FRAMES), onset - 1, "NO", split,
                             "vehicle still parallel in the adjacent lane without heading change",
                             "negative boundary control immediately before the objective onset"))
        if end + 1 <= min(frames - 1, end + CONTROL_FRAMES):
            rows.append(_row(route_dir.name, end + 1, min(frames - 1, end + CONTROL_FRAMES), "NO", split,
                             "vehicle settled ahead as an ordinary lead with no further lateral entry",
                             "negative boundary control immediately after the objective settle"))
    return {"trace": trace, "rows": rows, "unlabeled_test_windows": unlabeled_test}


def merge_with_v1(v1_path: pathlib.Path, proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
    """测试路线保留 v1 原行；其余 v1 行被客观窗口取代（含 Town13 左侧匝道改为 NO 对照）。"""

    v1_rows = [json.loads(line) for line in v1_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [r for r in v1_rows if r.get("expected_split") == "test"]
    superseded = [r for r in v1_rows if r.get("expected_split") != "test"]
    objective_routes = {r["route_id"] for r in proposal["rows"]}
    rows = list(kept)
    for old in superseded:
        if old["route_id"] in objective_routes:
            continue  # 该路线由客观窗口重新给出 YES/NO 行
        if old["ue3_answer"] == "YES":
            verdict = proposal["trace"].get(old["route_id"]) or {}
            if verdict.get("onset") is not None or "side" not in verdict:
                raise ValueError("v1 YES route without an objective no-entry verdict cannot be downgraded: " + old["route_id"])
            # 2026-10-30 客观轨迹：Town13 1586_x 的匝道车全程停留在左侧相邻车道
            # （|y|≥3.4 m，lane_id 从未等于 ego 车道），按提示词口径属于相邻车道并行，不是切入。
            rows.append(_row(old["route_id"], old["start_frame"], old["end_frame"], "NO", old["expected_split"],
                             "merging vehicle converges from the left ramp but stays in the adjacent lane (|y| >= 3.4 m, never ego lane)",
                             "v1 YES span superseded by objective trajectory; kept only as adjacent-lane hard negative",
                             superseded_v1_answer="YES"))
        else:
            rows.append(dict(old, boundary_note=old.get("boundary_note", "") + " (retained from v1)"))
    rows.extend(proposal["rows"])
    rows.sort(key=lambda r: (r["expected_split"], r["route_id"], r["start_frame"]))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    this = pathlib.Path(__file__).resolve()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--v1", default=str(this.with_name("highway_ue3_rgb_decisions_v1.jsonl")))
    parser.add_argument("--output", default=str(this.with_name("highway_ue3_rgb_decisions_v2.jsonl")))
    parser.add_argument("--trace-output", default="")
    parser.add_argument("--split-seed", type=int, default=20260819)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args(argv)
    proposal = propose(pathlib.Path(args.data_root).expanduser().resolve(), seed=args.split_seed,
                       test_ratio=args.test_ratio, val_ratio=args.val_ratio)
    rows = merge_with_v1(pathlib.Path(args.v1), proposal)
    output = pathlib.Path(args.output)
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    _, report = load_highway_ue3_decisions(output)  # 重叠/字段合法性由正式加载器把关
    summary = {
        "format": FORMAT,
        "output": str(output),
        "decision_report": report,
        "routes_scanned": len(proposal["trace"]),
        "objective_rows": len(proposal["rows"]),
        "unlabeled_test_windows": proposal["unlabeled_test_windows"],
    }
    if args.trace_output:
        pathlib.Path(args.trace_output).write_text(json.dumps({**summary, "trace": proposal["trace"]}, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "unlabeled_test_windows"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
