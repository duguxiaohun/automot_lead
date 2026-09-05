#!/usr/bin/env python3
"""复用已存在的全帧审计缓存，建立 RS/EVENT/scenario/Town 覆盖和轨迹复核索引。

此工具不宣称自动完成视觉审计，不生成 RGB 副本。输出保留源 sheet、逐帧速度、
车道身份以及映射结果；人工确认必须另写 review notes。只扫描已缓存的审计 route，
不是全量 LEAD 数据集。先剔除异常时长 route，再读取标注与 meta。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route
from qwen3vl_local.sft_new_loop_phase3.build_dataset import _event_codes, _last_bypass_frame, _rs_label
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import resolve_context_id
from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapped_contexts
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import load_route_trajectory, label_actions, action_evidence


def compact_route(path: Path, data_root: Path, review_root: Path, with_trajectory: bool) -> dict | None:
    """先验证 route 时长，再将已有审计缓存压缩为逐帧复核记录。"""
    scenario, town, run_id = path.relative_to(review_root).parts[:3]
    run = data_root / scenario / run_id
    if not run.is_dir() or is_abnormal_lead_route(run, scenario)[0]:
        return None
    source = json.loads(path.read_text())
    if source.get("status") != "success":
        return None
    annotations = sorted(source.get("annotations", []), key=lambda row: int(row["frame_id"]))
    gaps = _last_bypass_frame(annotations)
    trajectory = load_route_trajectory(run) if with_trajectory else None
    frames = []
    for ann in annotations:
        fid = int(ann["frame_id"])
        rs, events = _rs_label(ann), _event_codes(ann)
        contexts, mapping = mapped_contexts(scenario, run_id, fid, rs,
                                             str(ann.get("primary_event") or "UNKNOWN"), events)
        row = {"frame_id": fid, "rs": rs, "events": list(events),
               "context": list(contexts), "mapping_evidence": mapping,
               "frames_since_bypass": gaps.get(fid)}
        if trajectory and fid in trajectory.metas:
            meta = trajectory.metas[fid]
            row.update(speed=float(meta.get("speed", 0)), road_id=int(meta.get("road_id", 0)),
                       lane_id=int(meta.get("lane_id", 0)), is_junction=bool(meta.get("is_junction")),
                       theta=float(meta.get("theta", 0)),
                       goal_xy=trajectory.goal_ego_xy(fid),
                       lane_change=trajectory.lane_change(fid),
                       speeds_2s=trajectory._future_speeds(fid, 8))
            signals = trajectory.signals(fid)
            row["action_labels"] = label_actions(signals)
            row["action_evidence"] = action_evidence(signals)
        evidence = ann.get("evidence") or {}
        # 保存已有地图证据的来源，而非由 scenario 名臆造地图结构。
        row["map_evidence"] = {k: v for k, v in evidence.items()
                               if any(word in k for word in ("xodr", "lane_id", "road_id", "signal"))}
        frames.append(row)
    return {"scenario": scenario, "town": town, "route_id": run_id,
            "source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sheets": [str(p) for p in sorted(path.parent.glob("sheets/all_frames_*.jpg"))],
            "frames": frames}


def main() -> None:
    """生成机器扫描报告；明确分开机械覆盖与人工已看 RGB 覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "lead_data")
    parser.add_argument("--review-root", type=Path, default=ROOT / "keyframe_filter/collection_output/phase1_four_question_audit/full_route_rgb_label_review_20260809")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--with-trajectory", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.review_root.glob("*/*/*/route_annotations.json"))
    print(f"[discover] cached audit routes={len(paths)}", flush=True)
    counts, mapping, towns = Counter(), Counter(), Counter()
    facets = defaultdict(list)
    with (args.output_dir / "routes.jsonl").open("w") as handle:
        for index, path in enumerate(paths, 1):
            route = compact_route(path, args.data_root, args.review_root, args.with_trajectory)
            if route:
                handle.write(json.dumps(route, ensure_ascii=False) + "\n")
                s, t = route["scenario"], route["town"]
                counts["routes"] += 1
                towns[f"{s}/{t}"] += 1
                for frame in route["frames"]:
                    counts["frames"] += 1
                    key = f'{s}/{t}/{frame["rs"]}/{"+".join(frame["events"])}'
                    mapping[f'{frame["rs"]}/{"+".join(frame["events"])}/{frame["context"]}'] += 1
                    if len(facets[key]) < 3:
                        facets[key].append({"route_id": route["route_id"], "frame_id": frame["frame_id"]})
            else:
                counts["excluded_routes"] += 1
            if index % 20 == 0 or index == len(paths):
                print(f"[route] {index}/{len(paths)} valid={counts['routes']} {path.parent.name}", flush=True)
    report = {"scope": "existing_full_route_review_cache_only", "counts": dict(counts),
              "scenario_town_counts": dict(towns), "mapping_counts": dict(mapping),
              "facet_examples": dict(facets), "new_manual_rgb_reviews": 0}
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
