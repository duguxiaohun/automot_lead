#!/usr/bin/env python3
"""构建全量 4Hz action 索引，先排除异常 route，按物理路线分 train/val/test。"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route
from qwen3vl_local.action_prior.contracts import digest


def route_group(scenario, run):
    """同一 Town/route 不同 Rep/采集时间必须在同 split。"""
    stem = re.sub(r"_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$", "", run)
    stem = re.sub(r"_route0$", "", stem)
    stem = re.sub(r"_Rep\d+_", "_", stem)
    return f"{scenario}/{stem}"


def split_for(key, seed, val_ratio=0.1, test_ratio=0.1):
    """哈希划分不受遍历次序和候选数量影响。"""
    n = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16) / 2**64
    return (
        "test" if n < test_ratio else "val" if n < test_ratio + val_ratio else "train"
    )


def frame_ids(path, suffix):
    """按真实文件编号交集检查缺帧。"""
    return {int(p.stem) for p in path.glob("*" + suffix) if p.stem.isdigit()}


def main():
    """只构建引用索引，不复制 RGB/点云或读取未来状态作为条件。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="lead_data")
    p.add_argument("--output-dir", default="checkpoints/action_prior_data")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    args = p.parse_args()
    if (
        args.stride < 1
        or min(args.val_ratio, args.test_ratio) <= 0
        or args.val_ratio + args.test_ratio >= 1
    ):
        p.error("positive stride and nonempty train/val/test fractions required")
    root, out = Path(args.data_root).resolve(), Path(args.output_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / f"{s}.jsonl").exists() for s in ("train", "val", "test")):
        raise FileExistsError(
            f"{out}: use a new data directory; existing index is never overwritten"
        )
    print("[discover] discovering LEAD routes", flush=True)
    routes = sorted(
        p
        for s in root.iterdir()
        if s.is_dir()
        for p in s.iterdir()
        if p.is_dir()
        and (p / "rgb").is_dir()
        and (p / "metas").is_dir()
        and (p / "lidar").is_dir()
    )
    print(f"[discover] {len(routes)} routes", flush=True)
    counts, skipped, scenarios = Counter(), Counter(), Counter()
    handles = {
        s: (out / f"{s}.jsonl.tmp").open("w", encoding="utf-8")
        for s in ("train", "val", "test")
    }
    try:
        for i, route in enumerate(routes):
            scenario, run = route.parent.name, route.name
            abnormal, _ = is_abnormal_lead_route(route, scenario)
            print(
                f'\r[routes] {i+1}/{len(routes)} [{"=" * int(20*(i+1)/max(1,len(routes))):20s}] {scenario}/{run}',
                end="",
                flush=True,
            )
            if abnormal:
                skipped["abnormal_duration"] += 1
                continue
            common = (
                frame_ids(route / "rgb", ".jpg")
                & frame_ids(route / "metas", ".pkl")
                & frame_ids(route / "lidar", ".laz")
            )
            group = route_group(scenario, run)
            split = split_for(group, args.seed, args.val_ratio, args.test_ratio)
            for anchor in sorted(common)[:: args.stride]:
                # 旧 build_clip 仍取 +1/+2s 文件；保留其完整时窗并支持首帧 left-pad。
                if not all(k in common for k in range(max(0, anchor - 3), anchor + 9)):
                    skipped["missing_window"] += 1
                    continue
                row = dict(
                    schema="action_prior_data_v1",
                    scenario=scenario,
                    run_id=run,
                    route_id=run,
                    route_dir=f"{scenario}/{run}",
                    route_group=group,
                    split=split,
                    anchor=anchor,
                    rgb_frame_count=4,
                    rgb_frame_step=1,
                    bev_frame_count=1,
                    bev_frame_step=1,
                    frame_interval_s=0.25,
                    target_point_lookahead_s=1.0,
                    next_target_point_lookahead_s=2.0,
                    tp_mode="route_lookahead",
                    tp_min_lookahead_m=5.0,
                    use_final_goal=True,
                    future_waypoint_indices=[5, 10, 15, 20, 25, 30, 35, 40],
                )
                handles[split].write(json.dumps(row) + "\n")
                counts[split] += 1
                scenarios[f"{split}/{scenario}"] += 1
    finally:
        for f in handles.values():
            f.close()
        print()
    if any(counts[s] == 0 for s in handles):
        raise ValueError(
            f"empty split: {dict(counts)}; unfinished .tmp indices retained for diagnosis"
        )
    for s in handles:
        (out / f"{s}.jsonl.tmp").replace(out / f"{s}.jsonl")
    manifest = dict(
        schema="action_prior_data_v1",
        config=vars(args),
        counts=dict(counts),
        scenario_counts=dict(scenarios),
        skipped=dict(skipped),
        discovered_routes=len(routes),
        full_data=args.stride == 1,
        split_unit="scenario/Town/route, grouped across repetitions",
    )
    manifest["identity"] = digest(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
