#!/usr/bin/env python3
"""为 LeadMoT decoder 训练构建轻量 JSONL 索引。

JSONL 只保存 route 目录和 anchor 帧引用。RGB、LiDAR、BEV、Qwen prefill
这些重工作都留给 train_v1.py，因此这一步很快，也适合反复重建。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def _has_route_layout(path: Path) -> bool:
    """判断 ``path`` 本身是否是 LEAD route 目录。"""
    return (path / "rgb").is_dir() and (path / "metas").is_dir() and (path / "lidar").is_dir()


def _iter_route_dirs(data_root: Path) -> Iterable[Path]:
    """从 route、scenario 或 data root 路径中枚举 route 目录。"""
    if _has_route_layout(data_root):
        yield data_root
        return
    for scenario_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if _has_route_layout(scenario_dir):
            yield scenario_dir
            continue
        for route_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            if _has_route_layout(route_dir):
                yield route_dir


def _count_frames(route_dir: Path) -> int:
    """统计 rgb/meta/lidar 三个目录共同拥有的同步帧范围。"""
    rgb_count = len(list((route_dir / "rgb").glob("*.jpg")))
    meta_count = len(list((route_dir / "metas").glob("*.pkl")))
    lidar_count = len(list((route_dir / "lidar").glob("*.laz")))
    return min(rgb_count, meta_count, lidar_count)


def _route_name(data_root: Path, route_dir: Path) -> tuple[str, str]:
    """为 JSONL 行生成稳定的 ``scenario`` 和 ``route_id`` 字段。"""
    try:
        parts = route_dir.relative_to(data_root).parts
    except ValueError:
        parts = route_dir.parts
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return route_dir.parent.name, route_dir.name


def _choose_samples_by_route(samples: list[dict], target_total: int, rng: random.Random) -> list[dict]:
    """在单个 scenario 内按 route 均衡抽样。

    ``target_total <= 0`` 对齐 GoalGen 语义：保留所有有效 anchor。
    """
    if target_total <= 0 or len(samples) <= target_total:
        chosen = list(samples)
        rng.shuffle(chosen)
        return chosen

    buckets: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        buckets[str(sample["route_id"])].append(sample)

    # 先给每条 route 分配均匀 quota，再用剩余 anchor 补齐；
    # 这样小 debug 集不会被某条很长的 route 占满。
    chosen: list[dict] = []
    chosen_ids: set[int] = set()
    per_bucket = max(1, target_total // max(1, len(buckets)))
    for bucket_samples in buckets.values():
        picked = rng.sample(bucket_samples, per_bucket) if len(bucket_samples) > per_bucket else list(bucket_samples)
        for sample in picked:
            chosen.append(sample)
            chosen_ids.add(id(sample))

    if len(chosen) < target_total:
        remaining = [sample for sample in samples if id(sample) not in chosen_ids]
        chosen.extend(rng.sample(remaining, min(target_total - len(chosen), len(remaining))))
    elif len(chosen) > target_total:
        chosen = rng.sample(chosen, target_total)

    rng.shuffle(chosen)
    return chosen


def _split_train_val_by_route(samples: list[dict], val_ratio: float, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """按整条 route 切 train/val，而不是按 anchor 切。

    同一条 CARLA route 的相邻 anchor 共享大部分输入帧；
    如果按帧切分，近重复样本会泄漏到 validation。
    """
    by_route: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        route_key = f"{sample.get('scenario')}::{sample.get('route_id')}"
        by_route[route_key].append(sample)

    route_keys = sorted(by_route)
    rng.shuffle(route_keys)
    val_count = int(round(len(route_keys) * val_ratio)) if route_keys and val_ratio > 0 else 0
    if len(route_keys) > 1 and val_ratio > 0 and val_count == 0:
        val_count = 1
    if len(route_keys) > 1:
        val_count = min(val_count, len(route_keys) - 1)
    else:
        val_count = 0
    val_routes = set(route_keys[:val_count])

    train: list[dict] = []
    val: list[dict] = []
    for route_key, rows in by_route.items():
        (val if route_key in val_routes else train).extend(rows)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _build_samples(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """构建 JSONL 行和 stats，不加载重型 RGB/LiDAR 张量。"""
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    # 最早 anchor 必须留出足够历史帧，供 RGB 和 BEV history window 使用。
    history_margin = max(
        (args.rgb_frame_count - 1) * args.rgb_frame_step,
        (args.bev_frame_count - 1) * args.bev_frame_step,
    )
    # 这个 margin 供 target_point / next_target_point 使用的未来落盘帧使用。
    # future_waypoint_indices 索引的是 meta["future_positions"]，不是 route 帧文件。
    future_frame_margin = max(
        int(math.ceil(args.target_point_lookahead_s / args.frame_interval_s)),
        int(math.ceil(args.next_target_point_lookahead_s / args.frame_interval_s)),
    )
    rng = random.Random(args.seed)
    candidates_by_scenario: dict[str, list[dict]] = defaultdict(list)
    scenario_route_counts: Counter[str] = Counter()
    route_count = 0
    skipped_routes = 0

    for route_dir in _iter_route_dirs(data_root):
        route_count += 1
        frame_count = _count_frames(route_dir)
        first_anchor = max(history_margin, args.min_anchor)
        last_anchor_exclusive = frame_count - future_frame_margin
        if last_anchor_exclusive <= first_anchor:
            skipped_routes += 1
            continue

        # LEAD 数据是 4Hz，``stride=5`` 大约表示相邻 anchor 间隔 1 秒。
        anchors = list(range(first_anchor, last_anchor_exclusive, max(1, args.stride)))
        scenario, route_id = _route_name(data_root, route_dir)
        scenario_route_counts[scenario] += 1
        # 不再做 lzma/pickle 预校验：每 anchor 6 次解压在大数据集上让构建从分钟级
        # 变成几小时；train_v1 已经有 DDP-safe 占位 loss 兜底坏样本，预校验不再必要。
        route_candidates = [
            {
                "schema_version": 1,
                "scenario": scenario,
                "route_id": route_id,
                "route_dir": str(route_dir),
                "anchor": anchor,
                "frame_interval_s": args.frame_interval_s,
                "rgb_frame_count": args.rgb_frame_count,
                "rgb_frame_step": args.rgb_frame_step,
                "bev_frame_count": args.bev_frame_count,
                "bev_frame_step": args.bev_frame_step,
                "target_point_lookahead_s": args.target_point_lookahead_s,
                "next_target_point_lookahead_s": args.next_target_point_lookahead_s,
                "future_waypoint_indices": args.future_waypoint_indices,
            }
            for anchor in anchors
        ]
        candidates_by_scenario[scenario].extend(route_candidates)

    samples: list[dict] = []
    scenario_stats: dict[str, dict] = {}
    target_per_scenario = 50 if args.dry_run else args.samples_per_scenario
    for scenario, candidates in sorted(candidates_by_scenario.items()):
        chosen = _choose_samples_by_route(candidates, target_per_scenario, rng)
        by_route = Counter(str(sample["route_id"]) for sample in chosen)
        scenario_stats[scenario] = {
            "routes": int(scenario_route_counts[scenario]),
            "candidates": len(candidates),
            "chosen": len(chosen),
            "chosen_by_route": dict(sorted(by_route.items())),
        }
        print(
            f"[scenario] {scenario:42s} routes={scenario_route_counts[scenario]:4d} "
            f"candidates={len(candidates):7d} chosen={len(chosen):7d}"
        )
        samples.extend(chosen)
    rng.shuffle(samples)
    stats = {
        "config": vars(args),
        "data_root": str(data_root),
        "route_count": route_count,
        "skipped_routes": skipped_routes,
        "samples": len(samples),
        "scenario_stats": scenario_stats,
    }
    print(json.dumps({k: v for k, v in stats.items() if k != "scenario_stats"}, ensure_ascii=False))
    return samples, stats


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """用 UTF-8 newline 写出确定性 JSONL 行。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    """解析数据索引构建 CLI 参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="LEAD data root or one route directory.")
    parser.add_argument("--output-dir", default="checkpoints/leadmot_v1_data")
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stride", type=int, default=5, help="Default 5 keeps anchors roughly 1 second apart for LEAD 4Hz data.")
    parser.add_argument("--samples-per-scenario", type=int, default=0, help="0 keeps all valid anchors per scenario; positive values sample a route-balanced subset.")
    parser.add_argument("--dry-run", action="store_true", help="Keep at most 50 samples per scenario and still write train/val/stats.")
    parser.add_argument("--min-anchor", type=int, default=0)
    parser.add_argument("--frame-interval-s", type=float, default=0.25)
    parser.add_argument("--rgb-frame-count", type=int, default=4)
    parser.add_argument("--rgb-frame-step", type=int, default=1)
    parser.add_argument("--bev-frame-count", type=int, default=1)
    parser.add_argument("--bev-frame-step", type=int, default=1)
    parser.add_argument("--target-point-lookahead-s", type=float, default=1.5)
    parser.add_argument("--next-target-point-lookahead-s", type=float, default=3.0)
    parser.add_argument(
        "--future-waypoint-indices",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30, 35, 40],
        help="Indices into meta['future_positions']; LEAD default is 8 points over 2 seconds.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI 入口：构建 train/val JSONL 和 stats.json。"""
    args = parse_args()
    if not 0.0 <= args.val_ratio < 0.5:
        raise ValueError("--val-ratio must be in [0, 0.5)")
    samples, stats = _build_samples(args)
    rng = random.Random(args.seed)
    train_rows, val_rows = _split_train_val_by_route(samples, args.val_ratio, rng)

    output_dir = Path(args.output_dir)
    _write_jsonl(output_dir / "train.jsonl", train_rows)
    _write_jsonl(output_dir / "val.jsonl", val_rows)
    stats.update(
        {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
        }
    )
    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote train={len(train_rows)} val={len(val_rows)} to {output_dir}")


if __name__ == "__main__":
    main()
