#!/usr/bin/env python3
"""Build lightweight JSONL indexes for LeadMoT decoder training.

The JSONL keeps only route directory + anchor frame references. Heavy RGB,
LiDAR, BEV and Qwen prefill work stays in train_v1.py so this step is fast and
safe to rerun.
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import pickle
import random
from pathlib import Path
from typing import Iterable


def _has_route_layout(path: Path) -> bool:
    return (path / "rgb").is_dir() and (path / "metas").is_dir() and (path / "lidar").is_dir()


def _iter_route_dirs(data_root: Path) -> Iterable[Path]:
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
    rgb_count = len(list((route_dir / "rgb").glob("*.jpg")))
    meta_count = len(list((route_dir / "metas").glob("*.pkl")))
    lidar_count = len(list((route_dir / "lidar").glob("*.laz")))
    return min(rgb_count, meta_count, lidar_count)


def _frame_paths(route_dir: Path, frame_idx: int) -> tuple[Path, Path, Path]:
    stem = f"{frame_idx:04d}"
    return (
        route_dir / "rgb" / f"{stem}.jpg",
        route_dir / "metas" / f"{stem}.pkl",
        route_dir / "lidar" / f"{stem}.laz",
    )


def _load_meta(meta_path: Path) -> tuple[dict | None, str]:
    try:
        with lzma.open(meta_path, "rb") as f:
            meta = pickle.load(f)
    except Exception as exc:
        return None, f"bad meta {meta_path.name}: {exc}"
    if not isinstance(meta, dict):
        return None, f"bad meta {meta_path.name}: expected dict, got {type(meta).__name__}"
    return meta, ""


def _sample_is_readable(route_dir: Path, anchor: int, frame_count: int, args: argparse.Namespace) -> tuple[bool, str]:
    history_margin = max(
        (max(1, args.rgb_frame_count) - 1) * max(1, args.rgb_frame_step),
        (max(1, args.bev_frame_count) - 1) * max(1, args.bev_frame_step),
    )
    actual_start = max(0, anchor - history_margin)
    tp_offset = int(round(float(args.target_point_lookahead_s) / max(1e-6, float(args.frame_interval_s))))
    ntp_offset = int(round(float(args.next_target_point_lookahead_s) / max(1e-6, float(args.frame_interval_s))))

    meta_cache: dict[int, dict] = {}
    for frame_idx in range(actual_start, anchor + 1):
        rgb_path, meta_path, lidar_path = _frame_paths(route_dir, frame_idx)
        for path in (rgb_path, meta_path, lidar_path):
            if not path.exists():
                return False, f"missing {path.relative_to(route_dir)}"
        meta, reason = _load_meta(meta_path)
        if meta is None:
            return False, reason
        for key in ("pos_global", "theta"):
            if key not in meta:
                return False, f"missing {meta_path.name}[{key!r}]"
        meta_cache[frame_idx] = meta

        for future_idx in {
            min(max(0, frame_idx + max(0, tp_offset)), frame_count - 1),
            min(max(0, frame_idx + max(0, ntp_offset)), frame_count - 1),
        }:
            if future_idx in meta_cache:
                continue
            future_meta_path = route_dir / "metas" / f"{future_idx:04d}.pkl"
            if not future_meta_path.exists():
                return False, f"missing {future_meta_path.relative_to(route_dir)}"
            future_meta, reason = _load_meta(future_meta_path)
            if future_meta is None:
                return False, reason
            if "pos_global" not in future_meta:
                return False, f"missing {future_meta_path.name}['pos_global']"
            meta_cache[future_idx] = future_meta

    anchor_meta = meta_cache.get(anchor)
    if anchor_meta is None:
        return False, "anchor meta not loaded"
    for key in ("route", "future_positions"):
        if key not in anchor_meta:
            return False, f"missing anchor meta[{key!r}]"
    return True, ""


def _route_name(data_root: Path, route_dir: Path) -> tuple[str, str]:
    try:
        parts = route_dir.relative_to(data_root).parts
    except ValueError:
        parts = route_dir.parts
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return route_dir.parent.name, route_dir.name


def _build_samples(args: argparse.Namespace) -> list[dict]:
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    history_margin = max(
        (args.rgb_frame_count - 1) * args.rgb_frame_step,
        (args.bev_frame_count - 1) * args.bev_frame_step,
    )
    # This margin is for future saved frames used by target_point / next_target_point.
    # future_waypoint_indices index meta["future_positions"], not route frame files.
    future_frame_margin = max(
        int(math.ceil(args.target_point_lookahead_s / args.frame_interval_s)),
        int(math.ceil(args.next_target_point_lookahead_s / args.frame_interval_s)),
    )
    rng = random.Random(args.seed)
    samples: list[dict] = []
    route_count = 0
    skipped_routes = 0
    skipped_samples = 0

    for route_dir in _iter_route_dirs(data_root):
        route_count += 1
        frame_count = _count_frames(route_dir)
        first_anchor = max(history_margin, args.min_anchor)
        last_anchor_exclusive = frame_count - future_frame_margin
        if last_anchor_exclusive <= first_anchor:
            skipped_routes += 1
            continue

        anchors = list(range(first_anchor, last_anchor_exclusive, max(1, args.stride)))
        if args.samples_per_route > 0 and len(anchors) > args.samples_per_route:
            anchors = sorted(rng.sample(anchors, args.samples_per_route))

        scenario, route_id = _route_name(data_root, route_dir)
        for anchor in anchors:
            if args.check_readable:
                ok, reason = _sample_is_readable(route_dir, anchor, frame_count, args)
                if not ok:
                    skipped_samples += 1
                    if args.verbose_skips:
                        print(f"[skip] {route_dir}@{anchor:04d}: {reason}")
                    continue
            samples.append(
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
            )

    rng.shuffle(samples)
    print(
        json.dumps(
            {
                "data_root": str(data_root),
                "route_count": route_count,
                "skipped_routes": skipped_routes,
                "skipped_samples": skipped_samples,
                "samples": len(samples),
            },
            ensure_ascii=False,
        )
    )
    return samples


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="LEAD data root or one route directory.")
    parser.add_argument("--output-dir", default="checkpoints/leadmot_v1_data")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stride", type=int, default=5, help="Default 5 keeps anchors roughly 1 second apart for LEAD 4Hz data.")
    parser.add_argument("--samples-per-route", type=int, default=0, help="0 keeps all valid anchors.")
    parser.add_argument("--check-readable", action="store_true", help="Verify sampled history rgb/laz/meta and future TP/NTP meta before writing JSONL.")
    parser.add_argument("--verbose-skips", action="store_true")
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
    args = parse_args()
    if not 0.0 <= args.val_ratio < 0.5:
        raise ValueError("--val-ratio must be in [0, 0.5)")
    samples = _build_samples(args)
    val_count = int(round(len(samples) * args.val_ratio))
    if len(samples) > 1 and val_count == 0 and args.val_ratio > 0:
        val_count = 1
    val_rows = samples[:val_count]
    train_rows = samples[val_count:]

    output_dir = Path(args.output_dir)
    _write_jsonl(output_dir / "train.jsonl", train_rows)
    _write_jsonl(output_dir / "val.jsonl", val_rows)
    manifest = {
        "schema_version": 1,
        "args": vars(args),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote train={len(train_rows)} val={len(val_rows)} to {output_dir}")


if __name__ == "__main__":
    main()
