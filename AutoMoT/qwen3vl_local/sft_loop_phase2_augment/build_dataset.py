#!/usr/bin/env python3
"""Build Phase2 route-disjoint RS binary rows from the existing frame labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional

_THIS = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS.parents[2]
_PROJECT_ROOT = _THIS.parents[3]
for _path in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route  # noqa: E402
from qwen3vl_local.sft_loop_phase1.audit_matrix import _iter_routes_stream, _rgb_path  # noqa: E402
from qwen3vl_local.sft_loop_phase2_augment import DATASET_NAME  # noqa: E402
from qwen3vl_local.sft_loop_phase2_augment.prompts import ANSWER_KEYS  # noqa: E402
from qwen3vl_local.sft_loop_phase2_augment.visual_audit import (  # noqa: E402
    DEFAULT_COVERAGE_MANIFEST,
    frame_visual_risk,
    load_review_coverage,
)

RGB_HISTORY_COUNT = 4


def _stable_unit(value: str) -> float:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)


def _split(scenario: str, route_id: str, seed: int, test_ratio: float, val_ratio: float) -> str:
    value = _stable_unit(f"{seed}:{scenario}:{route_id}")
    return "test" if value < test_ratio else "val" if value < test_ratio + val_ratio else "train"


def _labels(annotation: Mapping[str, Any]) -> str:
    return str(annotation.get("primary_road_structure") or (annotation.get("frame_rs_annotation") or {}).get("label") or "UNKNOWN")


def _history(run_dir: pathlib.Path, frame_id: int) -> Optional[list[str]]:
    paths = []
    for idx in [max(0, frame_id - offset) for offset in reversed(range(RGB_HISTORY_COUNT))]:
        path = _rgb_path(run_dir, idx)
        if path is None:
            return None
        paths.append(str(path))
    return paths


def _answers(rs: str) -> Dict[str, bool]:
    return {key: rs == key.replace("RS", "R") for key in ANSWER_KEYS}


def _town(annotation: Mapping[str, Any], route: Mapping[str, Any]) -> str:
    return str((annotation.get("evidence") or {}).get("xml_town") or route.get("xml_town") or "UNKNOWN")


def _assert_coverage(counts: Mapping[str, Counter[str]], *, val_ratio: float) -> Dict[str, Dict[str, int]]:
    expected = [f"{key}:{answer}" for key in ANSWER_KEYS for answer in ("YES", "NO")]
    availability = {split: {name: int(counts[split][name]) for name in expected} for split in ("train", "val", "test")}
    required = ["train", "test"] + (["val"] if val_ratio > 0 else [])
    missing = {split: [name for name, count in availability[split].items() if count <= 0] for split in required}
    missing = {split: names for split, names in missing.items() if names}
    if missing:
        raise ValueError(f"route-disjoint Phase2 split lacks required focus bins: {missing}")
    return availability


def iter_rows(args: argparse.Namespace, risk_stats: Optional[Counter[str]] = None) -> Iterable[Dict[str, Any]]:
    collection_dir = pathlib.Path(args.collection_dir)
    data_root = pathlib.Path(args.data_root)
    selected = None if args.scenarios == "all" else {item.strip() for item in args.scenarios.split(",") if item.strip()}
    seen = 0
    for result_path in sorted(collection_dir.glob("*_result.json")):
        scenario = result_path.stem.removesuffix("_result")
        if scenario == "noScenarios" or (selected is not None and scenario not in selected):
            continue
        for route in _iter_routes_stream(result_path):
            route_id = str(route.get("route_id") or "")
            if not route_id or str(route.get("status")) == "data_missing_skip":
                continue
            run_dir = data_root / scenario / route_id
            abnormal, _ = is_abnormal_lead_route(run_dir, scenario)
            if abnormal or not run_dir.is_dir():
                continue
            seen += 1
            if int(args.progress_every_routes) > 0 and seen % int(args.progress_every_routes) == 0:
                print(f"[phase2-build] routes={seen} last={scenario}/{route_id}", flush=True)
            if args.max_routes > 0 and seen > args.max_routes:
                return
            split = _split(scenario, route_id, int(args.split_seed), float(args.test_ratio), float(args.val_ratio))
            for ann in route.get("annotations", []) or []:
                try:
                    frame_id = int(ann.get("frame_id"))
                except (TypeError, ValueError):
                    continue
                rs = _labels(ann)
                if rs not in {"R1", "R2", "R3", "R4", "R5"}:
                    continue
                risk, reasons = frame_visual_risk(ann)
                if risk and risk_stats is not None:
                    risk_stats["risk_frames_seen"] += 1
                    for reason in reasons:
                        risk_stats[f"reason/{reason}"] += 1
                if risk and not args.include_visual_risk:
                    if risk_stats is not None:
                        risk_stats["risk_frames_excluded"] += 1
                    continue
                if risk and risk_stats is not None:
                    risk_stats["risk_frames_retained"] += 1
                history = _history(run_dir, frame_id)
                if history is None:
                    continue
                yield {
                    "dataset_name": DATASET_NAME,
                    "scenario": scenario,
                    "route_id": route_id,
                    "town": _town(ann, route),
                    "split": split,
                    "frame_id": frame_id,
                    "rs": rs,
                    "event": str(ann.get("primary_event") or "UNKNOWN"),
                    "answers": _answers(rs),
                    "is_highway_negative": rs == "R3",
                    "visual_label_risk": risk,
                    "visual_label_risk_reasons": reasons,
                    "history_rgb_paths": history,
                    "latest_rgb_path": history[-1],
                }


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    review_coverage, coverage_source = load_review_coverage(
        review_root=pathlib.Path(args.review_root),
        coverage_manifest=pathlib.Path(args.coverage_manifest),
    )
    missing_review = [
        f"{scenario}/{town}"
        for scenario, towns in sorted(review_coverage.items())
        for town, item in sorted(towns.items())
        if int(item.get("completed_routes", 0)) < 1
    ]
    if missing_review:
        raise ValueError(f"Phase2 requires one completed full-frame RGB review per scenario/Town; missing={missing_review[:20]}")
    target = out_dir / "frame_index.jsonl"
    temporary = out_dir / ".frame_index.jsonl.tmp"
    temporary.unlink(missing_ok=True)
    answer_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    counters: Counter[str] = Counter()
    routes: Dict[str, set[str]] = defaultdict(set)
    highway: Counter[str] = Counter()
    risk_stats: Counter[str] = Counter()
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in iter_rows(args, risk_stats):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                split = row["split"]
                counters[f"frames/{split}"] += 1
                counters[f"frames/{split}/{row['scenario']}"] += 1
                routes[split].add(f"{row['scenario']}/{row['route_id']}")
                highway[f"{split}/{'R3_ALL_NO' if row['is_highway_negative'] else 'NON_R3'}"] += 1
                for key in ANSWER_KEYS:
                    answer_counts[split][f"{key}:{'YES' if row['answers'][key] else 'NO'}"] += 1
        availability = _assert_coverage(answer_counts, val_ratio=float(args.val_ratio))
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(target)
    manifest = {
        "format": "sft_loop_phase2_augment_frame_index",
        "dataset_name": DATASET_NAME,
        "frame_index": str(target),
        "rs_label_contract": "RS1/R2/R4/R5 are true exactly when primary_road_structure is R1/R2/R4/R5; R3 highway/ramp frames are all four NO negatives.",
        "visual_risk_contract": "visual_label_risk never changes GT. It is excluded by default and can be retained with --include-visual-risk for robustness experiments.",
        "include_visual_risk": bool(args.include_visual_risk),
        "full_frame_rgb_review_coverage": {
            "review_root": str(args.review_root),
            "coverage_manifest": str(args.coverage_manifest),
            "coverage_source": coverage_source,
            "scenarios": len(review_coverage),
            "scenario_town_pairs": sum(len(towns) for towns in review_coverage.values()),
            "completed_routes": sum(int(item.get("completed_routes", 0)) for towns in review_coverage.values() for item in towns.values()),
        },
        "split_seed": int(args.split_seed),
        "test_ratio": float(args.test_ratio),
        "val_ratio": float(args.val_ratio),
        "counts": dict(counters),
        "route_counts": {split: len(value) for split, value in sorted(routes.items())},
        "focus_bin_availability": availability,
        "highway_negative_counts": dict(highway),
        "visual_label_risk_counts": dict(sorted(risk_stats.items())),
        "focus_bin_coverage_contract": "Every split contains all four RS questions with YES/NO available; augment train/eval samples the three prompt variants with a 2:1:1 target ratio and records per-variant balance.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection-dir", default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output"))
    p.add_argument("--data-root", default=str(_AUTOMOT_ROOT / "lead_data"))
    p.add_argument("--output-dir", default=str(_AUTOMOT_ROOT / "checkpoints/sft_loop_phase2_augment_data"))
    p.add_argument(
        "--review-root",
        default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output/phase1_four_question_audit/full_route_rgb_label_review_20260809"),
        help="completed all-frame RGB review root; every scenario/Town must have at least one reviewed route",
    )
    p.add_argument(
        "--coverage-manifest",
        default=str(DEFAULT_COVERAGE_MANIFEST),
        help="bundled compact coverage proof used when --review-root is absent on a remote machine",
    )
    p.add_argument("--scenarios", default="all")
    p.add_argument("--split-seed", type=int, default=20260813)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--progress-every-routes", type=int, default=100)
    p.add_argument("--include-visual-risk", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    manifest = build_dataset(parse_args())
    print(f"sft_loop_phase2_augment dataset: frames={sum(value for key, value in manifest['counts'].items() if key.count('/') == 1)} output={manifest['frame_index']}")
