#!/usr/bin/env python3
"""汇总已有逐帧 RGB 审计，并标记不适合直接 EVENT 监督的风险帧。

不生成或提交新的 RGB 图。Phase1 的全帧 sheet 已覆盖每一个场景-Town 的可用审计
route：通常至少三条；数据不足的 Town 则审完全部可用 route。本工具只把这些证据索引
和 result.json 内已有的 review 原因写成轻量 JSON。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from typing import Any, Dict, Mapping, Tuple

_THIS = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS.parents[2]
DEFAULT_COVERAGE_MANIFEST = _THIS.with_name("event_rgb_audit_coverage.json")
if str(_AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTOMOT_ROOT))

from qwen3vl_local.sft_loop_phase1.audit_matrix import _iter_routes_stream  # noqa: E402


# These RS reasons say the annotation itself requests RGB confirmation for a geometry
# distinction used by this event phase. Explicit EVENT review flags are handled separately
# below: unlike RS, every event-review request is directly relevant to EVENT supervision.
VISUAL_RISK_REASONS = {
    "r2_lacks_xodr_opposite_lane_confirmation",
    "r4_bbox_tl_without_strong_context_requires_rgb_confirmation",
    "r4_meta_tl_without_strong_context_requires_rgb_confirmation",
    "signalized_r4_without_meta_tl_requires_rgb_confirmation",
    "nonsignalized_with_signal_topology_conflict",
    "r4_meta_tl_without_strong_context_review",
    "r4_bbox_tl_without_strong_context_review",
}


def frame_visual_risk(annotation: Mapping[str, Any]) -> Tuple[bool, list[str]]:
    """Return whether a frame needs explicit visual-label handling, never change GT."""

    rs_ann = annotation.get("frame_rs_annotation") or {}
    reasons = [str(x) for x in rs_ann.get("review_reasons", []) or []]
    evidence = annotation.get("evidence") or {}
    fired = [str(x) for x in evidence.get("rules_fired", []) or []]
    for item in fired:
        if item in VISUAL_RISK_REASONS or item.endswith("_without_strong_context_review"):
            reasons.append(item)
    selected = {
        f"rs:{reason}"
        for reason in reasons
        if reason in VISUAL_RISK_REASONS or reason.endswith("_without_strong_context_review")
    }

    # result.json keeps the same event audit both as a compact frame annotation and,
    # in some versions, as a sibling evidence block. Deduplicate the reasons, but do
    # not silently ignore an explicit review_required flag with an empty reason list.
    event_review_seen = False
    event_reason_seen = False
    for key in ("frame_event_annotation", "event_evidence"):
        event_ann = annotation.get(key) or {}
        if not isinstance(event_ann, Mapping):
            continue
        event_review_seen = event_review_seen or bool(event_ann.get("review_required", False))
        for reason in event_ann.get("review_reasons", []) or []:
            event_reason_seen = True
            selected.add(f"event:{reason}")
    if event_review_seen and not event_reason_seen:
        selected.add("event:review_required_without_reason")

    unique = sorted(selected)
    return bool(unique), unique


def _existing_review_coverage(review_root: pathlib.Path) -> Dict[str, Any]:
    """Load the already completed full-route review summaries, grouped by scenario/Town."""

    coverage: Dict[str, Any] = {}
    for path in sorted(review_root.glob("*/scenario_visual_review_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        towns: Dict[str, Any] = {}
        for town, summary in (payload.get("towns") or {}).items():
            completed = [r for r in summary.get("routes", []) if r.get("manual_rgb_review_status") == "completed"]
            towns[str(town)] = {
                "completed_routes": len(completed),
                "representative_route_ids": [str(r.get("route_id")) for r in completed],
                "first_full_frame_sheet": str((completed[0].get("full_frame_sheets") or [""])[0]) if completed else "",
            }
        coverage[path.parent.name] = towns
    return coverage


def _bundled_review_coverage(path: pathlib.Path) -> Dict[str, Any]:
    """读取可随代码提交的紧凑审计覆盖证明，不依赖本地 RGB sheet。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "sft_new_loop_phase2_rgb_audit_coverage_v1":
        raise ValueError(f"unsupported new Phase2 RGB coverage manifest: {path}")
    coverage: Dict[str, Any] = {}
    for scenario, towns in (payload.get("coverage") or {}).items():
        coverage[str(scenario)] = {
            str(town): {"completed_routes": int(count)}
            for town, count in (towns or {}).items()
        }
    return coverage


def load_review_coverage(
    *, review_root: pathlib.Path, coverage_manifest: pathlib.Path = DEFAULT_COVERAGE_MANIFEST
) -> Tuple[Dict[str, Any], str]:
    """优先读本地全帧审计；远程缺大产物时回退到随代码提交的覆盖证明。"""

    local = _existing_review_coverage(review_root) if review_root.is_dir() else {}
    if local:
        return local, "local_full_frame_review"
    if not coverage_manifest.is_file():
        raise FileNotFoundError(
            "no local full-frame RGB review and no bundled coverage manifest; "
            f"review_root={review_root} coverage_manifest={coverage_manifest}"
        )
    return _bundled_review_coverage(coverage_manifest), "bundled_coverage_manifest"


def build_visual_audit_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    """Write a small evidence/risk manifest; no RGB pixels are copied."""

    collection_dir = pathlib.Path(args.collection_dir)
    review_root = pathlib.Path(args.review_root)
    output_path = pathlib.Path(args.output)
    coverage, coverage_source = load_review_coverage(
        review_root=review_root,
        coverage_manifest=pathlib.Path(args.coverage_manifest),
    )
    risk_counts: Counter[str] = Counter()
    by_scenario: Dict[str, Dict[str, int]] = {}
    if args.scan_frame_risks:
        for result_path in sorted(collection_dir.glob("*_result.json")):
            scenario = result_path.stem.removesuffix("_result")
            if scenario == "noScenarios":
                continue
            counts = Counter()
            for route in _iter_routes_stream(result_path):
                if str(route.get("status")) == "data_missing_skip":
                    continue
                for ann in route.get("annotations", []) or []:
                    risk, reasons = frame_visual_risk(ann)
                    counts["frames"] += 1
                    if risk:
                        counts["visual_risk_frames"] += 1
                        for reason in reasons:
                            risk_counts[reason] += 1
            by_scenario[scenario] = dict(counts)
    missing = []
    for scenario, towns in sorted(coverage.items()):
        for town, item in sorted(towns.items()):
            if int(item["completed_routes"]) < 1:
                missing.append(f"{scenario}/{town}")
    payload = {
        "format": "sft_new_loop_phase2_visual_audit_manifest",
        "source_review_root": str(review_root),
        "coverage_source": coverage_source,
        "coverage_manifest": str(args.coverage_manifest),
        "coverage_contract": "Every scenario/Town must have at least one completed full-frame RGB review before direct-event prompt changes.",
        "scenario_town_coverage": coverage,
        "coverage_missing": missing,
        "reviewed_scenarios": len(coverage),
        "reviewed_scenario_town_pairs": sum(len(towns) for towns in coverage.values()),
        "reviewed_routes": sum(int(item["completed_routes"]) for towns in coverage.values() for item in towns.values()),
        "frame_risk_scan_enabled": bool(args.scan_frame_risks),
        "visual_risk_reason_counts": dict(sorted(risk_counts.items())),
        "per_scenario_frame_counts": by_scenario,
        "visual_risk_contract": "Risk flags retain the original primary RS/EVENT; they do not rewrite labels. Explicit EVENT review requests and selected RS geometry risks are recorded, and build_dataset.py defaults to clean-only rows.",
    }
    if missing:
        raise ValueError(f"missing completed RGB review for scenario/Town pairs: {missing[:20]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection-dir", default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output"))
    p.add_argument(
        "--review-root",
        default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output/phase1_four_question_audit/full_route_rgb_label_review_20260809"),
    )
    p.add_argument("--output", default=str(_AUTOMOT_ROOT / "checkpoints/sft_new_loop_phase2_data/visual_audit_manifest.json"))
    p.add_argument(
        "--coverage-manifest",
        default=str(DEFAULT_COVERAGE_MANIFEST),
        help="bundled compact coverage proof used when the local RGB audit artifacts are not present",
    )
    p.add_argument(
        "--scan-frame-risks",
        action="store_true",
        help="also stream every result.json frame to summarize risks; build_dataset.py already performs this once during real index construction",
    )
    return p.parse_args()


if __name__ == "__main__":
    result = build_visual_audit_manifest(parse_args())
    print(f"new Phase2 RGB audit: routes={result['reviewed_routes']} town_pairs={result['reviewed_scenario_town_pairs']}")
