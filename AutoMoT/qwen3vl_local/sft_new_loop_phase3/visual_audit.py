#!/usr/bin/env python3
"""汇总已有逐帧 RGB 审计，并标记不适合直接动作监督的风险帧。

不生成或提交新的 RGB 图。Phase1 的全帧 sheet 已覆盖每一个 scenario/Town 的可用
审计 route；本工具只把这些证据索引和 result.json 内已有的 review 原因写成轻量
JSON。风险判定与 `sft_new_loop_phase2.visual_audit` 完全同源，避免两个阶段对同
一帧给出不同的可用性结论。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from typing import Any, Dict, Tuple

_THIS = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS.parents[2]
DEFAULT_COVERAGE_MANIFEST = _THIS.with_name("action_rgb_audit_coverage.json")
if str(_AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTOMOT_ROOT))

from qwen3vl_local.sft_loop_phase1.audit_matrix import _iter_routes_stream  # noqa: E402
from qwen3vl_local.sft_new_loop_phase2.visual_audit import (  # noqa: E402
    _existing_review_coverage,
    frame_visual_risk,
)

COVERAGE_FORMAT = "sft_new_loop_phase3_rgb_audit_coverage_v1"


def _bundled_review_coverage(path: pathlib.Path) -> Dict[str, Any]:
    """读取随代码提交的紧凑审计覆盖证明，不依赖本地 RGB sheet。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != COVERAGE_FORMAT:
        raise ValueError(f"unsupported new Phase3 RGB coverage manifest: {path}")
    return {
        str(scenario): {str(town): {"completed_routes": int(count)} for town, count in (towns or {}).items()}
        for scenario, towns in (payload.get("coverage") or {}).items()
    }


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
    """写一份轻量证据/风险清单；不复制任何 RGB 像素。"""

    collection_dir = pathlib.Path(args.collection_dir)
    review_root = pathlib.Path(args.review_root)
    output_path = pathlib.Path(args.output)
    coverage, coverage_source = load_review_coverage(
        review_root=review_root, coverage_manifest=pathlib.Path(args.coverage_manifest)
    )
    risk_counts: Counter[str] = Counter()
    by_scenario: Dict[str, Dict[str, int]] = {}
    if args.scan_frame_risks:
        for result_path in sorted(collection_dir.glob("*_result.json")):
            scenario = result_path.stem.removesuffix("_result")
            if scenario == "noScenarios":
                continue
            counts: Counter[str] = Counter()
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
    missing = [
        f"{scenario}/{town}"
        for scenario, towns in sorted(coverage.items())
        for town, item in sorted(towns.items())
        if int(item["completed_routes"]) < 1
    ]
    payload = {
        "format": "sft_new_loop_phase3_visual_audit_manifest",
        "source_review_root": str(review_root),
        "coverage_source": coverage_source,
        "coverage_manifest": str(args.coverage_manifest),
        "coverage_contract": "Existing Phase1 RGB/RS review cache availability; this is not new manual action-label verification.",
        "action_boundary_verification_complete": False,
        "new_review_notes": str(pathlib.Path(__file__).with_name("rgb_mapping_review_20260905.jsonl")),
        "scenario_town_coverage": coverage,
        "coverage_missing": missing,
        "reviewed_scenarios": len(coverage),
        "reviewed_scenario_town_pairs": sum(len(towns) for towns in coverage.values()),
        "reviewed_routes": sum(
            int(item["completed_routes"]) for towns in coverage.values() for item in towns.values()
        ),
        "frame_risk_scan_enabled": bool(args.scan_frame_risks),
        "visual_risk_reason_counts": dict(sorted(risk_counts.items())),
        "per_scenario_frame_counts": by_scenario,
        "visual_risk_contract": "Risk flags retain the original primary RS/EVENT; they never rewrite labels. build_dataset.py defaults to clean-only rows.",
        "action_label_contract": "High-level action labels come from the run's own future meta trajectory, never from scenario names. Longitudinal labels use the future speed curve; lateral labels require a real OpenDRIVE lane-identity change so a curved lane can never become a lane change.",
    }
    if missing:
        raise ValueError(f"missing completed RGB review for scenario/Town pairs: {missing[:20]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数。"""

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection-dir", default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output"))
    p.add_argument(
        "--review-root",
        default=str(
            _AUTOMOT_ROOT
            / "keyframe_filter/collection_output/phase1_four_question_audit/full_route_rgb_label_review_20260809"
        ),
    )
    p.add_argument(
        "--output",
        default=str(_AUTOMOT_ROOT / "checkpoints/sft_new_loop_phase3_data/visual_audit_manifest.json"),
    )
    p.add_argument("--coverage-manifest", default=str(DEFAULT_COVERAGE_MANIFEST))
    p.add_argument("--scan-frame-risks", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    result = build_visual_audit_manifest(parse_args())
    print(
        f"new Phase3 RGB audit: routes={result['reviewed_routes']} "
        f"town_pairs={result['reviewed_scenario_town_pairs']}"
    )
