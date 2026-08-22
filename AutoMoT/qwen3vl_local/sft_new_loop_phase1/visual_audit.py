#!/usr/bin/env python3
"""Build the RGB audit manifest used by fused Phase1+Phase2 dataset creation.

The visual-risk rules are intentionally inherited from the latest
`sft_loop_phase2_augment` implementation. This local wrapper keeps the fused
package operationally self-contained while preserving Phase2 as the ROAD_STRUCTURE
authority.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


_THIS = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS.parents[2]
if str(_AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTOMOT_ROOT))

from qwen3vl_local.sft_loop_phase2_augment.visual_audit import (  # noqa: E402
    VISUAL_RISK_REASONS,
    build_visual_audit_manifest,
    frame_visual_risk,
    load_review_coverage,
)

DEFAULT_COVERAGE_MANIFEST = _THIS.with_name("phase2_rgb_audit_coverage.json")

__all__ = [
    "DEFAULT_COVERAGE_MANIFEST",
    "VISUAL_RISK_REASONS",
    "build_visual_audit_manifest",
    "frame_visual_risk",
    "load_review_coverage",
]


def parse_args() -> argparse.Namespace:
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
        default=str(_AUTOMOT_ROOT / "checkpoints/sft_new_loop_phase1_data/visual_audit_manifest.json"),
    )
    p.add_argument(
        "--coverage-manifest",
        default=str(DEFAULT_COVERAGE_MANIFEST),
        help="bundled compact Phase2 coverage proof used when local RGB audit artifacts are absent",
    )
    p.add_argument(
        "--scan-frame-risks",
        action="store_true",
        help="also stream every result.json frame to summarize risks; build_dataset.py performs this during index construction",
    )
    return p.parse_args()


if __name__ == "__main__":
    result = build_visual_audit_manifest(parse_args())
    print(
        "fused RGB audit: "
        f"routes={result['reviewed_routes']} town_pairs={result['reviewed_scenario_town_pairs']}"
    )
