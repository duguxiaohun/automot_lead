"""SFT v4 answer-token and private-field cleanup test."""

from __future__ import annotations

import json
import pathlib
import sys

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_v4.prompts import (
    build_step1_teacher_target,
    build_step2_teacher_target,
    build_step3_teacher_target,
    check_gt_leak_road_structure,
    check_gt_leak_scene,
    check_gt_leak_status_subgoal,
)


def main() -> None:
    """Answer tokens are valid supervision, but private field names must be cleaned."""

    results = {
        "road_structure": check_gt_leak_road_structure(
            "The correct road structure is JUNCTION.", "JUNCTION"
        ),
        "scene": check_gt_leak_scene("This clearly indicates Accident scene.", "Accident"),
        "status_subgoal": check_gt_leak_status_subgoal(
            "Current status is hazard_detect and subgoal max_brake_or_min_gap.",
            "hazard_detect",
            "max_brake_or_min_gap",
        ),
    }
    cleaned_targets = {
        "step1": build_step1_teacher_target(
            "Memory Judgment: Evidence supports GROUND_TRUTH_ROAD_STRUCTURE.", "JUNCTION"
        ),
        "step2": build_step2_teacher_target(
            "Memory Judgment: Evidence supports GROUND_TRUTH_SCENE rather than the believed scene.",
            "Accident",
        ),
        "step3": build_step3_teacher_target(
            "Memory Judgment: BELIEF_SUBGOAL should change to GROUND_TRUTH_SUBGOAL.",
            "hazard_detect",
            "max_brake_or_min_gap",
        ),
    }
    forbidden = ("GROUND_TRUTH_", "ANSWER_", "REFERENCE_", "BELIEF_", "BELIEVED_")
    clean_ok = all(not any(token in value for token in forbidden) for value in cleaned_targets.values())
    ok = all(value is False for value in results.values())
    print(
        json.dumps(
            {"ok": ok and clean_ok, "results": results, "cleaned_targets": cleaned_targets},
            ensure_ascii=False,
            indent=2,
        )
    )
    if not (ok and clean_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
