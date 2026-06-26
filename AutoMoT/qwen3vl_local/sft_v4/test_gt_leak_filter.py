"""SFT v4 legacy answer-token compatibility test."""

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
    check_gt_leak_road_structure,
    check_gt_leak_scene,
    check_gt_leak_status_subgoal,
)


def main() -> None:
    """Answer tokens in teacher analysis are valid supervision, not a skip signal."""

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
    ok = all(value is False for value in results.values())
    print(json.dumps({"ok": ok, "results": results}, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
