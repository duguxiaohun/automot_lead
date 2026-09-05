#!/usr/bin/env python3
"""确定 LEAD ego frame 的 y 轴左右符号：用已知左/右转 scenario 的 route 折线取证。"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter
from typing import Dict, List

import numpy as np

_THIS = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS.parents[2]
_PROJECT_ROOT = _THIS.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_new_loop_phase3.trajectory_action import load_route_trajectory  # noqa: E402

SCENARIOS = {
    "SignalizedJunctionLeftTurn": "LEFT",
    "NonSignalizedJunctionLeftTurn": "LEFT",
    "SignalizedJunctionRightTurn": "RIGHT",
    "NonSignalizedJunctionRightTurn": "RIGHT",
}


def main() -> None:
    """统计转弯 scenario 中 route 前方 30m 的横向 y 符号。"""

    data_root = _AUTOMOT_ROOT / "lead_data"
    report: Dict[str, Dict[str, int]] = {}
    for scenario, expected in SCENARIOS.items():
        scenario_dir = data_root / scenario
        if not scenario_dir.is_dir():
            continue
        counts: Counter[str] = Counter()
        runs = sorted(p for p in scenario_dir.iterdir() if p.is_dir())[:6]
        for run in runs:
            traj = load_route_trajectory(run)
            if traj is None:
                continue
            for frame_id in traj.frames:
                meta = traj.metas[frame_id]
                if not bool(meta.get("is_junction")):
                    continue
                route = np.asarray(meta.get("route", []), dtype=np.float64)
                if route.ndim != 2 or route.shape[0] < 25:
                    continue
                lateral = float(route[24][1])
                if abs(lateral) < 3.0:
                    continue
                counts["y_positive" if lateral > 0 else "y_negative"] += 1
        report[scenario] = {"expected_turn": expected, **dict(counts)}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
