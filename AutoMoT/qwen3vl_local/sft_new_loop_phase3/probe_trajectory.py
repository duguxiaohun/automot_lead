#!/usr/bin/env python3
"""Phase3 轨迹探查工具：把逐帧 RS/EVENT 标注与 meta 轨迹信号对齐后落盘。

用于 phase3 动作标定规则的调研与复核，不参与训练/评测。输出写到
``qwen3vl_local/sft_new_loop_phase3/probe_output/``。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List

import numpy as np

_THIS = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS.parents[2]
_PROJECT_ROOT = _THIS.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_loop_phase1.audit_matrix import _iter_routes_stream  # noqa: E402
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import (  # noqa: E402
    RouteTrajectory,
    load_route_trajectory,
)

DEFAULT_OUTPUT_DIR = _THIS.parent / "probe_output"


def _rows(traj: RouteTrajectory, annotations: List[Dict[str, Any]]) -> List[str]:
    """渲染逐帧对照行。"""

    out: List[str] = []
    for ann in annotations:
        try:
            fid = int(ann.get("frame_id"))
        except (TypeError, ValueError):
            continue
        sig = traj.signals(fid)
        if sig is None:
            continue
        out.append(
            f"f{fid:03d} rs={str(ann.get('primary_road_structure')):>3} "
            f"ev={str(ann.get('primary_event')):<6} "
            f"v={sig['speed']:5.2f} vmin={sig['speed_min']:5.2f} vmax={sig['speed_max']:5.2f} "
            f"lane={sig['lane_id']:+d}@{sig['road_id']} "
            f"lat={sig['lateral_shift']:+5.2f} "
            f"lc={sig['lane_change_direction'] or '-':<5} "
            f"goal=({sig['goal_x']:+7.1f},{sig['goal_y']:+7.1f}) "
            f"brake={int(sig['brake'])} junc={int(sig['is_junction'])}"
        )
    return out


def main() -> None:
    """CLI 入口。"""

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--route-substr", default="")
    ap.add_argument("--max-routes", type=int, default=1)
    ap.add_argument("--collection-dir", default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output"))
    ap.add_argument("--data-root", default=str(_AUTOMOT_ROOT / "lead_data"))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = ap.parse_args()

    result = pathlib.Path(args.collection_dir) / f"{args.scenario}_result.json"
    data_root = pathlib.Path(args.data_root)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shown = 0
    blocks: List[str] = []
    for route in _iter_routes_stream(result):
        route_id = str(route.get("route_id") or "")
        if args.route_substr and args.route_substr not in route_id:
            continue
        if str(route.get("status")) == "data_missing_skip":
            continue
        run_dir = data_root / args.scenario / route_id
        if not run_dir.is_dir():
            continue
        annotations = list(route.get("annotations") or [])
        if not annotations:
            continue
        traj = load_route_trajectory(run_dir)
        if traj is None:
            continue
        blocks.append(
            f"=== {args.scenario}/{route_id} town={route.get('xml_town')} frames={len(annotations)}"
        )
        blocks.extend(_rows(traj, annotations))
        shown += 1
        if shown >= args.max_routes:
            break

    tag = f"{args.scenario}{'_' + args.route_substr if args.route_substr else ''}"
    target = out_dir / f"probe_{tag}.txt"
    target.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    print(json.dumps({"routes": shown, "output": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
