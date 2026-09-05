#!/usr/bin/env python3
"""Phase3 动作标定的 RGB 复核工具：把逐帧 RGB 与派生动作标签拼成 contact sheet。

输出写到 ``qwen3vl_local/sft_new_loop_phase3/probe_output/``，用于人工逐帧确认
减速 / 停车 / 恢复 / 左变道 / 右变道 的区间划分是否与画面一致。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw

_THIS = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS.parents[2]
_PROJECT_ROOT = _THIS.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_loop_phase1.audit_matrix import _iter_routes_stream, _rgb_path  # noqa: E402
from qwen3vl_local.sft_new_loop_phase3.build_dataset import (  # noqa: E402
    _event_codes,
    _last_bypass_frame,
)
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import resolve_context_id  # noqa: E402
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import (  # noqa: E402
    label_actions,
    load_route_trajectory,
)

DEFAULT_OUTPUT_DIR = _THIS.parent / "probe_output"
TILE_WIDTH = 640
CAPTION_HEIGHT = 46


def _caption(
    frame_id: int,
    ann: Dict[str, Any],
    signals: Dict[str, Any],
    labels: Optional[Dict[str, bool]],
    frames_since_bypass: Optional[int],
) -> str:
    """一帧的标注摘要。"""

    actions = "-" if labels is None else (",".join(k for k, v in labels.items() if v) or "KEEP")
    context_id = resolve_context_id(
        str(ann.get("primary_road_structure")),
        _event_codes(ann),
        frames_since_bypass,
    )
    return (
        f"f{frame_id}  rs={ann.get('primary_road_structure')}  ev={ann.get('primary_event')}"
        f"  ctx={context_id or '-'}\n"
        f"v={signals['speed']:.1f} vmin={signals['speed_min']:.1f} vmax={signals['speed_max']:.1f}"
        f"  goal=({signals['goal_x']:+.0f},{signals['goal_y']:+.0f})  ->  {actions}"
    )


def build_sheet(
    *,
    scenario: str,
    route_id: str,
    frames: List[int],
    annotations: Dict[int, Dict[str, Any]],
    run_dir: pathlib.Path,
    output: pathlib.Path,
) -> Optional[pathlib.Path]:
    """渲染一条 route 的选定帧 contact sheet。"""

    traj = load_route_trajectory(run_dir)
    if traj is None:
        return None
    gaps = _last_bypass_frame([annotations[key] for key in sorted(annotations)])
    tiles: List[Image.Image] = []
    for frame_id in frames:
        rgb = _rgb_path(run_dir, frame_id)
        signals = traj.signals(frame_id)
        if rgb is None or signals is None:
            continue
        image = Image.open(rgb).convert("RGB")
        scale = TILE_WIDTH / image.width
        image = image.resize((TILE_WIDTH, max(1, int(image.height * scale))))
        tile = Image.new("RGB", (TILE_WIDTH, image.height + CAPTION_HEIGHT), (16, 16, 16))
        tile.paste(image, (0, 0))
        ImageDraw.Draw(tile).multiline_text(
            (6, image.height + 4),
            _caption(
                frame_id,
                annotations.get(frame_id, {}),
                signals,
                label_actions(signals),
                gaps.get(frame_id),
            ),
            fill=(255, 235, 120),
            spacing=3,
        )
        tiles.append(tile)
    if not tiles:
        return None
    sheet = Image.new("RGB", (TILE_WIDTH, sum(tile.height for tile in tiles)), (16, 16, 16))
    offset = 0
    for tile in tiles:
        sheet.paste(tile, (0, offset))
        offset += tile.height
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=88)
    return output


def main() -> None:
    """CLI 入口。"""

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--route-substr", default="")
    ap.add_argument("--event", default="", help="只渲染该 primary_event 的帧，例如 U-E2 / R-E2 / R-E3")
    ap.add_argument("--start-frame", type=int, default=-1)
    ap.add_argument("--frame-step", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--collection-dir", default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output"))
    ap.add_argument("--data-root", default=str(_AUTOMOT_ROOT / "lead_data"))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = ap.parse_args()

    result = pathlib.Path(args.collection_dir) / f"{args.scenario}_result.json"
    for route in _iter_routes_stream(result):
        route_id = str(route.get("route_id") or "")
        if args.route_substr and args.route_substr not in route_id:
            continue
        if str(route.get("status")) == "data_missing_skip":
            continue
        run_dir = pathlib.Path(args.data_root) / args.scenario / route_id
        if not run_dir.is_dir():
            continue
        annotations = {
            int(ann["frame_id"]): dict(ann)
            for ann in (route.get("annotations") or [])
            if ann.get("frame_id") is not None
        }
        candidates = sorted(annotations)
        if args.event:
            candidates = [f for f in candidates if str(annotations[f].get("primary_event")) == args.event]
        if args.start_frame >= 0:
            candidates = [f for f in candidates if f >= args.start_frame]
        frames = candidates[:: max(1, args.frame_step)][: args.max_frames]
        if not frames:
            continue
        tag = f"{args.scenario}_{route_id}_{args.event or 'all'}_{frames[0]}"
        output = pathlib.Path(args.output_dir) / f"sheet_{tag}.jpg"
        saved = build_sheet(
            scenario=args.scenario,
            route_id=route_id,
            frames=frames,
            annotations=annotations,
            run_dir=run_dir,
            output=output,
        )
        print(json.dumps({"route_id": route_id, "frames": frames, "sheet": str(saved)}, ensure_ascii=False))
        return
    print(json.dumps({"sheet": None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
