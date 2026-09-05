"""精确帧级 RGB/车道身份冲突；只否决不可靠横向候选，不猜新的变道真值。"""
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def _decisions():
    return [json.loads(line) for line in Path(__file__).with_name(
        "lateral_rgb_uncertainties_v1.jsonl").read_text().splitlines() if line.strip()]


def lateral_uncertainty(scenario, route_id, frame_id, horizon=12):
    """当前到未来窗口含未确认的 lane-id 切换时，整段横向标签保持未知。"""
    for row in _decisions():
        if row["scenario"] == scenario and row["route_id"] == route_id:
            if any(frame_id < f <= frame_id + horizon for f in row["transition_frames"]):
                return row
    return None
