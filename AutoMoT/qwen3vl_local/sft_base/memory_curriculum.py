"""SFT base memory 扰动课程。

本模块故意不 import torch，便于用纯 Python 测试分布边际。训练入口只负责把
FrameRow/Memory 传进来，真正的 wrong/UNKNOWN/dropout 逻辑都在这里。
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, List

from qwen3vl_local.sft_base.labels import (
    EVENT_CANDIDATES_BY_RS,
    EVENT_ORDER,
    RS_DESCRIPTIONS,
    collapse_regular_to_re,
)
from qwen3vl_local.sft_base.prompts import Memory, refresh_memory_goal


def stable_frame_seed(*, route_id: str, frame_id: int, seed: int, salt: str) -> int:
    """生成 route/frame/salt 稳定随机种子。"""

    src = f"{route_id}::{frame_id}::{seed}::{salt}".encode("utf-8")
    return int(hashlib.sha256(src).hexdigest(), 16) % (2**31)


def pick_different(rng: random.Random, current: str, candidates: List[str]) -> str:
    """从候选中挑一个不同于 current 的值，候选为空时保持原值。"""

    pool = [item for item in candidates if item and item != current]
    if not pool:
        return current
    return rng.choice(sorted(set(pool)))


def wrong_rs_for_frame(rng: random.Random, current: str, gt_rs: str) -> str:
    """生成 RS 对抗扰动。

    只保留 GT=R3 -> memory=R1 的方向，避免 R1 大类反向灌入大量假 R3 memory。
    """

    if gt_rs == "R3" and current != "R1":
        return "R1"
    return pick_different(rng, current, list(RS_DESCRIPTIONS))


def event_memory_pool_for_rs(frame: Any, rs_label: str) -> List[str]:
    """按给定 RS 生成内部自洽的 EVENT memory 候选。"""

    if rs_label not in RS_DESCRIPTIONS:
        return ["RE"]
    scenario_raw = getattr(frame, "raw", {}).get("scenario_event_candidates") or list(EVENT_ORDER)
    scenario_set = {str(code) for code in scenario_raw}
    raw = [code for code in EVENT_CANDIDATES_BY_RS.get(rs_label, []) if code in scenario_set]
    return collapse_regular_to_re(raw, rs_label)


def maybe_corrupt_memory(
    memory: Memory,
    *,
    frame: Any,
    route_id: str,
    frame_pos: int,
    seed: int,
    first_frame_unknown: bool,
    rs_wrong_prob: float,
    rs_unknown_prob: float,
    event_wrong_prob: float,
    event_unknown_prob: float,
    rs_wrong_event_unknown_prob: float,
    memory_dropout_prob: float,
) -> Memory:
    """训练时把 prompt 里的 memory 从“答案”降级成“不可靠先验”。"""

    mem = memory.copy()
    rng = random.Random(stable_frame_seed(route_id=route_id, frame_id=int(frame.frame_id), seed=seed, salt="memory"))
    if first_frame_unknown and frame_pos == 0:
        mem.rs_label = "UNKNOWN"
        mem.event_label = "UNKNOWN"
        return refresh_memory_goal(mem, getattr(frame, "ego_to_goal_xy", None))

    if rng.random() < max(0.0, float(memory_dropout_prob)):
        mem.hide_priors = True
        return refresh_memory_goal(mem, getattr(frame, "ego_to_goal_xy", None))

    rs_mode = "keep"
    rs_draw = rng.random()
    if rs_draw < max(0.0, float(rs_unknown_prob)):
        mem.rs_label = "UNKNOWN"
        rs_mode = "unknown"
    elif rs_draw < max(0.0, float(rs_unknown_prob)) + max(0.0, float(rs_wrong_prob)):
        mem.rs_label = wrong_rs_for_frame(rng, mem.rs_label, str(frame.rs_label))
        rs_mode = "wrong"

    event_draw = rng.random()
    if event_draw < max(0.0, float(event_unknown_prob)):
        mem.event_label = "UNKNOWN"
    elif rs_mode == "wrong":
        if rng.random() < min(1.0, max(0.0, float(rs_wrong_event_unknown_prob))):
            mem.event_label = "UNKNOWN"
        else:
            mem.event_label = pick_different(rng, mem.event_label, event_memory_pool_for_rs(frame, mem.rs_label))
    elif event_draw < max(0.0, float(event_unknown_prob)) + max(0.0, float(event_wrong_prob)):
        event_pool = ["RE"] + [code for code in EVENT_ORDER if code in set(getattr(frame, "event_candidates", []))]
        mem.event_label = pick_different(rng, mem.event_label, event_pool)
    return refresh_memory_goal(mem, getattr(frame, "ego_to_goal_xy", None))


def resample_event_memory_for_q2(
    memory: Memory,
    *,
    frame: Any,
    route_id: str,
    seed: int,
    event_wrong_prob: float,
    event_unknown_prob: float,
    keep_event_label: str,
) -> Memory:
    """Q1 后为 Q2 重新构造 EVENT memory。

    Q1 输出已经把 RS 更新到本帧答案语境；这里只在该 RS 的自洽候选池里重采
    EVENT，避免“纠正 RS 后沿用旧 RS 语境 EVENT”导致 Q2 prompt 大量 UNKNOWN 或不自洽。
    """

    mem = memory.copy()
    if mem.hide_priors:
        return refresh_memory_goal(mem, getattr(frame, "ego_to_goal_xy", None))
    rng = random.Random(stable_frame_seed(route_id=route_id, frame_id=int(frame.frame_id), seed=seed, salt="q2_event_memory"))
    pool = event_memory_pool_for_rs(frame, mem.rs_label)
    keep_label = str(keep_event_label)
    if keep_label not in pool:
        keep_label = "UNKNOWN"
    event_draw = rng.random()
    if event_draw < max(0.0, float(event_unknown_prob)):
        mem.event_label = "UNKNOWN"
    elif event_draw < max(0.0, float(event_unknown_prob)) + max(0.0, float(event_wrong_prob)):
        mem.event_label = pick_different(rng, keep_label, pool)
    else:
        mem.event_label = keep_label
    return refresh_memory_goal(mem, getattr(frame, "ego_to_goal_xy", None))
