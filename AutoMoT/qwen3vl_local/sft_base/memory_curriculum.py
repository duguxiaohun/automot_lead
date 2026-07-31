"""SFT base memory 扰动课程。

本模块故意不 import torch，便于用纯 Python 测试分布边际。训练入口只负责把
FrameRow/Memory 传进来，真正的 wrong/UNKNOWN/dropout 逻辑都在这里。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, List, Optional

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


def stable_route_seed(*, route_id: str, seed: int, salt: str) -> int:
    """生成 route/salt 稳定随机种子，用于顺序状态机。"""

    src = f"{route_id}::{seed}::{salt}".encode("utf-8")
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

    del frame
    if rs_label not in RS_DESCRIPTIONS:
        return ["RE"]
    raw = list(EVENT_CANDIDATES_BY_RS.get(rs_label, []))
    return collapse_regular_to_re(raw, rs_label)


@dataclass
class _SegmentState:
    """route 级扰动段状态。"""

    mode: str = "keep"
    value: Optional[str] = None
    remaining: int = 0


class RouteMemoryCorruptor:
    """把逐帧独立 memory 扰动改成 route 级块状扰动。

    sft_base 保持 off-policy，不用学生 rollout；这个状态机只模拟“错误 memory 会连续
    存在几帧，到期后恢复干净 GT 轨迹”的效果。状态按真实 route 帧推进，训练 repeat
    不会额外推进状态。
    """

    def __init__(
        self,
        *,
        route_id: str,
        seed: int,
        first_frame_unknown: bool,
        rs_wrong_prob: float,
        rs_unknown_prob: float,
        event_wrong_prob: float,
        event_unknown_prob: float,
        rs_wrong_event_unknown_prob: float,
        memory_dropout_prob: float,
        duration_min: int = 3,
        duration_max: int = 5,
    ) -> None:
        """初始化 route 级扰动状态。"""

        self.route_id = str(route_id)
        self.seed = int(seed)
        self.first_frame_unknown = bool(first_frame_unknown)
        self.rs_wrong_prob = max(0.0, float(rs_wrong_prob))
        self.rs_unknown_prob = max(0.0, float(rs_unknown_prob))
        self.event_wrong_prob = max(0.0, float(event_wrong_prob))
        self.event_unknown_prob = max(0.0, float(event_unknown_prob))
        self.rs_wrong_event_unknown_prob = min(1.0, max(0.0, float(rs_wrong_event_unknown_prob)))
        self.memory_dropout_prob = max(0.0, float(memory_dropout_prob))
        self.duration_min = max(1, int(duration_min))
        self.duration_max = max(self.duration_min, int(duration_max))
        self.rng = random.Random(stable_route_seed(route_id=self.route_id, seed=self.seed, salt="route_memory"))
        self.rs_state = _SegmentState()
        self.event_state = _SegmentState()
        self.q2_event_state = _SegmentState()

    def _duration(self) -> int:
        """采样一个扰动段长度。"""

        return self.rng.randint(self.duration_min, self.duration_max)

    def _draw_mode(self, *, unknown_prob: float, wrong_prob: float) -> str:
        """按 unknown/wrong/keep 概率采样新段类型。"""

        draw = self.rng.random()
        if draw < max(0.0, float(unknown_prob)):
            return "unknown"
        if draw < max(0.0, float(unknown_prob)) + max(0.0, float(wrong_prob)):
            return "wrong"
        return "keep"

    def _ensure_state(
        self,
        state: _SegmentState,
        *,
        unknown_prob: float,
        wrong_prob: float,
        wrong_value: Optional[str] = None,
    ) -> None:
        """必要时开启一个新扰动段。"""

        if state.remaining > 0:
            return
        state.mode = self._draw_mode(unknown_prob=unknown_prob, wrong_prob=wrong_prob)
        state.value = wrong_value if state.mode == "wrong" else None
        state.remaining = self._duration()

    def _tick(self, state: _SegmentState) -> None:
        """推进一个真实帧。"""

        state.remaining = max(0, int(state.remaining) - 1)

    def corrupt(self, memory: Memory, *, frame: Any, frame_pos: int) -> Memory:
        """生成 Q1 prompt 使用的 route 级块状扰动 memory。"""

        mem = memory.copy()
        if self.first_frame_unknown and int(frame_pos) == 0:
            mem.rs_label = "UNKNOWN"
            mem.event_label = "UNKNOWN"
            self.rs_state = _SegmentState()
            self.event_state = _SegmentState()
            self.q2_event_state = _SegmentState()
            return refresh_memory_goal(mem, getattr(frame, "ego_to_goal_xy", None))

        if self.rng.random() < self.memory_dropout_prob:
            mem.hide_priors = True
            return refresh_memory_goal(mem, getattr(frame, "ego_to_goal_xy", None))

        if self.rs_state.remaining <= 0:
            mode = self._draw_mode(unknown_prob=self.rs_unknown_prob, wrong_prob=self.rs_wrong_prob)
            value = wrong_rs_for_frame(self.rng, mem.rs_label, str(frame.rs_label)) if mode == "wrong" else None
            self.rs_state = _SegmentState(mode=mode, value=value, remaining=self._duration())
        if self.rs_state.mode == "unknown":
            mem.rs_label = "UNKNOWN"
        elif self.rs_state.mode == "wrong":
            value = self.rs_state.value or wrong_rs_for_frame(self.rng, mem.rs_label, str(frame.rs_label))
            if value == str(frame.rs_label):
                value = wrong_rs_for_frame(self.rng, mem.rs_label, str(frame.rs_label))
                self.rs_state.value = value
            mem.rs_label = value
        rs_mode = self.rs_state.mode
        self._tick(self.rs_state)

        if self.event_state.remaining <= 0:
            mode = self._draw_mode(unknown_prob=self.event_unknown_prob, wrong_prob=self.event_wrong_prob)
            if mode == "wrong" and rs_mode == "wrong" and self.rng.random() < self.rs_wrong_event_unknown_prob:
                mode = "unknown"
            value = None
            if mode == "wrong":
                value = pick_different(self.rng, mem.event_label, event_memory_pool_for_rs(frame, mem.rs_label))
            self.event_state = _SegmentState(mode=mode, value=value, remaining=self._duration())
        if self.event_state.mode == "unknown":
            mem.event_label = "UNKNOWN"
        elif self.event_state.mode == "wrong":
            pool = event_memory_pool_for_rs(frame, mem.rs_label)
            value = self.event_state.value
            if value not in pool:
                value = pick_different(self.rng, mem.event_label, pool)
                self.event_state.value = value
            mem.event_label = str(value)
        self._tick(self.event_state)
        return refresh_memory_goal(mem, getattr(frame, "ego_to_goal_xy", None))

    def resample_event_for_q2(
        self,
        memory: Memory,
        *,
        frame: Any,
        keep_event_label: str,
    ) -> Memory:
        """为 Q2 生成 route 级块状 EVENT memory。"""

        mem = memory.copy()
        if mem.hide_priors:
            return refresh_memory_goal(mem, getattr(frame, "ego_to_goal_xy", None))
        pool = event_memory_pool_for_rs(frame, mem.rs_label)
        keep_label = str(keep_event_label)
        if keep_label not in pool:
            keep_label = "UNKNOWN"
        if self.q2_event_state.remaining <= 0:
            mode = self._draw_mode(unknown_prob=self.event_unknown_prob, wrong_prob=self.event_wrong_prob)
            value = pick_different(self.rng, keep_label, pool) if mode == "wrong" else None
            self.q2_event_state = _SegmentState(mode=mode, value=value, remaining=self._duration())
        if self.q2_event_state.mode == "unknown":
            mem.event_label = "UNKNOWN"
        elif self.q2_event_state.mode == "wrong":
            value = self.q2_event_state.value
            if value not in pool:
                value = pick_different(self.rng, keep_label, pool)
                self.q2_event_state.value = value
            mem.event_label = str(value)
        else:
            mem.event_label = keep_label
        self._tick(self.q2_event_state)
        return refresh_memory_goal(mem, getattr(frame, "ego_to_goal_xy", None))


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
