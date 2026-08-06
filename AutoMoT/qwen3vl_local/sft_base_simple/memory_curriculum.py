"""SFT base simple memory 扰动课程。

本模块故意不 import torch，便于用纯 Python 测试分布边际。训练入口只负责把
FrameRow/Memory 传进来，真正的 wrong/UNKNOWN/dropout 逻辑都在这里。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, List, Optional

from qwen3vl_local.sft_base_simple.labels import (
    EVENT_CANDIDATES_BY_RS,
    EVENT_ORDER,
    RS_DESCRIPTIONS,
    collapse_regular_to_re,
    default_regular_event_for_rs,
    event_family_from_label,
    is_regular_event,
    is_unusual,
)
from qwen3vl_local.sft_base_simple.prompts import Memory, refresh_memory_goal


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
    """生成 ROAD 二分类意义上的对抗扰动。

    当前 baseline 只看 HIGHWAY / NON_HIGHWAY，所以 wrong memory 必须跨这个二分类
    边界：GT=R3 时扰成非高速，GT!=R3 时扰成高速。不能只在 R1/R2/R4/R5 内部
    换标签，否则 prompt 折叠后仍然是正确的 NON_HIGHWAY。
    """

    del rng, current
    if gt_rs == "R3":
        return "R1"
    return "R3"


def wrong_event_for_frame(rng: random.Random, current: str, frame: Any, rs_label: str) -> str:
    """生成 EVENT 二分类意义上的对抗扰动。

    GT 为 UE 时 wrong 必须给 RE；GT 为 RE 时 wrong 必须给当前 RS 下可见/合理的 UE。
    如果该 RS 静态表没有 UE，退回全局 UE，保证 PREVIOUS_EVENT 展示为真正错误的 family。
    """

    del current
    if bool(getattr(frame, "abnormal", False)):
        return default_regular_event_for_rs(str(rs_label))
    pool = [
        code
        for code in event_memory_pool_for_rs(frame, str(rs_label))
        if is_unusual(str(code))
    ]
    if not pool:
        pool = [code for code in EVENT_ORDER if is_unusual(str(code))]
    return rng.choice(sorted(set(pool)))


def event_memory_pool_for_rs(frame: Any, rs_label: str) -> List[str]:
    """按给定 RS 生成内部自洽的 EVENT memory 候选。"""

    del frame
    if rs_label not in RS_DESCRIPTIONS:
        return [default_regular_event_for_rs("R1")]
    raw = list(EVENT_CANDIDATES_BY_RS.get(rs_label, []))
    return collapse_regular_to_re(raw, rs_label)


def _clip_prob(value: float) -> float:
    """把概率裁剪到 [0, 1]。"""

    return min(1.0, max(0.0, float(value)))


def _scaled_event_probs(
    *,
    event_wrong_prob: float,
    event_unknown_prob: float,
    memory_dropout_prob: float,
    early_ue_age: Optional[int],
    early_ue_frames: int,
    early_ue_wrong_scale: float,
    early_ue_unknown_scale: float,
    early_ue_dropout_scale: float,
) -> tuple[float, float, float, bool]:
    """按 early-UE 课程放大 EVENT memory 噪声。

    base simple 仍保留 baseline 的基础 wrong/UNKNOWN/dropout 比例；只有当前帧
    处在连续 UE span 开头时，额外提高 EVENT memory 的不可信程度。这样首段
    UE 不会总是看到 PREVIOUS_EVENT=UE，模型必须从图像触发 UE。
    """

    is_early_ue = early_ue_age is not None and 0 <= int(early_ue_age) < max(0, int(early_ue_frames))
    wrong = _clip_prob(event_wrong_prob)
    unknown = _clip_prob(event_unknown_prob)
    dropout = _clip_prob(memory_dropout_prob)
    if is_early_ue:
        wrong = _clip_prob(float(event_wrong_prob) * max(0.0, float(early_ue_wrong_scale)))
        unknown = _clip_prob(float(event_unknown_prob) * max(0.0, float(early_ue_unknown_scale)))
        dropout = _clip_prob(float(memory_dropout_prob) * max(0.0, float(early_ue_dropout_scale)))
    if is_early_ue:
        # early-UE 期间保留 keep 概率地板，不让归一化把 keep 压成 0。
        # wrong+unknown 上限定在 0.85，留 >=15% 给 keep，使 UE 触发帧有机会
        # 看到 memory=RE -> answer=UE 的正确先验。
        ue_cap = 0.85
        total = wrong + unknown
        if total > ue_cap:
            scale = ue_cap / total
            wrong *= scale
            unknown *= scale
    else:
        total = wrong + unknown
        if total > 1.0:
            scale = 1.0 / total
            wrong *= scale
            unknown *= scale
    return (wrong, unknown, dropout, bool(is_early_ue))


@dataclass
class _SegmentState:
    """route 级扰动段状态。"""

    mode: str = "keep"
    value: Optional[str] = None
    remaining: int = 0


class RouteMemoryCorruptor:
    """把逐帧独立 memory 扰动改成 route 级块状扰动。

    sft_base_simple 保持 off-policy，不用学生 rollout；这个状态机只模拟“错误 memory 会连续
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
        early_ue_frames: int = 4,
        early_ue_wrong_scale: float = 1.75,
        early_ue_unknown_scale: float = 1.35,
        early_ue_dropout_scale: float = 1.50,
        early_ue_resample_prob: float = 0.70,
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
        self.early_ue_frames = max(0, int(early_ue_frames))
        self.early_ue_wrong_scale = max(0.0, float(early_ue_wrong_scale))
        self.early_ue_unknown_scale = max(0.0, float(early_ue_unknown_scale))
        self.early_ue_dropout_scale = max(0.0, float(early_ue_dropout_scale))
        self.early_ue_resample_prob = _clip_prob(early_ue_resample_prob)
        self.rng = random.Random(stable_route_seed(route_id=self.route_id, seed=self.seed, salt="route_memory"))
        self.rs_state = _SegmentState()
        self.event_state = _SegmentState()

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

    def corrupt(self, memory: Memory, *, frame: Any, frame_pos: int, early_ue_age: Optional[int] = None) -> Memory:
        """生成 Q1 prompt 使用的 route 级块状扰动 memory。"""

        mem = memory.copy()
        event_wrong_prob, event_unknown_prob, dropout_prob, is_early_ue = _scaled_event_probs(
            event_wrong_prob=self.event_wrong_prob,
            event_unknown_prob=self.event_unknown_prob,
            memory_dropout_prob=self.memory_dropout_prob,
            early_ue_age=early_ue_age,
            early_ue_frames=self.early_ue_frames,
            early_ue_wrong_scale=self.early_ue_wrong_scale,
            early_ue_unknown_scale=self.early_ue_unknown_scale,
            early_ue_dropout_scale=self.early_ue_dropout_scale,
        )
        if self.first_frame_unknown and int(frame_pos) == 0:
            mem.rs_label = "UNKNOWN"
            mem.event_label = "UNKNOWN"
            self.rs_state = _SegmentState()
            self.event_state = _SegmentState()
            return refresh_memory_goal(mem, getattr(frame, "ego_to_goal_xy", None))

        if self.rng.random() < dropout_prob:
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
            mode = self._draw_mode(unknown_prob=event_unknown_prob, wrong_prob=event_wrong_prob)
            if mode == "wrong" and rs_mode == "wrong" and self.rng.random() < self.rs_wrong_event_unknown_prob:
                mode = "unknown"
            value = None
            if mode == "wrong":
                value = wrong_event_for_frame(self.rng, mem.event_label, frame, mem.rs_label)
            self.event_state = _SegmentState(mode=mode, value=value, remaining=self._duration())
        if self.event_state.mode == "unknown":
            mem.event_label = "UNKNOWN"
        elif self.event_state.mode == "wrong":
            pool = event_memory_pool_for_rs(frame, mem.rs_label)
            value = self.event_state.value
            if value not in pool:
                value = wrong_event_for_frame(self.rng, mem.event_label, frame, mem.rs_label)
                self.event_state.value = value
            mem.event_label = str(value)
        self._tick(self.event_state)
        # Post-perturbation early-UE directional guard:
        # Apply perturbation first, then check if the result landed on UE family.
        # During early-UE, if event memory would be pushed to UE, force a redraw
        # to RE or UNKNOWN (controlled by early_ue_resample_prob).
        # This prevents the model from learning the shortcut
        # "memory says UE -> answer UE" at the start of every UE span.
        if is_early_ue:
            result_family = "UNKNOWN" if mem.event_label == "UNKNOWN" else event_family_from_label(mem.event_label)
            if result_family == "UE" and self.rng.random() < self.early_ue_resample_prob:
                pool = event_memory_pool_for_rs(frame, mem.rs_label)
                re_pool = [c for c in pool if not is_unusual(c)]
                if re_pool:
                    mem.event_label = self.rng.choice(sorted(set(re_pool)))
                else:
                    mem.event_label = "UNKNOWN"
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
    early_ue_age: Optional[int] = None,
    early_ue_frames: int = 4,
    early_ue_wrong_scale: float = 1.75,
    early_ue_unknown_scale: float = 1.35,
    early_ue_dropout_scale: float = 1.50,
) -> Memory:
    """训练时把 prompt 里的 memory 从“答案”降级成“不可靠先验”。"""

    mem = memory.copy()
    rng = random.Random(stable_frame_seed(route_id=route_id, frame_id=int(frame.frame_id), seed=seed, salt="memory"))
    event_wrong_prob, event_unknown_prob, dropout_prob, _is_early_ue = _scaled_event_probs(
        event_wrong_prob=event_wrong_prob,
        event_unknown_prob=event_unknown_prob,
        memory_dropout_prob=memory_dropout_prob,
        early_ue_age=early_ue_age,
        early_ue_frames=early_ue_frames,
        early_ue_wrong_scale=early_ue_wrong_scale,
        early_ue_unknown_scale=early_ue_unknown_scale,
        early_ue_dropout_scale=early_ue_dropout_scale,
    )
    if first_frame_unknown and frame_pos == 0:
        mem.rs_label = "UNKNOWN"
        mem.event_label = "UNKNOWN"
        return refresh_memory_goal(mem, getattr(frame, "ego_to_goal_xy", None))

    if rng.random() < dropout_prob:
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
    if event_draw < event_unknown_prob:
        mem.event_label = "UNKNOWN"
    elif rs_mode == "wrong":
        if rng.random() < min(1.0, max(0.0, float(rs_wrong_event_unknown_prob))):
            mem.event_label = "UNKNOWN"
        else:
            mem.event_label = wrong_event_for_frame(rng, mem.event_label, frame, mem.rs_label)
    elif event_draw < event_unknown_prob + event_wrong_prob:
        mem.event_label = wrong_event_for_frame(rng, mem.event_label, frame, mem.rs_label)
    return refresh_memory_goal(mem, getattr(frame, "ego_to_goal_xy", None))



    if is_early_ue:
        # early-UE 期间保留 keep 概率地板，不让归一化把 keep 压成 0。
        # wrong+unknown 上限定在 0.85，留 >=15% 给 keep，使 UE 触发帧有机会
        # 看到 "memory=RE -> answer=UE" 的正确先验。
        ue_cap = 0.85
        total = wrong + unknown
        if total > ue_cap:
            scale = ue_cap / total
            wrong *= scale
            unknown *= scale
    else:
        total = wrong + unknown
        if total > 1.0:
            scale = 1.0 / total
            wrong *= scale
            unknown *= scale
    return (wrong, unknown, dropout, bool(is_early_ue))
