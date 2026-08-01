"""SFT base prompt 与直接答案解析。

本路线复用 sft_v5 的 RS/EVENT 标签和 memory，但刻意去掉 CoT 与 teacher：
Qwen 只需要在两问里复制选项值，便于单独测试“直接答案监督”能不能学好。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
import hashlib
import random
from typing import Dict, List, Optional, Sequence, Tuple

from qwen3vl_local.sft_base.labels import (
    EVENT_LABEL_TO_TOKEN,
    EVENT_TOKEN_TO_LABEL,
    RS_DESCRIPTIONS,
    RS_LABEL_TO_TOKEN,
    RS_LABELS,
    RS_TOKEN_TO_LABEL,
    EventTarget,
    RSTarget,
    default_regular_event_for_rs,
    event_description_for_display,
    is_unusual,
)


# sft_base 不要求模型写 CoT，但仍保留 v5 的“弱证据不改 memory”策略。
# 这条短规则很重要：eval 时离散 memory 由学生输出自维护，如果 prompt 完全不约束
# memory 更新倾向，模型在远处/遮挡/雾天画面里会更容易把 RS/EVENT 改漂。
SYSTEM_PROMPT_BASE = """\
You are an autonomous driving agent. Use the stitched RGB history as visual context, ordered from oldest to newest. Keep the current memory by default and change it only when clear visual evidence supports the change. Treat weak, distant, foggy, or occluded evidence as uncertain. Answer only with the requested lines, copying one token exactly as it is written in the choices. Do not explain, do not use chain-of-thought, and do not mention ground truth, hidden labels, dataset rules, or scenario names."""

DEFAULT_W_RS = 1.2
DEFAULT_W_EVENT = 1.2


@dataclass
class Memory:
    """学生跨帧维护的轻量 memory。

    直接监督训练时使用 teacher-forced memory；eval 时则由模型自己的 Q1/Q2 输出维护。
    """

    rs_label: str
    event_label: str = "R-E1"
    ego_to_goal_x: Optional[float] = None
    ego_to_goal_y: Optional[float] = None
    hide_priors: bool = False

    @property
    def rs_token(self) -> str:
        if self.rs_label == "UNKNOWN":
            return "UNKNOWN"
        return RS_LABEL_TO_TOKEN.get(self.rs_label, "ORDINARY_ROAD")

    def copy(self) -> "Memory":
        return Memory(
            rs_label=self.rs_label,
            event_label=self.event_label,
            ego_to_goal_x=self.ego_to_goal_x,
            ego_to_goal_y=self.ego_to_goal_y,
            hide_priors=self.hide_priors,
        )

    def format_text(self, *, include_event: bool = True) -> str:
        # EGO_TO_GOAL_XY 是连续导航提示，不应像 RS/EVENT 那样跨帧沿用旧值；
        # train/eval 在每帧提问前都会 refresh_memory_goal，保证这里展示的是当前帧坐标。
        rs_token = self.rs_token
        event_token = "UNKNOWN" if self.event_label == "UNKNOWN" else EVENT_LABEL_TO_TOKEN.get(self.event_label, self.event_label)
        if self.ego_to_goal_x is None or self.ego_to_goal_y is None:
            goal_text = "UNKNOWN"
        else:
            goal_text = f"({self.ego_to_goal_x:+.1f}, {self.ego_to_goal_y:+.1f}) m"
        if self.hide_priors:
            return (
                "[MEMORY]\n"
                "PRIOR_STATE: HIDDEN_FOR_VISUAL_CHECK - classify from the RGB history, not from memory.\n"
                f"EGO_TO_GOAL_XY: {goal_text}\n"
                "[/MEMORY]"
            )
        return (
            "[MEMORY]\n"
            f"BELIEVED_RS: {rs_token}\n"
            + (f"BELIEVED_EVENT: {event_token}\n" if include_event else "")
            + f"EGO_TO_GOAL_XY: {goal_text}\n"
            "[/MEMORY]"
        )


def _stable_shuffle(items: Sequence[str], seed_text: Optional[str]) -> List[str]:
    """按 seed_text 稳定打乱展示顺序；None 时保持定义顺序。"""

    out = list(items)
    if seed_text:
        seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest(), 16) % (2**31)
        random.Random(seed).shuffle(out)
    return out


def rs_choices_block(choice_seed: Optional[str] = None) -> str:
    """渲染 Q1 的固定 RS token 选项，展示顺序可稳定打乱。"""

    lines = ["[RS_CHOICES]"]
    for label in _stable_shuffle(list(RS_LABELS), choice_seed):
        token = RS_LABEL_TO_TOKEN[label]
        lines.append(f"{token}. {RS_DESCRIPTIONS[label]}")
    lines.append("[/RS_CHOICES]")
    return "\n".join(lines)


def event_choices_block(
    candidates: Sequence[str],
    rs_label: str,
    regular_event_codes: Optional[Sequence[str]] = None,
) -> str:
    """渲染 Q2 的本帧 EVENT token 选项。

    `candidates` 是 build_dataset 存下来的有序候选 list（R-E* / U-E*），顺序即展示
    顺序，本身就是本帧的可复现随机结果，这里不再二次排序。
    """

    rs_token = RS_LABEL_TO_TOKEN.get(rs_label, "ORDINARY_ROAD")
    lines = [f"[EVENT_CHOICES under RS={rs_token}]"]
    for label in candidates:
        token = EVENT_LABEL_TO_TOKEN.get(label, label)
        lines.append(f"{token}. {event_description_for_display(label, rs_label, regular_event_codes)}")
    lines.append("[/EVENT_CHOICES]")
    return "\n".join(lines)


def build_q1_prompt(memory: Memory, *, choice_seed: Optional[str] = None) -> str:
    """Q1 student prompt：只问 RS，禁止输出解释。"""

    return "\n\n".join(
        [
            memory.format_text(include_event=False),
            rs_choices_block(choice_seed),
            (
                "[QUESTION_1]\n"
                "Look at the latest frame in the RGB history and choose the current road-structure token. "
                "If the evidence is weak, keep the memory unless it is clearly contradicted.\n\n"
                "Output exactly this line and nothing else:\n"
                "RS: <one RS token from RS_CHOICES>\n"
                "[/QUESTION_1]"
            ),
        ]
    )


def build_q1_target(*, rs_target: RSTarget, event_target: EventTarget) -> str:
    """Q1 直接监督答案。"""

    del event_target
    token = RS_LABEL_TO_TOKEN.get(rs_target.label, "ORDINARY_ROAD")
    return f"RS: {token}"


def build_q2_prompt(
    memory: Memory,
    *,
    candidates: Sequence[str],
    regular_event_codes: Optional[Sequence[str]] = None,
) -> str:
    """Q2 student prompt：只问 EVENT 选项。"""

    has_unusual_candidate = any(is_unusual(str(label)) for label in candidates)
    if has_unusual_candidate:
        question = (
            "[QUESTION_2]\n"
            "Choose the listed event token best supported by the latest frame. "
            "Choose a regular-driving token only if none of the listed unusual-event tokens is supported. "
            "Do not invent an event that is not listed.\n\n"
            "Output exactly this line and nothing else:\n"
            "EVENT: <one EVENT token from EVENT_CHOICES>\n"
            "[/QUESTION_2]"
        )
    elif memory.rs_label == "R3" and {"R-E1", "R-E2", "R-E3"}.issubset({str(label) for label in candidates}):
        question = (
            "[QUESTION_2]\n"
            "Choose the listed regular-driving token best matching the visible road geometry and ego motion "
            "in the latest frame. Use HIGHWAY_MANEUVER when lateral motion happens as part of a highway "
            "branch action such as joining from an acceleration lane, exiting to a ramp, taking a connector, "
            "or choosing a split. Use LANE_CHANGE only for an ordinary adjacent-lane move on the mainline "
            "without a ramp, connector, or split action. Use LANE_FOLLOWING when the ego stays stably in "
            "its current lane/path. Do not invent an event that is not listed.\n\n"
            "Output exactly this line and nothing else:\n"
            "EVENT: <one EVENT token from EVENT_CHOICES>\n"
            "[/QUESTION_2]"
        )
    else:
        question = (
            "[QUESTION_2]\n"
            "Choose the listed regular-driving token best matching the visible road geometry and ego motion "
            "in the latest frame. Separate stable lane following, visible lateral lane change, and "
            "highway/ramp/merge/exit structure when they are listed. Do not invent an event that is not listed.\n\n"
            "Output exactly this line and nothing else:\n"
            "EVENT: <one EVENT token from EVENT_CHOICES>\n"
            "[/QUESTION_2]"
        )
    return "\n\n".join(
        [
            memory.format_text(include_event=True),
            event_choices_block(candidates, memory.rs_label, regular_event_codes),
            question,
        ]
    )


def build_q2_target(
    memory: Memory,
    *,
    candidates: Sequence[str],
    event_target: EventTarget,
    regular_event_codes: Optional[Sequence[str]] = None,
) -> str:
    """Q2 直接监督答案。

    正常情况下调用方已经用 `event_in_candidates` 确认过真值在候选里；这里的兜底
    只在候选非空时取展示顺序第一项，保证 target 一定是本帧 prompt 列出过的 token。
    """

    del regular_event_codes
    labels = [str(item) for item in candidates]
    label = event_target.label if event_target.label in labels else (labels[0] if labels else default_regular_event_for_rs(memory.rs_label))
    token = EVENT_LABEL_TO_TOKEN.get(label, label)
    return f"EVENT: {token}"


_TOKEN_VALUE_RE = r"([A-Z][A-Z0-9_-]*)"
_Q1_RS_RE = re.compile(rf"(?im)^\s*RS\s*:\s*{_TOKEN_VALUE_RE}\b")
_Q2_EVENT_RE = re.compile(rf"(?im)^\s*EVENT\s*:\s*{_TOKEN_VALUE_RE}\b")


def parse_q1_output(text: str) -> Dict[str, Optional[str]]:
    """解析 Q1 输出。"""

    rs_match = _Q1_RS_RE.search(text or "")
    token = rs_match.group(1).upper().replace("-", "_") if rs_match else None
    return {
        "rs_token": token,
        "rs_label": RS_TOKEN_TO_LABEL.get(token) if token else None,
    }


def parse_q2_output(text: str, candidates: Sequence[str]) -> Dict[str, Optional[str]]:
    """解析 Q2 输出，并校验 token 是否是本帧列出过的候选。

    模型可能吐出一个合法的全局 EVENT token，但它并不在本帧候选里；这种情况按非法
    处理（`event_label=None`），不更新 memory，也不计为正确。
    """

    match = _Q2_EVENT_RE.search(text or "")
    token = match.group(1).upper().replace("-", "_") if match else None
    label = EVENT_TOKEN_TO_LABEL.get(token) if token else None
    allowed = {str(item) for item in candidates}
    if label not in allowed:
        label = None
    return {
        "event_token": token,
        "event_label": label,
    }


def _line_value_span(text: str, label: str) -> Optional[Tuple[int, int]]:
    pattern = re.compile(rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$")
    match = pattern.search(text or "")
    if not match:
        return None
    return match.start(1), match.end(1)


def target_spans_q1(text: str) -> Dict[str, Tuple[int, int]]:
    """Q1 只监督 RS 的值。"""

    spans: Dict[str, Tuple[int, int]] = {}
    rs_span = _line_value_span(text, "RS")
    if rs_span is not None:
        spans["rs"] = rs_span
    return spans


def target_spans_q2(text: str) -> Dict[str, Tuple[int, int]]:
    """Q2 只监督 EVENT 的值。"""

    event_span = _line_value_span(text, "EVENT")
    return {"event": event_span} if event_span is not None else {}


def loss_weights_q1() -> Dict[str, float]:
    return {"rs": DEFAULT_W_RS}


def loss_weights_q2() -> Dict[str, float]:
    return {"event": DEFAULT_W_EVENT}


def update_memory_after_q1(
    memory: Memory,
    *,
    student_rs_label: Optional[str],
) -> Memory:
    """Q1 后更新 memory；RS hypothesis 改变时旧 EVENT 失效。"""

    mem = memory.copy()
    if student_rs_label in RS_LABELS:
        if str(student_rs_label) != mem.rs_label:
            mem.event_label = "UNKNOWN"
        mem.rs_label = str(student_rs_label)
    return mem


def update_memory_after_q2(memory: Memory, *, student_event_label: Optional[str]) -> Memory:
    """Q2 后更新 EVENT memory；非法输出不污染状态。"""

    mem = memory.copy()
    if student_event_label:
        mem.event_label = str(student_event_label)
    return mem


def refresh_memory_goal(memory: Memory, ego_to_goal_xy: Optional[Sequence[float]]) -> Memory:
    """把当前帧 ego-frame goal 写入 memory，离散 RS/EVENT 保持不变。

    v5 早期逻辑只在 route 首帧/reset 时刷新 goal，后续 copy 旧坐标；这里显式每帧
    刷新，避免 prompt 里的连续导航目标 stale。这个函数只改坐标，不改 RS/EVENT，
    所以不会破坏 teacher-forced 或 student 自维护的离散状态。
    """

    mem = memory.copy()
    mem.ego_to_goal_x = None
    mem.ego_to_goal_y = None
    if ego_to_goal_xy is not None and len(ego_to_goal_xy) >= 2:
        try:
            mem.ego_to_goal_x = float(ego_to_goal_xy[0])
            mem.ego_to_goal_y = float(ego_to_goal_xy[1])
        except Exception:
            mem.ego_to_goal_x = None
            mem.ego_to_goal_y = None
    return mem


def reset_memory_for_frame(rs_target: RSTarget, ego_to_goal_xy: Optional[Sequence[float]] = None) -> Memory:
    """首帧或 reset 后恢复 GT RS + 当前 RS 的默认 regular EVENT。"""

    return refresh_memory_goal(Memory(rs_label=rs_target.label, event_label=default_regular_event_for_rs(rs_target.label)), ego_to_goal_xy)
