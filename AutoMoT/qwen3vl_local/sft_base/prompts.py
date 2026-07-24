"""SFT base prompt 与直接答案解析。

本路线复用 sft_v5 的 RS/EVENT 标签和 memory，但刻意去掉 CoT 与 teacher：
Qwen 只需要在两问里复制选项值，便于单独测试“直接答案监督”能不能学好。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from qwen3vl_local.sft_base.labels import (
    RS_LABEL_TO_OPTION,
    RS_OPTION_DESCRIPTIONS,
    RS_OPTION_TO_LABEL,
    EventTarget,
    RSTarget,
    event_description_for_display,
    option_for_event,
)


# sft_base 不要求模型写 CoT，但仍保留 v5 的“弱证据不改 memory”策略。
# 这条短规则很重要：eval 时离散 memory 由学生输出自维护，如果 prompt 完全不约束
# memory 更新倾向，模型在远处/遮挡/雾天画面里会更容易把 RS/EVENT 改漂。
SYSTEM_PROMPT_BASE = """\
You are an autonomous driving agent. Use the stitched RGB history as visual context, ordered from oldest to newest. Keep the current memory by default and change it only when clear visual evidence supports the change. Treat weak, distant, foggy, or occluded evidence as uncertain. Answer only with the requested option lines. Do not explain, do not use chain-of-thought, and do not mention ground truth, hidden labels, dataset rules, or scenario names."""

DEFAULT_W_RS = 1.2
DEFAULT_W_ABNORMAL = 0.8
DEFAULT_W_EVENT = 1.2


@dataclass
class Memory:
    """学生跨帧维护的轻量 memory。

    直接监督训练时使用 teacher-forced memory；eval 时则由模型自己的 Q1/Q2 输出维护。
    """

    rs_label: str
    event_label: str = "RE"
    ego_to_goal_x: Optional[float] = None
    ego_to_goal_y: Optional[float] = None

    @property
    def rs_option(self) -> str:
        return RS_LABEL_TO_OPTION.get(self.rs_label, "A")

    def copy(self) -> "Memory":
        return Memory(
            rs_label=self.rs_label,
            event_label=self.event_label,
            ego_to_goal_x=self.ego_to_goal_x,
            ego_to_goal_y=self.ego_to_goal_y,
        )

    def format_text(self) -> str:
        # EGO_TO_GOAL_XY 是连续导航提示，不应像 RS/EVENT 那样跨帧沿用旧值；
        # train/eval 在每帧提问前都会 refresh_memory_goal，保证这里展示的是当前帧坐标。
        rs_opt = self.rs_option
        rs_desc = RS_OPTION_DESCRIPTIONS.get(rs_opt, RS_OPTION_DESCRIPTIONS["A"])
        event_desc = event_description_for_display(self.event_label, self.rs_label)
        if self.ego_to_goal_x is None or self.ego_to_goal_y is None:
            goal_text = "UNKNOWN"
        else:
            goal_text = f"({self.ego_to_goal_x:+.1f}, {self.ego_to_goal_y:+.1f}) m"
        return (
            "[MEMORY]\n"
            f"BELIEVED_RS: {rs_opt} - {rs_desc}\n"
            f"BELIEVED_EVENT: {self.event_label} - {event_desc}\n"
            f"EGO_TO_GOAL_XY: {goal_text}\n"
            "[/MEMORY]"
        )


def rs_choices_block() -> str:
    """渲染 Q1 的固定 RS A-E 选项。"""

    lines = ["[RS_CHOICES]"]
    for option in ("A", "B", "C", "D", "E"):
        lines.append(f"{option}. {RS_OPTION_DESCRIPTIONS[option]}")
    lines.append("[/RS_CHOICES]")
    return "\n".join(lines)


def event_choices_block(
    option_map: Mapping[str, str],
    rs_label: str,
    regular_event_codes: Optional[Sequence[str]] = None,
) -> str:
    """渲染 Q2 的本帧随机 EVENT 选项。"""

    lines = [f"[EVENT_CHOICES under RS={RS_LABEL_TO_OPTION.get(rs_label, 'A')}]"]
    for letter in sorted(option_map):
        label = option_map[letter]
        lines.append(f"{letter}. {event_description_for_display(label, rs_label, regular_event_codes)}")
    lines.append("[/EVENT_CHOICES]")
    return "\n".join(lines)


def build_q1_prompt(memory: Memory) -> str:
    """Q1 student prompt：只问 RS 与 abnormal，禁止输出解释。"""

    return "\n\n".join(
        [
            memory.format_text(),
            rs_choices_block(),
            (
                "[QUESTION_1]\n"
                "Look at the latest frame in the RGB history and choose the current road-structure option. "
                "Also decide whether an unusual event is currently active or still affecting the ego vehicle. "
                "If the evidence is weak, keep the memory unless it is clearly contradicted.\n\n"
                "Output exactly these two lines and nothing else:\n"
                "RS: <A|B|C|D|E>\n"
                "ABNORMAL: <YES|NO>\n"
                "[/QUESTION_1]"
            ),
        ]
    )


def build_q1_target(*, rs_target: RSTarget, event_target: EventTarget) -> str:
    """Q1 直接监督答案。"""

    abnormal = "YES" if event_target.abnormal else "NO"
    return f"RS: {rs_target.option}\nABNORMAL: {abnormal}"


def build_q2_prompt(
    memory: Memory,
    *,
    option_map: Mapping[str, str],
    q1_abnormal: bool,
    regular_event_codes: Optional[Sequence[str]] = None,
) -> str:
    """Q2 student prompt：只问 EVENT 选项。"""

    if q1_abnormal:
        task = (
            "Question 1 said an unusual event may be active. Choose the listed event that best matches "
            "the latest frame. If none of the unusual-event options is supported, choose the regular-event option."
        )
    else:
        task = (
            "Question 1 said no unusual event is active. Still compare every listed option and choose the "
            "one best supported by the latest frame."
        )
    return "\n\n".join(
        [
            memory.format_text(),
            event_choices_block(option_map, memory.rs_label, regular_event_codes),
            (
                "[QUESTION_2]\n"
                f"{task} Do not invent an event that is not listed.\n\n"
                "Output exactly this line and nothing else:\n"
                "EVENT: <option letter>\n"
                "[/QUESTION_2]"
            ),
        ]
    )


def build_q2_target(
    memory: Memory,
    *,
    option_map: Mapping[str, str],
    event_target: EventTarget,
    regular_event_codes: Optional[Sequence[str]] = None,
) -> str:
    """Q2 直接监督答案。"""

    del memory, regular_event_codes
    option = option_for_event(event_target.label, option_map)
    if option is None:
        option = sorted(option_map)[0] if option_map else "A"
    return f"EVENT: {option}"


_Q1_RS_RE = re.compile(r"(?im)^\s*RS\s*:\s*([A-E])\b")
_Q1_ABNORMAL_RE = re.compile(r"(?im)^\s*ABNORMAL\s*:\s*(YES|NO)\b")
_Q2_EVENT_RE = re.compile(r"(?im)^\s*EVENT\s*:\s*([A-Z])\b")


def parse_q1_output(text: str) -> Dict[str, Optional[str]]:
    """解析 Q1 输出。"""

    rs_match = _Q1_RS_RE.search(text or "")
    abnormal_match = _Q1_ABNORMAL_RE.search(text or "")
    return {
        "rs_option": rs_match.group(1).upper() if rs_match else None,
        "rs_label": RS_OPTION_TO_LABEL.get(rs_match.group(1).upper()) if rs_match else None,
        "abnormal": abnormal_match.group(1).upper() if abnormal_match else None,
    }


def parse_q2_output(text: str, option_map: Mapping[str, str]) -> Dict[str, Optional[str]]:
    """解析 Q2 输出，并按本帧随机 option map 还原 EVENT label。"""

    match = _Q2_EVENT_RE.search(text or "")
    option = match.group(1).upper() if match else None
    return {
        "event_option": option,
        "event_label": option_map.get(option) if option is not None else None,
    }


def _line_value_span(text: str, label: str) -> Optional[Tuple[int, int]]:
    pattern = re.compile(rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$")
    match = pattern.search(text or "")
    if not match:
        return None
    return match.start(1), match.end(1)


def target_spans_q1(text: str) -> Dict[str, Tuple[int, int]]:
    """Q1 只监督 RS 与 ABNORMAL 的值。"""

    spans: Dict[str, Tuple[int, int]] = {}
    rs_span = _line_value_span(text, "RS")
    abnormal_span = _line_value_span(text, "ABNORMAL")
    if rs_span is not None:
        spans["rs"] = rs_span
    if abnormal_span is not None:
        spans["abnormal"] = abnormal_span
    return spans


def target_spans_q2(text: str) -> Dict[str, Tuple[int, int]]:
    """Q2 只监督 EVENT 的值。"""

    event_span = _line_value_span(text, "EVENT")
    return {"event": event_span} if event_span is not None else {}


def loss_weights_q1() -> Dict[str, float]:
    return {"rs": DEFAULT_W_RS, "abnormal": DEFAULT_W_ABNORMAL}


def loss_weights_q2() -> Dict[str, float]:
    return {"event": DEFAULT_W_EVENT}


def update_memory_after_q1(
    memory: Memory,
    *,
    student_rs_label: Optional[str],
    student_abnormal: Optional[bool],
) -> Memory:
    """Q1 后更新 memory；非法 RS 不污染状态。"""

    mem = memory.copy()
    if student_rs_label in RS_LABEL_TO_OPTION:
        mem.rs_label = str(student_rs_label)
    if student_abnormal is False:
        mem.event_label = "RE"
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
    """首帧或 reset 后恢复 GT RS + RE。"""

    return refresh_memory_goal(Memory(rs_label=rs_target.label, event_label="RE"), ego_to_goal_xy)
