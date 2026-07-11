"""SFT v5 prompt、Memory 与输出解析。

这里是 v5 文本协议的唯一来源。训练、评估和 probe 都从本文件 import，
避免不同入口对同一个 Q1/Q2 问题写出不一致格式。
"""

from __future__ import annotations

import dataclasses
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from qwen3vl_local.sft_v5.labels import (
    EVENT_DESCRIPTIONS,
    RS_LABEL_TO_OPTION,
    RS_OPTION_DESCRIPTIONS,
    RS_OPTION_TO_LABEL,
    UE_DESCRIPTIONS,
    EventTarget,
    RSTarget,
    event_description_for_display,
    option_for_event,
)


SYSTEM_PROMPT_V5 = """\
You are an autonomous driving agent. Use the stitched RGB history as visual context, ordered from oldest to newest. Keep the current memory by default and change it only when clear visual evidence supports the change. Describe weak, distant, foggy, or occluded evidence as uncertain. Never mention ground truth, answer keys, hidden labels, dataset rules, or scenario names."""

DEFAULT_W_ANALYSIS = 0.2
DEFAULT_W_RS = 1.2
DEFAULT_W_ABNORMAL = 0.8
DEFAULT_W_EVENT = 1.2

TEACHER_MAX_NEW_TOKENS_Q1 = 256
TEACHER_MAX_NEW_TOKENS_Q2 = 192


@dataclasses.dataclass
class Memory:
    """学生跨帧维护的 v5 记忆。

    v5 的 memory 故意只保留 RS 和 EVENT，避免把旧 v3/v4 的 scene/status/subgoal
    泄漏进新任务。Q1 判断 RS；Q2 判断 EVENT；若 Q1 RS 错，本帧停止，下一帧把
    memory 恢复到 GT RS + RE。
    """

    rs_label: str
    event_label: str = "RE"

    @property
    def rs_option(self) -> str:
        """返回 A-E 选项。"""

        return RS_LABEL_TO_OPTION.get(self.rs_label, "A")

    def copy(self) -> "Memory":
        """浅拷贝，便于状态机更新。"""

        return Memory(rs_label=self.rs_label, event_label=self.event_label)

    def format_text(self) -> str:
        """渲染给学生看的纯文本 memory。"""

        rs_opt = self.rs_option
        rs_desc = RS_OPTION_DESCRIPTIONS.get(rs_opt, RS_OPTION_DESCRIPTIONS["A"])
        event_desc = EVENT_DESCRIPTIONS.get(self.event_label)
        if self.event_label == "RE":
            event_desc = event_description_for_display("RE", self.rs_label)
        return (
            "[MEMORY]\n"
            f"BELIEVED_RS: {rs_opt} - {rs_desc}\n"
            f"BELIEVED_EVENT: {self.event_label} - {event_desc}\n"
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
    """渲染 Q2 的本帧随机 EVENT 选项。

    option_map 已经由 labels.stable_event_option_map 按 frame 可复现随机生成；这里
    只负责把 label 转成自然语言描述。
    """

    lines = [f"[EVENT_CHOICES under RS={RS_LABEL_TO_OPTION.get(rs_label, 'A')}]"]
    for letter in sorted(option_map):
        label = option_map[letter]
        lines.append(f"{letter}. {event_description_for_display(label, rs_label, regular_event_codes)}")
    lines.append("[/EVENT_CHOICES]")
    return "\n".join(lines)


def build_q1_student_prompt(memory: Memory) -> str:
    """Q1 student prompt。

    Student 不看 XML weather；天气只允许从 RGB 中观察，并写进 ANALYSIS。
    """

    return "\n\n".join([
        memory.format_text(),
        rs_choices_block(),
        (
            "[QUESTION_1]\n"
            "Analyze the latest frame in the RGB history.\n"
            "Decide:\n"
            "1. the current road-structure option from RS_CHOICES;\n"
            "2. whether an unusual event is currently happening or still affecting the ego vehicle.\n\n"
            "Use visible road geometry, lane layout, traffic lights or stop/yield cues, nearby actors, "
            "ego-path conflicts, and image-visible weather or visibility cues. Do not use a scenario name. "
            "If the evidence is weak, keep the memory unless contradicted.\n\n"
            "Output exactly:\n"
            "ANALYSIS: <2-5 sentences about weather/visibility, road structure, and whether an unusual event is present>\n"
            "RS: <A|B|C|D|E> - <copy the chosen option meaning in your own words>\n"
            "ABNORMAL: <YES|NO>\n"
            "[/QUESTION_1]"
        ),
    ])


def build_q1_teacher_prompt(
    memory: Memory,
    *,
    rs_target: RSTarget,
    event_target: EventTarget,
    weather_text: str,
) -> str:
    """Q1 teacher privileged prompt。

    Teacher 可以看 XML weather 与 GT，但 build_q1_teacher_target 会把 ANSWER/REFERENCE
    字段清洗掉，只给学生视角的监督文本。
    """

    abnormal = "YES" if event_target.abnormal else "NO"
    return "\n\n".join([
        memory.format_text(),
        rs_choices_block(),
        (
            "[REFERENCE]\n"
            f"XML_WEATHER: {weather_text}\n"
            f"ANSWER_RS: {rs_target.option} - {rs_target.description}\n"
            f"ANSWER_ABNORMAL: {abnormal}\n"
            f"ANSWER_EVENT_FOR_REASONING: {event_target.event_code}\n"
            "[/REFERENCE]\n\n"
            "[QUESTION_1_TEACHER]\n"
            "Write the same output format as the student. Use the reference only to make the "
            "analysis visually grounded and consistent. If XML weather conflicts with visible "
            "RGB weather or visibility, follow the RGB evidence. Do not mention the reference block, "
            "ground truth, answer keys, or hidden labels.\n"
            "[/QUESTION_1_TEACHER]"
        ),
    ])


def build_q1_teacher_target(
    *,
    rs_target: RSTarget,
    event_target: EventTarget,
    weather_text: str,
) -> str:
    """脚本化 Q1 teacher target，供 CE smoke 或 teacher 抽检兜底。

    OPSD 主训练默认用 teacher logits，不强制逐字模仿这段文本；但 inspect/probe 需要
    一个可读 target 来审计 prompt 合同。
    """

    abnormal = "YES" if event_target.abnormal else "NO"
    event_phrase = (
        UE_DESCRIPTIONS.get(event_target.label, event_target.label)
        if event_target.abnormal
        else "no unusual event visibly interrupts the ego vehicle"
    )
    return (
        f"ANALYSIS: The visible weather and visibility should be described from the RGB history. "
        f"The road layout supports option {rs_target.option} because {rs_target.description} "
        f"The current event evidence indicates {event_phrase}.\n"
        f"RS: {rs_target.option} - {rs_target.description}\n"
        f"ABNORMAL: {abnormal}"
    )


def build_q2_student_prompt(
    memory: Memory,
    *,
    option_map: Mapping[str, str],
    q1_abnormal: bool,
    regular_event_codes: Optional[Sequence[str]] = None,
) -> str:
    """Q2 student prompt。

    Q2 候选优先来自逐帧 allowed_events，缺失时才按 scenario ∩ 当前 RS fallback，
    并做 frame 级随机化。prompt 不能暴露 scenario 名。
    """

    if q1_abnormal:
        task = (
            "You judged in Question 1 that an unusual event is active. Choose the listed unusual event "
            "that most directly affects the ego vehicle right now. If the latest frame does not actually "
            "support any listed unusual event, or if no unusual event is listed, choose the regular-event "
            "option instead."
        )
    else:
        task = (
            "You judged in Question 1 that no unusual event is active, but you must still compare the "
            "regular-event option against the listed unusual-event candidates. If the only listed choice "
            "is RE, use the analysis to explain which regular behavior is visible under the current road structure."
        )
    return "\n\n".join([
        memory.format_text(),
        event_choices_block(option_map, memory.rs_label, regular_event_codes),
        (
            "[QUESTION_2]\n"
            "Decide the current event from EVENT_CHOICES. The choices have already been filtered to "
            "events that are possible for the current road structure and this route type. "
            f"{task} Do not invent an event that is not listed.\n\n"
            "Output exactly:\n"
            "ANALYSIS: <1-4 sentences explaining why the selected event is active or why regular behavior should continue>\n"
            "EVENT: <option letter> - <copy the chosen event meaning in your own words>\n"
            "[/QUESTION_2]"
        ),
    ])


def build_q2_teacher_prompt(
    memory: Memory,
    *,
    option_map: Mapping[str, str],
    q1_abnormal: bool,
    event_target: EventTarget,
    regular_event_codes: Optional[Sequence[str]] = None,
) -> str:
    """Q2 teacher privileged prompt。"""

    target_option = option_for_event(event_target.label, option_map) or "?"
    regular_codes = regular_event_codes if regular_event_codes is not None else event_target.regular_event_codes
    target_desc = event_description_for_display(event_target.label, memory.rs_label, regular_codes)
    return "\n\n".join([
        memory.format_text(),
        event_choices_block(option_map, memory.rs_label, regular_codes),
        (
            "[REFERENCE]\n"
            f"QUESTION_1_ABNORMAL: {'YES' if q1_abnormal else 'NO'}\n"
            f"ANSWER_EVENT: {target_option} - {target_desc}\n"
            f"ANSWER_EVENT_CODE: {event_target.event_code}\n"
            "[/REFERENCE]\n\n"
            "[QUESTION_2_TEACHER]\n"
            "Write the same output format as the student. Use the reference only to explain the visible "
            "event choice. Do not mention the reference block, ground truth, answer keys, or hidden labels.\n"
            "[/QUESTION_2_TEACHER]"
        ),
    ])


def build_q2_teacher_target(
    memory: Memory,
    *,
    option_map: Mapping[str, str],
    event_target: EventTarget,
    regular_event_codes: Optional[Sequence[str]] = None,
) -> str:
    """脚本化 Q2 teacher target，供 probe/smoke 审计。"""

    option = option_for_event(event_target.label, option_map)
    if option is None:
        # 候选池不含 GT 时只给一个可读诊断；主训练应记录 q2_candidate_mismatch。
        option = sorted(option_map)[0] if option_map else "A"
        chosen = option_map.get(option, "RE")
    else:
        chosen = event_target.label
    regular_codes = regular_event_codes if regular_event_codes is not None else event_target.regular_event_codes
    desc = event_description_for_display(chosen, memory.rs_label, regular_codes)
    if chosen == "RE":
        analysis = (
            "The latest frame does not show one of the listed unusual events interrupting the ego path, "
            "so the regular behavior for the current road structure should continue."
        )
    else:
        analysis = f"The latest frame supports the listed unusual event: {desc}"
    return f"ANALYSIS: {analysis}\nEVENT: {option} - {desc}"


_Q1_RS_RE = re.compile(r"(?im)^\s*RS\s*:\s*([A-E])\b")
_Q1_ABNORMAL_RE = re.compile(r"(?im)^\s*ABNORMAL\s*:\s*(YES|NO)\b")
_Q2_EVENT_RE = re.compile(r"(?im)^\s*EVENT\s*:\s*([A-Z])\b")


def parse_q1_output(text: str) -> Dict[str, Optional[str]]:
    """解析 Q1 student 输出。"""

    rs_match = _Q1_RS_RE.search(text or "")
    abnormal_match = _Q1_ABNORMAL_RE.search(text or "")
    return {
        "rs_option": rs_match.group(1).upper() if rs_match else None,
        "rs_label": RS_OPTION_TO_LABEL.get(rs_match.group(1).upper()) if rs_match else None,
        "abnormal": abnormal_match.group(1).upper() if abnormal_match else None,
    }


def parse_q2_output(text: str, option_map: Mapping[str, str]) -> Dict[str, Optional[str]]:
    """解析 Q2 student 输出，并按本帧 event_option_map 还原 label。"""

    match = _Q2_EVENT_RE.search(text or "")
    option = match.group(1).upper() if match else None
    return {
        "event_option": option,
        "event_label": option_map.get(option) if option is not None else None,
    }


def update_memory_after_q1(
    memory: Memory,
    *,
    student_rs_label: Optional[str],
    student_abnormal: Optional[bool],
) -> Memory:
    """Q1 后的 memory 更新。

    只有合法 RS 才写入 memory。若 Q1 判断 no abnormal，则 event 立即回到 RE；
    若判断 abnormal，则等待 Q2 进一步选择具体事件。
    """

    mem = memory.copy()
    if student_rs_label in RS_LABEL_TO_OPTION:
        mem.rs_label = str(student_rs_label)
    if student_abnormal is False:
        mem.event_label = "RE"
    return mem


def update_memory_after_q2(memory: Memory, *, student_event_label: Optional[str]) -> Memory:
    """Q2 后的 memory 更新。非法输出不污染 memory。"""

    mem = memory.copy()
    if student_event_label:
        mem.event_label = str(student_event_label)
    return mem


def reset_memory_for_frame(rs_target: RSTarget) -> Memory:
    """RS 错误后，下一有效帧恢复 GT RS + RE。"""

    return Memory(rs_label=rs_target.label, event_label="RE")


def _line_value_span(text: str, label: str) -> Optional[Tuple[int, int]]:
    """返回某个输出行冒号后的值 span。"""

    pattern = re.compile(rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$")
    match = pattern.search(text or "")
    if not match:
        return None
    return match.start(1), match.end(1)


def _analysis_span(text: str) -> Tuple[int, int]:
    """返回 ANALYSIS 值 span；没有时退化为空 span。"""

    span = _line_value_span(text, "ANALYSIS")
    if span is not None:
        return span
    return 0, 0


def target_spans_q1(text: str) -> Dict[str, Tuple[int, int]]:
    """Q1 的 token loss 字符 span。"""

    spans: Dict[str, Tuple[int, int]] = {"analysis": _analysis_span(text)}
    rs_span = _line_value_span(text, "RS")
    abnormal_span = _line_value_span(text, "ABNORMAL")
    if rs_span is not None:
        spans["rs"] = rs_span
    if abnormal_span is not None:
        spans["abnormal"] = abnormal_span
    return spans


def target_spans_q2(text: str) -> Dict[str, Tuple[int, int]]:
    """Q2 的 token loss 字符 span。"""

    spans: Dict[str, Tuple[int, int]] = {"analysis": _analysis_span(text)}
    event_span = _line_value_span(text, "EVENT")
    if event_span is not None:
        spans["event"] = event_span
    return spans


_PRIVATE_MARKERS = re.compile(r"ANSWER_|GROUND_TRUTH|REFERENCE|XML_WEATHER", re.IGNORECASE)


def check_no_private_markers(text: str) -> bool:
    """检查 teacher target 是否泄漏私有字段名。"""

    return _PRIVATE_MARKERS.search(text or "") is None


def loss_weights_q1() -> Dict[str, float]:
    """Q1 默认 loss 权重。"""

    return {
        "analysis": DEFAULT_W_ANALYSIS,
        "rs": DEFAULT_W_RS,
        "abnormal": DEFAULT_W_ABNORMAL,
    }


def loss_weights_q2() -> Dict[str, float]:
    """Q2 默认 loss 权重。"""

    return {
        "analysis": DEFAULT_W_ANALYSIS,
        "event": DEFAULT_W_EVENT,
    }
