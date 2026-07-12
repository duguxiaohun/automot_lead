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
You are an autonomous driving agent. Use the stitched RGB history as visual context, ordered from oldest to newest. Focus on traffic lights/signs, nearby vehicles/pedestrians/obstacles, lane markings and road structure, and key factors affecting ego decisions. Keep the current memory by default and change it only when clear visual evidence supports the change. Describe weak, distant, foggy, or occluded evidence as uncertain. Never mention ground truth, answer keys, hidden labels, dataset rules, or scenario names."""

# loss 权重只用于训练时的 token span 加权。结构化分析段低权重，让模型学习“怎么解释”，
# 但不要让冗长自然语言压过 RS/ABNORMAL/EVENT 这几个离散答案 token。
DEFAULT_W_ANALYSIS = 0.2
DEFAULT_W_RS = 1.2
DEFAULT_W_ABNORMAL = 0.8
DEFAULT_W_EVENT = 1.2

TEACHER_MAX_NEW_TOKENS_Q1 = 256
TEACHER_MAX_NEW_TOKENS_Q2 = 192


@dataclasses.dataclass
class Memory:
    """学生跨帧维护的 v5 记忆。

    v5 的离散状态只保留 RS 和 EVENT，避免把旧 v3/v4 的 scene/status/subgoal
    泄漏进新任务。EGO_TO_GOAL_XY 不是标签，而是导航输入：它来自当前帧 meta
    的 `next_target_points[-1]` 转 ego frame，用来帮助模型在路口/匝道等场景里
    判断道路结构与行驶方向。
    """

    rs_label: str
    event_label: str = "RE"
    ego_to_goal_x: Optional[float] = None
    ego_to_goal_y: Optional[float] = None

    @property
    def rs_option(self) -> str:
        """返回 A-E 选项。"""

        return RS_LABEL_TO_OPTION.get(self.rs_label, "A")

    def copy(self) -> "Memory":
        """浅拷贝，便于状态机更新。"""

        return Memory(
            rs_label=self.rs_label,
            event_label=self.event_label,
            ego_to_goal_x=self.ego_to_goal_x,
            ego_to_goal_y=self.ego_to_goal_y,
        )

    def _road_description(self) -> str:
        """返回 memory 中使用的道路结构自然语言描述，不带 A-E 选项字母。"""

        return RS_OPTION_DESCRIPTIONS.get(self.rs_option, RS_OPTION_DESCRIPTIONS["A"])

    def _event_description(self) -> str:
        """返回 memory 中使用的事件自然语言描述，不带 RE/U-E 标签。"""

        event_desc = EVENT_DESCRIPTIONS.get(self.event_label)
        if self.event_label == "RE":
            event_desc = event_description_for_display("RE", self.rs_label)
        return event_desc or EVENT_DESCRIPTIONS["RE"]

    def _goal_text(self) -> str:
        """按 v4 同款格式渲染当前帧目的地相对坐标。"""

        if self.ego_to_goal_x is None or self.ego_to_goal_y is None:
            return "UNKNOWN"
        return f"({self.ego_to_goal_x:+.1f}, {self.ego_to_goal_y:+.1f}) m"

    def format_text(self, *, include_event: bool = True) -> str:
        """渲染给学生看的纯文本 memory。

        Q1 只看 road-only memory，不能提前暴露上一次 EVENT；Q2 才在 Q1 的
        RS 判断基础上继续使用 EVENT memory。渲染文本只写自然语言描述，不写
        A-E 或 RE/U-E 标签，避免模型把局部选项编号当成跨帧状态。
        """

        # Memory 是学生唯一可见的跨帧状态。它不包含 scenario、GT、置信度或 event_code。
        lines = [
            "[MEMORY]",
            f"BELIEVED_RS: {self._road_description()}",
        ]
        if include_event:
            lines.append(f"BELIEVED_EVENT: {self._event_description()}")
        lines.append(f"EGO_TO_GOAL_XY={self._goal_text()}")
        lines.append("[/MEMORY]")
        return "\n".join(lines)

    def format_q1_text(self) -> str:
        """Q1 使用 road-only memory。"""

        return self.format_text(include_event=False)

    def format_q2_text(self) -> str:
        """Q2 使用 road + event memory。"""

        return self.format_text(include_event=True)


def _structured_q1_format() -> str:
    """Q1 的学生/老师共享输出合同。"""

    return (
        "Scene Description: <1-2 concise sentences about visible weather/visibility, lane markings, road layout, traffic lights/signs, surrounding motion, and goal direction>\n"
        "Critical Object Description: <1-2 concise sentences naming up to 2-3 key actors or map cues, their locations/actions, likely next motion, and why they matter to ego>\n"
        "Reasoning on Intent: <1-2 concise sentences using motion, signals, lanes, ego state, and EGO_TO_GOAL_XY to decide RS and abnormality>\n"
        "RS: <A|B|C|D|E> - <copy the chosen option meaning in your own words>\n"
        "ABNORMAL: <YES|NO>"
    )


def _structured_q2_format() -> str:
    """Q2 的学生/老师共享输出合同。"""

    return (
        "Scene Description: <one concise sentence continuing from Question 1 and the current RS>\n"
        "Critical Object Description: <1-2 concise sentences naming up to 2-3 event-relevant actors or cues, or stating that no critical object is visible>\n"
        "Reasoning on Intent: <1-2 concise sentences explaining why the selected event is active or why regular behavior continues>\n"
        "EVENT: <option letter> - <copy the chosen event meaning in your own words>"
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

    Student 不看 XML weather；天气只允许从 RGB 中观察，并写进 Scene Description。
    """

    return "\n\n".join([
        memory.format_q1_text(),
        rs_choices_block(),
        (
            "[QUESTION_1]\n"
            "Analyze the latest frame in the RGB history.\n"
            "Decide:\n"
            "1. the current road-structure option from RS_CHOICES;\n"
            "2. whether an unusual event is currently happening or still affecting the ego vehicle.\n\n"
            "Use visible road geometry, lane layout, traffic lights or stop/yield cues, nearby actors, "
            "ego-path conflicts, and image-visible weather or visibility cues. Do not use a scenario name. "
            "If the evidence is weak, keep the memory unless contradicted. Keep the CoT concise.\n\n"
            "Output exactly these lines:\n"
            f"{_structured_q1_format()}\n"
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
        memory.format_q1_text(),
        rs_choices_block(),
        (
            "[REFERENCE]\n"
            f"XML_WEATHER: {weather_text}\n"
            f"ANSWER_RS: {rs_target.option} - {rs_target.description}\n"
            f"ANSWER_ABNORMAL: {abnormal}\n"
            f"ANSWER_EVENT_FOR_REASONING: {event_target.event_code}\n"
            "[/REFERENCE]\n\n"
            "[QUESTION_1_TEACHER]\n"
            "Write the same structured output format as the student. Start directly with "
            "`Scene Description:` and do not copy MEMORY, RS_CHOICES, REFERENCE, or this instruction. "
            "Use the reference only to make the visible analysis grounded and consistent. If XML weather "
            "conflicts with visible RGB weather or visibility, follow the RGB evidence. Do not mention "
            "the reference block, ground truth, answer keys, or hidden labels. Keep the CoT concise.\n\n"
            "Output exactly these lines:\n"
            f"{_structured_q1_format()}\n"
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
    ).rstrip(".")
    return "\n".join(
        [
            f"Scene Description: Describe the visible weather, lane markings, traffic controls, road layout, surrounding motion, and goal direction; the road layout supports option {rs_target.option}.",
            "Critical Object Description: Name the most relevant actor, obstacle, signal, or map cue that affects the ego path; if none is visible, state that no critical object is present.",
            f"Reasoning on Intent: The road-structure evidence supports {rs_target.option}: {rs_target.description}. The event evidence indicates that {event_phrase}.",
            f"RS: {rs_target.option} - {rs_target.description}",
            f"ABNORMAL: {abnormal}",
        ]
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
        # Q1 已经说 abnormal=yes 时，Q2 仍允许选择 RE：这是为了处理 Q1 误报或
        # 视觉证据不足的情况，不能因为第一问异常就硬塞 UE。
        task = (
            "You judged in Question 1 that an unusual event is active. Choose the listed unusual event "
            "that most directly affects the ego vehicle right now. If the latest frame does not actually "
            "support any listed unusual event, or if no unusual event is listed, choose the regular-event "
            "option instead."
        )
    else:
        # Q1 说 no abnormal 时也列出 UE 候选，让模型显式比较“保持 RE”与“确有异常”。
        # 这能训练模型在弱证据下保持 RE，而不是被候选中的 UE 诱导。
        task = (
            "You judged in Question 1 that no unusual event is active, but you must still compare the "
            "regular-event option against the listed unusual-event candidates. If the only listed choice "
            "is RE, use the analysis to explain which regular behavior is visible under the current road structure."
        )
    return "\n\n".join([
        memory.format_q2_text(),
        event_choices_block(option_map, memory.rs_label, regular_event_codes),
        (
            "[QUESTION_2]\n"
            "Decide the current event from EVENT_CHOICES. The choices have already been filtered to "
            "events that are possible for the current road structure and this route type. "
            f"{task} Do not invent an event that is not listed. Keep the CoT concise.\n\n"
            "Output exactly these lines:\n"
            f"{_structured_q2_format()}\n"
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
        memory.format_q2_text(),
        event_choices_block(option_map, memory.rs_label, regular_codes),
        (
            "[REFERENCE]\n"
            f"QUESTION_1_ABNORMAL: {'YES' if q1_abnormal else 'NO'}\n"
            f"ANSWER_EVENT: {target_option} - {target_desc}\n"
            f"ANSWER_EVENT_CODE: {event_target.event_code}\n"
            "[/REFERENCE]\n\n"
            "[QUESTION_2_TEACHER]\n"
            "Write the same structured output format as the student. Start directly with "
            "`Scene Description:` and do not copy MEMORY, EVENT_CHOICES, REFERENCE, or this instruction. "
            "Use the reference only to explain the visible event choice. Do not mention the reference block, "
            "ground truth, answer keys, or hidden labels. Keep the CoT concise.\n\n"
            "Output exactly these lines:\n"
            f"{_structured_q2_format()}\n"
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
        reasoning = (
            "The latest frame does not show one of the listed unusual events interrupting the ego path, "
            "so the regular behavior for the current road structure should continue."
        )
    else:
        reasoning = f"The latest frame supports the listed unusual event: {desc}"
    return "\n".join(
        [
            "Scene Description: Continue from the current road-structure decision and inspect the latest frame for event evidence.",
            "Critical Object Description: Name the actor, obstacle, signal, or map cue that drives the event choice; if none matters, state that no critical object is visible.",
            f"Reasoning on Intent: {reasoning}",
            f"EVENT: {option} - {desc}",
        ]
    )


_Q1_RS_RE = re.compile(r"(?im)^\s*RS\s*:\s*([A-E])\b")
_Q1_ABNORMAL_RE = re.compile(r"(?im)^\s*ABNORMAL\s*:\s*(YES|NO)\b")
_Q2_EVENT_RE = re.compile(r"(?im)^\s*EVENT\s*:\s*([A-Z])\b")
# 解析器刻意只看行首字段，不试图理解整段 analysis。这样学生可以自由解释，
# 但离散答案必须落在固定字段上，便于 eval/probe 和 memory 状态机稳定读取。


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


def reset_memory_for_frame(rs_target: RSTarget, ego_to_goal_xy: Optional[Sequence[float]] = None) -> Memory:
    """RS 错误后，下一有效帧恢复 GT RS + RE。"""

    gx: Optional[float] = None
    gy: Optional[float] = None
    if ego_to_goal_xy is not None and len(ego_to_goal_xy) >= 2:
        try:
            gx = float(ego_to_goal_xy[0])
            gy = float(ego_to_goal_xy[1])
        except Exception:
            gx = None
            gy = None
    return Memory(rs_label=rs_target.label, event_label="RE", ego_to_goal_x=gx, ego_to_goal_y=gy)


def _line_value_span(text: str, label: str) -> Optional[Tuple[int, int]]:
    """返回某个输出行冒号后的值 span。"""

    pattern = re.compile(rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$")
    match = pattern.search(text or "")
    if not match:
        return None
    return match.start(1), match.end(1)


_ANALYSIS_HEADING_RE = re.compile(
    r"(?im)^\s*(WEATHER|SCENE DESCRIPTION|CRITICAL OBJECT DESCRIPTION|REASONING|REASONING ON INTENT|MEMORY JUDGMENT|ANALYSIS)\s*:",
)


def _analysis_span(text: str, *, terminal_label: str) -> Tuple[int, int]:
    """返回结构化分析段 span；没有时退化为空 span。

    v5 新格式不再只有一行 `ANALYSIS:`，而是把天气、场景、关键物体、推理和
    memory 判断拆成多行。训练时仍把这些行统一归为低权重 analysis span，
    离散字段 `RS/ABNORMAL/EVENT` 单独加权。
    """

    first = _ANALYSIS_HEADING_RE.search(text or "")
    if not first:
        span = _line_value_span(text, "ANALYSIS")
        return span if span is not None else (0, 0)
    terminal = re.search(rf"(?im)^\s*{re.escape(terminal_label)}\s*:", text or "")
    end = terminal.start() if terminal else len(text or "")
    return first.start(), max(first.start(), end)


def target_spans_q1(text: str) -> Dict[str, Tuple[int, int]]:
    """Q1 的 token loss 字符 span。"""

    spans: Dict[str, Tuple[int, int]] = {"analysis": _analysis_span(text, terminal_label="RS")}
    rs_span = _line_value_span(text, "RS")
    abnormal_span = _line_value_span(text, "ABNORMAL")
    if rs_span is not None:
        spans["rs"] = rs_span
    if abnormal_span is not None:
        spans["abnormal"] = abnormal_span
    return spans


def target_spans_q2(text: str) -> Dict[str, Tuple[int, int]]:
    """Q2 的 token loss 字符 span。"""

    spans: Dict[str, Tuple[int, int]] = {"analysis": _analysis_span(text, terminal_label="EVENT")}
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
