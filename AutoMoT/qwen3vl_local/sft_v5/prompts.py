"""SFT v5 prompt、Memory 与输出解析。

这里是 v5 文本协议的唯一来源。训练、评估和 probe 都从本文件 import，
避免不同入口对同一个 Q1/Q2 问题写出不一致格式。

调用顺序是：用 ``reset_memory_for_frame`` 初始化，构造 Q1 prompt 并解析输出，调用
``update_memory_after_q1``；只有 RS 正确时再构造 Q2 prompt、解析 EVENT 并调用
``update_memory_after_q2``。teacher builder 只供 OPSD 私有 forward 和审计产物使用。
"""

from __future__ import annotations

import dataclasses
import hashlib
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
You are an autonomous driving agent. Use the stitched RGB history as visual context, ordered from oldest to newest. Focus on traffic lights/signs, nearby vehicles/pedestrians/obstacles, lane markings and road structure, and key factors affecting ego decisions. Memory is only an unverified previous hypothesis: it may be stale or wrong, so decide from current visual evidence first and change memory whenever the evidence contradicts it. Describe weak, distant, foggy, or occluded evidence as uncertain. Never mention ground truth, answer keys, hidden labels, dataset rules, or scenario names."""

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

        return RS_LABEL_TO_OPTION.get(self.rs_label, "?")

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

        if self.rs_label not in RS_LABEL_TO_OPTION:
            return "No reliable previous road-structure hypothesis is available."
        return RS_OPTION_DESCRIPTIONS[self.rs_option]

    def _event_description(self) -> str:
        """返回 memory 中使用的事件自然语言描述，不带 RE/U-E 标签。"""

        if self.event_label not in EVENT_DESCRIPTIONS:
            return "No reliable previous event hypothesis is available."
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
            f"PREVIOUS_RS_HYPOTHESIS: {self._road_description()}",
        ]
        if include_event:
            lines.append(f"PREVIOUS_EVENT_HYPOTHESIS: {self._event_description()}")
        lines.append("MEMORY_RELIABILITY: unverified previous model output; it may be stale or wrong")
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
        "RS: <A|B|C|D|E>\n"
        "ABNORMAL: <YES|NO>"
    )


def _structured_q2_format() -> str:
    """Q2 的学生/老师共享输出合同。"""

    return (
        "Scene Description: <one concise sentence continuing from Question 1 and the current RS>\n"
        "Critical Object Description: <1-2 concise sentences naming up to 2-3 event-relevant actors or cues, or stating that no critical object is visible>\n"
        "Reasoning on Intent: <1-2 concise sentences explaining why the selected event is active or why regular behavior continues>\n"
        "EVENT: <option letter>"
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
            "First reach an independent decision from the RGB evidence. Treat PREVIOUS_RS_HYPOTHESIS as "
            "a fallible temporal hint, not as an answer. If visible geometry or traffic control contradicts "
            "it, the final RS must follow the current image. Keep the CoT concise.\n\n"
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
            f"RS: {rs_target.option}",
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
            f"EVENT: {option}",
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


def should_trigger_q2(*, student_rs_label: Optional[str], target_rs_label: str) -> bool:
    """只有本帧 Q1 的 RS 回答正确时才允许进入 Q2。

    RS memory 错误不会让 Q1 休眠：训练/eval/probe 在下一有效帧仍要重新运行 Q1，
    直到学生自行纠正，或训练期 curriculum 达到 patience 后执行兜底修复。这个函数
    只定义“当前帧是否追问 EVENT”，不能被用来跳过下一帧 Q1。
    """

    return student_rs_label == str(target_rs_label)


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

    只有合法 RS 才写入 memory。Q1 的 ABNORMAL 只是进入 Q2 前的粗判断，不能在
    脚本里直接把 EVENT 改成 RE；否则错误 EVENT memory 会在 Q2 真正比较视觉证据前
    被自动清掉，模型永远学不到 EVENT 自主纠偏。
    """

    mem = memory.copy()
    if student_rs_label in RS_LABEL_TO_OPTION:
        mem.rs_label = str(student_rs_label)
    return mem


def update_memory_after_q2(memory: Memory, *, student_event_label: Optional[str]) -> Memory:
    """Q2 后的 memory 更新。非法输出不污染 memory。"""

    mem = memory.copy()
    if student_event_label:
        mem.event_label = str(student_event_label)
    return mem


def update_memory_navigation(
    memory: Memory,
    ego_to_goal_xy: Optional[Sequence[float]],
) -> Memory:
    """只刷新当前帧导航坐标，不改 RS/EVENT 离散状态。

    连续序列中的 RS/EVENT 应由学生上一帧输出维护；但 ``EGO_TO_GOAL_XY`` 是当前帧
    meta 提供的外部导航输入，必须每帧刷新。测试时调用本函数不会形成 GT 标签纠错，
    因为它只写坐标；训练的错误后 GT reset 仍由外层显式调用
    :func:`reset_memory_for_frame`。
    """

    mem = memory.copy()
    if ego_to_goal_xy is None or len(ego_to_goal_xy) < 2:
        mem.ego_to_goal_x = None
        mem.ego_to_goal_y = None
        return mem
    try:
        mem.ego_to_goal_x = float(ego_to_goal_xy[0])
        mem.ego_to_goal_y = float(ego_to_goal_xy[1])
    except Exception:
        mem.ego_to_goal_x = None
        mem.ego_to_goal_y = None
    return mem


def reset_memory_for_frame(rs_target: RSTarget, ego_to_goal_xy: Optional[Sequence[float]] = None) -> Memory:
    """构造 GT RS + RE reference/兼容初始化；正式训练的延迟修复由 curriculum 管理。"""

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


@dataclasses.dataclass(frozen=True)
class MemoryCurriculumConfig:
    """训练期错误记忆课程参数。

    RS 是慢变量，默认给更长的自主修复窗口并只按较疏的 review interval 执行强制
    修复；EVENT 是快变量，每帧都复核，较短 patience 后才使用 GT 兜底。扰动只在
    当前 memory 原本正确时注入；若学生复制了错误 hypothesis，它会自然延续到后续帧，
    从而形成真正的 closed-loop 纠偏样本，而不是每帧互不关联的随机噪声。
    """

    rs_error_patience: int = 4
    event_error_patience: int = 3
    rs_repair_interval: int = 2
    event_repair_interval: int = 1
    rs_corrupt_prob: float = 0.06
    rs_unknown_prob: float = 0.02
    event_corrupt_prob: float = 0.10
    event_unknown_prob: float = 0.05
    rs_initial_gt_prob: float = 0.5
    event_initial_gt_prob: float = 0.5

    def validate(self) -> None:
        """拒绝会破坏概率或延迟语义的配置。"""

        for name in (
            "rs_error_patience",
            "event_error_patience",
            "rs_repair_interval",
            "event_repair_interval",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be >= 1")
        for prefix in ("rs", "event"):
            corrupt = float(getattr(self, f"{prefix}_corrupt_prob"))
            unknown = float(getattr(self, f"{prefix}_unknown_prob"))
            initial = float(getattr(self, f"{prefix}_initial_gt_prob"))
            if corrupt < 0.0 or unknown < 0.0 or corrupt + unknown > 1.0:
                raise ValueError(f"{prefix} corruption/unknown probabilities must sum to [0, 1]")
            if not 0.0 <= initial <= 1.0:
                raise ValueError(f"{prefix}_initial_gt_prob must be in [0, 1]")


@dataclasses.dataclass
class MemoryCurriculumState:
    """单条 route 的训练期延迟修复状态。"""

    frames_seen: int = 0
    rs_error_streak: int = 0
    event_error_streak: int = 0
    rs_repair_pending: bool = False
    event_repair_pending: bool = False


def _stable_unit_interval(seed: int, *parts: object) -> float:
    """把 route/frame/key 稳定映射到 [0, 1)，保证 DDP 与重跑可复现。"""

    payload = "|".join([str(int(seed)), *(str(part) for part in parts)]).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
    return float(value) / float(1 << 64)


def _stable_different_label(
    current: str,
    choices: Sequence[str],
    *,
    seed: int,
    key: Sequence[object],
) -> str:
    """从候选中可复现地选择一个不同标签。"""

    alternatives = [str(item) for item in choices if str(item) != str(current)]
    if not alternatives:
        return "UNKNOWN"
    draw = _stable_unit_interval(seed, *key, "choice")
    index = min(len(alternatives) - 1, int(draw * len(alternatives)))
    return alternatives[index]


def prepare_training_memory(
    memory: Optional[Memory],
    state: MemoryCurriculumState,
    config: MemoryCurriculumConfig,
    *,
    gt_rs_label: str,
    gt_event_label: str,
    accepted_event_labels: Optional[Sequence[str]] = None,
    event_corruption_choices: Optional[Sequence[str]] = None,
    ego_to_goal_xy: Optional[Sequence[float]],
    route_key: str,
    frame_id: int,
    epoch: int,
    seed: int,
) -> Tuple[Memory, Dict[str, object]]:
    """在当前帧 prompt 前执行延迟修复、初始化和错误记忆注入。

    返回的审计字段会进入训练窗口统计。强制修复只在 patience 已耗尽且当前维度的
    review interval 到期时发生；EVENT 默认每帧 review，RS 默认每 2 帧 review。
    """

    config.validate()
    frame_ordinal = int(state.frames_seen)
    key = (str(route_key), int(frame_id), int(epoch), frame_ordinal)
    initialized = memory is None
    if initialized:
        keep_rs = _stable_unit_interval(seed, *key, "init_rs") < float(config.rs_initial_gt_prob)
        keep_event = _stable_unit_interval(seed, *key, "init_event") < float(config.event_initial_gt_prob)
        memory = Memory(
            rs_label=str(gt_rs_label) if keep_rs else "UNKNOWN",
            event_label=str(gt_event_label) if keep_event else "UNKNOWN",
        )
    memory = update_memory_navigation(memory, ego_to_goal_xy)
    audit: Dict[str, object] = {
        "memory_initialized": initialized,
        "memory_rs_forced_repair": False,
        "memory_event_forced_repair": False,
        "memory_rs_injected_wrong": False,
        "memory_rs_injected_unknown": False,
        "memory_event_injected_wrong": False,
        "memory_event_injected_unknown": False,
    }

    rs_review_due = frame_ordinal % int(config.rs_repair_interval) == 0
    event_review_due = frame_ordinal % int(config.event_repair_interval) == 0
    if state.rs_repair_pending and rs_review_due:
        memory.rs_label = str(gt_rs_label)
        state.rs_error_streak = 0
        state.rs_repair_pending = False
        audit["memory_rs_forced_repair"] = True
    # 只扰动当前正确状态；已经错误/UNKNOWN 的 memory 应交给学生自行修复，不能每帧
    # 再换一个随机答案，否则学到的是无结构噪声而不是连续 closed-loop recovery。
    if memory.rs_label == str(gt_rs_label) and not bool(audit["memory_rs_forced_repair"]):
        draw = _stable_unit_interval(seed, *key, "augment_rs")
        if draw < float(config.rs_unknown_prob):
            memory.rs_label = "UNKNOWN"
            audit["memory_rs_injected_unknown"] = True
        elif draw < float(config.rs_unknown_prob + config.rs_corrupt_prob):
            memory.rs_label = _stable_different_label(
                str(gt_rs_label),
                tuple(RS_LABEL_TO_OPTION),
                seed=seed,
                key=(*key, "augment_rs"),
            )
            audit["memory_rs_injected_wrong"] = True
    # EVENT 只有在 RS 扰动完成后仍保持正确时才有机会进入 Q2。RS memory 已知错误/
    # UNKNOWN 时保留 EVENT pending，不在一个注定无法可靠构造 EVENT 选择题的帧里
    # 悄悄修复或注入 EVENT。
    event_gate_ready = memory.rs_label == str(gt_rs_label)
    if state.event_repair_pending and event_review_due and event_gate_ready:
        memory.event_label = str(gt_event_label)
        state.event_error_streak = 0
        state.event_repair_pending = False
        audit["memory_event_forced_repair"] = True
    accepted_events = {
        str(item)
        for item in (accepted_event_labels or (gt_event_label,))
        if str(item) in EVENT_DESCRIPTIONS
    } or {str(gt_event_label)}
    if (
        event_gate_ready
        and memory.event_label in accepted_events
        and not bool(audit["memory_event_forced_repair"])
    ):
        draw = _stable_unit_interval(seed, *key, "augment_event")
        if draw < float(config.event_unknown_prob):
            memory.event_label = "UNKNOWN"
            audit["memory_event_injected_unknown"] = True
        elif draw < float(config.event_unknown_prob + config.event_corrupt_prob):
            # 优先从本帧 Q2 真正会展示的候选里构造“看起来合理但错误”的 EVENT
            # hypothesis；单选 RE 等没有替代项时再退回全局事件表，保证仍能形成
            # stale-memory 纠偏样本。
            local_choices = tuple(
                dict.fromkeys(
                    str(item)
                    for item in (event_corruption_choices or ())
                    if str(item) in EVENT_DESCRIPTIONS
                )
            )
            local_choices = tuple(item for item in local_choices if item not in accepted_events)
            if not local_choices:
                local_choices = tuple(item for item in EVENT_DESCRIPTIONS if item not in accepted_events)
            memory.event_label = _stable_different_label(
                str(gt_event_label),
                local_choices,
                seed=seed,
                key=(*key, "augment_event"),
            )
            audit["memory_event_injected_wrong"] = True

    audit.update(
        {
            "memory_rs_input_label": memory.rs_label,
            "memory_event_input_label": memory.event_label,
            "memory_rs_review_due": rs_review_due,
            "memory_event_review_due": event_review_due,
            "memory_event_gate_ready": event_gate_ready,
            "memory_rs_error_streak_before": int(state.rs_error_streak),
            "memory_event_error_streak_before": int(state.event_error_streak),
        }
    )
    state.frames_seen += 1
    return memory, audit


def observe_training_memory(
    state: MemoryCurriculumState,
    config: MemoryCurriculumConfig,
    *,
    rs_correct: bool,
    event_checked: bool,
    event_correct: bool,
) -> Dict[str, object]:
    """观察学生本帧结果，更新两个维度各自的错误 streak 与延迟修复请求。"""

    if rs_correct:
        rs_self_recovered = state.rs_error_streak > 0 or state.rs_repair_pending
        state.rs_error_streak = 0
        state.rs_repair_pending = False
    else:
        rs_self_recovered = False
        state.rs_error_streak += 1
        if state.rs_error_streak >= int(config.rs_error_patience):
            state.rs_repair_pending = True

    event_self_recovered = False
    if event_checked:
        if event_correct:
            event_self_recovered = state.event_error_streak > 0 or state.event_repair_pending
            state.event_error_streak = 0
            state.event_repair_pending = False
        else:
            state.event_error_streak += 1
            if state.event_error_streak >= int(config.event_error_patience):
                state.event_repair_pending = True
    return {
        "memory_rs_self_recovered_after_streak": rs_self_recovered,
        "memory_event_self_recovered_after_streak": event_self_recovered,
        "memory_rs_error_streak_after": int(state.rs_error_streak),
        "memory_event_error_streak_after": int(state.event_error_streak),
        "memory_rs_repair_pending": bool(state.rs_repair_pending),
        "memory_event_repair_pending": bool(state.event_repair_pending),
        "memory_any_repair_pending": bool(
            state.rs_repair_pending or state.event_repair_pending
        ),
    }


def _line_value_span(text: str, label: str) -> Optional[Tuple[int, int]]:
    """返回某个输出行冒号后的值 span。"""

    pattern = re.compile(rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$")
    match = pattern.search(text or "")
    if not match:
        return None
    return match.start(1), match.end(1)


def _line_choice_span(text: str, label: str, choices: str) -> Optional[Tuple[int, int]]:
    """只返回离散选项字符 span，避免错误答案后的长描述稀释纠偏梯度。"""

    match = re.search(
        rf"(?im)^\s*{re.escape(label)}\s*:\s*([{re.escape(choices)}])\b",
        text or "",
    )
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
    rs_span = _line_choice_span(text, "RS", "ABCDE")
    abnormal_span = _line_value_span(text, "ABNORMAL")
    if rs_span is not None:
        spans["rs"] = rs_span
    if abnormal_span is not None:
        spans["abnormal"] = abnormal_span
    return spans


def target_spans_q2(text: str) -> Dict[str, Tuple[int, int]]:
    """Q2 的 token loss 字符 span。"""

    spans: Dict[str, Tuple[int, int]] = {"analysis": _analysis_span(text, terminal_label="EVENT")}
    event_span = _line_choice_span(text, "EVENT", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
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
