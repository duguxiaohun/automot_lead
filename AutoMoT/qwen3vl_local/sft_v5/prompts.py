"""SFT v5 prompt、Memory 与输出解析。

这里是 v5 文本协议的唯一来源。训练、评估和 probe 都从本文件 import，
避免不同入口对同一个 Q1/Q2 问题写出不一致格式。

调用顺序是：用 ``reset_memory_for_frame`` 初始化；Q1 是低频 RS_SLOW，只有到达
可复现的随机复核间隔或处于 RS recovery 时才运行；Q2 是逐帧 EVENT_FAST，直接在显式标注
``[RE | REGULAR]`` / ``[UE | UNUSUAL]`` 的混合候选里选择。Q2 不再依赖
独立 ABNORMAL 问题：选择 RE 就表示 regular/normal，选择任意 UE 就表示
unusual/abnormal。两问都保留三段式视觉分析。

阅读本文件时可以按四层理解：

1. ``Memory`` 负责“上一帧学生输出如何作为下一帧不可信提示词”；
2. ``build_*_prompt`` / ``parse_*_output`` 定义模型实际看到和输出的文本协议；
3. ``prepare_training_memory`` / ``observe_training_memory`` 实现错误记忆注入、连续错误
   计数和延迟兜底修复；
4. ``target_spans_*`` 把自由生成文本里的分析段与最终选项字符映射成 OPSD loss 区间。

训练、eval、probe 不应在各自文件里复制上述规则；否则很容易出现训练时能纠偏、测试时
却直接复用旧 memory 的协议漂移。
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
    EventTarget,
    RSTarget,
    event_description_for_display,
    option_for_event,
)


SYSTEM_PROMPT_V5 = """\
You are an autonomous driving agent. Use the stitched RGB history as visual context, ordered from oldest to newest. Focus on traffic lights/signs, nearby vehicles/pedestrians/obstacles, lane markings and road structure, and key factors affecting ego decisions. Memory is only an unverified previous hypothesis: it may be stale or wrong, so decide from current visual evidence first and change memory whenever the evidence contradicts it. Memory age measures how many 4 Hz frames the hypothesis has remained unchanged; older hypotheses require stronger visual verification, especially for dynamic events. Describe weak, distant, foggy, or occluded evidence as uncertain. Never mention ground truth, answer keys, hidden labels, dataset rules, or scenario names."""

# loss 权重只用于训练时的 token span 加权。结构化分析段低权重，让模型学习“怎么解释”，
# 但不要让冗长自然语言压过 RS/EVENT 这两个离散答案 token。
DEFAULT_W_ANALYSIS = 0.2
DEFAULT_W_RS = 1.2
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

    ``rs_label`` / ``event_label`` 保存内部 canonical label，只有渲染 prompt 时才转换成
    自然语言。``*_age_frames`` 是两个 hypothesis 各自连续未变的 4Hz 帧数：
    label 改变时归零，进入下一帧时加一。RS 和 EVENT 必须独立计时，因为
    道路结构的先验衰减速度和瞬时交通事件不同。调用者更新状态时必须先
    ``copy``，避免同一 timestep 的 student、teacher、reference 分支原地污染。
    """

    rs_label: str
    event_label: str = "RE"
    ego_to_goal_x: Optional[float] = None
    ego_to_goal_y: Optional[float] = None
    rs_age_frames: int = 0
    event_age_frames: int = 0

    @property
    def rs_option(self) -> str:
        """把内部 R1-R5 转成当前 Q1 使用的 A-E；未知 memory 返回 ``?``。

        A-E 只是当前 prompt 的局部选择题编号，不能写回跨帧 memory；跨帧状态始终保存
        R1-R5，防止选项顺序以后变化时旧 memory 语义跟着漂移。
        """

        return RS_LABEL_TO_OPTION.get(self.rs_label, "?")

    def copy(self) -> "Memory":
        """复制六个标量字段，供 Q1/Q2/reference 分支做不可变式更新。

        字段目前都是字符串、浮点或 ``None``，无需 ``deepcopy``；显式构造也能让后续新增
        字段时更容易在 code review 中发现漏拷贝。
        """

        return Memory(
            rs_label=self.rs_label,
            event_label=self.event_label,
            ego_to_goal_x=self.ego_to_goal_x,
            ego_to_goal_y=self.ego_to_goal_y,
            rs_age_frames=int(self.rs_age_frames),
            event_age_frames=int(self.event_age_frames),
        )

    def _road_description(self) -> str:
        """返回 memory 中使用的道路结构自然语言描述，不带 A-E 选项字母。

        UNKNOWN/非法标签统一渲染成“没有可靠先验”，而不是默认 R1；这样错误格式不会被
        静默变成普通道路答案，也能训练模型在 no-prior 输入上依赖当前 RGB。
        """

        if self.rs_label not in RS_LABEL_TO_OPTION:
            return "No reliable previous road-structure hypothesis is available."
        return RS_OPTION_DESCRIPTIONS[self.rs_option]

    def _event_description(self) -> str:
        """返回 memory 中使用的事件自然语言描述，不带 RE/U-E 标签。

        RE 的自然语言依赖当前 RS，例如路口下的 regular 是遵守信号灯而不是普通跟车，
        因此必须动态展开；UE 则可直接使用固定描述。
        """

        if self.event_label not in EVENT_DESCRIPTIONS:
            return "No reliable previous event hypothesis is available."
        event_desc = EVENT_DESCRIPTIONS.get(self.event_label)
        if self.event_label == "RE":
            event_desc = event_description_for_display("RE", self.rs_label)
        return event_desc or EVENT_DESCRIPTIONS["RE"]

    def _goal_text(self) -> str:
        """按 v4 同款格式渲染当前帧目的地相对坐标。

        正负号固定显示，坐标语义为 ego frame ``x_forward, y_left``。缺失时保留 UNKNOWN
        只为健壮性；正式 build_dataset 已要求缺坐标的帧直接跳过。
        """

        if self.ego_to_goal_x is None or self.ego_to_goal_y is None:
            return "UNKNOWN"
        return f"({self.ego_to_goal_x:+.1f}, {self.ego_to_goal_y:+.1f}) m"

    def format_text(self, *, include_event: bool = True) -> str:
        """渲染给学生看的纯文本 memory。

        Q1 只看 road-only memory，不能提前暴露上一次 EVENT；Q2 才在 Q1 的
        RS 判断基础上继续使用 EVENT memory。渲染文本只写自然语言描述，不写
        A-E 或 RE/U-E 标签，避免模型把局部选项编号当成跨帧状态。
        """

        # Memory 是学生唯一可见的跨帧状态。它不包含 scenario、GT、置信度或 event_code，
        # 否则 teacher-private 标注会绕过视觉推理直接泄漏给 student。
        lines = [
            "[MEMORY]",
            f"PREVIOUS_RS_HYPOTHESIS: {self._road_description()}",
            f"PREVIOUS_RS_HYPOTHESIS_AGE: {max(0, int(self.rs_age_frames))} frames "
            f"({max(0, int(self.rs_age_frames)) / 4.0:.2f} s at 4 Hz; 0 means newly initialized or changed)",
        ]
        if include_event:
            # Q1 专门判断慢变量 RS，所以不展示 EVENT；Q2 才读取上一帧 EVENT hypothesis。
            lines.append(f"PREVIOUS_EVENT_HYPOTHESIS: {self._event_description()}")
            lines.append(
                f"PREVIOUS_EVENT_HYPOTHESIS_AGE: {max(0, int(self.event_age_frames))} frames "
                f"({max(0, int(self.event_age_frames)) / 4.0:.2f} s at 4 Hz; 0 means newly initialized or changed)"
            )
        # reliability 不是装饰文本：它明确阻止模型把 memory 当作确定答案，也是错误
        # memory curriculum 能产生纠偏学习信号的必要 prompt 条件。
        lines.append("MEMORY_RELIABILITY: unverified previous model output; it may be stale or wrong")
        lines.append(f"EGO_TO_GOAL_XY={self._goal_text()}")
        lines.append("[/MEMORY]")
        return "\n".join(lines)

    def format_q1_text(self) -> str:
        """渲染 RS_SLOW 的 road-only memory，不包含旧 EVENT hypothesis。"""

        return self.format_text(include_event=False)

    def format_q2_text(self) -> str:
        """渲染 EVENT_FAST 的完整 memory，包含 RS、EVENT 与当前导航坐标。"""

        return self.format_text(include_event=True)


def _structured_q1_format() -> str:
    """返回 RS_SLOW 的学生/老师共享输出合同。

    这里故意只返回格式模板，不拼 memory 和候选项：student/teacher 的可见信息不同，
    但最终输出骨架必须完全相同，后面的正则解析和 token span 才能共用。
    """

    return (
        "Scene Description: <1-2 concise sentences about visible weather/visibility, lane markings, road layout, traffic lights/signs, surrounding motion, and goal direction>\n"
        "Critical Object Description: <1-2 concise sentences naming up to 2-3 key actors or map cues, their locations/actions, likely next motion, and why they matter to ego>\n"
        "Reasoning on Intent: <1-2 concise sentences using road geometry, signals, lanes, ego state, and EGO_TO_GOAL_XY to decide RS>\n"
        "RS: <A|B|C|D|E>"
    )


def _structured_q2_format() -> str:
    """返回 EVENT_FAST 的学生/老师共享输出合同。

    EVENT_FAST 仍要求重新分析当前图像，不能把上一帧的 NORMAL/ABNORMAL 或 EVENT
    直接复制过来；末行只输出本帧随机候选对应的字母。
    """

    return (
        "Scene Description: <1-2 concise sentences about the latest frame under the current RS>\n"
        "Critical Object Description: <1-2 concise sentences naming up to 2-3 event-relevant actors or cues, or stating that no critical object is visible>\n"
        "Reasoning on Intent: <1-2 concise sentences explaining why the selected event is active or why regular behavior continues>\n"
        "EVENT: <option letter>"
    )


def rs_choices_block() -> str:
    """渲染 Q1 的固定 RS A-E 选项。

    RS 选项顺序固定，方便长期保持 R1-R5 与 A-E 的一一映射。跨帧 memory 不保存
    字母，只保存 canonical R 标签；本函数仅负责当前 prompt 的展示层。
    """

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

    ``option_map`` 已经由 ``labels.stable_event_option_map`` 按 frame 可复现随机生成；
    这里不再重排，只把 canonical label 转成学生能理解的自然语言描述。每个选项都显式
    标注 REGULAR/UNUSUAL，使原先的 EVENT_FAST_1 和 EVENT_FAST_2 合成一次选择。
    """

    lines = [
        "[EVENT_FAMILY_LEGEND]",
        "[RE | REGULAR] = regular/normal driving behavior; no unusual event is actively affecting ego.",
        "[UE | UNUSUAL] = an unusual/abnormal event is actively affecting ego.",
        "[/EVENT_FAMILY_LEGEND]",
        f"[EVENT_CHOICES under RS={RS_LABEL_TO_OPTION.get(rs_label, 'A')}]",
    ]
    # 按字母排序只影响显示顺序，不改变 option_map 的 label 绑定关系。
    for letter in sorted(option_map):
        label = option_map[letter]
        event_family = "RE | REGULAR" if label == "RE" else "UE | UNUSUAL"
        lines.append(
            f"{letter}. [{event_family}] "
            f"{event_description_for_display(label, rs_label, regular_event_codes)}"
        )
    lines.append("[/EVENT_CHOICES]")
    return "\n".join(lines)


def build_q1_student_prompt(memory: Memory) -> str:
    """低频 RS_SLOW student prompt。

    Student 不看 XML weather；天气只允许从 RGB 中观察，并写进 Scene Description。
    调用者应在本帧决定 ``should_run_rs_slow=True`` 后才构造本 prompt；稳定快帧不会
    产生 Q1 token，也不会用旧 RS 分析冒充当前帧分析。
    """

    # 三块按“过去的假设 → 当前可选语义 → 当前问题”排列。memory reliability 会提示
    # 模型先看图再判断，候选描述则把 R1-R5 的工程语义完整翻译给基础模型。
    return "\n\n".join([
        memory.format_q1_text(),
        rs_choices_block(),
        (
            "[QUESTION_1]\n"
            "This is the low-frequency road-structure review. Analyze the latest frame in the RGB history "
            "and decide the current road-structure option from RS_CHOICES.\n\n"
            "Use visible road geometry, lane layout, traffic lights or stop/yield cues, "
            "EGO_TO_GOAL_XY, and image-visible weather or visibility cues. Do not use a scenario name. "
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
    weather_text: str,
) -> str:
    """Q1 teacher privileged prompt。

    Teacher 可以看 XML weather 与 GT，但这些信息只用于生成更稳定的 privileged logits。
    Student forward 使用 :func:`build_q1_student_prompt`，不会看到 REFERENCE；生成出的
    文本还要经过 private-marker 检查，防止把 ANSWER/REFERENCE 字样蒸馏给学生。
    """

    # teacher 和 student 使用同一份 memory/choices，唯一额外信息放在显式 REFERENCE
    # 块中，便于 inspect_teacher.py 把特权输入与学生输入并排审计。
    return "\n\n".join([
        memory.format_q1_text(),
        rs_choices_block(),
        (
            "[REFERENCE]\n"
            f"XML_WEATHER: {weather_text}\n"
            f"ANSWER_RS: {rs_target.option} - {rs_target.description}\n"
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
    weather_text: str,
) -> str:
    """脚本化 Q1 teacher target，供 CE smoke 或 teacher 抽检兜底。

    OPSD 主训练默认用 teacher logits，不强制逐字模仿这段文本；但 inspect/probe 需要
    一个可读 target 来审计 prompt 合同。``weather_text`` 保留在接口中以与 teacher
    构造器对齐，target 本身不复述 XML 天气，避免形成不可见信息泄漏。
    """

    return "\n".join(
        [
            f"Scene Description: Describe the visible weather, lane markings, traffic controls, road layout, surrounding motion, and goal direction; the road layout supports option {rs_target.option}.",
            "Critical Object Description: Name the most relevant lane boundary, traffic control, map cue, or occluding actor needed to identify the road structure.",
            f"Reasoning on Intent: The road-structure evidence supports {rs_target.option}: {rs_target.description}.",
            f"RS: {rs_target.option}",
        ]
    )


def build_q2_student_prompt(
    memory: Memory,
    *,
    option_map: Mapping[str, str],
    regular_event_codes: Optional[Sequence[str]] = None,
) -> str:
    """Q2 student prompt。

    Q2 候选优先来自逐帧 allowed_events，缺失时才按 scenario ∩ 当前 RS fallback，
    并做 frame 级随机化。prompt 不能暴露 scenario 名。

    慢帧调用时，这段 user turn 接在当帧 Q1 assistant token/KV 后；快帧没有 Q1 turn，
    调用者必须把本帧 RGB 与本 prompt 做 fresh prefill。两种路径都必须重新读取当前图像。
    """

    # Q2 memory 含 RS 与 EVENT 两个“未验证假设”。当前 RS 决定候选语义，旧 EVENT 只
    # 提供时间连续性，不能绕过最新 RGB 直接成为答案。
    return "\n\n".join([
        memory.format_q2_text(),
        event_choices_block(option_map, memory.rs_label, regular_event_codes),
        (
            "[QUESTION_2]\n"
            "This is the per-frame event review. Decide the current event directly from EVENT_CHOICES. "
            "Every choice is explicitly marked [RE | REGULAR] for regular/normal behavior or "
            "[UE | UNUSUAL] for an unusual/abnormal event. Compare all listed RE and UE choices against "
            "the latest RGB evidence; do not perform a "
            "separate normal/abnormal classification and do not blindly copy PREVIOUS_EVENT_HYPOTHESIS. "
            "The choices have already been filtered for the current road structure and route type. "
            "Do not invent an unlisted event. Keep the CoT concise.\n\n"
            "Output exactly these lines:\n"
            f"{_structured_q2_format()}\n"
            "[/QUESTION_2]"
        ),
    ])


def build_q2_teacher_prompt(
    memory: Memory,
    *,
    option_map: Mapping[str, str],
    event_target: EventTarget,
    regular_event_codes: Optional[Sequence[str]] = None,
) -> str:
    """构造 Q2 teacher 的 privileged prompt。

    ``event_target`` 只出现在 REFERENCE 中；学生看到的 memory、候选表和输出格式与
    teacher 一致。regular 原始 code 只用于细化 RE 文案和审计，最终标签仍折叠为 RE。
    """

    # target_option 必须反查本帧 option_map，不能假设某个固定字母恒等于 RE/某个 UE。
    target_option = option_for_event(event_target.label, option_map) or "?"
    regular_codes = regular_event_codes if regular_event_codes is not None else event_target.regular_event_codes
    target_desc = event_description_for_display(event_target.label, memory.rs_label, regular_codes)
    return "\n\n".join([
        memory.format_q2_text(),
        event_choices_block(option_map, memory.rs_label, regular_codes),
        (
            "[REFERENCE]\n"
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
    """脚本化 Q2 teacher target，供 probe/smoke 审计。

    正常数据构建已保证 GT 在 ``option_map`` 中。若遇到旧 index 的 candidate mismatch，
    这里选择首个候选只为生成可读诊断文本；训练主路径会单独统计并跳过/标记不一致，
    不能把这个兜底解释成新的监督规则。
    """

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
_Q2_EVENT_RE = re.compile(r"(?im)^\s*EVENT\s*:\s*([A-Z])\b")
# 解析器刻意只看行首字段，不试图理解整段 analysis。这样学生可以自由解释，
# 但离散答案必须落在固定字段上，便于 eval/probe 和 memory 状态机稳定读取。


def parse_q1_output(text: str) -> Dict[str, Optional[str]]:
    """解析 Q1 student 输出，返回局部选项和 canonical RS label。

    只认独立 ``RS:`` 行；分析正文里偶然提到 A-E 不会误触发。格式非法时两个字段均为
    ``None``，上层会进入逐帧 RS recovery，并跳过当前帧 EVENT。
    """

    rs_match = _Q1_RS_RE.search(text or "")
    return {
        "rs_option": rs_match.group(1).upper() if rs_match else None,
        "rs_label": RS_OPTION_TO_LABEL.get(rs_match.group(1).upper()) if rs_match else None,
    }


def should_trigger_q2(*, student_rs_label: Optional[str], target_rs_label: str) -> bool:
    """只有本帧 Q1 的 RS 回答正确时才允许进入 Q2。

    RS memory 错误不会让 Q1 休眠：训练/eval/probe 在下一有效帧仍要重新运行 Q1，
    直到学生自行纠正，或训练期 curriculum 达到 patience 后执行兜底修复。这个函数
    只定义“当前帧是否追问 EVENT”，不能被用来跳过下一帧 Q1。
    """

    return student_rs_label == str(target_rs_label)


def should_run_event_fast(
    *,
    rs_slow_ran: bool,
    q1_rs_correct: bool,
    memory_rs_label: Optional[str],
    target_rs_label: str,
) -> bool:
    """统一慢帧/快帧的 EVENT_FAST RS gate。

    慢帧刚运行过 RS_SLOW，必须以本帧 Q1 是否解析且答对为准；旧 memory 即使碰巧
    正确，也不能掩盖当前 Q1 的错误或非法输出。快帧没有 Q1，才允许复用稳定 RS
    memory。train/eval/probe 应共用这个合同。
    """

    if rs_slow_ran:
        return bool(q1_rs_correct)
    return memory_rs_label == str(target_rs_label)


def parse_q2_output(text: str, option_map: Mapping[str, str]) -> Dict[str, Optional[str]]:
    """解析 Q2 student 输出，并按本帧 ``event_option_map`` 还原 label。

    同一个字母在不同帧可能对应不同事件，所以绝不能用全局 A/B/C 规则解码。解析到
    不在本帧映射中的字母时保留字母供格式审计，但 ``event_label`` 返回 ``None``。
    """

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
) -> Memory:
    """Q1 后的 memory 更新。

    只有合法 RS 才写入 memory。新 label 与输入 hypothesis 不同时才把
    ``rs_age_frames`` 归零；周期复核后仍回答同一 RS 表示该假设继续成立，
    age 保持连续，不伪造成“刚刚变化”。RS_SLOW 不输出 EVENT family。
    """

    mem = memory.copy()
    if student_rs_label in RS_LABEL_TO_OPTION:
        next_label = str(student_rs_label)
        if next_label != mem.rs_label:
            mem.rs_label = next_label
            mem.rs_age_frames = 0
    return mem


def update_memory_after_q2(memory: Memory, *, student_event_label: Optional[str]) -> Memory:
    """用 EVENT_FAST 的合法输出更新 EVENT memory；非法输出保持上一状态。

    本函数不判断答案是否正确：closed-loop memory 必须记录学生真实预测，错误预测才会
    在后续帧形成纠偏样本。只有 EVENT label 真正变化时才将独立 age 归零；
    重复确认同一事件不重置时钟。GT 强制修复只允许由训练期 curriculum 执行。
    """

    mem = memory.copy()
    if student_event_label:
        next_label = str(student_event_label)
        if next_label != mem.event_label:
            mem.event_label = next_label
            mem.event_age_frames = 0
    return mem


def advance_memory_age(memory: Memory) -> Memory:
    """进入下一个有效 4Hz frame 时，独立累加 RS/EVENT hypothesis age。

    该函数只表示时间流逝，不刷新导航坐标，也不修改 label。首帧初始化
    时不调用，所以初始 age=0；之后每个真实有效帧恰好调用一次。
    padding、缺图 skip 不调用，避免 DDP 对齐位置伪造时间。
    """

    mem = memory.copy()
    mem.rs_age_frames = max(0, int(mem.rs_age_frames)) + 1
    mem.event_age_frames = max(0, int(mem.event_age_frames)) + 1
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
    """构造 GT RS + RE 的 reference/兼容初始化 memory。

    该 helper 主要供测试、reference 分支和旧调用接口使用；正式训练必须调用
    :func:`prepare_training_memory`，否则每帧用 GT 重置会重新制造“复制 memory 即 99%”
    的捷径，并破坏延迟纠偏课程。
    """

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


def initialize_student_memory(
    rs_target: RSTarget,
    ego_to_goal_xy: Optional[Sequence[float]] = None,
    *,
    mode: str = "unknown",
) -> Memory:
    """初始化 eval/probe 中一条 student 序列的首帧 memory。

    默认 ``unknown`` 同时把 RS/EVENT 设为 UNKNOWN，只保留当前导航坐标，代表真实系统
    刚启动时没有上一帧预测。``ground_truth`` 保留旧评估兼容行为：RS 使用当前 GT、
    EVENT 使用 RE。reference 分支仍应直接调用 :func:`reset_memory_for_frame`，不能把本
    helper 的 UNKNOWN 当作 reference 真值。
    """

    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "ground_truth":
        return reset_memory_for_frame(rs_target, ego_to_goal_xy=ego_to_goal_xy)
    if normalized_mode != "unknown":
        raise ValueError(
            f"student initial memory mode must be 'unknown' or 'ground_truth', got {mode!r}"
        )
    memory = Memory(rs_label="UNKNOWN", event_label="UNKNOWN")
    return update_memory_navigation(memory, ego_to_goal_xy)


@dataclasses.dataclass(frozen=True)
class MemoryCurriculumConfig:
    """训练期错误记忆课程参数。

    RS 是慢变量，默认给更长的自主修复窗口并只按较疏的 review interval 执行修复；
    EVENT 是快变量，每帧都复核，使用独立 patience。正式默认是 ``ground_truth``
    延迟硬修复：只有连续错误用完 patience 并到达 review slot 后才写回答案，
    不是错误后下一帧立刻纠正。``unknown`` 保留为软擦除消融；它不保证学生能
    退出 UNKNOWN，因此不作为正式长训默认。

    扰动只在当前 memory 原本正确时注入；若学生复制错误 hypothesis，它会自然延续到
    后续帧，从而形成真正的 closed-loop 纠偏样本，而不是每帧互不关联的随机噪声。
    """

    rs_slow_interval: int = 4
    rs_slow_interval_jitter: int = 1
    rs_error_patience: int = 4
    event_error_patience: int = 3
    rs_repair_interval: int = 2
    event_repair_interval: int = 1
    rs_repair_mode: str = "ground_truth"
    event_repair_mode: str = "ground_truth"
    # RS 扰动是“每个当前正确 frame”的条件概率。由于扰动会额外触发
    # RS_SLOW，5% contradiction + 7% omission 在理想当帧纠偏下会映射为
    # Q1 样本约 61% aligned / 23% omission / 16% contradiction，而非 88/7/5。
    rs_corrupt_prob: float = 0.05
    rs_unknown_prob: float = 0.07
    # EVENT 在每个 RS gate 正确帧都训练；这里把“有资格注入 EVENT 的帧”设成
    # 55% aligned / 25% omission / 20% contradiction。由于 RS augmentation 先于
    # EVENT 且会拦下一部分注入，理想当帧纠偏的最终 Q2 实测约为 60/22/17；
    # closed-loop 持续错误还会继续改变分布，因此最终必须以 TB 实测为准。
    event_corrupt_prob: float = 0.20
    event_unknown_prob: float = 0.25
    rs_initial_gt_prob: float = 0.5
    event_initial_gt_prob: float = 0.5

    def validate(self) -> None:
        """拒绝会破坏概率或延迟语义的配置。

        interval/patience 为 0 会让取模或恢复状态失去含义；同一维度的 wrong/UNKNOWN
        概率之和不能超过 1，剩余概率自然代表“保持正确 memory”。训练入口在启动时和
        调度函数里都会调用本校验，使错误环境变量尽早失败。
        """

        for name in (
            "rs_slow_interval",
            "rs_error_patience",
            "event_error_patience",
            "rs_repair_interval",
            "event_repair_interval",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be >= 1")
        if int(self.rs_slow_interval_jitter) < 0:
            raise ValueError("rs_slow_interval_jitter must be >= 0")
        if int(self.rs_slow_interval) - int(self.rs_slow_interval_jitter) < 1:
            raise ValueError(
                "rs_slow_interval - rs_slow_interval_jitter must be >= 1"
            )
        for prefix in ("rs", "event"):
            corrupt = float(getattr(self, f"{prefix}_corrupt_prob"))
            unknown = float(getattr(self, f"{prefix}_unknown_prob"))
            initial = float(getattr(self, f"{prefix}_initial_gt_prob"))
            if corrupt < 0.0 or unknown < 0.0 or corrupt + unknown > 1.0:
                raise ValueError(f"{prefix} corruption/unknown probabilities must sum to [0, 1]")
            if not 0.0 <= initial <= 1.0:
                raise ValueError(f"{prefix}_initial_gt_prob must be in [0, 1]")
            repair_mode = str(getattr(self, f"{prefix}_repair_mode"))
            if repair_mode not in {"unknown", "ground_truth"}:
                raise ValueError(
                    f"{prefix}_repair_mode must be 'unknown' or 'ground_truth', got {repair_mode!r}"
                )


@dataclasses.dataclass
class MemoryCurriculumState:
    """单条 route 的训练期延迟修复状态。

    ``frames_seen`` 是本 route 已准备过的真实帧数；``last_rs_query_ordinal`` 记录上次
    实际执行 RS_SLOW 的序号。RS 除了 error streak 还需要 ``recovery_active``，因为
    一旦慢问答错，后续帧要临时恢复成逐帧慢思考，直到学生自己答对或延迟修复生效。
    EVENT 本来逐帧执行，只需 streak/pending 两组状态。
    """

    frames_seen: int = 0
    last_rs_query_ordinal: int = -1
    rs_recovery_active: bool = False
    rs_error_streak: int = 0
    event_error_streak: int = 0
    rs_repair_pending: bool = False
    event_repair_pending: bool = False


def rs_slow_interval_for_state(
    state: MemoryCurriculumState,
    config: MemoryCurriculumConfig,
    *,
    schedule_key: Optional[str] = None,
    schedule_seed: int = 0,
) -> int:
    """返回当前上一次 RS query 之后应等待的可复现随机帧数。

    默认 ``base=4, jitter=1`` 在 3/4/5 中均匀选择，平均频率仍约 1Hz，
    但不再让所有 route 锁在 1/5/9 或 0/4/8 的固定相位。hash 同时包含
    route/window key 和上次 query ordinal：同一 seed 重跑完全一致，每次 query
    更新 ``last_rs_query_ordinal`` 后又会得到新间隔。

    ``schedule_key=None`` 时返回固定 base，保留单元测试和旧外部调用的
    确定语义；正式 train/eval/probe 都会显式传 route key。
    """

    config.validate()
    base = int(config.rs_slow_interval)
    jitter = int(config.rs_slow_interval_jitter)
    if jitter <= 0 or schedule_key is None:
        return base
    low = base - jitter
    high = base + jitter
    draw = _stable_unit_interval(
        int(schedule_seed),
        str(schedule_key),
        int(state.last_rs_query_ordinal),
        "rs_slow_interval",
    )
    # draw 严格小于 1，乘以闭区间大小后取整即可均匀映射到
    # [low, high]；min 是浮点边界的防御性保护。
    offset = min(high - low, int(draw * (high - low + 1)))
    return low + offset


def should_run_rs_slow(
    state: MemoryCurriculumState,
    config: MemoryCurriculumConfig,
    *,
    memory: Memory,
    gt_rs_label: Optional[str],
    frame_ordinal: Optional[int] = None,
    schedule_key: Optional[str] = None,
    schedule_seed: int = 0,
) -> Tuple[bool, str]:
    """决定当前帧是否执行低频 RS_SLOW，并返回可审计原因。

    稳定且正确的 RS 默认按 ``rs_slow_interval ± jitter`` 复核。首帧、UNKNOWN、已知
    memory 与当前训练 GT 不一致、上一轮 RS 错误 recovery、以及 delayed repair pending
    都立即执行 RS。真实部署以及默认 eval/probe 必须把 ``gt_rs_label`` 传 ``None``，
    依赖周期复核、非法输出以及 RS 变化后的确认帧进入 recovery。只有训练课程或显式
    oracle 对照评估才允许传 GT；否则调度时机本身就会泄漏答案。
    """

    config.validate()
    ordinal = int(state.frames_seen if frame_ordinal is None else frame_ordinal)
    if state.last_rs_query_ordinal < 0:
        return True, "route_start"
    if state.rs_recovery_active or state.rs_error_streak > 0 or state.rs_repair_pending:
        return True, "recovery"
    if memory.rs_label not in RS_LABEL_TO_OPTION:
        return True, "unknown_memory"
    if gt_rs_label is not None and memory.rs_label != str(gt_rs_label):
        return True, "memory_mismatch"
    scheduled_interval = rs_slow_interval_for_state(
        state,
        config,
        schedule_key=schedule_key,
        schedule_seed=schedule_seed,
    )
    if ordinal - int(state.last_rs_query_ordinal) >= scheduled_interval:
        return True, "periodic"
    return False, "reuse_stable_rs"


def observe_inference_rs_schedule(
    state: MemoryCurriculumState,
    *,
    rs_checked: bool,
    memory_rs_label_before: Optional[str],
    student_rs_label: Optional[str],
) -> Dict[str, object]:
    """用无 GT 信号更新真实推理可执行的 RS 调度状态。

    推理端无法知道一个合法 R1-R5 是否“答错”，因此不能复用训练期的 GT correctness
    streak。本函数只使用模型公开输出：非法/UNKNOWN 输出会进入逐帧 recovery；合法输出
    若相对输入 memory 发生变化，也会要求下一帧再做一次 RS_SLOW 确认。下一次输出与
    当前 memory 一致后退出 recovery，之后恢复可复现的随机周期复核。

    这仍不能识别“模型稳定地复制了同一个合法但错误的 RS”，该情况只能等周期复核或
    未来的置信度/几何一致性检测。返回字段只用于 eval/probe 审计，不实施 GT 修复。
    """

    if not rs_checked:
        return {
            "inference_rs_output_valid": None,
            "inference_rs_changed": None,
            "memory_rs_recovery_active": bool(state.rs_recovery_active),
            "memory_rs_last_query_ordinal": int(state.last_rs_query_ordinal),
        }

    # frames_seen 由 eval/probe 在当前帧结束前写成“已消费帧数”；减一得到当前 ordinal，
    # 与训练期 observe_training_memory 的 last-query 口径保持一致。
    state.last_rs_query_ordinal = max(0, int(state.frames_seen) - 1)
    valid = student_rs_label in RS_LABEL_TO_OPTION
    previous_valid = memory_rs_label_before in RS_LABEL_TO_OPTION
    changed = bool(
        valid
        and (
            not previous_valid
            or str(student_rs_label) != str(memory_rs_label_before)
        )
    )
    if not valid:
        # 无法解析意味着当前 memory 没有获得可靠确认；下一帧继续慢问。
        state.rs_recovery_active = True
        state.rs_error_streak += 1
    elif changed:
        # 合法变化可能是真实路口切换，也可能是一次误判。保留新 student memory，但要求
        # 下一帧再确认一次；若再次输出相同标签，下面的 stable 分支会退出 recovery。
        state.rs_recovery_active = True
        state.rs_error_streak = 0
    else:
        state.rs_recovery_active = False
        state.rs_error_streak = 0
    # 推理端没有训练期 repair；显式清空 pending，防止复用 state 时意外触发 GT 写回。
    state.rs_repair_pending = False
    return {
        "inference_rs_output_valid": bool(valid),
        "inference_rs_changed": bool(changed),
        "memory_rs_recovery_active": bool(state.rs_recovery_active),
        "memory_rs_last_query_ordinal": int(state.last_rs_query_ordinal),
    }


def _stable_unit_interval(seed: int, *parts: object) -> float:
    """把 route/frame/key 稳定映射到 ``[0, 1)``，保证 DDP 与重跑可复现。

    这里不使用进程级 ``random`` 状态：length-balanced sampler 会让不同 rank 以不同
    顺序处理 route，若依赖调用顺序，同一帧在不同卡上可能得到不同扰动。SHA256 把
    route、frame、epoch 和用途共同固定为一次无状态随机抽样。
    """

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
    """从候选中可复现地选择一个与 ``current`` 不同的标签。

    所有候选都与当前标签相同时返回 UNKNOWN；调用者把它视作无先验扰动，不会伪装成
    “错误但其实仍正确”的增强样本。
    """

    alternatives = [str(item) for item in choices if str(item) != str(current)]
    if not alternatives:
        return "UNKNOWN"
    draw = _stable_unit_interval(seed, *key, "choice")
    index = min(len(alternatives) - 1, int(draw * len(alternatives)))
    return alternatives[index]


def _repair_memory_label(mode: str, ground_truth_label: str) -> str:
    """按课程配置返回修复后写入 prompt memory 的标签。

    ``ground_truth`` 是正式默认，但调用它时已经用完 patience/review，
    所以是延迟硬修复而非即时纠正。``unknown`` 只移除持续错误的先验，作为
    不泄漏答案的软擦除消融。调用前配置已经过 validate，这里仍拒绝
    未知值，避免未来单独调用 helper 时静默选择一种修复。
    """

    if str(mode) == "unknown":
        return "UNKNOWN"
    if str(mode) == "ground_truth":
        return str(ground_truth_label)
    raise ValueError(f"unsupported repair mode: {mode!r}")


def _set_memory_hypothesis(memory: Memory, *, dimension: str, label: str) -> None:
    """原地写入训练期 RS/EVENT hypothesis，发生变化时将对应 age 归零。

    ``prepare_training_memory`` 在自己的局部 copy 上调用该 helper，因此
    这里允许原地更新。统一 helper 可防止 corruption 重置了 age、repair 却忘了，
    或 RS 变化误清空 EVENT 时钟。label 未变时保留 age，严格实现“从 memory
    改变开始重新计时”。
    """

    normalized = str(dimension).strip().lower()
    if normalized == "rs":
        if memory.rs_label != str(label):
            memory.rs_label = str(label)
            memory.rs_age_frames = 0
        return
    if normalized == "event":
        if memory.event_label != str(label):
            memory.event_label = str(label)
            memory.event_age_frames = 0
        return
    raise ValueError(f"dimension must be 'rs' or 'event', got {dimension!r}")


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

    调用顺序固定为：route 初始化 → 刷新当前导航坐标 → 执行到期修复 → 在“原本正确”
    的维度注入 wrong/UNKNOWN → 记录 prompt 实际输入。这个顺序保证 repair 当帧不会又
    被立即污染，也保证错误 memory 能连续保留几帧供学生尝试自我纠正。
    """

    config.validate()
    # frame_ordinal 与原始 frame_id 分离：原始 ID 可能不连续，课程 patience/interval
    # 应按真正进入该 route 序列的帧数计算。
    frame_ordinal = int(state.frames_seen)
    key = (str(route_key), int(frame_id), int(epoch), frame_ordinal)
    initialized = memory is None
    if initialized:
        # route 首帧也不是固定 GT：一半从正确 hypothesis 开始，一半从 UNKNOWN 开始，
        # 让模型同时学会“保持正确”和“没有先验时独立看图”。RS/EVENT 独立抽样。
        keep_rs = _stable_unit_interval(seed, *key, "init_rs") < float(config.rs_initial_gt_prob)
        keep_event = _stable_unit_interval(seed, *key, "init_event") < float(config.event_initial_gt_prob)
        memory = Memory(
            rs_label=str(gt_rs_label) if keep_rs else "UNKNOWN",
            event_label=str(gt_event_label) if keep_event else "UNKNOWN",
        )
    else:
        # 只有真正进入下一个有效训练帧才增加 age。此函数在缺图/padding
        # 过滤之后调用，所以不会把伪 timestep 算成时间衰减。
        memory = advance_memory_age(memory)
    # 导航坐标属于当前帧外部输入，必须每帧刷新；这一步不会改离散状态或泄漏 GT。
    memory = update_memory_navigation(memory, ego_to_goal_xy)
    audit: Dict[str, object] = {
        "memory_frame_ordinal": frame_ordinal,
        "memory_initialized": initialized,
        "memory_rs_forced_repair": False,
        "memory_event_forced_repair": False,
        "memory_rs_injected_wrong": False,
        "memory_rs_injected_unknown": False,
        "memory_event_injected_wrong": False,
        "memory_event_injected_unknown": False,
        "memory_rs_repaired_to_unknown": False,
        "memory_rs_repaired_to_ground_truth": False,
        "memory_event_repaired_to_unknown": False,
        "memory_event_repaired_to_ground_truth": False,
    }

    # pending 只表示“patience 已耗尽”，实际干预还要等各自 repair
    # review slot。默认干预是延迟写回 GT，显式 unknown 模式则擦除先验；
    # 两种模式都不会在错误发生的下一帧立刻介入。
    rs_review_due = frame_ordinal % int(config.rs_repair_interval) == 0
    event_review_due = frame_ordinal % int(config.event_repair_interval) == 0
    if state.rs_repair_pending and rs_review_due:
        # 正式默认在这个延迟 slot 写回 GT，防止早期学生长期卡在错误/
        # UNKNOWN 而饿饿 EVENT 训练。该帧会单列为 repair-after-recovery，不算自主纠偏。
        _set_memory_hypothesis(
            memory,
            dimension="rs",
            label=_repair_memory_label(config.rs_repair_mode, str(gt_rs_label)),
        )
        state.rs_error_streak = 0
        state.rs_repair_pending = False
        audit["memory_rs_forced_repair"] = True
        audit[f"memory_rs_repaired_to_{config.rs_repair_mode}"] = True
    # 只扰动当前正确状态；已经错误/UNKNOWN 的 memory 应交给学生自行修复，不能每帧
    # 再换一个随机答案，否则学到的是无结构噪声而不是连续 closed-loop recovery。
    if memory.rs_label == str(gt_rs_label) and not bool(audit["memory_rs_forced_repair"]):
        draw = _stable_unit_interval(seed, *key, "augment_rs")
        if draw < float(config.rs_unknown_prob):
            # UNKNOWN 是这一帧新形成的 no-prior 状态，所以对应 age 从 0 开始。不能为了
            # 模拟“陈旧”而随意伪造较大 age；若学生没有纠正，后续真实帧会让它自然变旧。
            _set_memory_hypothesis(memory, dimension="rs", label="UNKNOWN")
            audit["memory_rs_injected_unknown"] = True
        elif draw < float(config.rs_unknown_prob + config.rs_corrupt_prob):
            _set_memory_hypothesis(
                memory,
                dimension="rs",
                label=_stable_different_label(
                    str(gt_rs_label),
                    tuple(RS_LABEL_TO_OPTION),
                    seed=seed,
                    key=(*key, "augment_rs"),
                ),
            )
            audit["memory_rs_injected_wrong"] = True
    # EVENT 只有在 RS 扰动完成后仍保持正确时才有机会进入 Q2。RS memory 已知错误/
    # UNKNOWN 时保留 EVENT pending，不在一个注定无法可靠构造 EVENT 选择题的帧里
    # 悄悄修复或注入 EVENT。
    event_gate_ready = memory.rs_label == str(gt_rs_label)
    if state.event_repair_pending and event_review_due and event_gate_ready:
        # EVENT 也在独立 patience 后才修复；正式默认写回 GT，软擦除消融
        # 才写 UNKNOWN。Q2 仍会在当帧重新读 RGB，这次介入不会被计成学生自救。
        _set_memory_hypothesis(
            memory,
            dimension="event",
            label=_repair_memory_label(config.event_repair_mode, str(gt_event_label)),
        )
        state.event_error_streak = 0
        state.event_repair_pending = False
        audit["memory_event_forced_repair"] = True
        audit[f"memory_event_repaired_to_{config.event_repair_mode}"] = True
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
            # EVENT no-prior 同样从改变当帧 age=0 起算。EVENT_FAST 若继续复制 UNKNOWN/
            # 错误事件，closed-loop 才会在后续帧形成 age>0 的 omission/contradiction。
            _set_memory_hypothesis(memory, dimension="event", label="UNKNOWN")
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
            _set_memory_hypothesis(
                memory,
                dimension="event",
                label=_stable_different_label(
                    str(gt_event_label),
                    local_choices,
                    seed=seed,
                    key=(*key, "augment_event"),
                ),
            )
            audit["memory_event_injected_wrong"] = True

    # 保存的是“送进本帧 prompt 的最终值”，便于计算注入样本占比、修复后准确率和
    # memory-copy 指标；不能只记录注入前的内部状态。
    audit.update(
        {
            "memory_rs_input_label": memory.rs_label,
            "memory_event_input_label": memory.event_label,
            "memory_rs_input_age_frames": int(memory.rs_age_frames),
            "memory_event_input_age_frames": int(memory.event_age_frames),
            "memory_rs_review_due": rs_review_due,
            "memory_event_review_due": event_review_due,
            "memory_event_gate_ready": event_gate_ready,
            "memory_rs_error_streak_before": int(state.rs_error_streak),
            "memory_event_error_streak_before": int(state.event_error_streak),
        }
    )
    # observe_training_memory 在本帧模型输出后调用，因此这里先递增；observe 用
    # frames_seen - 1 还原当前 ordinal 并记录 last_rs_query_ordinal。
    state.frames_seen += 1
    return memory, audit


def observe_training_memory(
    state: MemoryCurriculumState,
    config: MemoryCurriculumConfig,
    *,
    rs_correct: bool,
    rs_checked: bool = True,
    event_checked: bool,
    event_correct: bool,
    rs_forced_repair: bool = False,
    event_forced_repair: bool = False,
) -> Dict[str, object]:
    """观察学生本帧结果，更新两个维度各自的错误 streak 与延迟修复请求。

    ``rs_checked`` 只有实际运行 RS_SLOW 时才为真；稳定快帧复用 RS memory，不得把
    “没有问 Q1”误记成一次正确或错误。``event_checked`` 同理，只在 RS gate 通过且
    EVENT_FAST 真正运行时更新 EVENT streak。forced-repair 帧即使答对也不能记为“学生
    在干预前自主恢复”；它会单列为 repair 后恢复。返回字典直接并入训练审计指标。
    """

    rs_self_recovered = False
    if rs_checked:
        # 正确回答会清空 recovery/pending。只有当帧没有发生 repair
        # 干预时，才把它计为“干预前自主纠偏”；repair 帧答对进入单独计数。
        # 错误回答则让下一帧 RS_SLOW 立即运行，并在连续达到 patience 后请求修复。
        state.last_rs_query_ordinal = max(0, int(state.frames_seen) - 1)
        if rs_correct:
            rs_self_recovered = bool(
                not rs_forced_repair
                and (
                    state.rs_recovery_active
                    or state.rs_error_streak > 0
                    or state.rs_repair_pending
                )
            )
            state.rs_error_streak = 0
            state.rs_repair_pending = False
            state.rs_recovery_active = False
        else:
            state.rs_recovery_active = True
            state.rs_error_streak += 1
            if state.rs_error_streak >= int(config.rs_error_patience):
                state.rs_repair_pending = True

    event_self_recovered = False
    if event_checked:
        # EVENT 是快变量，独立维护 streak；RS gate 失败的帧 event_checked=False，
        # 既不奖励也不惩罚 EVENT memory，避免把上游 RS 错误算成事件错误。
        if event_correct:
            event_self_recovered = bool(
                not event_forced_repair
                and (state.event_error_streak > 0 or state.event_repair_pending)
            )
            state.event_error_streak = 0
            state.event_repair_pending = False
        else:
            state.event_error_streak += 1
            if state.event_error_streak >= int(config.event_error_patience):
                state.event_repair_pending = True
    return {
        "memory_rs_self_recovered_after_streak": rs_self_recovered,
        "memory_event_self_recovered_after_streak": event_self_recovered,
        "memory_rs_recovered_after_forced_repair": bool(rs_checked and rs_correct and rs_forced_repair),
        "memory_event_recovered_after_forced_repair": bool(
            event_checked and event_correct and event_forced_repair
        ),
        "memory_rs_error_streak_after": int(state.rs_error_streak),
        "memory_event_error_streak_after": int(state.event_error_streak),
        "memory_rs_repair_pending": bool(state.rs_repair_pending),
        "memory_rs_recovery_active": bool(state.rs_recovery_active),
        "memory_rs_last_query_ordinal": int(state.last_rs_query_ordinal),
        "memory_event_repair_pending": bool(state.event_repair_pending),
        "memory_any_repair_pending": bool(
            state.rs_repair_pending or state.event_repair_pending
        ),
    }


def _line_value_span(text: str, label: str) -> Optional[Tuple[int, int]]:
    """返回某个输出行冒号后的完整值字符区间 ``[start, end)``。

    字符区间随后由 tokenizer offset mapping 转成 token 权重；没找到独立字段行时返回
    ``None``，而不是误把正文中的同名词当监督值。
    """

    pattern = re.compile(rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$")
    match = pattern.search(text or "")
    if not match:
        return None
    return match.start(1), match.end(1)


def _line_choice_span(text: str, label: str, choices: str) -> Optional[Tuple[int, int]]:
    """只返回离散选项字符 span，避免答案后的长描述稀释纠偏梯度。

    RS/EVENT 的高权重监督只覆盖一个选项 token；分析段仍由低权重 span 学语言推理。
    如果模型输出 ``RS: A because ...``，这里也只监督 ``A``。
    """

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
    离散字段 `RS/EVENT` 单独加权。
    """

    first = _ANALYSIS_HEADING_RE.search(text or "")
    if not first:
        span = _line_value_span(text, "ANALYSIS")
        return span if span is not None else (0, 0)
    terminal = re.search(rf"(?im)^\s*{re.escape(terminal_label)}\s*:", text or "")
    end = terminal.start() if terminal else len(text or "")
    return first.start(), max(first.start(), end)


def target_spans_q1(text: str) -> Dict[str, Tuple[int, int]]:
    """返回 Q1 的低权重分析 span 与高权重 RS 字符 span。

    缺少合法 ``RS`` 行时不创建 ``rs`` key，训练侧可据此识别无离散监督样本；analysis
    即使为空仍保留 ``(0, 0)``，使调用接口稳定。
    """

    spans: Dict[str, Tuple[int, int]] = {"analysis": _analysis_span(text, terminal_label="RS")}
    rs_span = _line_choice_span(text, "RS", "ABCDE")
    if rs_span is not None:
        spans["rs"] = rs_span
    return spans


def target_spans_q2(text: str) -> Dict[str, Tuple[int, int]]:
    """返回 Q2 的低权重分析 span 与高权重 EVENT 字符 span。

    EVENT 字母范围允许 A-Z，是因为候选数由逐帧 allowed_events 决定；真正是否有效还要
    由本帧 ``option_map`` 解析，span 层只负责定位输出合同中的字符。
    """

    spans: Dict[str, Tuple[int, int]] = {"analysis": _analysis_span(text, terminal_label="EVENT")}
    event_span = _line_choice_span(text, "EVENT", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if event_span is not None:
        spans["event"] = event_span
    return spans


_PRIVATE_MARKERS = re.compile(r"ANSWER_|GROUND_TRUTH|REFERENCE|XML_WEATHER", re.IGNORECASE)


def check_no_private_markers(text: str) -> bool:
    """检查 student-facing teacher target 是否泄漏私有字段名。

    返回 ``False`` 时调用者必须拒绝该 target；这是一道词面防线，不能替代 student /
    teacher prompt 分离，但能捕获 teacher 复述 ANSWER、REFERENCE 或 XML_WEATHER 的错误。
    """

    return _PRIVATE_MARKERS.search(text or "") is None


def loss_weights_q1() -> Dict[str, float]:
    """返回 Q1 span 名到默认权重的映射，供训练、probe 和 mask 合同测试共用。"""

    return {
        "analysis": DEFAULT_W_ANALYSIS,
        "rs": DEFAULT_W_RS,
    }


def loss_weights_q2() -> Dict[str, float]:
    """返回 Q2 span 名到默认权重的映射，避免 eval/probe 自行复制常量。"""

    return {
        "analysis": DEFAULT_W_ANALYSIS,
        "event": DEFAULT_W_EVENT,
    }
