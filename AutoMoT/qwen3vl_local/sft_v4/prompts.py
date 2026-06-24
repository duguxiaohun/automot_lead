"""SFT v4 的 prompt、Memory 状态机与输出解析工具。

v4 仍沿用 v2 的串行选择题外壳：先选 ``SCENE``，再在该场景的事件序列中选
``STATUS`` / ``SUBGOAL``。区别是训练单元不再是单帧，而是一整段 sub-scenario
时间序列。学生模型在序列推进时维护一个纯文本 ``Memory``，每一帧的 step2/step3
都读这个 memory，并用自己的输出更新下一帧 memory。

本文件只负责“文本协议”和“状态转移”，不读图片、不加载模型。这样 train/eval/probe
三条入口能复用同一套语义，避免 prompt 或 memory 更新规则在不同脚本里悄悄分叉。
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import re
from typing import Dict, List, Optional, Tuple

from qwen3vl_local.prompt_pipeline import (
    EVENT_DESCRIPTIONS,
    SCENARIO_LABELS,
    get_full_sequence,
)


DATASET_VERSION = "sft_v4_sequence"

# Teacher 生成上下限：
# - max 给完整分析（2~4 句 + 视觉细节）留余量；
# - min 兜底防止 base Qwen 早停成 "I am." 这种 3~4 token 输出。
# 新版老师 prompt 强制写完整推理且不输出任何标签，所以 max 可以放宽。
TEACHER_MAX_NEW_TOKENS_STEP1 = 128
TEACHER_MAX_NEW_TOKENS_STEP2 = 160
TEACHER_MAX_NEW_TOKENS_STEP3 = 160

TEACHER_MIN_NEW_TOKENS_STEP1 = 32
TEACHER_MIN_NEW_TOKENS_STEP2 = 48
TEACHER_MIN_NEW_TOKENS_STEP3 = 48

DEFAULT_W_ANALYSIS = 0.2
DEFAULT_W_SCENE = 1.0
DEFAULT_W_STATUS = 1.0
DEFAULT_W_SUBGOAL = 1.0

SYSTEM_PROMPT_V4 = """\
You are an autonomous driving agent analyzing a sequence of camera frames.
You receive 4 stitched RGB frames ordered oldest to newest. Each stitched
frame contains left, front and right camera views.
You maintain a plain-text memory of your believed scene, status and subgoal.
Copy scenario names and event names verbatim from the provided option lists.
Do not invent scenario names or event names.

This frame is handled as a KV-cache conversation:
1. Step 1 describes only the current visual evidence, without using memory.
2. Step 2 (student) receives MEMORY and SCENE_CHOICES, then writes 2 to 4
   first-person evidence sentences and exactly one line "SCENE: <name>".
3. Step 3 (student) receives MEMORY and EVENT_OPTIONS for the chosen scene,
   then writes 2 to 4 first-person evidence sentences and exactly two lines
   "STATUS: <event>" and "SUBGOAL: <event>".
Later user turns are incremental. Reuse the visual context and prior assistant
turns already present in KV cache; do not expect the image instructions or
global rules to be repeated.

When you are acting as the privileged teacher (steps marked TEACHER):
- A VERDICT is provided that tells you whether the student's memory entry is
  already correct (KEEP) or wrong (CHANGE).
- Your sole job is to write 3 to 4 first-person evidence sentences in the
  student's voice that justify the VERDICT, using only visible cues and the
  memory text. Mention concrete elements (road geometry, agents, signals,
  vehicle motion, weather) that drive the conclusion.
- You MUST NOT output any "SCENE:", "STATUS:" or "SUBGOAL:" line. The final
  labels are appended automatically and are not part of your reply.
- You MUST NOT name any scenario/event token verbatim. Describe the situation
  in plain language instead. Never say "the ground truth is" or similar."""


def initial_event(scene: str) -> str:
    """返回某个场景的 canonical init 状态 token。

    这里不硬编码 ``"initial"``，而是从 ``prompt_pipeline.get_full_sequence`` 读取，
    保证未来如果事件命名变动，v4 的 memory 初始化仍跟全局 prompt 字典同步。
    """

    seq = get_full_sequence(scene)
    return seq[0]


def first_subgoal(scene: str) -> str:
    """返回某个场景 init 之后的第一个子目标 token。

    当 memory.scene 被学生改写时，status/subgoal 必须重置为
    ``initial_event(scene)`` + ``first_subgoal(scene)``，这是用户指定的“换场景就从
    新场景第一子目标重新开始”的状态机规则。
    """

    seq = get_full_sequence(scene)
    return seq[1] if len(seq) > 1 else seq[0]


@dataclasses.dataclass
class Memory:
    """学生在外循环帧之间携带的纯文本记忆。

    这个对象不是神经网络 memory，也不跨帧保留 KV cache；它只是被格式化进 prompt 的
    文本状态。训练时它由学生自己的 step2/step3 输出更新，teacher 也读取同一个学生
    memory，再用 GT hindsight 以学生口吻写纠错分析。
    """

    scene: str
    status: str
    subgoal: str
    ego_to_goal_x: float
    ego_to_goal_y: float

    def copy(self) -> "Memory":
        """复制 memory，避免状态更新函数原地修改调用方对象。"""

        return Memory(
            scene=self.scene,
            status=self.status,
            subgoal=self.subgoal,
            ego_to_goal_x=self.ego_to_goal_x,
            ego_to_goal_y=self.ego_to_goal_y,
        )

    def format_text(self) -> str:
        """按固定 prompt 协议渲染 ``[MEMORY]`` 文本块。

        文本中不仅写 token 名，也写自然语言描述和完整事件链，目的是让模型在不知道
        当前 memory 是否正确的情况下，也能理解“我现在相信的场景/状态/子目标”到底
        对应什么交通语义。
        """

        seq = get_full_sequence(self.scene)
        scene_desc = SCENARIO_LABELS.get(self.scene, self.scene)
        status_desc = EVENT_DESCRIPTIONS.get(self.status, self.status)
        subgoal_desc = EVENT_DESCRIPTIONS.get(self.subgoal, self.subgoal)
        return (
            "[MEMORY]\n"
            f"BELIEVED_SCENE: {self.scene}\n"
            f"  Description: {scene_desc}\n"
            f"  EventSequence: {' -> '.join(seq)}\n"
            f"BELIEVED_STATUS: {self.status}\n"
            f"  Description: {status_desc}\n"
            f"BELIEVED_SUBGOAL: {self.subgoal}\n"
            f"  Description: {subgoal_desc}\n"
            f"EGO_TO_GOAL_XY: x={self.ego_to_goal_x:+.1f} m, y={self.ego_to_goal_y:+.1f} m\n"
            "[/MEMORY]"
        )


def init_memory(
    *,
    run_id: str,
    sub_scenario_id: str,
    ego_to_goal_x: float,
    ego_to_goal_y: float,
    gt_scene: str | None = None,
    p_init_correct: float = 0.5,
    seed_salt: str = "",
) -> Memory:
    """在 ``t = anchor[1] - delta`` 初始化学生 memory。

    D3v4 拍板：默认 50% 直接给 GT scene，50% 从非 GT scene 均匀随机抽取。这样
    Phase A 同时覆盖“对的别改”和“错的要翻”两类监督；如需退回 v3 的 100% 错场景，
    设置 ``p_init_correct=0.0`` 即可。``seed_salt`` 用于 collector id / policy version
    注入，让多 collector 对同一 episode 能产生可复现但不同的初始扰动。
    """

    # 防止命令行传入越界值：小于 0 当 0，大于 1 当 1。这样 collector 的 env 调参
    # 不会因为拼错概率导致随机分支异常。
    p = min(1.0, max(0.0, float(p_init_correct)))
    # seed 只由数据身份 + collector/policy salt 决定，不使用全局 random 状态。
    # 好处是：同一配置重跑可复现；不同 collector / snapshot 版本又能产生不同扰动。
    seed_src = f"{run_id}::{sub_scenario_id}::{seed_salt}".encode("utf-8")
    seed = int(hashlib.sha256(seed_src).hexdigest(), 16) % (2**31)
    rng = random.Random(seed)
    candidates = sorted(SCENARIO_LABELS.keys())
    if gt_scene is not None and gt_scene in SCENARIO_LABELS and rng.random() < p:
        # 初始正确样本：训练学生在 Phase A 学会“memory 已经对时不要乱改”。
        scene = gt_scene
    else:
        if gt_scene is not None and gt_scene in SCENARIO_LABELS and len(candidates) > 1:
            # 初始错误样本：排除 GT 后随机抽，让学生学习“看到证据后翻转 scene”。
            candidates = [s for s in candidates if s != gt_scene]
        scene = rng.choice(candidates)
    return Memory(
        scene=scene,
        status=initial_event(scene),
        subgoal=first_subgoal(scene),
        ego_to_goal_x=ego_to_goal_x,
        ego_to_goal_y=ego_to_goal_y,
    )


def force_memory_to_gt_scene(memory: Memory, *, gt_scene: str) -> Memory:
    """Phase B 帧开头的弱纠偏：只把 scene 拉回 GT，status/subgoal 保留。

    D2 拍板：Phase B 的本意是“假设场景认知已被矫正，让学生顺着真实事件链推进
    status/subgoal”，而不是把 status/subgoal 每帧拉回 init。所以这里只当 memory.scene
    与 GT 不一致时，才走与 step2 翻转完全一致的“scene change → status=init,
    subgoal=first_subgoal”重置路径；scene 已经等于 GT 时全 no-op，让上一帧 step3 推
    进过的 status/subgoal 自然跨帧延续，e2/e3 才有监督密度。
    """

    mem = memory.copy()
    if mem.scene == gt_scene:
        return mem
    mem.scene = gt_scene
    mem.status = initial_event(gt_scene)
    mem.subgoal = first_subgoal(gt_scene)
    return mem


def inject_phase_b_noise(memory: Memory, *, gt_scene: str, rng: random.Random, prob: float) -> Tuple[Memory, bool]:
    """Phase B weak-correction noise: optionally switch to a random non-GT scene."""

    p = min(1.0, max(0.0, float(prob)))
    if p <= 0.0 or rng.random() >= p:
        return memory.copy(), False
    candidates = [s for s in sorted(SCENARIO_LABELS) if s != gt_scene]
    if not candidates:
        return memory.copy(), False
    scene = rng.choice(candidates)
    mem = memory.copy()
    mem.scene = scene
    mem.status = initial_event(scene)
    mem.subgoal = first_subgoal(scene)
    return mem, True


def should_trigger_step3(*, memory_scene_after_step2: str, gt_scene: str) -> bool:
    """step3 触发判定：与 step2 是否翻转无关，只看 step2 后 memory.scene 是否 = GT。

    把这个布尔规则单独抽出来，是为了能在 ``test_memory_update.py`` 中直接对状态
    机做单元测试，而不必拉起整个 train loop。
    """

    return memory_scene_after_step2 == gt_scene


def update_memory_after_step2(memory: Memory, *, student_scene: Optional[str]) -> Memory:
    """根据学生 step2 的 ``SCENE`` 输出更新 memory.scene。

    - 输出非法 scene：忽略，memory 保持不变；
    - 输出等于当前 memory.scene：视作 keep，status/subgoal 不动；
    - 输出不同合法 scene：视作场景翻转，同时强制重置 status/subgoal。
    """

    mem = memory.copy()
    if not validate_scene(student_scene):
        return mem
    assert student_scene is not None
    if student_scene != mem.scene:
        mem.scene = student_scene
        mem.status = initial_event(student_scene)
        mem.subgoal = first_subgoal(student_scene)
    return mem


def update_memory_after_step3(
    memory: Memory,
    *,
    student_status: Optional[str],
    student_subgoal: Optional[str],
) -> Memory:
    """根据学生 step3 输出更新 status/subgoal。

    只有当前 memory.scene 事件序列中的合法 event 才会生效。非法输出不写入 memory，
    避免一次格式错误污染后续整段 episode。
    """

    mem = memory.copy()
    if validate_event(mem.scene, student_status):
        assert student_status is not None
        mem.status = student_status
    if validate_event(mem.scene, student_subgoal):
        assert student_subgoal is not None
        mem.subgoal = student_subgoal
    return mem


def scenario_choices_block() -> str:
    """渲染全量场景选择表，格式与 v2 的 scene 选择题保持一致。"""

    lines = ["[SCENE_CHOICES]"]
    for name in sorted(SCENARIO_LABELS):
        lines.append(f"- {name}: {SCENARIO_LABELS[name]}")
    lines.append("[/SCENE_CHOICES]")
    return "\n".join(lines)


def _event_sequence_block(scene: str) -> str:
    """渲染某个 scene 对应的事件序列和事件描述。"""

    seq = get_full_sequence(scene)
    lines = [f"EVENT_SEQUENCE: {' -> '.join(seq)}"]
    for event in seq:
        lines.append(f"- {event}: {EVENT_DESCRIPTIONS.get(event, event)}")
    return "\n".join(lines)


def build_step1_user_prompt(image_count: int) -> str:
    """构造 step1 user prompt：只允许看图，不允许使用 memory。"""

    return (
        f"[STEP1]\n{image_count} images are ordered oldest to newest; the last image is now.\n"
        "Describe visible surroundings and recent motion in at most 3 concise sentences "
        "(no more than 60 tokens). Do not use memory, scenario names, status names, or subgoal names."
    )


def build_step2_student_prompt(memory: Memory) -> str:
    """构造学生 step2 prompt。

    学生只能看到当前 memory 和场景候选表；它需要先写短分析，再输出一个合法
    ``SCENE`` 名。prompt 中不出现 GT。
    """

    return (
        f"{memory.format_text()}\n\n"
        f"{scenario_choices_block()}\n\n"
        "[STEP2]\n"
        "Using the cached visual context and MEMORY, first write at most 2 first-person evidence "
        "sentences (no more than 40 tokens) explaining whether BELIEVED_SCENE should be kept "
        "or corrected. Then output exactly one final line by copying one option name verbatim:\n"
        "SCENE: <scenario_name>"
    )


def build_step2_teacher_prompt(memory: Memory, gt_scene: str) -> str:
    """构造 teacher step2 prompt。

    设计要点：
    - 老师不再看到 GT scene 的字面 token；只看到 VERDICT (KEEP / CHANGE) 与对 GT 场景
      的自然语言描述，避免老师直接复述 GT 名词。
    - 老师**禁止输出** ``SCENE:`` 标签行；标签由 ``build_step2_teacher_target`` 拼回去。
    - 老师必须写 3~4 句以学生口吻陈述的视觉证据，解释为什么 memory 该保持或该改写，
      并落到具体的可视线索（道路几何、其它车辆、行人、信号、天气、运动状态）。
    """

    verdict = "KEEP" if memory.scene == gt_scene else "CHANGE"
    memory_scene_desc = SCENARIO_LABELS.get(memory.scene, memory.scene)
    gt_scene_desc = SCENARIO_LABELS.get(gt_scene, gt_scene)

    if verdict == "KEEP":
        verdict_line = (
            "VERDICT: KEEP — BELIEVED_SCENE is already consistent with what the camera shows. "
            "Argue why the visible cues support the current memory entry."
        )
        focus_line = (
            f"Reference the memory's own description ({memory_scene_desc}) in plain language and "
            "list which visible features in the latest frames make it the correct interpretation."
        )
    else:
        verdict_line = (
            "VERDICT: CHANGE — BELIEVED_SCENE does not match what the camera shows. "
            "Argue why the visible cues contradict the current memory entry and what the scene "
            "actually looks like, without naming any scenario class verbatim."
        )
        focus_line = (
            f"Contrast the memory's own description ({memory_scene_desc}) with the visible cues, "
            f"and describe the actual situation in plain language (it resembles: {gt_scene_desc}). "
            "Do not write the scenario class token; describe it in free words."
        )

    return (
        f"{memory.format_text()}\n\n"
        f"{scenario_choices_block()}\n\n"
        "[STEP2_TEACHER]\n"
        f"{verdict_line}\n"
        f"{focus_line}\n"
        "Write 3 to 4 first-person evidence sentences (roughly 60-120 tokens). Cover at least one of:\n"
        "  - road geometry (junction / straight / merge / curve)\n"
        "  - other agents (vehicles, pedestrians, cyclists, their motion)\n"
        "  - signals or signs (traffic lights, stop signs, markings)\n"
        "  - ego motion and recent change between the 4 frames\n"
        "  - weather / lighting / occlusion cues\n"
        "Do not output any line starting with 'SCENE:', 'STATUS:' or 'SUBGOAL:'. "
        "Do not mention 'ground truth' or 'verdict'. Do not copy any scenario or event token "
        "from the option lists verbatim. Plain prose only."
    )


def build_step2_teacher_target(analysis: str, gt_scene: str) -> str:
    """把 teacher step2 分析与 GT scene 拼成 student 的 teacher-forced target。

    新口径下老师不再输出 SCENE 标签，``analysis`` 应该就是纯分析文本；
    但为了兼容旧 trajectory 与万一老师漏写违规输出标签，依然先 strip 标签行再 append。
    """

    cleaned = _strip_label_lines(analysis).strip()
    if not cleaned:
        cleaned = "I observe the current driving scene from the camera frames."
    return f"{cleaned}\nSCENE: {gt_scene}".strip()


def build_step3_student_prompt(memory: Memory) -> str:
    """构造学生 step3 prompt。

    只有当 step2 后 memory.scene 已经等于 GT scene 时才会进入 step3；此时学生在该
    scene 的事件序列中选择当前 ``STATUS`` 和下一步 ``SUBGOAL``。
    """

    return (
        f"{memory.format_text()}\n\n"
        f"[EVENT_OPTIONS]\n{_event_sequence_block(memory.scene)}\n[/EVENT_OPTIONS]\n\n"
        "[STEP3]\n"
        "Using the cached visual context and MEMORY, first write at most 2 first-person evidence "
        "sentences (no more than 40 tokens) explaining whether BELIEVED_STATUS and BELIEVED_SUBGOAL "
        "should be kept or corrected. Then output exactly these two final lines by copying event "
        "names verbatim:\n"
        "STATUS: <event_name>\n"
        "SUBGOAL: <event_name>"
    )


def build_step3_teacher_prompt(memory: Memory, gt_status: str, gt_subgoal: str) -> str:
    """构造 teacher step3 prompt。

    设计要点同 step2：
    - 老师不再看到 GT status / subgoal 的字面 event token；只看到 KEEP / CHANGE
      裁定与目标 event 的自然语言描述；
    - 老师禁止输出任何 ``STATUS:`` / ``SUBGOAL:`` 标签行；
    - 老师必须写 3~4 句以学生口吻陈述的视觉证据，覆盖 ego 动作、前方阻挡、对手行为
      与下一步合理意图等可视线索。
    """

    status_keep = memory.status == gt_status
    subgoal_keep = memory.subgoal == gt_subgoal
    if status_keep and subgoal_keep:
        verdict = "KEEP"
        verdict_line = (
            "VERDICT: KEEP — BELIEVED_STATUS and BELIEVED_SUBGOAL are both consistent with the "
            "latest observations. Argue why the visible cues support keeping both."
        )
    else:
        verdict = "CHANGE"
        verdict_line = (
            "VERDICT: CHANGE — at least one of BELIEVED_STATUS / BELIEVED_SUBGOAL is no longer "
            "consistent with what the camera shows. Argue what the ego currently does and what "
            "the next reasonable intent is, based on visible cues."
        )

    memory_status_desc = EVENT_DESCRIPTIONS.get(memory.status, memory.status)
    memory_subgoal_desc = EVENT_DESCRIPTIONS.get(memory.subgoal, memory.subgoal)
    gt_status_desc = EVENT_DESCRIPTIONS.get(gt_status, gt_status)
    gt_subgoal_desc = EVENT_DESCRIPTIONS.get(gt_subgoal, gt_subgoal)

    if verdict == "KEEP":
        focus_line = (
            f"Reference the believed current event ({memory_status_desc}) and the believed next "
            f"sub-goal ({memory_subgoal_desc}) in plain language. Explain which visible elements "
            "make the current sub-goal still pending and the next step still appropriate."
        )
    else:
        focus_line = (
            f"Contrast the believed current event ({memory_status_desc}) and believed next sub-goal "
            f"({memory_subgoal_desc}) with the observed driving phase, which actually looks like: "
            f"current event = {gt_status_desc}; next sub-goal = {gt_subgoal_desc}. "
            "Describe these phases in free words; do not copy any event token from the option list."
        )

    return (
        f"{memory.format_text()}\n\n"
        f"[EVENT_OPTIONS]\n{_event_sequence_block(memory.scene)}\n[/EVENT_OPTIONS]\n\n"
        "[STEP3_TEACHER]\n"
        f"{verdict_line}\n"
        f"{focus_line}\n"
        "Write 3 to 4 first-person evidence sentences (roughly 60-120 tokens). Cover at least one of:\n"
        "  - ego speed / braking / steering trend across the 4 frames\n"
        "  - immediate hazard (oncoming vehicle, pedestrian, parked obstacle, leading car)\n"
        "  - whether the ego is still approaching, yielding, holding or already resuming\n"
        "  - what cue the ego should wait for before transitioning to the next sub-goal\n"
        "Do not output any line starting with 'SCENE:', 'STATUS:' or 'SUBGOAL:'. "
        "Do not mention 'ground truth' or 'verdict'. Do not copy any scenario or event token "
        "from the option lists verbatim. Plain prose only."
    )


def build_step3_teacher_target(analysis: str, gt_status: str, gt_subgoal: str) -> str:
    """把 teacher step3 分析与 GT status/subgoal 拼成监督 target。

    新口径下老师不再输出 STATUS/SUBGOAL 标签，``analysis`` 应该就是纯分析文本；
    依然 strip 标签行兜底兼容。
    """

    cleaned = _strip_label_lines(analysis).strip()
    if not cleaned:
        cleaned = "I observe the current driving phase from the camera frames."
    return f"{cleaned}\nSTATUS: {gt_status}\nSUBGOAL: {gt_subgoal}".strip()


_SCENE_RE = re.compile(r"^\s*SCENE\s*:\s*([^\s]+)", re.IGNORECASE | re.MULTILINE)
_STATUS_RE = re.compile(r"^\s*STATUS\s*:\s*([^\s]+)", re.IGNORECASE | re.MULTILINE)
_SUBGOAL_RE = re.compile(r"^\s*SUBGOAL\s*:\s*([^\s]+)", re.IGNORECASE | re.MULTILINE)

# 用来把老师万一漏写的整行标签从分析文本里剥掉，避免污染监督 target。
_LABEL_LINE_RE = re.compile(r"^\s*(?:SCENE|STATUS|SUBGOAL)\s*:.*$", re.IGNORECASE | re.MULTILINE)


def _strip_label_lines(text: str) -> str:
    """删除分析文本中的所有 ``SCENE:`` / ``STATUS:`` / ``SUBGOAL:`` 行。"""

    if not text:
        return ""
    return _LABEL_LINE_RE.sub("", text).strip()


def parse_output(text: str) -> Dict[str, Optional[str]]:
    """从自由生成文本中解析 ``SCENE`` / ``STATUS`` / ``SUBGOAL`` 值。

    解析器只取行首 ``KEY: value``，避免分析段落里偶然提到某个字段名时误判。
    返回值可能为 ``None``，调用方必须再用 ``validate_scene`` / ``validate_event``
    做合法性检查。
    """

    scene = _SCENE_RE.search(text or "")
    status = _STATUS_RE.search(text or "")
    subgoal = _SUBGOAL_RE.search(text or "")
    return {
        "scene": scene.group(1).strip() if scene else None,
        "status": status.group(1).strip() if status else None,
        "subgoal": subgoal.group(1).strip() if subgoal else None,
    }


def analysis_before_choices(text: str) -> str:
    """返回第一条标签行之前的分析文本。"""

    cuts = [m.start() for m in (_SCENE_RE.search(text), _STATUS_RE.search(text), _SUBGOAL_RE.search(text)) if m]
    return text[: min(cuts)].strip() if cuts else text.strip()


def target_spans_scene(assistant_text: str) -> Dict[str, Tuple[int, int]]:
    """返回 step2 中需要打高权重 CE 的 scene 值字符 span。"""

    m = _SCENE_RE.search(assistant_text)
    return {"scene": (m.start(1), m.end(1))} if m else {}


def target_spans_status(assistant_text: str) -> Dict[str, Tuple[int, int]]:
    """返回 step3 中需要打高权重 CE 的 status/subgoal 值字符 span。"""

    spans: Dict[str, Tuple[int, int]] = {}
    for key, regex in (("status", _STATUS_RE), ("subgoal", _SUBGOAL_RE)):
        m = regex.search(assistant_text)
        if m:
            spans[key] = (m.start(1), m.end(1))
    return spans


def _variants(name: str) -> List[str]:
    """生成 GT token 的泄露检测变体：原名、空格版、去下划线版。"""

    vals = {name, name.replace("_", " "), name.replace("_", "")}
    return [x.lower() for x in vals if x]


def check_gt_leak_scene(analysis_text: str, gt_scene: str) -> bool:
    """检测 step2 分析是否泄露 GT scene 字面名。"""

    lower = analysis_text.lower()
    return any(v in lower for v in _variants(gt_scene))


def check_gt_leak_status_subgoal(analysis_text: str, gt_status: str, gt_subgoal: str) -> bool:
    """检测 step3 分析是否泄露 GT status/subgoal 字面名。"""

    lower = analysis_text.lower()
    return any(v in lower for name in (gt_status, gt_subgoal) for v in _variants(name))


def validate_scene(scene: Optional[str]) -> bool:
    """检查 scene 是否来自全局场景白名单。"""

    return bool(scene and scene in SCENARIO_LABELS)


def validate_event(scene: str, event: Optional[str]) -> bool:
    """检查 event 是否属于当前 scene 的合法事件序列。"""

    if not event:
        return False
    try:
        return event in get_full_sequence(scene)
    except Exception:
        return False


def next_event(scene: str, status: str) -> str:
    """返回某个 status 后面的下一事件；非法 status 时回退到第一子目标。"""

    seq = get_full_sequence(scene)
    try:
        idx = seq.index(status)
    except ValueError:
        return seq[1] if len(seq) > 1 else seq[0]
    return seq[idx + 1] if idx + 1 < len(seq) else seq[-1]

