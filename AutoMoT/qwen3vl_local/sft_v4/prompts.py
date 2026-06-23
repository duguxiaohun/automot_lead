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

# Teacher 生成上限比 prompt 中要求的 token 数略宽松：
# prompt 用保守上限约束风格，generate 上限给格式行和分词误差留余量。
TEACHER_MAX_NEW_TOKENS_STEP1 = 80
TEACHER_MAX_NEW_TOKENS_STEP2 = 60
TEACHER_MAX_NEW_TOKENS_STEP3 = 60

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
2. Step 2 receives MEMORY and SCENE_CHOICES, then writes at most 2 short
   first-person evidence sentences and exactly one line "SCENE: <name>".
3. Step 3 receives MEMORY and EVENT_OPTIONS for the selected scene, then writes
   at most 2 short first-person evidence sentences and exactly two lines
   "STATUS: <event>" and "SUBGOAL: <event>".
Later user turns are incremental. Reuse the visual context and prior assistant
turns already present in KV cache; do not expect the image instructions or
global rules to be repeated. Teacher-only GROUND_TRUTH blocks are privileged:
use them to choose the final labels, but never mention literal ground-truth
tokens in analysis text."""


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
) -> Memory:
    """在 ``t = anchor[1] - delta`` 初始化学生 memory。

    初始 scene 从 ``SCENARIO_LABELS \\ {gt_scene}`` 均匀随机抽取，用 run/sub-scenario
    id 固定随机种子保证可复现。排除 GT 的理由（D3 拍板）：Phase A 的稀缺信号是
    “看到证据 → 推翻错误 memory”，混入“初始就对”的 episode 与 Phase B 的反向监督
    重复，浪费样本预算。eval/部署时初始恰好等于 GT 的情形由 Phase B 训练样本覆盖，
    不存在 OOD 风险。
    """

    seed_src = f"{run_id}::{sub_scenario_id}".encode("utf-8")
    seed = int(hashlib.sha256(seed_src).hexdigest(), 16) % (2**31)
    rng = random.Random(seed)
    candidates = sorted(SCENARIO_LABELS.keys())
    if gt_scene is not None and gt_scene in SCENARIO_LABELS and len(candidates) > 1:
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

    Teacher 额外看到 ``[GROUND_TRUTH] SCENE``，但分析文本必须以学生口吻解释证据，
    不能直接复述 GT token；最终标签行固定输出 GT scene。
    """

    return (
        f"{memory.format_text()}\n\n"
        f"{scenario_choices_block()}\n\n"
        "[GROUND_TRUTH]\n"
        f"SCENE: {gt_scene}\n"
        "[/GROUND_TRUTH]\n\n"
        "[STEP2_TEACHER]\n"
        "Use privileged GT only for the final label. First write at most 2 first-person evidence "
        "sentences in the student's voice, based only on visual evidence and MEMORY. Never mention "
        "the literal ground-truth scenario name or say that ground truth was provided. Then output "
        "exactly this final label line:\n"
        f"SCENE: {gt_scene}"
    )


def build_step2_teacher_target(analysis: str, gt_scene: str) -> str:
    """把 teacher step2 分析与 GT scene 拼成 student 的 teacher-forced target。"""

    return f"{analysis.strip()}\nSCENE: {gt_scene}".strip()


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

    Teacher 看到 GT status/subgoal，用它们决定最终两行标签；分析仍然必须围绕学生
    memory 和视觉证据，避免泄露 GT 字面 token。
    """

    return (
        f"{memory.format_text()}\n\n"
        f"[EVENT_OPTIONS]\n{_event_sequence_block(memory.scene)}\n[/EVENT_OPTIONS]\n\n"
        "[GROUND_TRUTH]\n"
        f"STATUS: {gt_status}\n"
        f"SUBGOAL: {gt_subgoal}\n"
        "[/GROUND_TRUTH]\n\n"
        "[STEP3_TEACHER]\n"
        "Use privileged GT only for the final labels. First write at most 2 first-person evidence "
        "sentences in the student's voice, based only on visual evidence, MEMORY, and EVENT_OPTIONS. "
        "Never mention literal ground-truth status/subgoal names or say that ground truth was provided. "
        "Then output exactly these final label lines:\n"
        f"STATUS: {gt_status}\n"
        f"SUBGOAL: {gt_subgoal}"
    )


def build_step3_teacher_target(analysis: str, gt_status: str, gt_subgoal: str) -> str:
    """把 teacher step3 分析与 GT status/subgoal 拼成监督 target。"""

    return f"{analysis.strip()}\nSTATUS: {gt_status}\nSUBGOAL: {gt_subgoal}".strip()


_SCENE_RE = re.compile(r"^\s*SCENE\s*:\s*([^\s]+)", re.IGNORECASE | re.MULTILINE)
_STATUS_RE = re.compile(r"^\s*STATUS\s*:\s*([^\s]+)", re.IGNORECASE | re.MULTILINE)
_SUBGOAL_RE = re.compile(r"^\s*SUBGOAL\s*:\s*([^\s]+)", re.IGNORECASE | re.MULTILINE)


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

