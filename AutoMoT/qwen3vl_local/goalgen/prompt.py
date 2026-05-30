"""Teacher-forced prompt：告诉 Qwen 真值 STATUS / SUBGOAL，不让它生成。

和 §14.10 的范式 A prompt 区别：

- system 去掉 "Output format: ANALYSIS/STATUS/SUBGOAL ..." 那段，因为我们不 decode。
- system 改成"会被告知当前 STATUS 和 SUBGOAL；请在视觉和语言上下文里把它消化"。
- user 直接把真值 STATUS、SUBGOAL（带事件描述）写出来；不要求模型回任何字段。

复用 prompt_pipeline.DrivingMemory + EVENT_DESCRIPTIONS，不再单独搭一套场景表。

注意：本文件所有进模型的字符串都是英文。中文只允许出现在 docstring/注释里。
理由见 memory:feedback-prompts-english-only：base 模型 Qwen3-VL-4B-Instruct 在
英文上训练分布更稠密；SFT v1 + 范式 A prompt 全英文，GoalGen teacher-forced
也保持英文，避免 KV 分布割裂导致下游 DiT 跟不上。
"""

from __future__ import annotations

from typing import Dict, List

from ..prompt_pipeline import EVENT_DESCRIPTIONS, DrivingMemory


# 与 prompt_pipeline._SYSTEM_PROMPT 措辞风格保持一致：
# - 同样以 "You are an autonomous driving agent ..." 开头；
# - 同样的 Input / Terminology 结构；
# - 唯一差异是去掉"Output format"段，改写为"this is teacher-forced, don't output STATUS/SUBGOAL"。
# 这种"同模板换 task 描述"的写法让两条路径的 prefill KV 在前几个 token 分布相近，
# 复用 Qwen 已经学到的范式 A 任务表征，对下游 DiT 取 KV 更稳定。
_TEACHER_SYSTEM_PROMPT = """\
You are an autonomous driving agent controlling an ego vehicle.

Input:
- You receive a short RGB clip ordered oldest to newest.
- Each frame is a stitched three-camera view: left, front, and right.
- The last frame is the current moment; earlier frames show how the scene \
has evolved.
- The current STATUS and the next SUBGOAL will be given to you with their \
semantic descriptions. They are ground truth — you do NOT need to predict \
them.

Purpose of this turn:
- You are NOT asked to output ANALYSIS, STATUS, or SUBGOAL.
- You must internally build a solid scene understanding: first, what the ego \
vehicle is currently doing at the given STATUS; second, what visual or \
dynamic change must occur in the scene before the SUBGOAL is reached.
- After this turn, your KV cache will be consumed by a downstream latent \
image generator. That downstream model predicts what the scene will look \
like once SUBGOAL is reached. Your job is therefore to compress the visual \
observation together with the given STATUS and SUBGOAL into an informative \
hidden state.

Terminology:
- SCENARIO indicates the driving challenge type.
- EVENT_SEQUENCE is the ordered state machine for this scenario.
- STATUS is the event the ego vehicle is currently in. It is ground truth.
- SUBGOAL is the event immediately after STATUS in EVENT_SEQUENCE. It is \
also ground truth."""


def build_teacher_system_prompt() -> str:
    """返回 teacher-forced 模式下固定的 system prompt（全英文）。"""

    return _TEACHER_SYSTEM_PROMPT


def _format_memory_block(memory: DrivingMemory) -> str:
    """把 memory 中关键字段格式化成 user prompt 里的 [GROUND_TRUTH_STATE] 块。

    与范式 A 的 [MEMORY] 块区别：这里强调"这是答案，不要你输出"，并把 STATUS 与
    SUBGOAL 的语义描述全部直接写出来。

    格式约定：
    - 用 [GROUND_TRUTH_STATE] / [/GROUND_TRUTH_STATE] 包起来，prompt 一眼可识别；
    - SCENARIO 行后面用 `#` 注释场景人类可读 label，便于读 prompt 时不查表；
    - EVENT_SEQUENCE 用 ` -> ` 拼，与范式 A 风格一致；
    - STATUS / SUBGOAL 各写一行 + 一行 meaning，让 Qwen 在 KV 里把"事件 token"和
      "语义描述"绑成相邻 token，对下游 DiT 取 KV 时更有信息密度。

    注意：所有 label / meaning 都是英文（feedback-prompts-english-only）。
    """

    # 把整个事件链每一步都展开成 "event: description"。
    # 之前只写 EVENT_SEQUENCE token 串，Qwen 在 KV 里只能拿到事件名 token 没有语义解释；
    # 全量列出后，DiT 通过 KV 读到的"每个子任务是什么"是密集的语义 token，对生成模型
    # 的语义引导更稳。token 成本 ~5 事件 × 30 token ≈ +6%，相对 prefill ~2300 token 可忽略。
    seq_desc_lines = [
        f"- {event}: {EVENT_DESCRIPTIONS.get(event, event)}"
        for event in memory.event_sequence
    ]
    seq_desc_str = "\n".join(seq_desc_lines)
    status_desc = EVENT_DESCRIPTIONS.get(memory.status, memory.status)
    subgoal_desc = EVENT_DESCRIPTIONS.get(memory.subgoal, memory.subgoal)

    # STATUS / SUBGOAL 单独再标一次 + 重复一次 meaning：
    # 序列里已经写过一遍各事件描述了，这里再单独点名"current event / next event to reach"，
    # 是为了让 Qwen attention 在 KV 里形成强对应——序列描述提供"全景"，
    # 这两行提供"焦点"，对下游 DiT 取 KV 时定位最有用的 token 帮助更大。
    return (
        "[GROUND_TRUTH_STATE]\n"
        f"SCENARIO: {memory.scenario}  # {memory.scenario_label}\n"
        "EVENT_SEQUENCE (each step explained in order):\n"
        f"{seq_desc_str}\n"
        f"STATUS (ground truth, current event): {memory.status}\n"
        f"  meaning: {status_desc}\n"
        f"SUBGOAL (ground truth, the next event to reach): {memory.subgoal}\n"
        f"  meaning: {subgoal_desc}\n"
        "[/GROUND_TRUTH_STATE]"
    )


def build_teacher_user_prompt(
    memory: DrivingMemory,
    image_description: str = "Refer to the visual observation(s) above.",
) -> str:
    """构造 teacher-forced 模式下的 user prompt（全英文）。

    与范式 A 区别：末尾不要求模型输出 ANALYSIS/STATUS/SUBGOAL；改为提示
    "compress the current driving state into your hidden representation"，
    并让模型只回一个 "ok" 短 token 收口，方便 KV cache 到此截断。
    """

    block = _format_memory_block(memory)
    return (
        f"{image_description}\n\n"
        f"{block}\n\n"
        "Using the visual observations above together with the ground-truth "
        "STATUS and SUBGOAL, compress the current driving state into your "
        "internal hidden representation. Do NOT output any analysis text; "
        "reply with a single short token: \"ok\"."
    )


def describe_image_inputs(num_images: int) -> str:
    """与 standalone 范式 A runner 完全一致的图片说明句子（全英文）。"""

    if num_images <= 0:
        return "No visual observation is provided for this step."
    if num_images == 1:
        return "The image above is the visual observation at the current moment."
    return (
        f"The {num_images} images above are ordered oldest to newest; "
        "the last image is the current moment."
    )
