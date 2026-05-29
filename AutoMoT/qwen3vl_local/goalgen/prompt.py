"""Teacher-forced prompt：告诉 Qwen 真值 STATUS / SUBGOAL，不让它生成。

和 §14.10 的范式 A prompt 区别：

- system 去掉 "Output format: ANALYSIS/STATUS/SUBGOAL ..." 那段，因为我们不 decode。
- system 改成"你会被告知当前 STATUS 和 SUBGOAL；请在视觉和语言上下文里把它消化"。
- user 直接把真值 STATUS、SUBGOAL（带事件描述）写出来；不要求模型回任何字段。

复用 prompt_pipeline.DrivingMemory + EVENT_DESCRIPTIONS，不再单独搭一套场景表。
"""

from __future__ import annotations

from typing import Dict, List

from ..prompt_pipeline import EVENT_DESCRIPTIONS, DrivingMemory


_TEACHER_SYSTEM_PROMPT = """\
You are an autonomous driving agent controlling an ego vehicle.

Input:
- You receive a short RGB clip ordered oldest to newest.
- Each frame is a stitched three-camera view: left, front, and right.
- The last frame is the current moment; earlier frames show how the scene has evolved.
- You will be TOLD the current STATUS and the next SUBGOAL together with their semantic descriptions; these are ground truth, not a question.

What this conversation is for:
- We are not asking you to predict STATUS or SUBGOAL.
- Build an internal grounded understanding of: (a) what the ego vehicle is doing right now under the given STATUS, and (b) what visual / dynamic change must happen for the SUBGOAL to be reached.
- Your KV cache after reading this turn will be consumed downstream by a latent image generator that predicts how the scene will look once the SUBGOAL is reached. So your job is to consolidate the visual scene, the given STATUS and the given SUBGOAL into rich hidden states.

Definitions:
- SCENARIO names the driving challenge (e.g., Accident, MergerIntoSlowTraffic).
- EVENT_SEQUENCE is the ordered state machine for this scenario.
- STATUS is the event the ego is currently at (ground truth, given to you).
- SUBGOAL is the immediate next event in EVENT_SEQUENCE that the ego should reach (ground truth, given to you)."""


def build_teacher_system_prompt() -> str:
    """返回 teacher-forced 模式下固定的 system prompt。"""

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
    """

    seq_str = " -> ".join(memory.event_sequence)
    status_desc = EVENT_DESCRIPTIONS.get(memory.status, memory.status)
    subgoal_desc = EVENT_DESCRIPTIONS.get(memory.subgoal, memory.subgoal)

    return (
        "[GROUND_TRUTH_STATE]\n"
        f"SCENARIO: {memory.scenario}  # {memory.scenario_label}\n"
        f"EVENT_SEQUENCE: {seq_str}\n"
        f"STATUS (ground truth, current): {memory.status}\n"
        f"  meaning: {status_desc}\n"
        f"SUBGOAL (ground truth, next event to reach): {memory.subgoal}\n"
        f"  meaning: {subgoal_desc}\n"
        "[/GROUND_TRUTH_STATE]"
    )


def build_teacher_user_prompt(
    memory: DrivingMemory,
    image_description: str = "Refer to the visual observation(s) above.",
) -> str:
    """构造 teacher-forced 模式下的 user prompt。

    与范式 A 区别：末尾不要求模型输出 ANALYSIS/STATUS/SUBGOAL；改为提示
    "internalize this state into your hidden representation"，便于 KV cache 收口。
    """

    block = _format_memory_block(memory)
    return (
        f"{image_description}\n\n"
        f"{block}\n\n"
        "Given the visual observations and the ground-truth STATUS / SUBGOAL above, "
        "internalize this driving state into your hidden representation. "
        "Do not output any analysis text. Acknowledge briefly with a single word: \"ok\"."
    )


def describe_image_inputs(num_images: int) -> str:
    """与 standalone 范式 A runner 完全一致的图片说明句子。"""

    if num_images <= 0:
        return "No visual observations are provided for this step."
    if num_images == 1:
        return "The image above is the current visual observation."
    return (
        f"The {num_images} images above are ordered oldest to newest; "
        "the last image is the current moment."
    )
