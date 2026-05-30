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
你是一个自动驾驶智能体，正在控制自车。

输入说明：
- 你会收到一小段按时间排序的 RGB 图像，顺序是从最旧到最新。
- 每一帧都是三视角拼接图，包含左视、前视、右视。
- 最后一帧代表当前时刻，前面的帧展示场景如何演化。
- 当前 STATUS 和下一步 SUBGOAL 会直接告诉你，并附带语义说明；它们是真值，不是让你猜的问题。

这轮对话的目的：
- 不要求你预测 STATUS 或 SUBGOAL。
- 你需要在内部建立扎实的场景理解：第一，自车在给定 STATUS 下当前正在做什么；第二，场景中需要出现什么视觉或动态变化，才算到达 SUBGOAL。
- 你读完这轮输入后产生的 KV cache 会被下游潜变量图像生成器使用；下游模型要预测到达 SUBGOAL 时画面会是什么样。因此你的任务是把视觉场景、给定 STATUS、给定 SUBGOAL 压缩成信息丰富的隐藏状态。

术语说明：
- SCENARIO 表示驾驶挑战类型。
- EVENT_SEQUENCE 是该场景的有序状态机。
- STATUS 是自车当前所在事件，是真值。
- SUBGOAL 是 EVENT_SEQUENCE 中自车马上要到达的下一个事件，也是真值。"""


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

    # STATUS / SUBGOAL 仍单独标出 + 重复一次 meaning：
    # 序列里已经写过一遍各事件描述了，这里再单独点名"现在你在哪、下一站是哪"，
    # 是为了让 Qwen attention 在 KV 里形成强对应——序列描述提供"全景"，
    # 这两行提供"焦点"，对下游 DiT 取 KV 时定位最有用的 token 帮助更大。
    return (
        "[GROUND_TRUTH_STATE]\n"
        f"SCENARIO: {memory.scenario}  # {memory.scenario_label}\n"
        "EVENT_SEQUENCE（按顺序解释每一步）:\n"
        f"{seq_desc_str}\n"
        f"STATUS（真值，当前所在事件）: {memory.status}\n"
        f"  含义: {status_desc}\n"
        f"SUBGOAL（真值，下一步要到达的事件）: {memory.subgoal}\n"
        f"  含义: {subgoal_desc}\n"
        "[/GROUND_TRUTH_STATE]"
    )


def build_teacher_user_prompt(
    memory: DrivingMemory,
    image_description: str = "Refer to the visual observation(s) above.",
) -> str:
    """构造 teacher-forced 模式下的 user prompt。

    与范式 A 区别：末尾不要求模型输出 ANALYSIS/STATUS/SUBGOAL；改为提示
    "把当前驾驶状态压缩进隐藏表示"，便于 KV cache 收口。
    """

    block = _format_memory_block(memory)
    return (
        f"{image_description}\n\n"
        f"{block}\n\n"
        "请结合上面的视觉观测以及真值 STATUS / SUBGOAL，"
        "把当前驾驶状态压缩进你的隐藏表示中。"
        "不要输出分析文字，只用一个词简短确认：\"ok\"。"
    )


def describe_image_inputs(num_images: int) -> str:
    """与 standalone 范式 A runner 完全一致的图片说明句子。"""

    if num_images <= 0:
        return "这一步没有提供视觉观测。"
    if num_images == 1:
        return "上面的图像是当前时刻的视觉观测。"
    return (
        f"上面的 {num_images} 张图像按时间从旧到新排列；"
        "最后一张图像是当前时刻。"
    )
