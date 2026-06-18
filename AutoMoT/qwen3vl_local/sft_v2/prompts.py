"""SFT v2 串行选择题 prompt 工具。

本文件是 SFT v2 的唯一 prompt 来源，负责把原来的自由文本 ANALYSIS 路线拆成
两个严格的选择题阶段：

1. 第一阶段只看 RGB 历史帧和全部场景候选，输出一行 ``SCENE: ...``。
2. 第二阶段在同一条对话后追加一个 user turn，只给定“已选场景”的事件序列，
   输出两行 ``STATUS`` / ``SUBGOAL``。

这样做的目的不是让模型自由解释，而是把任务约束成可解析、可验证、可精确打
loss 的离散状态机判断。eval 时第二阶段会使用模型自己预测的 scene；如果 scene
合法但错误，也会继续沿该 scene 的 EVENT_SEQUENCE 推理。
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from qwen3vl_local.prompt_pipeline import (
    EVENT_DESCRIPTIONS,
    SCENARIO_EVENT_SEQUENCES,
    SCENARIO_LABELS,
    get_full_sequence,
)


DATASET_VERSION = "sft_v2_serial_choice"


SCENE_SYSTEM_PROMPT = """\
You are an autonomous driving serial scene/status classifier.

Input:
- You receive a short RGB clip ordered oldest to newest.
- Each frame is a stitched three-camera view: left, front, and right.
- The newest frame is the current moment.

Rules:
- This is a choice task, not free-form generation.
- Copy scenario names and event names verbatim from the choices provided in
  the current user message.
- Do not invent scenario names or event names.
- Do not output ANALYSIS or explanations.
- If the visual evidence is ambiguous, prefer the previous status hint when it
  is provided and do not advance STATUS without a clear visual transition."""


STATUS_SYSTEM_PROMPT = """\
You are an autonomous driving status and subgoal classifier.

Input:
- You receive a short RGB clip ordered oldest to newest.
- Each frame is a stitched three-camera view: left, front, and right.
- The newest frame is the current moment.
- You also receive exactly one selected SCENE and that scene's EVENT_SEQUENCE.

Task:
1. Use only the selected SCENE's EVENT_SEQUENCE.
2. Choose STATUS by copying exactly one event token from that sequence.
3. Choose SUBGOAL as the immediate next event after STATUS in that same
   EVENT_SEQUENCE. If STATUS is final, SUBGOAL is final.

Output exactly two lines and nothing else:
STATUS: <event_name>
SUBGOAL: <event_name>

Rules:
- This is a choice task, not free-form generation.
- STATUS and SUBGOAL must be copied verbatim from the selected scene's
  EVENT_SEQUENCE.
- Do not invent event names.
- Do not output SCENE, ANALYSIS, or explanations.
- If the visual evidence is ambiguous, prefer the previous status hint and do
  not advance STATUS without a clear visual transition."""


# 兼容旧调用方的别名；新 SFT v2 主流程直接使用 SCENE_SYSTEM_PROMPT。
SYSTEM_PROMPT = SCENE_SYSTEM_PROMPT


def scenario_choices_block() -> str:
    """生成第一阶段可选场景列表。

    返回内容会直接塞进 user prompt。场景名是模型必须复制的合法 token，后面的
    人类可读说明只提供语义提示，不参与合法性校验。
    """

    lines = ["[SCENE_CHOICES]"]
    for name in sorted(SCENARIO_LABELS):
        # 固定排序让数据集和评估 prompt 可复现，避免同一数据多次构建时 prompt 抖动。
        lines.append(f"- {name}: {SCENARIO_LABELS[name]}")
    lines.append("[/SCENE_CHOICES]")
    return "\n".join(lines)


def event_choices_block() -> str:
    """生成兼容旧单阶段调用的全量事件表。

    SFT v2 主流程第二阶段只会展示一个 selected scene 的事件序列；这个函数保留给
    旧的 one-stage caller 或临时诊断使用。
    """

    lines = ["[SCENE_EVENT_CHOICES]"]
    for scenario in sorted(SCENARIO_EVENT_SEQUENCES):
        seq = get_full_sequence(scenario)
        lines.append(f"- {scenario}: {' -> '.join(seq)}")
        for event in seq:
            # EVENT_DESCRIPTIONS 只是自然语言解释，真正的可选值仍然是序列里的事件 token。
            lines.append(f"  * {event}: {EVENT_DESCRIPTIONS.get(event, event)}")
    lines.append("[/SCENE_EVENT_CHOICES]")
    return "\n".join(lines)


def build_scene_user_prompt(
    *,
    image_count: int,
) -> str:
    """构造第一阶段 user prompt。

    这个 prompt 只要求输出 SCENE，不暴露 STATUS/SUBGOAL 事件序列，避免模型在
    第一阶段被状态判断细节干扰。
    """

    return (
        f"The {image_count} images above are ordered oldest to newest; "
        "the last image is the current moment.\n\n"
        f"{scenario_choices_block()}\n\n"
        "Choose SCENE from SCENE_CHOICES. Output exactly one line and nothing "
        "else:\nSCENE: <scenario_name>"
    )


def selected_event_block(scene: str) -> str:
    """构造第二阶段的 selected scene 事件块。

    第二阶段必须只从这里列出的 EVENT_SEQUENCE 里复制 STATUS/SUBGOAL。训练和 eval
    都依赖这个约束来判断模型是否遵守“按预测场景继续推理”的串行协议。
    """

    seq = get_full_sequence(scene)
    lines = [
        "[SELECTED_SCENE]",
        f"SCENE: {scene}",
        f"DESCRIPTION: {SCENARIO_LABELS.get(scene, scene)}",
        "[/SELECTED_SCENE]",
        "",
        "[EVENT_SEQUENCE]",
        " -> ".join(seq),
    ]
    for event in seq:
        lines.append(f"- {event}: {EVENT_DESCRIPTIONS.get(event, event)}")
    lines.append("[/EVENT_SEQUENCE]")
    return "\n".join(lines)


def build_status_user_prompt(
    *,
    image_count: int,
    selected_scene: str,
    previous_status: str,
    previous_subgoal: str,
) -> str:
    """构造第二阶段 user prompt。

    参数 ``selected_scene`` 可以是真实场景，也可以是 eval 中模型预测出的合法但错误
    场景。调用方必须保证 previous_status / previous_subgoal 已经映射到 selected
    scene 的事件空间，否则 prompt 内部会自相矛盾。
    """

    return (
        f"The {image_count} images above are ordered oldest to newest; "
        "the last image is the current moment.\n\n"
        f"{selected_event_block(selected_scene)}\n\n"
        "[PREVIOUS_STATUS_HINT]\n"
        f"STATUS: {previous_status}\n"
        f"SUBGOAL: {previous_subgoal}\n"
        "[/PREVIOUS_STATUS_HINT]\n\n"
        "Task:\n"
        "1. Use only the selected scene's EVENT_SEQUENCE above.\n"
        "2. Choose STATUS by copying exactly one event token from that sequence.\n"
        "3. Choose SUBGOAL as the immediate next event after STATUS in that same sequence. "
        "If STATUS is final, SUBGOAL is final.\n\n"
        "Output exactly two lines and nothing else:\n"
        "STATUS: <event_name>\n"
        "SUBGOAL: <event_name>\n\n"
        "Do not output SCENE, ANALYSIS, or explanations."
    )


def build_user_prompt(
    *,
    image_count: int,
    previous_status: str,
    previous_subgoal: str,
) -> str:
    """构造旧版单阶段 prompt。

    新训练/eval 不走这个函数；它只作为兼容入口保留，方便和旧 SFT 或早期诊断脚本
    对比。
    """

    return (
        f"{build_scene_user_prompt(image_count=image_count)}\n\n"
        f"{event_choices_block()}\n\n"
        "[PREVIOUS_STATUS_HINT]\n"
        f"STATUS: {previous_status}\n"
        f"SUBGOAL: {previous_subgoal}\n"
        "[/PREVIOUS_STATUS_HINT]\n\n"
        "Output SCENE, STATUS, and SUBGOAL now."
    )


def next_event(scenario: str, status: str) -> str:
    """返回某个场景下 STATUS 的下一阶段事件。

    状态机完整序列是 ``initial -> middle[0..2] -> final``。如果当前 status 已经是
    final，就保持 final；如果传入非法 status，则保守返回 final，避免构建数据时崩溃。
    """

    seq = get_full_sequence(scenario)
    try:
        idx = seq.index(status)
    except ValueError:
        return "final"
    return seq[idx + 1] if idx + 1 < len(seq) else "final"


def format_assistant(scene: str, status: str, subgoal: Optional[str] = None) -> str:
    """格式化旧单阶段监督文本。

    SFT v2 主流程更常用 ``format_scene_assistant`` 和 ``format_status_assistant``，
    这个函数保留给旧接口。
    """

    if subgoal is None:
        subgoal = next_event(scene, status)
    return f"SCENE: {scene}\nSTATUS: {status}\nSUBGOAL: {subgoal}"


def format_scene_assistant(scene: str) -> str:
    """格式化第一阶段 assistant 监督文本，只包含 SCENE 值。"""

    return f"SCENE: {scene}"


def format_status_assistant(status: str, subgoal: str) -> str:
    """格式化第二阶段 assistant 监督文本，只包含 STATUS/SUBGOAL。"""

    return f"STATUS: {status}\nSUBGOAL: {subgoal}"


_SCENE_RE = re.compile(r"^\s*SCENE\s*:\s*(\S+)", re.MULTILINE | re.IGNORECASE)
_STATUS_RE = re.compile(r"^\s*STATUS\s*:\s*(\S+)", re.MULTILINE | re.IGNORECASE)
_SUBGOAL_RE = re.compile(r"^\s*SUBGOAL\s*:\s*(\S+)", re.MULTILINE | re.IGNORECASE)


def parse_output(text: str) -> Dict[str, Optional[str]]:
    """解析模型输出中的 SCENE / STATUS / SUBGOAL。

    解析逻辑刻意宽松，只取每行冒号后的第一个非空 token。这样模型多输出解释时，
    指标仍能抓到关键选择值；格式是否干净由 raw dump 另行诊断。
    """

    scene_m = _SCENE_RE.search(text or "")
    status_m = _STATUS_RE.search(text or "")
    subgoal_m = _SUBGOAL_RE.search(text or "")
    return {
        "scene": scene_m.group(1).strip() if scene_m else None,
        "status": status_m.group(1).strip() if status_m else None,
        "subgoal": subgoal_m.group(1).strip() if subgoal_m else None,
    }


def validate_choice(scene: Optional[str], status: Optional[str], subgoal: Optional[str]) -> Dict[str, bool]:
    """校验解析结果是否满足串行选择题约束。

    这里不判断是否等于 GT，只判断“模型输出是否在预测 scene 的合法事件序列里”以及
    subgoal 是否正好是 status 的下一阶段。真正的准确率在 eval.py 里计算。
    """

    scene_ok = bool(scene in SCENARIO_EVENT_SEQUENCES)
    if not scene_ok:
        return {
            "scene_valid": False,
            "status_valid_for_scene": False,
            "subgoal_valid_for_scene": False,
            "subgoal_matches_status": False,
        }
    seq = get_full_sequence(str(scene))
    status_ok = bool(status in seq)
    subgoal_ok = bool(subgoal in seq)
    expected = next_event(str(scene), str(status)) if status_ok else None
    return {
        "scene_valid": True,
        "status_valid_for_scene": status_ok,
        "subgoal_valid_for_scene": subgoal_ok,
        "subgoal_matches_status": bool(subgoal_ok and subgoal == expected),
    }


def extract_gt(assistant_text: str) -> Dict[str, str]:
    """从 assistant 监督文本里提取标签字段。

    数据加载阶段用它兼容旧 row 结构；如果 choice_meta 里有显式 target，会优先使用
    choice_meta。
    """

    parsed = parse_output(assistant_text)
    return {
        "scene": parsed.get("scene") or "",
        "status": parsed.get("status") or "",
        "subgoal": parsed.get("subgoal") or "",
    }


def target_spans(assistant_text: str) -> Dict[str, Tuple[int, int]]:
    """返回监督文本中各个“值 token”的字符区间。

    train.py 会把这些字符区间映射到 tokenizer offset，只给值 token 打 loss；冒号、
    字段名和换行等格式 token 权重保持 0。
    """

    spans: Dict[str, Tuple[int, int]] = {}
    for key, regex in (("scene", _SCENE_RE), ("status", _STATUS_RE), ("subgoal", _SUBGOAL_RE)):
        m = regex.search(assistant_text)
        if m:
            spans[key] = (m.start(1), m.end(1))
    return spans
