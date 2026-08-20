"""Phase3 的 EVENT 级 YES/NO prompt、目标渲染与严格解析。

本阶段假设上一层 loop 已经得到一个 ROAD_STRUCTURE answer，并把它以
Phase2 风格的 assistant 片段放进同一轮上下文里。模型当前只负责在该 RS gate
下判断少数安全关键 UE：直道/双向路问 UE1、UE3、UE5；路口问 UE6。
训练会额外注入少量 wrong-RS 上下文，要求输出全部事件 NO 且
``INVALID_RS_CONTEXT: YES``，但 prompt 明确要求谨慎使用这个标签。
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from qwen3vl_local.sft_loop_phase3.history_rgb import (
    DEFAULT_HISTORY_RGB_MODE,
    HISTORY_RGB_MODE_ALL4,
    history_rgb_prompt_description,
    validate_history_rgb_mode,
)


PROMPT_NAME = "sft_loop_phase3_event_gate_visual_v2"
EVENT_KEYS = ("UE1", "UE3", "UE5", "UE6")
INVALID_KEY = "INVALID_RS_CONTEXT"
ANSWER_KEYS = (*EVENT_KEYS, INVALID_KEY)
ANSWER_VALUES = ("YES", "NO")
VARIANT_WEIGHTS = {"all_random_order": 1}
VARIANT_ORDER = tuple(VARIANT_WEIGHTS.keys())
SUBSET_COUNTS: Tuple[int, ...] = ()
GROUP_DEFINITIONS: Dict[str, Tuple[str, str, str, set[str]]] = {}
RS_CONTEXT_KEYS = ("CTX_R1", "CTX_R2", "CTX_R4", "CTX_R5")
RS_CONTEXT_TO_PHASE2 = {
    "R1": {"RS1": "YES", "RS2": "NO", "RS4": "NO", "RS5": "NO"},
    "R2": {"RS1": "NO", "RS2": "YES", "RS4": "NO", "RS5": "NO"},
    "R4": {"RS1": "NO", "RS2": "NO", "RS4": "YES", "RS5": "NO"},
    "R5": {"RS1": "NO", "RS2": "NO", "RS4": "NO", "RS5": "YES"},
}


SYSTEM_PROMPT = """You are the third perception step of an autonomous-driving agent.
The input is a stitched three-camera RGB history, ordered from oldest to newest. Classify only the newest moment. A previous ROAD_STRUCTURE step is shown as a text answer block; treat it as the current gate, but still check whether that gate is visually applicable. Use only visible RGB evidence and the provided gate. Do not use scenario names, dataset labels, maps, hidden metadata, or future frames."""


@dataclass(frozen=True)
class QuestionSpec:
    """一个需要模型回答的 YES/NO 问题。"""

    output_key: str
    metric_key: str
    question_id: str
    question: str
    answer: bool


@dataclass(frozen=True)
class PromptSpec:
    """一次 forward 内的事件问题集合。"""

    variant: str
    questions: Tuple[QuestionSpec, ...]
    seed_key: str
    rs_context: str
    invalid_context: bool

    @property
    def output_keys(self) -> Tuple[str, ...]:
        """返回严格输出行顺序。"""

        return tuple(q.output_key for q in self.questions)


EVENT_DEFINITIONS = {
    "UE1": """UE1 - lead vehicle hard braking / sudden slowdown:
YES only when a vehicle already in ego's forward path or same-lane following relation visibly brakes or slows enough to interrupt normal following. Use brake lights, rapid closing distance across the history, a newly formed queue in ego's lane, or clear ego-path deceleration cues. NO for ordinary steady following, normal red-light queueing, a static obstacle, a side-crossing vehicle, an early/ambiguous history with no motion cue, or a distant slow vehicle with no sudden interaction.""",
    "UE3": """UE3 - dynamic vehicle cut-in / dynamic occupation:
YES only when another vehicle is moving into, cutting across, pulling out into, or about to occupy ego's immediate future corridor, forcing ego to yield, slow, or stop. Use lateral motion across frames, vehicle nose/body entering ego lane, parking-side pull-out, side-lane cut-in, or dynamic crossing into the path. NO for ego's own planned lane change/merge, stopped accident or blocked-traffic scenes, ordinary adjacent-lane traffic, distant vehicles, dark/weak evidence with no visible path entry, or a vehicle that remains outside the ego corridor.""",
    "UE5": """UE5 - abnormal oncoming invasion:
YES only when an oncoming/opposite-direction vehicle intrudes into ego's lane or usable corridor and ego must yield or wait. The key evidence is the other vehicle invading ego's side, not ego borrowing the opposite lane to pass an obstacle. NO for normal oncoming traffic in its own lane, ego's own TwoWays detour, ordinary narrow-road sharing without invasion, or a signal/priority conflict at an intersection.""",
    "UE6": """UE6 - rule-violating vehicle conflict at an intersection:
YES only inside a local R4/R5 junction gate when ego should have priority or a legal phase but another vehicle violates the rule and occupies the conflict path. Look for a crossing/oncoming/turning vehicle entering against the signal or right-of-way and forcing ego to stop despite priority. NO for ordinary turning/crossing vehicles that follow their lane or yield, normal red-light waiting, normal yielding, blocked traffic queue, pedestrian/cyclist crossing, defective signal hardware, a vehicle merely present in the junction, or non-junction cut-in.""",
}


def _stable_rng(*parts: object) -> random.Random:
    """从任意字段构造可复现 RNG。"""

    payload = ":".join(str(p) for p in parts)
    return random.Random(hashlib.sha256(payload.encode("utf-8")).hexdigest())


def rs_context_from_answers(answers: Mapping[str, bool]) -> str:
    """从数据行的 CTX_R* one-hot 字段恢复本轮仿真 RS gate。"""

    positives = [key for key in RS_CONTEXT_KEYS if bool(answers.get(key, False))]
    if len(positives) != 1:
        return "UNKNOWN"
    return positives[0].replace("CTX_", "")


def event_keys_for_rs(rs_context: str) -> Tuple[str, ...]:
    """返回给定 RS gate 下应询问的 phase3 事件键。"""

    rs = str(rs_context)
    if rs in {"R1", "R2"}:
        return ("UE1", "UE3", "UE5")
    if rs in {"R4", "R5"}:
        return ("UE6",)
    return EVENT_KEYS


def _question(key: str, answers: Mapping[str, bool]) -> QuestionSpec:
    """构造一个事件/invalid 问题。"""

    if key == INVALID_KEY:
        return QuestionSpec(
            output_key=INVALID_KEY,
            metric_key=INVALID_KEY,
            question_id=INVALID_KEY,
            question="Is the previous ROAD_STRUCTURE gate visually inapplicable to this newest frame?",
            answer=bool(answers.get(INVALID_KEY, False)),
        )
    return QuestionSpec(
        output_key=key,
        metric_key=key,
        question_id=key,
        question={
            "UE1": "Is there a lead-vehicle hard braking or sudden slowdown event now?",
            "UE3": "Is there a dynamic vehicle cut-in or dynamic occupation of ego's path now?",
            "UE5": "Is an oncoming vehicle abnormally invading ego's usable corridor now?",
            "UE6": "Is there a rule-violating vehicle conflict inside this junction now?",
        }[key],
        answer=bool(answers.get(key, False)),
    )


def make_prompt_spec(
    *,
    variant: str,
    answers: Mapping[str, bool],
    seed_key: str,
    focus: str = "UE1",
    subset_count: int = 1,
    group_id: str = "",
    detail_key: str = "",
) -> PromptSpec:
    """按当前 RS gate 构造一次 phase3 事件筛查问题。"""

    del focus, subset_count, group_id, detail_key
    if str(variant) != "all_random_order":
        raise ValueError(f"unknown phase3 variant: {variant}")
    rs_context = rs_context_from_answers(answers)
    rng = _stable_rng("phase3_event_spec", seed_key, rs_context)
    event_keys = list(event_keys_for_rs(rs_context))
    rng.shuffle(event_keys)
    questions = tuple(_question(key, answers) for key in event_keys)
    questions = (*questions, _question(INVALID_KEY, answers))
    return PromptSpec(
        variant="all_random_order",
        questions=questions,
        seed_key=str(seed_key),
        rs_context=rs_context,
        invalid_context=bool(answers.get(INVALID_KEY, False)),
    )


def prompt_spec_to_json(spec: PromptSpec) -> Dict[str, object]:
    """把 PromptSpec 写入 case/audit JSON。"""

    return {
        "variant": spec.variant,
        "seed_key": spec.seed_key,
        "rs_context": spec.rs_context,
        "invalid_context": bool(spec.invalid_context),
        "output_keys": list(spec.output_keys),
        "questions": [
            {
                "output_key": q.output_key,
                "metric_key": q.metric_key,
                "question_id": q.question_id,
                "question": q.question,
                "answer": bool(q.answer),
            }
            for q in spec.questions
        ],
    }


def _phase2_context_block(rs_context: str) -> str:
    """渲染上一层 ROAD_STRUCTURE 的仿真回答块。"""

    answers = RS_CONTEXT_TO_PHASE2.get(str(rs_context), {})
    if not answers:
        return "RS1: NO\nRS2: NO\nRS4: NO\nRS5: NO"
    return "\n".join(f"{key}: {answers[key]}" for key in ("RS1", "RS2", "RS4", "RS5"))


def build_phase2_context_user_prompt(*, spec: PromptSpec) -> str:
    """构造 synthetic Phase2 ROAD_STRUCTURE user turn。

    Phase3 训练/eval 会把这个 turn 与下面的 assistant answer 放在真正的 chat
    history 里，让 KV 分布更接近真实 loop 中“先答 RS，再继续问 EVENT”的形态。
    """

    return f"""
[PROMPT_NAME]
sft_loop_phase2_synthetic_rs_context_for_phase3
[/PROMPT_NAME]

[TASK]
Use the newest RGB frame to answer the ROAD_STRUCTURE gate for the next event step.
[/TASK]

[ROAD_STRUCTURE_CHOICES]
RS1: non-junction one-way or same-direction local road.
RS2: non-junction narrow/two-way local road.
RS4: signal-controlled local intersection.
RS5: unsignalized or priority-controlled local intersection.
[/ROAD_STRUCTURE_CHOICES]

[OUTPUT]
Output exactly these lines and nothing else:
RS1: <YES or NO>
RS2: <YES or NO>
RS4: <YES or NO>
RS5: <YES or NO>
[/OUTPUT]
""".strip()


def build_phase2_context_assistant(*, spec: PromptSpec) -> str:
    """构造 synthetic Phase2 assistant answer。"""

    return _phase2_context_block(spec.rs_context)


def build_phase3_prompt(
    *,
    spec: Optional[PromptSpec] = None,
    audit: bool = False,
    history_rgb_mode: str = DEFAULT_HISTORY_RGB_MODE,
    include_inline_phase2_context: bool = False,
) -> str:
    """构造 phase3 事件 prompt；默认匹配真实多轮 chat 的后一轮 user turn。

    `include_inline_phase2_context=True` 只用于需要单字符串可读审计视图的场景；
    train/eval 的实际模型输入应通过 `build_phase3_messages` 放入上一轮 assistant。
    """

    mode = validate_history_rgb_mode(history_rgb_mode)
    history = history_rgb_prompt_description(mode)
    endpoint_notice = "" if mode == HISTORY_RGB_MODE_ALL4 else " Only the first and fourth frames of the original four-frame history are visible; do not assume intermediate evidence exists."
    if spec is None:
        spec = make_prompt_spec(
            variant="all_random_order",
            answers={"CTX_R1": True, "UE1": False, "UE3": False, "UE5": False, "UE6": False, INVALID_KEY: False},
            seed_key="default",
        )
    event_defs = "\n\n".join(EVENT_DEFINITIONS[key] for key in spec.output_keys if key in EVENT_DEFINITIONS)
    question_lines = "\n".join(
        f"{idx}. {q.output_key}: {q.question}" for idx, q in enumerate(spec.questions, start=1)
    )
    if audit:
        answer_lines = "\n".join(f"{q.output_key}: <YES or NO>" for q in spec.questions)
        evidence_lines = "\n".join(
            f"EVIDENCE_{q.output_key}: <one short RGB cue; max 14 words>" for q in spec.questions
        )
        output_lines = f"{answer_lines}\n{evidence_lines}"
    else:
        output_lines = "\n".join(f"{q.output_key}: <YES or NO>" for q in spec.questions)
    output_label = "AUDIT_OUTPUT" if audit else "OUTPUT"
    phase2_inline = (
        f"""

[PHASE2_RS_CONTEXT]
The previous ROAD_STRUCTURE step has already answered:
{_phase2_context_block(spec.rs_context)}
[/PHASE2_RS_CONTEXT]
"""
        if include_inline_phase2_context
        else ""
    )
    text = f"""
[PROMPT_NAME]
{PROMPT_NAME}
[/PROMPT_NAME]
{phase2_inline}

[VISUAL_CHECK_ORDER]
Classify the newest frame using the {history}. First verify whether the previous ROAD_STRUCTURE gate is visually applicable: RS1/RS2 are non-junction road gates; RS4/RS5 are local intersection gates. Then answer only the event questions listed in [QUESTIONS]. Use older frames only to confirm motion, suddenness, relative movement, or whether a conflict has entered the ego corridor.{endpoint_notice}
[/VISUAL_CHECK_ORDER]

[QUESTIONS]
{question_lines}
[/QUESTIONS]

[DECISION_RULES]
{event_defs}

DECISION ORDER:
1. If the RS gate is clearly inapplicable, answer every UE line NO and {INVALID_KEY}: YES.
2. If the RS gate is visually plausible, keep {INVALID_KEY}: NO and judge only the listed UE questions.
3. Prefer RE/all-NO when motion, priority violation, lane intrusion, or hard-braking evidence is weak in RGB.

RE / REGULAR:
If none of the listed UE questions is visibly true and the RS gate is applicable, answer all UE lines NO and {INVALID_KEY}: NO. Do not classify which regular event it is in this phase.

INVALID_RS_CONTEXT:
Answer {INVALID_KEY}: YES only when the previous RS gate is clearly incompatible with the newest RGB. Examples: the context says RS1/RS2 but the newest frame is a highway/ramp or a local intersection, or the context says RS4/RS5 but the newest frame is a plain road/highway with no local junction control. Use this label cautiously. A hard case, fog/night, blocked traffic, unusual congestion, ordinary red light, or absence of a UE event is not enough. When {INVALID_KEY}: YES, all UE lines must be NO because the event question was asked under the wrong gate.

BOUNDARIES:
UE2 static obstacles, UE4 pedestrians/cyclists, UE7 defective traffic lights, and UE8 blocked intersections are not target abnormal classes here. Treat them as RE for this phase unless the wrong RS gate itself is visibly inapplicable.
[/DECISION_RULES]

[{output_label}]
Output exactly these lines and nothing else:
{output_lines}
[/{output_label}]
""".strip()
    return text


def build_phase3_messages(
    *,
    images: Sequence[object],
    spec: PromptSpec,
    audit: bool = False,
    history_rgb_mode: str = DEFAULT_HISTORY_RGB_MODE,
    target: Optional[str] = None,
) -> List[Dict[str, object]]:
    """构造真实 chat message 序列。

    多张 RGB 只放在第一轮 user turn；Phase2 的 RS answer 是上一轮 assistant，
    Phase3 的 EVENT prompt 是后一轮 user。训练时 target 作为最后 assistant turn
    追加；自由生成/eval 时不追加。
    """

    first_content: List[Dict[str, object]] = [{"type": "image", "image": image} for image in images]
    first_content.append({"type": "text", "text": build_phase2_context_user_prompt(spec=spec)})
    messages: List[Dict[str, object]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": first_content},
        {"role": "assistant", "content": build_phase2_context_assistant(spec=spec)},
        {
            "role": "user",
            "content": build_phase3_prompt(
                spec=spec,
                audit=audit,
                history_rgb_mode=history_rgb_mode,
                include_inline_phase2_context=False,
            ),
        },
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def build_phase3_target(spec: PromptSpec) -> str:
    """Render the strict target for this phase3 spec."""

    return "\n".join(f"{q.output_key}: {'YES' if bool(q.answer) else 'NO'}" for q in spec.questions)


def parse_phase3_output(text: str, *, spec: PromptSpec) -> Dict[str, Optional[bool]]:
    """严格解析当前 spec 期待的 YES/NO 行，漏答/重复返回 None。"""

    out: Dict[str, Optional[bool]] = {q.output_key: None for q in spec.questions}
    for q in spec.questions:
        matches = re.findall(rf"(?im)^\s*{re.escape(q.output_key)}\s*:\s*(YES|NO)\s*$", text or "")
        if len(matches) == 1:
            out[q.output_key] = matches[0] == "YES"
    return out


def spec_answers(spec: PromptSpec) -> Dict[str, bool]:
    """返回当前 spec 的 GT 答案。"""

    return {q.output_key: bool(q.answer) for q in spec.questions}


def spec_metric_items(spec: PromptSpec) -> Iterable[Tuple[str, str, bool]]:
    """Yield (output_key, metric_key, gt_bool) for metrics."""

    for q in spec.questions:
        yield q.output_key, q.metric_key, bool(q.answer)


def phase3_prompt_sha256(
    *,
    spec: Optional[PromptSpec] = None,
    audit: bool = False,
    history_rgb_mode: str = DEFAULT_HISTORY_RGB_MODE,
) -> str:
    """Return a prompt-contract fingerprint persisted with each adapter/eval."""

    if spec is not None:
        payload = "\n\0\n".join(
            [
                SYSTEM_PROMPT,
                build_phase2_context_user_prompt(spec=spec),
                build_phase2_context_assistant(spec=spec),
                build_phase3_prompt(
                    spec=spec,
                    audit=audit,
                    history_rgb_mode=history_rgb_mode,
                    include_inline_phase2_context=False,
                ),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    parts = [
        json.dumps(
            {
                "prompt_name": PROMPT_NAME,
                "system_prompt": SYSTEM_PROMPT,
                "event_definitions": EVENT_DEFINITIONS,
                "variant_order": list(VARIANT_ORDER),
                "variant_weights": dict(VARIANT_WEIGHTS),
                "answer_keys": list(ANSWER_KEYS),
                "rs_context_to_phase2": RS_CONTEXT_TO_PHASE2,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    ]
    for rs_context, invalid in itertools.product(("R1", "R2", "R4", "R5"), (False, True)):
        answers = {f"CTX_{rs_context}": True, INVALID_KEY: invalid}
        for key in EVENT_KEYS:
            answers[key] = False
        spec_obj = make_prompt_spec(variant="all_random_order", answers=answers, seed_key=f"sha:{rs_context}:{invalid}")
        parts.append(build_phase2_context_user_prompt(spec=spec_obj))
        parts.append(build_phase2_context_assistant(spec=spec_obj))
        parts.append(
            build_phase3_prompt(
                spec=spec_obj,
                audit=audit,
                history_rgb_mode=history_rgb_mode,
                include_inline_phase2_context=False,
            )
        )
    return hashlib.sha256("\n\0\n".join(parts).encode("utf-8")).hexdigest()
