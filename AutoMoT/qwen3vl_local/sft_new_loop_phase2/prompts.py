"""新 Phase2 的单轮 EVENT YES/NO prompt、目标渲染与严格解析。

本路线不再伪造上一轮 ROAD_STRUCTURE user/assistant 对话，也不把任何 RS 标签写进
模型输入。每个样本只给 RGB history 和当前 EVENT 问题组：道路走廊问
UE1/UE3/UE5，局部路口问 UE6。问题组与图像道路布局明显不相容时，所有 UE
必须回答 NO，并把 ``INVALID_EVENT_CONTEXT`` 回答为 YES。

事件边界直接复用 2026-07 全帧 RGB 审计和旧 Phase3 prompt v2 的已验证口径：
UE3 保留 “about to occupy / dynamic crossing” 的早期可见交互，不收紧为已经完全
进入 ego path；低能见度、拥堵、普通红灯等待或单纯没有 UE 都不能触发 invalid。
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from qwen3vl_local.sft_new_loop_phase2.history_rgb import (
    DEFAULT_HISTORY_RGB_MODE,
    HISTORY_RGB_MODE_ALL4,
    history_rgb_prompt_description,
    validate_history_rgb_mode,
)


PROMPT_NAME = "sft_new_loop_phase2_direct_event_visual_v1"
EVENT_KEYS = ("UE1", "UE3", "UE5", "UE6")
INVALID_KEY = "INVALID_EVENT_CONTEXT"
ANSWER_KEYS = (*EVENT_KEYS, INVALID_KEY)
ANSWER_VALUES = ("YES", "NO")
VARIANT_WEIGHTS = {"all_random_order": 1}
VARIANT_ORDER = tuple(VARIANT_WEIGHTS.keys())
SUBSET_COUNTS: Tuple[int, ...] = ()
GROUP_DEFINITIONS: Dict[str, Tuple[str, str, str, set[str]]] = {}

ROAD_DOMAIN = "ROAD_CORRIDOR"
JUNCTION_DOMAIN = "LOCAL_JUNCTION"
QUESTION_DOMAINS = (ROAD_DOMAIN, JUNCTION_DOMAIN)
DOMAIN_ANSWER_KEYS = {
    ROAD_DOMAIN: "DOMAIN_ROAD_CORRIDOR",
    JUNCTION_DOMAIN: "DOMAIN_LOCAL_JUNCTION",
}


SYSTEM_PROMPT = """You are the event-perception step of an autonomous-driving agent.
The input is a stitched three-camera RGB history ordered from oldest to newest. Classify only the newest moment. Answer the listed event questions directly from visible RGB evidence. No previous ROAD_STRUCTURE answer is provided or implied. Do not use scenario names, dataset labels, maps, hidden metadata, or future frames."""


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
    """一次 forward 内的 EVENT 问题集合。"""

    variant: str
    questions: Tuple[QuestionSpec, ...]
    seed_key: str
    question_domain: str
    invalid_context: bool

    @property
    def output_keys(self) -> Tuple[str, ...]:
        """返回严格输出行顺序。"""

        return tuple(q.output_key for q in self.questions)


EVENT_DEFINITIONS = {
    "UE1": """UE1 - lead vehicle hard braking / sudden slowdown:
YES only when a vehicle already in ego's forward path or same-lane following relation visibly brakes or slows enough to interrupt normal following. Use brake lights, rapid closing distance across the history, a newly formed queue in ego's lane, or clear ego-path deceleration cues. NO for ordinary steady following, normal red-light queueing, a static obstacle, a side-crossing vehicle, an early or ambiguous history with no motion cue, or a distant slow vehicle with no sudden interaction.""",
    "UE3": """UE3 - dynamic vehicle cut-in / dynamic occupation:
YES when another vehicle is moving into, cutting across, pulling out into, or visibly about to occupy ego's immediate future corridor, forcing ego to yield, slow, or stop. Use lateral motion across frames, vehicle nose or body entering ego lane, parking-side pull-out, side-lane cut-in, or dynamic crossing into the path. Do not require the vehicle to be fully centered in ego's lane. NO for ego's own planned lane change or merge, stopped accident or blocked-traffic scenes, ordinary adjacent-lane traffic, distant vehicles, weak evidence with no visible path entry, or a vehicle that remains outside the ego corridor.""",
    "UE5": """UE5 - abnormal oncoming invasion:
YES only when an oncoming or opposite-direction vehicle intrudes into ego's lane or usable corridor and ego must yield or wait. The key evidence is the other vehicle invading ego's side, not ego borrowing the opposite lane to pass an obstacle. NO for normal oncoming traffic in its own lane, ego's own TwoWays detour, ordinary narrow-road sharing without invasion, distant headlights, or a signal or priority conflict at an intersection.""",
    "UE6": """UE6 - rule-violating vehicle conflict at an intersection:
YES only at a visible local junction when ego should have priority or a legal phase but another vehicle violates the rule and occupies the conflict path. Look for a crossing, oncoming, or turning vehicle entering against the signal or right-of-way and forcing ego to stop despite priority. NO for ordinary turning or crossing vehicles that follow their lane or yield, normal red-light waiting, normal yielding, blocked traffic queue, pedestrian or cyclist crossing, defective signal hardware, a vehicle merely present in the junction, or a non-junction cut-in.""",
}

DOMAIN_DESCRIPTIONS = {
    ROAD_DOMAIN: """This question set applies to a continuous travel corridor rather than a local junction conflict. Ordinary local roads, narrow or two-way roads, and highways or ramps are valid inputs. A highway frame is not invalid merely because no target UE is present.""",
    JUNCTION_DOMAIN: """This question set applies only when the newest frame visibly belongs to a local intersection approach, entry, conflict area, or immediate exit where signal or right-of-way interaction can be judged.""",
}


def _stable_rng(*parts: object) -> random.Random:
    """从任意字段构造可复现 RNG。"""

    payload = ":".join(str(p) for p in parts)
    return random.Random(hashlib.sha256(payload.encode("utf-8")).hexdigest())


def question_domain_from_answers(answers: Mapping[str, bool]) -> str:
    """从内部 one-hot 字段恢复当前问题组；这些字段永远不写入 prompt。"""

    positives = [domain for domain, key in DOMAIN_ANSWER_KEYS.items() if bool(answers.get(key, False))]
    if len(positives) != 1:
        return "UNKNOWN"
    return positives[0]


def event_keys_for_domain(question_domain: str) -> Tuple[str, ...]:
    """返回当前问题适用域需要询问的 UE。"""

    if question_domain == ROAD_DOMAIN:
        return ("UE1", "UE3", "UE5")
    if question_domain == JUNCTION_DOMAIN:
        return ("UE6",)
    return EVENT_KEYS


def _question(key: str, answers: Mapping[str, bool]) -> QuestionSpec:
    """构造一个 EVENT/invalid 问题。"""

    if key == INVALID_KEY:
        return QuestionSpec(
            output_key=INVALID_KEY,
            metric_key=INVALID_KEY,
            question_id=INVALID_KEY,
            question="Is this event question set clearly inapplicable to the newest frame's road layout?",
            answer=bool(answers.get(INVALID_KEY, False)),
        )
    questions = {
        "UE1": "Is there a lead-vehicle hard braking or sudden slowdown event now?",
        "UE3": "Is there a dynamic vehicle cut-in or dynamic occupation of ego's path now?",
        "UE5": "Is an oncoming vehicle abnormally invading ego's usable corridor now?",
        "UE6": "Is there a rule-violating vehicle conflict inside this junction now?",
    }
    return QuestionSpec(
        output_key=key,
        metric_key=key,
        question_id=key,
        question=questions[key],
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
    """按内部 question domain 构造一次直接 EVENT 筛查。"""

    del focus, subset_count, group_id, detail_key
    if str(variant) != "all_random_order":
        raise ValueError(f"unknown direct-event variant: {variant}")
    question_domain = question_domain_from_answers(answers)
    if question_domain not in QUESTION_DOMAINS:
        raise ValueError(f"answers must contain exactly one question-domain flag: {answers}")
    rng = _stable_rng("new_phase2_direct_event_spec", seed_key, question_domain)
    event_keys = list(event_keys_for_domain(question_domain))
    rng.shuffle(event_keys)
    questions = tuple(_question(key, answers) for key in event_keys)
    questions = (*questions, _question(INVALID_KEY, answers))
    return PromptSpec(
        variant="all_random_order",
        questions=questions,
        seed_key=str(seed_key),
        question_domain=question_domain,
        invalid_context=bool(answers.get(INVALID_KEY, False)),
    )


def prompt_spec_to_json(spec: PromptSpec) -> Dict[str, object]:
    """把 PromptSpec 写入 case/audit JSON。"""

    return {
        "variant": spec.variant,
        "seed_key": spec.seed_key,
        "question_domain": spec.question_domain,
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


def build_event_prompt(
    *,
    spec: Optional[PromptSpec] = None,
    audit: bool = False,
    history_rgb_mode: str = DEFAULT_HISTORY_RGB_MODE,
) -> str:
    """构造不含 synthetic RS 对话的单轮 EVENT prompt。"""

    mode = validate_history_rgb_mode(history_rgb_mode)
    history = history_rgb_prompt_description(mode)
    endpoint_notice = "" if mode == HISTORY_RGB_MODE_ALL4 else " Only the first and fourth frames of the original four-frame history are visible; do not assume intermediate evidence exists."
    if spec is None:
        spec = make_prompt_spec(
            variant="all_random_order",
            answers={
                DOMAIN_ANSWER_KEYS[ROAD_DOMAIN]: True,
                "UE1": False,
                "UE3": False,
                "UE5": False,
                "UE6": False,
                INVALID_KEY: False,
            },
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
    text = f"""
[PROMPT_NAME]
{PROMPT_NAME}
[/PROMPT_NAME]

[VISUAL_CHECK_ORDER]
Use the {history}. First decide whether the listed question set is applicable to the newest frame's visible road layout. Then answer the listed UE questions directly. Use older frames only to confirm motion, suddenness, relative movement, or whether a conflict is entering the ego corridor.{endpoint_notice}
[/VISUAL_CHECK_ORDER]

[QUESTION_SCOPE]
{DOMAIN_DESCRIPTIONS[spec.question_domain]}
[/QUESTION_SCOPE]

[QUESTIONS]
{question_lines}
[/QUESTIONS]

[DECISION_RULES]
{event_defs}

DECISION ORDER:
1. If this question set is clearly inapplicable to the visible road layout, answer every UE line NO and {INVALID_KEY}: YES.
2. If the scope is visually plausible, keep {INVALID_KEY}: NO and judge the listed UE questions.
3. Prefer valid RE/all-NO when motion, priority violation, lane intrusion, or hard-braking evidence is weak.

RE / REGULAR / HIGHWAY HARD NEGATIVE:
If the scope is applicable but no listed UE is visibly true, answer all UE lines NO and {INVALID_KEY}: NO. Do not classify which regular event it is. Highways and ramps are valid ROAD_CORRIDOR negatives and must not become invalid only because all UE answers are NO.

INVALID_EVENT_CONTEXT:
Use {INVALID_KEY}: YES only for a clear geometry mismatch between the requested question set and the newest frame. A ROAD_CORRIDOR set is invalid on a clearly local junction conflict frame; a LOCAL_JUNCTION set is invalid on a clear continuous road or highway frame with no local junction context. Use this label cautiously. Fog, night, occlusion, congestion, a static crash, an ordinary queue, ordinary red-light waiting, hard visual ambiguity, or absence of a target UE is not enough. When invalid is YES, all UE lines must be NO.

BOUNDARIES:
UE2 static obstacles, UE4 pedestrians or cyclists, UE7 defective traffic lights, and UE8 blocked intersections are not target abnormal classes here. Treat them as valid RE/all-NO when the question scope itself still matches the visible road layout.
[/DECISION_RULES]

[{output_label}]
Output exactly these lines and nothing else:
{output_lines}
[/{output_label}]
""".strip()
    return text


def build_event_messages(
    *,
    images: Sequence[object],
    spec: PromptSpec,
    audit: bool = False,
    history_rgb_mode: str = DEFAULT_HISTORY_RGB_MODE,
    target: Optional[str] = None,
) -> List[Dict[str, object]]:
    """构造单轮 image+text user message；没有任何 synthetic assistant 前缀。"""

    content: List[Dict[str, object]] = [{"type": "image", "image": image} for image in images]
    content.append(
        {
            "type": "text",
            "text": build_event_prompt(spec=spec, audit=audit, history_rgb_mode=history_rgb_mode),
        }
    )
    messages: List[Dict[str, object]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def build_event_target(spec: PromptSpec) -> str:
    """渲染严格的 YES/NO target。"""

    return "\n".join(f"{q.output_key}: {'YES' if bool(q.answer) else 'NO'}" for q in spec.questions)


def parse_event_output(
    text: str,
    *,
    spec: PromptSpec,
    audit: bool = False,
) -> Dict[str, Optional[bool]]:
    """按完整输出合同严格解析当前 spec。

    production 必须只包含按 ``spec.output_keys`` 顺序排列的 ``KEY: YES|NO``；
    audit 必须紧接同顺序的 ``EVIDENCE_KEY: ...``，证据非空且不超过 14 个空白分词。
    行乱序、缺行、重复、额外解释或任意尾随文本都会让整条输出失效，所有值返回 None，
    从而让 generation ``format_valid`` 与 semantic ``exact`` 同时失败。
    """

    invalid: Dict[str, Optional[bool]] = {q.output_key: None for q in spec.questions}
    lines = (text or "").strip().splitlines()
    answer_count = len(spec.questions)
    expected_count = answer_count * (2 if audit else 1)
    if len(lines) != expected_count:
        return invalid

    parsed: Dict[str, Optional[bool]] = {}
    for line, question in zip(lines[:answer_count], spec.questions):
        match = re.fullmatch(rf"{re.escape(question.output_key)}: (YES|NO)", line)
        if match is None:
            return invalid
        parsed[question.output_key] = match.group(1) == "YES"

    if audit:
        for line, question in zip(lines[answer_count:], spec.questions):
            prefix = f"EVIDENCE_{question.output_key}: "
            if not line.startswith(prefix):
                return invalid
            evidence = line[len(prefix) :].strip()
            if not evidence or len(evidence.split()) > 14:
                return invalid
    return parsed


def spec_answers(spec: PromptSpec) -> Dict[str, bool]:
    """返回当前 spec 的 GT 答案。"""

    return {q.output_key: bool(q.answer) for q in spec.questions}


def spec_metric_items(spec: PromptSpec) -> Iterable[Tuple[str, str, bool]]:
    """迭代 ``(output_key, metric_key, gt_bool)``。"""

    for q in spec.questions:
        yield q.output_key, q.metric_key, bool(q.answer)


def event_prompt_sha256(
    *,
    spec: Optional[PromptSpec] = None,
    audit: bool = False,
    history_rgb_mode: str = DEFAULT_HISTORY_RGB_MODE,
) -> str:
    """返回单轮 prompt 合同指纹，供 adapter/eval 兼容校验。"""

    if spec is not None:
        payload = "\n\0\n".join(
            [SYSTEM_PROMPT, build_event_prompt(spec=spec, audit=audit, history_rgb_mode=history_rgb_mode)]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    parts = [
        json.dumps(
            {
                "prompt_name": PROMPT_NAME,
                "system_prompt": SYSTEM_PROMPT,
                "event_definitions": EVENT_DEFINITIONS,
                "domain_descriptions": DOMAIN_DESCRIPTIONS,
                "variant_order": list(VARIANT_ORDER),
                "variant_weights": dict(VARIANT_WEIGHTS),
                "answer_keys": list(ANSWER_KEYS),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    ]
    for question_domain, invalid in itertools.product(QUESTION_DOMAINS, (False, True)):
        answers = {key: False for key in EVENT_KEYS}
        answers[DOMAIN_ANSWER_KEYS[question_domain]] = True
        answers[INVALID_KEY] = invalid
        spec_obj = make_prompt_spec(
            variant="all_random_order",
            answers=answers,
            seed_key=f"sha:{question_domain}:{invalid}",
        )
        parts.append(build_event_prompt(spec=spec_obj, audit=audit, history_rgb_mode=history_rgb_mode))
    return hashlib.sha256("\n\0\n".join(parts).encode("utf-8")).hexdigest()
