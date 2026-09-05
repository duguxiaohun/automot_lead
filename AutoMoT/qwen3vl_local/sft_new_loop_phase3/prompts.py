"""新 Phase3 的单轮 high-level ACTION YES/NO prompt、目标渲染与严格解析。

输入合同：一个 system turn + 一个 user turn。user turn 里包含
四帧（或两端点）拼接 RGB history、由 Phase1/Phase2 或常规候选步骤提出的场景上下文文本、
route 目标点的 ego 相对坐标，以及本次要回答的 high-level 动作问题。

本阶段只有五个 high-level 动作：``DECELERATE / STOP / RESUME /
LANE_CHANGE_LEFT / LANE_CHANGE_RIGHT``。它们按问题域被复用：

* ``LONGITUDINAL_YIELD``（U-E1 / U-E3 / U-E5 / U-E6 / U-E7 / R-E5）
  只问 ``DECELERATE / STOP / RESUME``；
* ``FULL_MANEUVER``（U-E2、U-E4、R-E2 目标变道/恢复、R-E3 合流/驶出）
  问全部五行。

标签口径来自 2026-09-04 的逐帧 meta 轨迹 + RGB 复核（见
`probe_trajectory.py` / `render_action_contact_sheet.py` 的 probe_output 产物）：

* 纵向动作只由未来 2s 的真实速度曲线决定，STOP 用 1.5s 即时窗，因此“已经停稳
  但马上起步”属于 RESUME 而不是继续 STOP；
* 横向动作只由 OpenDRIVE 车道身份的真实切换决定。弯道会让 steer/yaw 长期非零却
  不换车道，所以 prompt 必须显式禁止用转向角、车道线在画面里横扫或车头偏角当作
  变道证据；
* ``LANE_CHANGE_LEFT`` 在 R2 借对向车道绕障时成立，回原车道则是
  ``LANE_CHANGE_RIGHT``；两者都以自车航向为参照，不是以画面为参照。

``INVALID_ACTION_CONTEXT=YES`` 表示“本题道路前提明显错误，或道路正确但事件被可见证据明确反驳”，此时所有动作行必须为 NO。夜间、雾、遮挡、拥堵、或者“当前不需要任何动作”
都不是 invalid：不需要动作时应当所有动作行为 NO 且 invalid 也为 NO。
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import (
    ACTION_KEYS,
    CONTEXT_BY_ID,
    CONTEXT_IDS,
    DOMAIN_ACTION_KEYS,
    DOMAIN_LONGITUDINAL,
    DOMAIN_MANEUVER,
    QUESTION_DOMAINS,
    ROAD_STRUCTURE_TEXT,
)
from qwen3vl_local.sft_new_loop_phase3.history_rgb import (
    DEFAULT_HISTORY_RGB_MODE,
    HISTORY_RGB_MODE_ALL4,
    history_rgb_prompt_description,
    validate_history_rgb_mode,
)
from qwen3vl_local.sft_new_loop_phase3.navigation_goal import render_navigation_goal


PROMPT_NAME = "sft_new_loop_phase3_high_level_action_v4_context_recheck"
INVALID_KEY = "INVALID_ACTION_CONTEXT"
ANSWER_KEYS: Tuple[str, ...] = (*ACTION_KEYS, INVALID_KEY)
ANSWER_VALUES = ("YES", "NO")
VARIANT_WEIGHTS = {"all_random_order": 1}
TRAIN_VARIANT_WEIGHTS = dict(VARIANT_WEIGHTS)
VARIANT_ORDER = tuple(VARIANT_WEIGHTS.keys())
SUBSET_COUNTS: Tuple[int, ...] = ()
GROUP_DEFINITIONS: Dict[str, Tuple[str, str, str, set]] = {}

ACTION_HORIZON_TEXT = (
    "The answer is a high-level plan for the next few seconds after the newest frame, "
    "about two seconds for the speed lines and about three seconds for the lane lines."
)

SYSTEM_PROMPT = """You are the high-level decision step of an autonomous-driving agent.
The input is a stitched three-camera RGB history ordered from oldest to newest, a road structure and driving situation proposed by an earlier perception or transition step for the newest frame, and the route target point in ego coordinates. Check the proposed context against the RGB before choosing the high-level actions ego should take next. Answer every listed action question independently from the visible RGB history, the given situation and the route target. Do not use scenario names, dataset labels, maps, hidden state, or future frames."""


ACTION_DEFINITIONS: Dict[str, str] = {
    "DECELERATE": """DECELERATE - clearly reduce speed without coming to rest:
YES when the first meaningful speed change is a slowdown in the next about two seconds, without a sustained near-stop in the immediate one-and-a-half-second window. Examples include holding a safe lead gap, yielding to an intruding actor, and waiting for a usable lane-change gap. A possible stop beyond that immediate window does not by itself cancel DECELERATE. NO when STOP applies, acceleration comes first, or the speed only jitters around the same cruising value.""",
    "STOP": """STOP - come to rest, or stay at rest, and wait:
YES when ego should reach a sustained near-stop within about one and a half seconds, or continue waiting at rest, including staying stopped while an obstacle, a queue, a crossing user or a conflicting vehicle still blocks the path. NO when ego only slows down but keeps rolling, and NO when ego is currently at rest but should already be pulling away inside that window.""",
    "RESUME": """RESUME - clearly build speed toward normal travel speed:
YES when the available path permits a sustained speed gain in the next about two seconds. This includes pulling away from a standstill, gaining speed through a usable bypass gap, and recovering after a blocking actor or conflict clears enough. A previous stop or a completed yield is not implied. NO for a brief isolated speed pulse followed by renewed slowing, NO when a meaningful slowdown or a stop still comes first inside that window, and NO for steady cruising with no real speed gain. Starting to roll does not by itself prove a stop-sign obligation or a lane-change gap is satisfied.""",
    "LANE_CHANGE_LEFT": """LANE_CHANGE_LEFT - move out of the current lane into the lane on ego's left:
YES when ego should cross the left lane boundary and occupy the neighbouring lane on its left within the next about three seconds. This includes borrowing the opposing lane to get around a blockage, moving left into a main-line lane while merging, and moving left back toward the route-target lane. Left is relative to ego's own heading, not to the image. NO when ego only follows a curved lane: a bend makes the steering angle, the vehicle heading and the lane markings sweep across the image while ego stays between the same two lane boundaries, and that is lane keeping, not a lane change. NO when ego stays inside its lane while passing a slower or stopped vehicle, when ego only follows a ramp or connecting road that physically becomes the next lane without crossing a lane boundary, and NO when another vehicle rather than ego is the one changing lane.""",
    "LANE_CHANGE_RIGHT": """LANE_CHANGE_RIGHT - move out of the current lane into the lane on ego's right:
YES when ego should cross the right lane boundary and occupy the neighbouring lane on its right within the next about three seconds. This includes returning from a borrowed or opposing lane back into ego's own lane after a blockage, moving right into a deceleration or exit lane, and moving right toward the route-target lane. Right is relative to ego's own heading, not to the image. The same negative boundaries as the left line apply: a curved lane, an in-lane pass, a ramp that becomes the next lane without a boundary crossing, and another vehicle's lane change are all NO.""",
}


DOMAIN_DESCRIPTIONS: Dict[str, str] = {
    DOMAIN_LONGITUDINAL: (
        "This question set covers speed control only. No lane-change line is asked here; "
        "unasked lateral actions remain unknown and must not be interpreted as NO."
    ),
    DOMAIN_MANEUVER: (
        "This question set covers both speed control and a possible lane change, because the given "
        "situation can require ego to leave or to re-enter a lane."
    ),
}


@dataclass(frozen=True)
class QuestionSpec:
    """一个需要模型回答的 YES/NO 动作问题。"""

    output_key: str
    metric_key: str
    question_id: str
    question: str
    answer: bool


@dataclass(frozen=True)
class PromptSpec:
    """一次 forward 内的动作问题集合与场景上下文。"""

    variant: str
    questions: Tuple[QuestionSpec, ...]
    seed_key: str
    context_id: str
    question_domain: str
    road_structure: str
    invalid_context: bool
    context_detail: str = ""
    goal_xy: Optional[Tuple[float, float]] = None

    @property
    def output_keys(self) -> Tuple[str, ...]:
        """返回严格输出行顺序。"""

        return tuple(q.output_key for q in self.questions)


ACTION_QUESTIONS: Dict[str, str] = {
    "DECELERATE": "Should ego clearly reduce speed without stopping now?",
    "STOP": "Should ego stop and wait now?",
    "RESUME": "Should ego accelerate back toward normal travel speed now?",
    "LANE_CHANGE_LEFT": "Should ego change into the lane on its left now?",
    "LANE_CHANGE_RIGHT": "Should ego change into the lane on its right now?",
}


def _stable_rng(*parts: object) -> random.Random:
    """从任意字段构造可复现 RNG。"""

    payload = ":".join(str(p) for p in parts)
    return random.Random(hashlib.sha256(payload.encode("utf-8")).hexdigest())


def _question(key: str, answers: Mapping[str, bool]) -> QuestionSpec:
    """构造一个动作 / invalid 问题。"""

    if key == INVALID_KEY:
        return QuestionSpec(
            output_key=INVALID_KEY,
            metric_key=INVALID_KEY,
            question_id=INVALID_KEY,
            question="Does the visible RGB clearly contradict the proposed road structure or driving event, even if the road structure itself is correct?",
            answer=bool(answers.get(INVALID_KEY, False)),
        )
    return QuestionSpec(
        output_key=key,
        metric_key=key,
        question_id=key,
        question=ACTION_QUESTIONS[key],
        answer=bool(answers.get(key, False)),
    )


def action_keys_for_domain(question_domain: str) -> Tuple[str, ...]:
    """返回当前问题域需要询问的动作行。"""

    return DOMAIN_ACTION_KEYS.get(str(question_domain), ACTION_KEYS)


def make_prompt_spec(
    *,
    variant: str,
    answers: Mapping[str, bool],
    seed_key: str,
    context_id: str,
    road_structure: str,
    goal_xy: Optional[Sequence[float]] = None,
    context_detail: str = "",
    focus: str = "",
    subset_count: int = 1,
    group_id: str = "",
    detail_key: str = "",
) -> PromptSpec:
    """按上下文构造一次 high-level 动作提问。"""

    del focus, subset_count, group_id, detail_key
    if str(variant) != "all_random_order":
        raise ValueError(f"unknown phase3 action variant: {variant}")
    context = CONTEXT_BY_ID.get(str(context_id))
    if context is None:
        raise ValueError(f"unknown phase3 action context: {context_id!r}")
    if str(road_structure) not in ROAD_STRUCTURE_TEXT:
        raise ValueError(f"unknown road structure for phase3 prompt: {road_structure!r}")
    rng = _stable_rng("new_phase3_action_spec", seed_key, context.context_id)
    keys = list(action_keys_for_domain(context.question_domain))
    rng.shuffle(keys)
    questions = tuple(_question(key, answers) for key in keys)
    questions = (*questions, _question(INVALID_KEY, answers))
    goal = None if goal_xy is None else (float(goal_xy[0]), float(goal_xy[1]))
    return PromptSpec(
        variant="all_random_order",
        questions=questions,
        seed_key=str(seed_key),
        context_id=context.context_id,
        question_domain=context.question_domain,
        road_structure=str(road_structure),
        invalid_context=bool(answers.get(INVALID_KEY, False)),
        goal_xy=goal,
        context_detail=str(context_detail),
    )


def prompt_spec_to_json(spec: PromptSpec) -> Dict[str, object]:
    """把 PromptSpec 写入 case/audit JSON。"""

    return {
        "variant": spec.variant,
        "seed_key": spec.seed_key,
        "context_id": spec.context_id,
        "context_detail": spec.context_detail,
        "question_domain": spec.question_domain,
        "road_structure": spec.road_structure,
        "invalid_context": bool(spec.invalid_context),
        "goal_xy": list(spec.goal_xy) if spec.goal_xy is not None else None,
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


def _scene_context_block(spec: PromptSpec) -> str:
    """渲染来自上游事实或常规候选步骤、需要与RGB核对的场景前提。"""

    context = CONTEXT_BY_ID[spec.context_id]
    return (
        "[SCENE_CONTEXT]\n"
        f"ROAD_STRUCTURE: The newest frame is on {ROAD_STRUCTURE_TEXT[spec.road_structure]}.\n"
        f"SITUATION: The proposed current situation is that {context.situation_text}.\n"
        f"SCOPE: {context.scope_text}\n"
        f"OBSERVED_HISTORY: {spec.context_detail or 'No additional transition history is asserted.'}\n"
        "This context is a premise, not a label to repeat. Check it against the RGB history: if the "
        "visible road structure contradicts the stated ROAD_STRUCTURE or visible evidence clearly rules out the stated situation, the whole question set is invalid.\n"
        "[/SCENE_CONTEXT]"
    )


def build_action_prompt(
    *,
    spec: Optional[PromptSpec] = None,
    audit: bool = False,
    history_rgb_mode: str = DEFAULT_HISTORY_RGB_MODE,
) -> str:
    """构造单轮 high-level 动作 prompt。"""

    mode = validate_history_rgb_mode(history_rgb_mode)
    history = history_rgb_prompt_description(mode)
    endpoint_notice = (
        ""
        if mode == HISTORY_RGB_MODE_ALL4
        else " Only the first and fourth frames of the original four-frame history are visible; do not assume intermediate evidence exists."
    )
    if spec is None:
        spec = make_prompt_spec(
            variant="all_random_order",
            answers={key: False for key in ANSWER_KEYS},
            seed_key="default",
            context_id="LEAD_BRAKE",
            road_structure="R1",
            goal_xy=(40.0, -2.0),
        )
    action_defs = "\n\n".join(
        ACTION_DEFINITIONS[key] for key in spec.output_keys if key in ACTION_DEFINITIONS
    )
    question_lines = "\n".join(
        f"{idx}. {q.output_key}: {q.question}" for idx, q in enumerate(spec.questions, start=1)
    )
    if audit:
        answer_lines = "\n".join(f"{q.output_key}: <YES or NO>" for q in spec.questions)
        evidence_lines = "\n".join(
            f"EVIDENCE_{q.output_key}: <visible RGB or route-target cue; max 14 words>"
            for q in spec.questions
        )
        output_lines = f"{answer_lines}\n{evidence_lines}"
        audit_contract = f"""

[AUDIT_EVIDENCE_CONTRACT]
Every EVIDENCE line is mandatory and must contain a non-empty visible cue, including for a NO answer. For NO, state the visible absence or boundary, such as "lead gap steady, no braking cue"; never leave text after the colon blank. For {INVALID_KEY}, describe either the visible layout mismatch for YES or why the given situation remains visually plausible for NO. Keep every cue at 14 words or fewer.
[/AUDIT_EVIDENCE_CONTRACT]"""
    else:
        output_lines = "\n".join(f"{q.output_key}: <YES or NO>" for q in spec.questions)
        audit_contract = ""
    output_label = "AUDIT_OUTPUT" if audit else "OUTPUT"

    return f"""
[PROMPT_NAME]
{PROMPT_NAME}
[/PROMPT_NAME]

[VISUAL_CHECK_ORDER]
Use the {history}. First read the given road structure and situation, then confirm from the newest frame that the visible layout can host that situation. Then read ego's own motion across the history: whether the lead gap is closing, whether ego is already braking or already stopped, and whether ego is still centred between the same two lane boundaries. Older frames only establish how the newest state developed. {ACTION_HORIZON_TEXT}{endpoint_notice}
[/VISUAL_CHECK_ORDER]

{_scene_context_block(spec)}

{render_navigation_goal(spec.goal_xy)}

[QUESTION_SCOPE]
{DOMAIN_DESCRIPTIONS[spec.question_domain]}
[/QUESTION_SCOPE]

[QUESTIONS]
{question_lines}
[/QUESTIONS]

[DECISION_RULES]
{action_defs}

DECISION ORDER:
1. If the stated road structure is clearly wrong or the given situation is clearly contradicted by visible evidence, answer every action line NO and {INVALID_KEY}: YES.
2. Otherwise keep {INVALID_KEY}: NO and judge every listed action line on its own.
3. When no listed action is needed, answer every action line NO and keep {INVALID_KEY}: NO. Holding the current lane at the current speed is a normal, valid outcome, not an invalid context.

LONGITUDINAL EXCLUSIVITY:
{ACTION_KEYS[0]}, {ACTION_KEYS[1]} and {ACTION_KEYS[2]} describe the same speed decision at different levels, so at most one of them is YES. First apply STOP for a sustained near-stop within about one and a half seconds, except for an already stopped ego continuously pulling away. Otherwise use the first meaningful speed change within about two seconds: slowdown means DECELERATE and a sustained speed gain means RESUME. An isolated acceleration pulse is insufficient for RESUME. Ignore brief near-zero flicker and ordinary speed noise. STOP and a lane-change YES can coexist because the speed and lane horizons differ; they do not request simultaneous stopping and lateral motion.

LATERAL EVIDENCE BOUNDARY:
A lane change means ego crosses a lane boundary and ends up in a different lane. Steering input, a vehicle yaw offset, lane markings sweeping across the image, or the road bending are not lane-change evidence: on a curved lane ego keeps the same two lane boundaries and both lane lines stay NO. Do not turn another vehicle's cut-in, ego passing a stopped or slower vehicle inside its own lane, or a ramp that physically becomes the next lane into a lane change. Left and right are relative to ego's heading. When a lane change is required, exactly one side is YES.

ROUTE TARGET USE:
The route target offset says where the navigation goal lies relative to ego now, with negative y on ego's left and positive y on ego's right. It is the final destination, not the centre of the next required lane. Its sign cannot choose a lane-change side: a destination far to the left can still require returning right after bypassing an obstacle. Determine the immediate target lane from visible boundaries, the established bypass history and any available current navigation instruction. a target far ahead and slightly off centre is normal lane following, and a lane change still needs visible room and a visible boundary to cross.

{INVALID_KEY}:
Use {INVALID_KEY}: YES only for a clear mismatch between the given situation and the newest frame, for example a blocked-lane bypass situation on a clean open corridor with nothing blocking ego, a local-junction conflict situation on a continuous limited-access highway with no junction, or a ramp merge and exit situation on an ordinary local surface street with no ramp, merge or exit geometry. Night, fog, glare, occlusion, congestion, an ordinary queue, or simply needing no action are not invalid. When invalid is YES, every action line must be NO.
[/DECISION_RULES]
{audit_contract}

[{output_label}]
Output exactly these lines and nothing else:
{output_lines}
[/{output_label}]
""".strip()


def build_action_messages(
    *,
    images: Sequence[object],
    spec: PromptSpec,
    audit: bool = False,
    history_rgb_mode: str = DEFAULT_HISTORY_RGB_MODE,
    target: Optional[str] = None,
) -> List[Dict[str, object]]:
    """构造单轮 image+text user message。"""

    content: List[Dict[str, object]] = [{"type": "image", "image": image} for image in images]
    content.append(
        {
            "type": "text",
            "text": build_action_prompt(spec=spec, audit=audit, history_rgb_mode=history_rgb_mode),
        }
    )
    messages: List[Dict[str, object]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def build_action_target(spec: PromptSpec) -> str:
    """渲染严格的 YES/NO target。"""

    return "\n".join(f"{q.output_key}: {'YES' if bool(q.answer) else 'NO'}" for q in spec.questions)


def action_prompt_sha256(*, audit: bool = False, history_rgb_mode: str = DEFAULT_HISTORY_RGB_MODE) -> str:
    """返回完整 prompt 表面的指纹，任何措辞/顺序变化都会改变它。"""

    parts: List[str] = [
        json.dumps(
            {
                "prompt_name": PROMPT_NAME,
                "system_prompt": SYSTEM_PROMPT,
                "answer_keys": list(ANSWER_KEYS),
                "question_domains": list(QUESTION_DOMAINS),
                "domain_action_keys": {k: list(v) for k, v in DOMAIN_ACTION_KEYS.items()},
                "context_ids": list(CONTEXT_IDS),
                "variant_order": list(VARIANT_ORDER),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    ]
    dummy = {key: False for key in ANSWER_KEYS}
    for context_id in CONTEXT_IDS:
        context = CONTEXT_BY_ID[context_id]
        for road_structure in context.allowed_rs:
            for goal in ((40.0, -2.0), (12.0, 18.0)):
                spec = make_prompt_spec(
                    variant="all_random_order",
                    answers=dummy,
                    seed_key="fingerprint",
                    context_id=context_id,
                    road_structure=road_structure,
                    goal_xy=goal,
                )
                parts.append(build_action_prompt(spec=spec, audit=audit, history_rgb_mode=history_rgb_mode))
    for domain, keys in sorted(DOMAIN_ACTION_KEYS.items()):
        for order in itertools.permutations(keys):
            parts.append(f"{domain}:{'|'.join(order)}")
    encoded = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_action_answer_lines(text: str, *, spec: PromptSpec) -> Dict[str, Optional[bool]]:
    """只解析开头的严格答案行，供 audit 区分语义答案与 evidence 格式。"""

    invalid: Dict[str, Optional[bool]] = {q.output_key: None for q in spec.questions}
    lines = (text or "").strip().splitlines()
    answer_count = len(spec.questions)
    if len(lines) < answer_count:
        return invalid
    parsed: Dict[str, Optional[bool]] = {}
    for line, question in zip(lines[:answer_count], spec.questions):
        match = re.fullmatch(rf"{re.escape(question.output_key)}: (YES|NO)", line)
        if match is None:
            return invalid
        parsed[question.output_key] = match.group(1) == "YES"
    return parsed


def parse_action_output(
    text: str,
    *,
    spec: PromptSpec,
    audit: bool = False,
) -> Dict[str, Optional[bool]]:
    """按完整输出合同严格解析当前 spec；任何越界文本都会让整条输出失效。"""

    invalid: Dict[str, Optional[bool]] = {q.output_key: None for q in spec.questions}
    lines = (text or "").strip().splitlines()
    answer_count = len(spec.questions)
    expected_count = answer_count * (2 if audit else 1)
    if len(lines) != expected_count:
        return invalid

    parsed = parse_action_answer_lines(text, spec=spec)
    if any(value is None for value in parsed.values()):
        return invalid

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


def spec_metric_items(spec: PromptSpec) -> Tuple[Tuple[str, str, bool], ...]:
    """返回 (output_key, metric_key, answer)，供训练/评测统一归因。"""

    return tuple((q.output_key, q.metric_key, bool(q.answer)) for q in spec.questions)
