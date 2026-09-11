"""新 Phase3 的单轮 high-level ACTION prompt、目标渲染与严格解析。

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

* 纵向预测从当前开始的动作阶段；当前持续等待优先于随后释放。反复增速再制动的歧义窗隔离；
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
    history_rgb_prompt_description,
    validate_history_rgb_mode,
)
from qwen3vl_local.sft_new_loop_phase3.navigation_goal import render_navigation_goal


PROMPT_NAME = "sft_new_loop_phase3_high_level_action_v7_compact_observed_forecast"
INVALID_KEY = "INVALID_ACTION_CONTEXT"
ANSWER_KEYS: Tuple[str, ...] = (*ACTION_KEYS, INVALID_KEY)
ANSWER_VALUES = ("YES", "NO")
VARIANT_WEIGHTS = {"all_random_order": 1}
TRAIN_VARIANT_WEIGHTS = dict(VARIANT_WEIGHTS)
VARIANT_ORDER = tuple(VARIANT_WEIGHTS.keys())
SUBSET_COUNTS: Tuple[int, ...] = ()
GROUP_DEFINITIONS: Dict[str, Tuple[str, str, str, set]] = {}

# 规则只说一次；不把安全建议混入采集行为预测，详见 20260911 RGB 审计。
SYSTEM_PROMPT = "Predict the recorded ego vehicle's next actions. Follow the requested YES/NO format."
CHOICE_SYSTEM_PROMPT = "Predict the recorded ego vehicle's next action. Output one listed high-level action only."
ACTION_OUTPUT_MODES = ("binary", "choice")
CHOICE_OUTPUT_KEY = "ACTION_CHOICE"

# 候选释义只解释动作含义；时间窗和阈值由下方共用规则限定，不暗示该帧真值。
CHOICE_ACTION_DESCRIPTIONS: Dict[str, str] = {
    "DECELERATE": "Reduce speed meaningfully without meeting the STOP condition.",
    "STOP": "Reach or remain at a sustained near-stop, including continued waiting.",
    "RESUME": "Sustain a speed increase; a previous stop is not required.",
    "LANE_CHANGE_LEFT": "Cross an ego lane boundary to the left after the newest frame, relative to ego's heading.",
    "LANE_CHANGE_RIGHT": "Cross an ego lane boundary to the right after the newest frame, relative to ego's heading.",
}

SPEED_RULES = """Speed: next 2 seconds, at most one YES.
STOP: two consecutive 4-Hz samples at or below 0.5 m/s within 1.5 seconds. Include the current sample: still waiting at the next sample counts, even if ego accelerates later. STOP takes priority.
Otherwise, use the FIRST qualifying change from current speed: a drop of at least max(1.2 m/s, 20%) means DECELERATE; a gain of that size for two consecutive samples means RESUME. An isolated gain is insufficient. If neither qualifies, all speed answers are NO. A stop beyond 1.5 seconds does not cancel DECELERATE."""

LANE_RULES = """Lane: next 3 seconds, at most one side YES.
Predict the FIRST crossing of an ego lane boundary after the newest frame: LANE_CHANGE_LEFT or LANE_CHANGE_RIGHT, relative to ego's heading. Ignore later return crossings and crossings already in the input.
Steering input, a curved lane, an in-lane pass, a connecting road without a boundary crossing, and another vehicle's lane change do not count. Speed and lane YES can coexist."""

# 只压缩已知旧索引模板；未知在线历史原样保留，不补写未观察到的绕障状态。
HISTORY_TEXT_COMPACT = {
    "Earlier ego encountered a static blockage in its normal path. Check from the visible history whether it actually left that lane and whether recovery is still pending. Waiting for a gap does not end a pending recovery state.":
        "Earlier blockage in ego's normal path; departure and recovery are not confirmed. Check RGB.",
    "No static-obstacle bypass history is asserted. Use visible lane geometry and the current navigation requirement to distinguish a target lane change from lane keeping.":
        "No bypass history is asserted.",
    "Concurrent observed condition: another vehicle is violating the junction rule and entering ego's conflict path while ego should have priority.":
        "Also reported: another vehicle enters ego's junction path against priority.",
    "Concurrent observed condition: installed traffic signals have an established malfunction, so ego cannot rely on a red or green phase and must watch every approach.":
        "Also reported: established traffic-signal malfunction.",
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
    action_output_mode: str = "binary"
    context_detail: str = ""
    goal_xy: Optional[Tuple[float, float]] = None
    current_speed_mps: Optional[float] = None

    @property
    def output_keys(self) -> Tuple[str, ...]:
        """返回严格输出行顺序。"""

        return tuple(q.output_key for q in self.questions)

    @property
    def target_output_keys(self) -> Tuple[str, ...]:
        """返回实际文本 target 的行键；choice 把多个语义标签压成一个动作词组。"""

        return (CHOICE_OUTPUT_KEY,) if self.action_output_mode == "choice" else self.output_keys


ACTION_QUESTIONS: Dict[str, str] = {
    "DECELERATE": "Will the first meaningful speed change be a reduction, without an immediate sustained near-stop?",
    "STOP": "Will ego reach or remain at a sustained near-stop in the immediate 1.5-second window?",
    "RESUME": "Will the first meaningful speed change be a sustained speed gain within two seconds?",
    "LANE_CHANGE_LEFT": "Will the FIRST lane-boundary crossing within three seconds be to ego's left?",
    "LANE_CHANGE_RIGHT": "Will the FIRST lane-boundary crossing within three seconds be to ego's right?",
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


def validate_action_output_mode(mode: str) -> str:
    """校验二值逐题或事件条件选择题输出合同。"""

    value = str(mode).strip().lower()
    if value not in ACTION_OUTPUT_MODES:
        raise ValueError(f"unknown action output mode {mode!r}; expected one of {ACTION_OUTPUT_MODES}")
    return value


def choice_options(spec: PromptSpec) -> Tuple[str, ...]:
    """返回当前 context 的随机顺序三/五个 high-level 动作词组。

    choice 不另加 ``NONE``、invalid 或组合动作：它严格等于 context_taxonomy.py
    定义的三选一/五选一。顺序仅由 case seed 稳定决定；目标动作词组本身不变。
    全 NO、invalid、多动作标签由训练/评测采样层显式排除。
    """

    keys = [question.output_key for question in spec.questions if question.output_key != INVALID_KEY]
    _stable_rng("phase3_choice_option_order", spec.seed_key, spec.context_id).shuffle(keys)
    return tuple(keys)


def choice_action_for_answers(spec: PromptSpec) -> Optional[str]:
    """返回唯一正 high-level 动作；不可表示标签返回 None，交给采样层剔除。"""

    if bool(spec.invalid_context):
        return None
    answers = spec_answers(spec)
    positive = [key for key in spec.output_keys if key != INVALID_KEY and bool(answers[key])]
    return positive[0] if len(positive) == 1 else None


def choice_rejection_reason(spec: PromptSpec) -> Optional[str]:
    """说明为何旧多标签行不能进入严格 high-level 单选数据。"""

    if bool(spec.invalid_context):
        return "invalid_context"
    positives = sum(bool(value) for key, value in spec_answers(spec).items() if key != INVALID_KEY)
    if positives == 0:
        return "no_high_level_action"
    if positives > 1:
        return "multiple_high_level_actions"
    return None


def choice_target_action(spec: PromptSpec) -> str:
    """返回唯一正 high-level 动作词组，作为 choice 的完整监督文本。"""

    action = choice_action_for_answers(spec)
    if action in choice_options(spec):
        return str(action)
    raise ValueError(
        "choice mode requires exactly one positive context action and INVALID_ACTION_CONTEXT=NO: "
        f"context={spec.context_id} answers={spec_answers(spec)}"
    )


def make_prompt_spec(
    *,
    variant: str,
    answers: Mapping[str, bool],
    seed_key: str,
    context_id: str,
    road_structure: str,
    goal_xy: Optional[Sequence[float]] = None,
    context_detail: str = "",
    current_speed_mps: Optional[float] = None,
    focus: str = "",
    subset_count: int = 1,
    group_id: str = "",
    detail_key: str = "",
    action_output_mode: str = "binary",
) -> PromptSpec:
    """按上下文构造一次 high-level 动作提问。"""

    del focus, subset_count, group_id, detail_key
    if str(variant) != "all_random_order":
        raise ValueError(f"unknown phase3 action variant: {variant}")
    output_mode = validate_action_output_mode(action_output_mode)
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
        action_output_mode=output_mode,
        goal_xy=goal,
        context_detail=str(context_detail),
        current_speed_mps=current_speed_mps,
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
        "action_output_mode": spec.action_output_mode,
        "current_speed_mps": spec.current_speed_mps,
        "goal_xy": list(spec.goal_xy) if spec.goal_xy is not None else None,
        "output_keys": list(spec.output_keys),
        "target_output_keys": list(spec.target_output_keys),
        "choice_options": list(choice_options(spec)) if spec.action_output_mode == "choice" else [],
        "choice_target_action": (
            choice_target_action(spec)
            if spec.action_output_mode == "choice" and choice_action_for_answers(spec) is not None else None
        ),
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
    """仅提供可核查事实，不给建议动作；保留未确认历史的边界。"""
    context = CONTEXT_BY_ID[spec.context_id]
    detail = HISTORY_TEXT_COMPACT.get(spec.context_detail, spec.context_detail)
    history = f"\nHistory: {detail}" if detail else ""
    return (
        "[SCENE_CONTEXT]\n"
        f"Proposed road: {ROAD_STRUCTURE_TEXT[spec.road_structure]}.\n"
        f"Situation: {context.situation_text}. {context.scope_text}"
        f"{history}\n[/SCENE_CONTEXT]"
    )


def build_action_prompt(
    *,
    spec: Optional[PromptSpec] = None,
    audit: bool = False,
    history_rgb_mode: str = DEFAULT_HISTORY_RGB_MODE,
) -> str:
    """为小型 Qwen 保留必要合同，删除重复问法、流程和无关横向说明。"""
    mode = validate_history_rgb_mode(history_rgb_mode)
    if spec is None:
        spec = make_prompt_spec(
            variant="all_random_order", answers={key: False for key in ANSWER_KEYS},
            seed_key="default", context_id="LEAD_BRAKE", road_structure="R1",
            goal_xy=(40.0, -2.0),
        )
    speed = "unknown" if spec.current_speed_mps is None else f"{spec.current_speed_mps:.3f} m/s"
    output_mode = validate_action_output_mode(spec.action_output_mode)
    if output_mode == "choice":
        options = "\n".join(
            f"- {action}: {CHOICE_ACTION_DESCRIPTIONS[action]}" for action in choice_options(spec)
        )
        lane_rule = (
            "For lane options, use only the FIRST ego lane-boundary crossing within 3 seconds; "
            "ignore crossings already in the input and later return crossings. "
            "A curve, steering, or another vehicle changing lanes is not ego lane change."
            if spec.question_domain == DOMAIN_MANEUVER else ""
        )
        return f"""RGB: {history_rgb_prompt_description(mode)}. Each image is left/front/right stitched views.
Predict actual driving, not recommended driving. Only past RGB and current state are observed.

{_scene_context_block(spec)}
Current speed: {speed}.
{render_navigation_goal(spec.goal_xy)}

Choose exactly one listed high-level action. Speed rules: predict the first qualifying change in the next 2 seconds. STOP means two consecutive 4-Hz samples at or below 0.5 m/s within 1.5 seconds, including the current sample; STOP has priority. Otherwise DECELERATE needs a drop of at least max(1.2 m/s,20%) from current speed. RESUME needs that size gain for two consecutive samples; an isolated gain is insufficient. A stop beyond 1.5 seconds does not cancel DECELERATE.
{lane_rule}

Choices:
{options}

Output exactly one listed action phrase (the name before ':'), with no description or extra text:
<ACTION_NAME>""".strip()
    lane = "\n\n" + LANE_RULES if spec.question_domain == DOMAIN_MANEUVER else ""
    output = "\n".join(f"{q.output_key}: <YES or NO>" for q in spec.questions)
    if audit:
        output += "\n" + "\n".join(f"EVIDENCE_{q.output_key}: <cue>" for q in spec.questions)
        output += "\nEach cue: 1-14 words; use 'unclear' when not observable. Never leave it blank."
    return f"""RGB: {history_rgb_prompt_description(mode)}. Each image is left/front/right stitched views.
Predict actual driving, not recommended driving. Only past RGB and current state are observed.

{_scene_context_block(spec)}
Current speed: {speed}.
{render_navigation_goal(spec.goal_xy)}

{SPEED_RULES}{lane}

INVALID_ACTION_CONTEXT: YES only if RGB clearly contradicts the proposed road or event; then all actions NO. Otherwise NO. Poor visibility, an occluded event, or no required action alone is not invalid. All actions NO is a valid prediction.

Output these lines in order, with no extra text:
{output}""".strip()


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
        {"role": "system", "content": CHOICE_SYSTEM_PROMPT if spec.action_output_mode == "choice" else SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def build_action_target(spec: PromptSpec) -> str:
    """渲染严格的 YES/NO 或选择题 target。"""

    if spec.action_output_mode == "choice":
        return choice_target_action(spec)
    return "\n".join(f"{q.output_key}: {'YES' if bool(q.answer) else 'NO'}" for q in spec.questions)


def action_prompt_sha256(
    *, audit: bool = False, history_rgb_mode: str = DEFAULT_HISTORY_RGB_MODE, action_output_mode: str = "binary"
) -> str:
    """返回完整 prompt 表面的指纹，任何措辞/顺序变化都会改变它。"""

    output_mode = validate_action_output_mode(action_output_mode)
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
    # binary 的序列化保持旧指纹，便于加载既有 YES/NO adapter；choice 自身另行绑定模式。
    if output_mode != "binary":
        parts.append(json.dumps(
            {
                "action_output_mode": output_mode,
                "choice_output_key": CHOICE_OUTPUT_KEY,
                "choice_system_prompt": CHOICE_SYSTEM_PROMPT,
                "choice_target_format": "one high-level action phrase",
                "choice_option_order": "stable seed shuffle",
            },
            sort_keys=True,
        ))
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
                    current_speed_mps=8.125,
                    action_output_mode=output_mode,
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
    if spec.action_output_mode == "choice":
        if not lines:
            return invalid
        action = lines[0]
        if action not in choice_options(spec):
            return invalid
        return {key: (key == action) if key != INVALID_KEY else False for key in invalid}
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
    answer_count = 1 if spec.action_output_mode == "choice" else len(spec.questions)
    expected_count = 1 if spec.action_output_mode == "choice" else answer_count * (2 if audit else 1)
    if len(lines) != expected_count:
        return invalid

    parsed = parse_action_answer_lines(text, spec=spec)
    if any(value is None for value in parsed.values()):
        return invalid

    if audit and spec.action_output_mode == "choice":
        # 单选输出合同严格只允许一个动作词组；audit 复用 production prompt 和 parser。
        return parsed
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
