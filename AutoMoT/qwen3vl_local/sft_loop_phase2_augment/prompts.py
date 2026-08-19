"""Phase2 augment 的随机 ROAD_STRUCTURE 问法、目标渲染与严格解析。

三类问法共用同一张 RGB history，并在一次 assistant 生成里连续回答所有问题：

1. ``all_random_order``：仍问 RS1/RS2/RS4/RS5 四题，但顺序可复现随机。
2. ``subset_random``：随机问 1/2/3 个 RS 细问题，允许全 NO；全 NO 只表示
   “被问到的题都不是”，不再暗含高速。
3. ``hierarchical_probe``：先问 HIGHWAY，再问一个组级几何问题，最后问一个
   RS 细问题；组级问题使用专门的视觉定义，避免把两个 RS prompt 简单拼接。
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from qwen3vl_local.sft_loop_phase2_augment.history_rgb import (
    DEFAULT_HISTORY_RGB_MODE,
    HISTORY_RGB_MODE_ALL4,
    history_rgb_prompt_description,
    validate_history_rgb_mode,
)


PROMPT_NAME = "sft_loop_phase2_rs_augmented_visual_v2"
ANSWER_KEYS = ("RS1", "RS2", "RS4", "RS5")
ANSWER_VALUES = ("YES", "NO")
VARIANT_WEIGHTS = {"all_random_order": 2, "subset_random": 1, "hierarchical_probe": 1}
VARIANT_ORDER = tuple(VARIANT_WEIGHTS.keys())
SUBSET_COUNTS = (1, 2, 3)


SYSTEM_PROMPT = """You are the second perception step of an autonomous-driving agent.
The input is a stitched three-camera RGB history, ordered from oldest to newest. Classify only the newest moment. Inspect the complete left/front/right scene and use older frames only to confirm road geometry, motion, visibility, or an approaching junction. Do not use scenario names, dataset labels, maps, hidden metadata, or memory. The questions ask about the driving-rule structure visible now, not about the event: an obstacle, a pedestrian, turning vehicle, rain, darkness, or braking does not by itself decide the road structure."""


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
    """一次 forward 内的增强问法。"""

    variant: str
    questions: Tuple[QuestionSpec, ...]
    seed_key: str

    @property
    def output_keys(self) -> Tuple[str, ...]:
        """返回严格输出行顺序。"""

        return tuple(q.output_key for q in self.questions)


RS_DEFINITIONS = {
    "RS1": """RS1:
Ask: "Is the ego currently on an ordinary same-direction drivable road, outside a controlling junction, opposing-lane-sharing constraint, and highway/ramp decision structure?"
YES when the usable path is an ordinary surface-road lane or ordinary same-direction roadway: lane markings or road edges continue ahead, ordinary following/lane keeping/ordinary lane change is the main rule, and there is no active junction control or need to negotiate an opposing lane. This includes local, urban, suburban, rural, parking-side, roundabout, and ordinary surface-road segments. Roadside rails, walls, trees, or darkness do not by themselves make an ordinary road non-RS1.
NO when an unseparated opposing direction governs the usable corridor, local signal/priority intersection control governs the ego path, or the road is a limited-access highway/ramp/merge/exit structure. Several same-direction high-speed lanes with continuous shoulders, medians, guardrails/barriers, grade-separated edges, or another separated carriageway are a controlled high-speed corridor and are RS1=NO even when this four-frame window shows no merge arrow, exit sign, or gore. Do not call a road RS1 merely because it is straight, empty, wide, rainy, foggy, or has a lead vehicle. A roundabout is RS1 unless a separate signal/priority junction visibly governs the ego path.""",
    "RS2": """RS2:
Ask: "Does the opposing direction currently constrain the ego's usable corridor, so the ego may need to share, borrow, yield around, or reason about the oncoming lane?"
YES needs visible road-layout evidence, not a scenario name or a nearby parked car. Strong evidence is a narrow two-way corridor with a centre line and little physical separation, an oncoming vehicle/vehicle front or headlight line occupying the adjacent opposing space immediately ahead, or a fixed obstruction/parked vehicles/door leaving only a passage that requires entering or yielding to the opposite lane. A narrow street with recurring opposing headlights/vehicle fronts across its centre line and parked vehicles at the curb is RS2 even when ego has not yet crossed the line. Four nominal lanes can still be RS2 when parked vehicles or obstacles remove the side lanes so that the remaining usable space is effectively one lane each way.
NO for a normal same-direction multi-lane road, ordinary traffic in a separate opposing carriageway behind a median/barrier, a curbside parked vehicle that leaves ego's lane open, or a road where the opposite lane is visible but clearly does not constrain ego's current usable space. A double-yellow line, fog, parked cars beside a fully open ego lane, or a vehicle merely turning/crossing ahead is not enough without an immediate opposing-lane sharing or yielding decision.""",
    "RS4": """RS4:
Ask: "Is the ego in the approach, stop line, conflict area, or immediate exit of an intersection where traffic-signal hardware is the governing right-of-way rule?"
YES when the current visible junction has readable signal heads/masts/overhead signal arms controlling the ego approach or the intersecting approaches, together with local junction geometry such as a cross street, T-junction, turning pocket, stop line, crosswalk, median opening, or crossing traffic. A signal head/arm plus a local stop line or crosswalk is sufficient even when rain or fog partly hides the cross street. Actively inspect the upper front view and both side views: the physical signal head/arm can be small, high, partly masked, or on the far side. A failed, dark, flashing, or contradictory signal system is still RS4 when visible traffic-signal hardware is the rule source at this junction; Phase1 handles whether it is abnormal.
NO for a distant tiny signal that does not govern the current junction, a traffic light seen down another road, a normal road with a far signal but no local junction geometry, a pedestrian signal only, a signalized junction already clearly left behind, or a vehicle running a normal red light. Do not infer RS4 from a scenario name, a light reflection, a green/red pixel, or an at-grade road alone.""",
    "RS5": """RS5:
Ask: "Is the ego in the approach, conflict area, or immediate exit of an intersection whose rule is priority, stop/yield, gap acceptance, or geometry rather than a working traffic signal?"
YES when a cross street, T-junction, angled junction, side-road mouth, curb break into another road, stop/yield control, STOP marking, visible priority conflict, crossing traffic, or clear junction box is local to ego and no working traffic-signal rule governs that conflict. A readable STOP or yield sign beside the ego approach is a strong RS5 cue when the road visibly meets/crosses another road, even in fog. T-shaped intersections and unsignalized local turning conflicts count when the road connection itself is visible. The vehicle may be turning or going straight; the deciding evidence is the local no-signal/priority junction structure.
NO for ordinary road bends, driveways, parking exits, a far side street with no immediate conflict, a roundabout, or a junction visibly governed by working traffic-signal hardware. Do not turn RS5 into a catch-all for any turn, any braking, any pedestrian/vehicle event, any crosswalk, or a missing/too-small signal in fog.""",
}


GROUP_DEFINITIONS = {
    "PLAIN_LANE_FOLLOWING_CORRIDOR": (
        "PLAIN_LANE_FOLLOWING_CORRIDOR",
        "Is the ego's current decision mainly ordinary same-direction lane following on a surface road?",
        """PLAIN_LANE_FOLLOWING_CORRIDOR:
YES when the newest frame is best explained as a plain surface-road corridor: ego follows an ordinary same-direction lane or same-direction roadway, lane/curb/road-edge evidence continues ahead, and no local junction rule or opposing-lane sharing constraint governs the immediate path.
NO when the visible road is a highway/ramp/merge/exit/controlled high-speed corridor, a narrow or blocked two-way corridor where the opposing lane constrains ego, or a local junction approach/conflict/exit governed by signal, stop/yield, priority, gap acceptance, or intersection geometry.""",
        {"R1"},
    ),
    "OPEN_SURFACE_PATH": (
        "OPEN_SURFACE_PATH",
        "Is the ego's path mainly an open surface-road corridor rather than a local junction conflict area?",
        """OPEN_SURFACE_PATH:
YES when the useful decision is to trace a continuing surface-road corridor ahead: ordinary same-direction road, ordinary lane keeping/lane change, or a narrow two-way road where the corridor continues and opposing traffic may matter. It may contain parked cars, shoulders, sidewalks, or rural road edges, but the newest moment is not governed by a local intersection box.
NO when the current local structure is an intersection approach/conflict/exit, whether controlled by traffic lights or by priority/stop/yield/gap acceptance. Also NO for highway/ramp/connector topology.""",
        {"R1", "R2"},
    ),
    "JUNCTION_CONTROL_ZONE": (
        "JUNCTION_CONTROL_ZONE",
        "Is the ego currently in a local intersection control zone, regardless of whether the rule source is a traffic signal or priority/stop/yield?",
        """JUNCTION_CONTROL_ZONE:
YES when the newest frame is governed by a local intersection approach, stop line, crosswalk, conflict area, immediate exit, crossing street, T-junction, angled junction, median opening, or side-road conflict. This asks whether a junction control zone is local to ego, not whether a particular signal is normal or abnormal.
NO for a continuing non-junction road, roundabout without a separate visible signal/priority junction, highway/ramp/connector, or a far side road that does not create an immediate conflict.""",
        {"R4", "R5"},
    ),
    "LOCAL_RIGHT_OF_WAY_RULE": (
        "LOCAL_RIGHT_OF_WAY_RULE",
        "Is the ego currently governed by a local right-of-way rule rather than simple corridor following?",
        """LOCAL_RIGHT_OF_WAY_RULE:
YES when ego is in a local junction rule zone: traffic-signal hardware, stop/yield signs, STOP text, stop lines, crosswalks tied to a junction, or visible no-signal priority/gap-acceptance geometry such as a cross street, T-junction, angled junction, side-road conflict, median opening, or crossing traffic that can govern ego now.
NO for ordinary same-direction road continuation, highway/ramp/merge/exit topology, or a narrow opposing-lane constraint where the immediate issue is shared corridor width rather than a local intersection right-of-way rule.""",
        {"R4", "R5"},
    ),
    "CONSTRAINED_SHARED_SPACE": (
        "CONSTRAINED_SHARED_SPACE",
        "Does ego need to reason about shared or conflicting space with another direction of travel now?",
        """CONSTRAINED_SHARED_SPACE:
YES when ego's usable space is constrained by an oncoming/opposing direction or by a local junction conflict where another approach can cross or enter the ego path. Evidence can be a narrow two-way passage, relevant oncoming lane, cross street, T-junction, stop/yield conflict, or signalized intersection box.
NO for ordinary same-direction surface-road continuation and for highway/ramp controlled corridors where directions are separated and the immediate task is not a local cross/opposing conflict.""",
        {"R2", "R4", "R5"},
    ),
}


def _stable_rng(*parts: object) -> random.Random:
    """从任意字段构造可复现 RNG。"""

    payload = ":".join(str(p) for p in parts)
    return random.Random(hashlib.sha256(payload.encode("utf-8")).hexdigest())


def _rs_from_answers(answers: Mapping[str, bool]) -> str:
    """由四问 one-hot / all-no 标签恢复 R1-R5，all-no 表示 R3。"""

    positive = [key for key in ANSWER_KEYS if bool(answers.get(key, False))]
    if len(positive) == 1:
        return positive[0].replace("RS", "R")
    if not positive:
        return "R3"
    return "MULTI"


def _answer_for_metric(metric_key: str, answers: Mapping[str, bool]) -> bool:
    """计算一个问题在当前标签上的 YES/NO 真值。"""

    if metric_key in ANSWER_KEYS:
        return bool(answers.get(metric_key, False))
    rs = _rs_from_answers(answers)
    if metric_key == "HIGHWAY":
        return rs == "R3"
    if metric_key.startswith("GROUP:"):
        group_id = metric_key.split(":", 1)[1]
        return rs in GROUP_DEFINITIONS[group_id][3]
    raise KeyError(metric_key)


def _question(output_key: str, metric_key: str, question_id: str, question: str, answers: Mapping[str, bool]) -> QuestionSpec:
    """构造单题 spec。"""

    return QuestionSpec(
        output_key=output_key,
        metric_key=metric_key,
        question_id=question_id,
        question=question,
        answer=_answer_for_metric(metric_key, answers),
    )


def _rs_question(key: str, answers: Mapping[str, bool]) -> QuestionSpec:
    """构造 RS 细问题。"""

    question = {
        "RS1": "Is the ego currently on an ordinary same-direction surface-road corridor?",
        "RS2": "Does the opposing direction currently constrain ego's usable corridor?",
        "RS4": "Is ego governed by traffic-signal hardware at a local intersection now?",
        "RS5": "Is ego governed by a no-signal priority/stop/yield/gap-acceptance intersection now?",
    }[key]
    return _question(key, key, key, question, answers)


def _sample_subset(keys: Sequence[str], *, focus: str, count: int, rng: random.Random) -> Tuple[str, ...]:
    """抽一个包含 focus 的子集，并随机排列输出顺序。"""

    others = [key for key in keys if key != focus]
    rng.shuffle(others)
    selected = [focus] + others[: max(0, int(count) - 1)]
    rng.shuffle(selected)
    return tuple(selected)


def make_prompt_spec(
    *,
    variant: str,
    answers: Mapping[str, bool],
    seed_key: str,
    focus: str = "RS1",
    subset_count: int = 1,
    group_id: str = "JUNCTION_CONTROL_ZONE",
    detail_key: str = "RS4",
) -> PromptSpec:
    """按指定增强类型构造一次 forward 的问题列表。"""

    variant = str(variant)
    if variant not in VARIANT_WEIGHTS:
        raise ValueError(f"unknown augment variant: {variant}")
    rng = _stable_rng("phase2_augment_spec", seed_key, variant, focus, subset_count, group_id, detail_key)
    if variant == "all_random_order":
        order = list(ANSWER_KEYS)
        rng.shuffle(order)
        questions = tuple(_rs_question(key, answers) for key in order)
    elif variant == "subset_random":
        count = int(subset_count)
        if count not in SUBSET_COUNTS:
            raise ValueError(f"subset_count must be 1/2/3, got {subset_count}")
        order = _sample_subset(ANSWER_KEYS, focus=str(focus), count=count, rng=rng)
        questions = tuple(_rs_question(key, answers) for key in order)
    else:
        if group_id not in GROUP_DEFINITIONS:
            raise ValueError(f"unknown group_id: {group_id}")
        if detail_key not in ANSWER_KEYS:
            raise ValueError(f"unknown detail_key: {detail_key}")
        group = GROUP_DEFINITIONS[group_id]
        questions = (
            _question(
                "HIGHWAY",
                "HIGHWAY",
                "HIGHWAY",
                "Is the ego path currently a limited-access highway, ramp, connector, merge, split, or exit structure?",
                answers,
            ),
            _question("GROUP", f"GROUP:{group_id}", group_id, group[1], answers),
            _question("DETAIL", detail_key, detail_key, _rs_question(detail_key, answers).question, answers),
        )
    return PromptSpec(variant=variant, questions=questions, seed_key=str(seed_key))


def prompt_spec_to_json(spec: PromptSpec) -> Dict[str, object]:
    """把 PromptSpec 写入 case/audit JSON。"""

    return {
        "variant": spec.variant,
        "seed_key": spec.seed_key,
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


def _definition_ids(spec: PromptSpec) -> Tuple[str, ...]:
    """返回本 prompt 需要展示的定义块 id。"""

    ids: List[str] = []
    for q in spec.questions:
        if q.question_id in ANSWER_KEYS and q.question_id not in ids:
            ids.append(q.question_id)
        if q.metric_key.startswith("GROUP:"):
            group_id = q.metric_key.split(":", 1)[1]
            if group_id not in ids:
                ids.append(group_id)
    return tuple(ids)


def build_phase2_prompt(
    *,
    spec: Optional[PromptSpec] = None,
    audit: bool = False,
    history_rgb_mode: str = DEFAULT_HISTORY_RGB_MODE,
) -> str:
    """Build one augmented Phase2 prompt; audit asks for visible evidence."""

    mode = validate_history_rgb_mode(history_rgb_mode)
    history = history_rgb_prompt_description(mode)
    endpoint_notice = "" if mode == HISTORY_RGB_MODE_ALL4 else " Only the first and fourth frames of the original four-frame history are visible; do not assume intermediate evidence exists."
    audit_notice = (
        " In audit mode, write all requested answer lines first, then one short evidence line per answer. "
        "Do not list many absent features, do not repeat phrases, and do not continue after the requested lines."
        if audit
        else ""
    )
    if spec is None:
        spec = make_prompt_spec(
            variant="all_random_order",
            answers={key: False for key in ANSWER_KEYS},
            seed_key="default",
            focus="RS1",
        )
    definition_blocks: List[str] = []
    for item in _definition_ids(spec):
        if item in RS_DEFINITIONS:
            definition_blocks.append(RS_DEFINITIONS[item])
        elif item in GROUP_DEFINITIONS:
            definition_blocks.append(GROUP_DEFINITIONS[item][2])
    question_lines = "\n".join(
        f"{idx}. {q.output_key}: {q.question}" for idx, q in enumerate(spec.questions, start=1)
    )
    if audit:
        answer_lines = "\n".join(f"{q.output_key}: <YES or NO>" for q in spec.questions)
        evidence_lines = "\n".join(
            f"EVIDENCE_{q.output_key}: <one short RGB cue; max 12 words>" for q in spec.questions
        )
        output_lines = f"{answer_lines}\n{evidence_lines}"
    else:
        output_lines = "\n".join(f"{q.output_key}: <YES or NO>" for q in spec.questions)
    output_label = "AUDIT_OUTPUT" if audit else "OUTPUT"
    text = f"""
[PROMPT_NAME]
{PROMPT_NAME}
[/PROMPT_NAME]

[AUGMENT_VARIANT]
{spec.variant}
[/AUGMENT_VARIANT]

[VISUAL_CHECK_ORDER]
Classify the newest frame using the {history}. First trace the ego vehicle's usable corridor and lane boundaries. Then inspect whether visible limited-access highway/ramp topology, opposing-direction sharing, or local junction control governs the ego path now. Scan the full left/front/right scene for shoulders, barriers, medians, sidewalks, side roads, cross streets, stop/yield signs, signal heads/masts/arms, stop lines, crosswalks, oncoming lanes, lane splits, gore areas, merges, exits, and connectors. Answer only the questions listed in [QUESTIONS]. Each answer must be decided from that question's visible RGB evidence. Previously generated answer lines are not evidence, and must not be used to infer later answers. All NO means only that none of the asked questions is supported; it does not imply highway or any unasked road type.{endpoint_notice}{audit_notice}
[/VISUAL_CHECK_ORDER]

[QUESTIONS]
{question_lines}
[/QUESTIONS]

[DECISION_RULES]
{chr(10).join(definition_blocks)}

HIGHWAY/RAMP ROBUSTNESS:
A limited-access highway mainline, ramp, merge, split, exit, connector, gore area, or controlled high-speed corridor is a visual road type. Positive clues include several same-direction high-speed lanes with continuous guardrails/barriers/shoulders, grade separation, lane convergence/divergence, gore, exit/merge structure, or no ordinary surface-street access. Do not call it ordinary RS1 merely because lane markings continue. Do not treat another carriageway behind a barrier as an opposing-lane constraint, or a distant lamp/bridge as a local junction.

INDEPENDENT ANSWER CHECK:
For each listed question, output only the YES or NO supported by that question's RGB evidence. Do not answer unlisted RS questions. Do not fill in missing RS labels from an all-NO pattern.
[/DECISION_RULES]

[{output_label}]
Output exactly these lines and nothing else:
{output_lines}
[/{output_label}]
""".strip()
    return text


def build_phase2_target(spec: PromptSpec) -> str:
    """Render the strict target for this augmented spec."""

    return "\n".join(f"{q.output_key}: {'YES' if bool(q.answer) else 'NO'}" for q in spec.questions)


def parse_phase2_output(text: str, *, spec: PromptSpec) -> Dict[str, Optional[bool]]:
    """Strictly parse expected YES/NO values without guessing omitted lines."""

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


def phase2_prompt_sha256(
    *,
    spec: Optional[PromptSpec] = None,
    audit: bool = False,
    history_rgb_mode: str = DEFAULT_HISTORY_RGB_MODE,
) -> str:
    """Return a prompt-contract fingerprint persisted with each adapter/eval.

    ``spec=None`` deliberately hashes the whole augment prompt surface, not one
    default example. This catches edits to subset/group/hierarchical wording
    before old adapters are compared against a changed prompt contract.
    """

    if spec is not None:
        payload = SYSTEM_PROMPT + "\n" + build_phase2_prompt(spec=spec, audit=audit, history_rgb_mode=history_rgb_mode)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    dummy_answers = {key: False for key in ANSWER_KEYS}
    stable_groups = {
        key: {
            "id": value[0],
            "question": value[1],
            "definition": value[2],
            "positive_rs": sorted(value[3]),
        }
        for key, value in GROUP_DEFINITIONS.items()
    }
    stable_contract = {
        "prompt_name": PROMPT_NAME,
        "variant_order": list(VARIANT_ORDER),
        "variant_weights": dict(VARIANT_WEIGHTS),
        "subset_counts": list(SUBSET_COUNTS),
        "system_prompt": SYSTEM_PROMPT,
        "rs_definitions": dict(RS_DEFINITIONS),
        "group_definitions": stable_groups,
    }
    parts = [
        json.dumps(stable_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    ]
    for perm in itertools.permutations(ANSWER_KEYS):
        questions = tuple(_rs_question(key, dummy_answers) for key in perm)
        all_spec = PromptSpec(variant="all_random_order", questions=questions, seed_key="fingerprint")
        parts.append(build_phase2_prompt(spec=all_spec, audit=audit, history_rgb_mode=history_rgb_mode))
    for count in SUBSET_COUNTS:
        for subset in itertools.permutations(ANSWER_KEYS, count):
            questions = tuple(_rs_question(key, dummy_answers) for key in subset)
            subset_spec = PromptSpec(variant="subset_random", questions=questions, seed_key="fingerprint")
            parts.append(build_phase2_prompt(spec=subset_spec, audit=audit, history_rgb_mode=history_rgb_mode))
    for group_id in GROUP_DEFINITIONS:
        for detail_key in ANSWER_KEYS:
            hier_spec = make_prompt_spec(
                variant="hierarchical_probe",
                answers=dummy_answers,
                seed_key="fingerprint",
                group_id=group_id,
                detail_key=detail_key,
            )
            parts.append(build_phase2_prompt(spec=hier_spec, audit=audit, history_rgb_mode=history_rgb_mode))
    payload = "\n\0\n".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
