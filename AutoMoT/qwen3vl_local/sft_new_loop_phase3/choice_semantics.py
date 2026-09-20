"""Phase3 choice/binary 共用正向保持标签与动作因果；不改变 action_prior 的旧 NONE 协议。

五个原始动作位是轨迹证据，KEEP 是证据完整且当前题域没有阶段转换时的正向选择。
目的只是场景条件下的解释，不能用来证明驾驶员意图、空隙安全或未来动作。
"""
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import (
    ACTION_KEYS, CONTEXT_BY_ID, DOMAIN_MANEUVER,
)
from qwen3vl_local.sft_new_loop_phase3.primary_action import primary_action

PRIMARY_CHOICE_VERSION = "primary_choice_v3_choice_and_binary_keep"
KEEP_ACTION = "KEEP"
PRIMARY_CHOICE_RULES = (
    "Choose STOP first; otherwise the first upcoming lane crossing takes priority over "
    "its accompanying speed change; otherwise choose the speed action or KEEP. Preparatory slowing may happen before the selected crossing. "
    "KEEP does not mean the event has ended or visibility is poor."
)
LONGITUDINAL_CHOICE_RULES = (
    "Choose STOP first; otherwise choose the speed action or KEEP. "
    "KEEP does not mean the event has ended or visibility is poor."
)


def primary_choice(answers, allowed_actions=ACTION_KEYS):
    """缺失/非布尔证据和无效前提不能变成 KEEP；速度与首跨优先级沿用 v1。"""
    keys = tuple(allowed_actions)
    if not keys or any(type(answers.get(key)) is not bool for key in keys):
        raise ValueError("primary choice requires complete boolean action evidence")
    if answers.get("INVALID_ACTION_CONTEXT", False) is not False:
        raise ValueError("invalid context cannot supply a primary choice")
    selected = primary_action(answers, keys)
    return KEEP_ACTION if selected == "NONE" else selected


def binary_answers(answers, allowed_actions=ACTION_KEYS):
    """保留纵横原始正动作，按当前题域补显式 KEEP；未知和无效不能冒充保持。"""
    allowed = tuple(allowed_actions)
    invalid = answers.get("INVALID_ACTION_CONTEXT", False)
    if type(invalid) is not bool:
        raise ValueError("invalid context flag must be boolean")
    if invalid:
        return {**dict.fromkeys(ACTION_KEYS, False), KEEP_ACTION: False, "INVALID_ACTION_CONTEXT": True}
    chosen = primary_choice(answers, allowed)
    return {**{key: answers.get(key, False) if key in allowed else False for key in ACTION_KEYS},
            KEEP_ACTION: chosen == KEEP_ACTION, "INVALID_ACTION_CONTEXT": False}


def choice_annotation(labels, context_id, evidence):
    """验证完整轨迹证据后写正向标签及保持的题域；审计字段不能进入模型输入。"""
    context = CONTEXT_BY_ID[context_id]
    decision = evidence.get("longitudinal_decision", {})
    if decision.get("eligible") is not True:
        raise ValueError("primary choice requires eligible longitudinal evidence")
    maneuver = context.question_domain == DOMAIN_MANEUVER
    if maneuver and evidence.get("lateral_observation_complete") is not True:
        raise ValueError("maneuver choice requires complete lateral evidence")
    chosen = primary_choice(labels, context.action_keys)
    speed_choice = primary_action(labels, ("DECELERATE", "STOP", "RESUME"))
    if speed_choice != decision.get("action"):
        raise ValueError("primary choice disagrees with longitudinal evidence")
    if maneuver:
        direction = evidence.get("lane_change_direction") or ""
        if direction not in ("", "LEFT", "RIGHT") or any(
            labels[key] != (direction == side)
            for key, side in (("LANE_CHANGE_LEFT", "LEFT"), ("LANE_CHANGE_RIGHT", "RIGHT"))
        ):
            raise ValueError("primary choice disagrees with lateral evidence")
    return {
        "primary_action_version": PRIMARY_CHOICE_VERSION,
        "primary_action": chosen,
        "keep_scope": ("lane_and_speed" if maneuver else "speed_only") if chosen == KEEP_ACTION else None,
        "primary_action_evidence_status": "complete_in_question_domain",
    }


def count_keep_prediction(counts, *, gt_keep, predicted_keep):
    """KEEP 单独统计支持、精确率和召回，格式错误不能当作保持。"""
    counts["KEEP/gt_yes"] += int(gt_keep)
    counts["KEEP/pred_yes"] += int(predicted_keep)
    counts["KEEP/recall_hit"] += int(gt_keep and predicted_keep)
    counts["KEEP/precision_hit"] += int(gt_keep and predicted_keep)


def validate_choice_row(row):
    """读取训练/评估索引时验证原始类型和正向标签，不能先 bool() 抹掉未知。"""
    if row.get("primary_action_version") != PRIMARY_CHOICE_VERSION:
        raise ValueError("primary action label/version mismatch; rebuild index")
    answers = row.get("answers")
    if not isinstance(answers, dict) or any(
        type(answers.get(key)) is not bool for key in (*ACTION_KEYS, KEEP_ACTION, "INVALID_ACTION_CONTEXT")
    ):
        raise ValueError("index requires complete boolean action evidence")
    invalid = row.get("invalid_action_context")
    if type(invalid) is not bool or answers["INVALID_ACTION_CONTEXT"] != invalid:
        raise ValueError("inconsistent invalid context in index")
    expected_answers = binary_answers(answers, CONTEXT_BY_ID[row["context_id"]].action_keys)
    if answers != expected_answers:
        raise ValueError("inconsistent KEEP or out-of-domain action evidence in index")
    if invalid:
        expected = dict(primary_action_version=PRIMARY_CHOICE_VERSION, primary_action=None,
                        keep_scope=None, primary_action_evidence_status="invalid_context")
        if any(answers[key] for key in ACTION_KEYS):
            raise ValueError("invalid context must not supply positive actions")
    else:
        expected = choice_annotation(answers, row["context_id"], row["action_evidence"])
    if any(key not in row or row[key] != value for key, value in expected.items()):
        raise ValueError("primary choice evidence/scope mismatch; rebuild index")


# 每个动作一段连续因果句；只查公开 context，不查答案、未来速度或采集 scenario。
CONTEXT_ACTION_DESCRIPTIONS = {
    "LEAD_BRAKE": {
        "DECELERATE": "As the braking lead vehicle reduces the following gap, slow to avoid closing on it and preserve reaction space while tracking its speed.",
        "STOP": "When the lead vehicle leaves too little forward space, stop or continue waiting behind it to prevent a rear-end collision until the gap opens.",
        "RESUME": "As the lead vehicle pulls away and following space opens, gain speed to continue following while keeping separation.",
        "KEEP": "Continue the current speed stage while tracking the lead vehicle and adjusting the following gap to preserve reaction space."
    },
    "STATIC_BLOCKAGE": {
        "DECELERATE": "As the obstruction restricts the path, slow to preserve collision clearance and reaction time while assessing adjacent-lane vehicles, approaching traffic and gaps, creating time to prepare a possible bypass.",
        "STOP": "When the obstruction or passing traffic blocks progress, stop or keep waiting to avoid collision while checking passing clearance and waiting for a usable bypass gap.",
        "RESUME": "As forward space opens during or after the bypass, gain speed to use that space while maintaining clearance from the obstruction and nearby traffic.",
        "KEEP": "Continue in the current lane and speed stage while monitoring obstruction clearance and nearby traffic, including after a completed bypass lane change while the event remains active.",
        "LANE_CHANGE_LEFT": "When a usable bypass or recovery gap opens, cross the left boundary to pass the obstruction or regain the route lane, checking adjacent vehicles, oncoming traffic and passing clearance.",
        "LANE_CHANGE_RIGHT": "When a usable bypass or recovery gap opens, cross the right boundary to pass the obstruction or regain the route lane, keeping clearance from adjacent vehicles and the obstruction."
    },
    "DYNAMIC_CUTIN": {
        "DECELERATE": "As the entering vehicle reduces forward space, slow to avoid closing on its path and create room for it to establish a position ahead.",
        "STOP": "When the entering vehicle leaves insufficient room to proceed, stop or keep waiting so it can pass or settle ahead without a collision.",
        "RESUME": "As space opens behind or beyond the entering vehicle, gain speed to continue progress while adapting to its motion.",
        "KEEP": "Continue the current speed stage while adjusting separation from the entering vehicle as it moves across or becomes established ahead."
    },
    "VULNERABLE_CROSSING": {
        "DECELERATE": "As the pedestrian or cyclist approaches or occupies the travel corridor, slow to reduce collision risk and preserve passing space while judging their motion.",
        "STOP": "When the pedestrian or cyclist occupies the corridor or leaves too little passing space, stop or continue yielding to let them move through.",
        "RESUME": "As space opens around or beyond the pedestrian or cyclist, gain speed to continue progress while preserving clearance from them.",
        "KEEP": "Continue in the current lane and speed stage while monitoring clearance from the pedestrian or cyclist, including an in-lane pass or progress after they move aside.",
        "LANE_CHANGE_LEFT": "When a usable left-lane gap provides room to pass the pedestrian or cyclist or recover the route lane, cross left while preserving clearance from them and nearby vehicles.",
        "LANE_CHANGE_RIGHT": "When a usable right-lane gap provides room to pass the pedestrian or cyclist or recover the route lane, cross right while preserving clearance from them and nearby vehicles."
    },
    "ONCOMING_INVASION": {
        "DECELERATE": "As the oncoming vehicle intrudes into the corridor, slow to reduce closing speed and leave passing space while judging the overlap between its path and ego.",
        "STOP": "When the intruding vehicle leaves too little room to pass, stop or continue waiting so it can move through before ego occupies the same space.",
        "RESUME": "As the intruder moves aside or passing space opens, gain speed to continue progress with sufficient separation, even while the other vehicle remains nearby.",
        "KEEP": "Continue the current speed stage while adjusting to the intruder's approach and passing clearance; an ongoing risk response can occur before the other vehicle passes."
    },
    "JUNCTION_RULE_CONFLICT": {
        "DECELERATE": "If the crossing vehicle threatens ego's path despite ego's priority, slow to avoid a collision and gain time to judge whether their arrival paths overlap.",
        "STOP": "When the priority-violating vehicle occupies the conflict area, stop or keep waiting before its path so it can pass without a collision.",
        "RESUME": "As the crossing vehicle moves away or a usable passage opens, gain speed to proceed through the junction while tracking remaining conflicts.",
        "KEEP": "Continue the current speed stage while tracking crossing traffic and adjusting separation around the reported junction conflict."
    },
    "SIGNAL_FAILURE": {
        "DECELERATE": "Because the faulty signals cannot reliably establish who proceeds, slow to assess traffic on each approach and time entry into a usable crossing gap.",
        "STOP": "When conflicting traffic blocks passage or its priority is unresolved, stop or keep waiting to yield and assess the gap without relying on the faulty signals.",
        "RESUME": "As conflicting traffic leaves a usable passage, gain speed to proceed through the faulty-signal junction while continuing to check other approaches.",
        "KEEP": "Continue the current speed stage while assessing crossing traffic and priority under the established signal fault."
    },
    "POST_BYPASS_RETURN": {
        "DECELERATE": "As traffic or an obstruction restricts the route-lane transition, slow to preserve clearance and fit a target-lane gap while checking the departure, bypass or recovery stage.",
        "STOP": "When the forward corridor or target-lane gap is blocked, stop or keep waiting to let traffic pass and create an opportunity for the pending transition.",
        "RESUME": "As forward space opens, gain speed to make progress or match target-lane traffic in preparation for a possible transition.",
        "KEEP": "While a transition remains pending or a crossing is already complete, continue in the current lane and speed stage while monitoring traffic for the remaining route requirement.",
        "LANE_CHANGE_LEFT": "When a usable left-lane gap supports the current route requirement, cross left to begin the maneuver or recover the route lane, using visible history to identify the stage.",
        "LANE_CHANGE_RIGHT": "When a usable right-lane gap supports the current route requirement, cross right to begin the maneuver or recover the route lane, keeping clearance from traffic and any remaining obstruction."
    },
    "UNSIGNALIZED_PRIORITY": {
        "DECELERATE": "On the junction approach, slow to assess stop/yield requirements and crossing traffic, gaining time to enter without conflicting with traffic that has priority.",
        "STOP": "When a stop requirement or priority traffic prevents entry, stop or continue waiting before the conflict area until the approach requirement and traffic gap allow progress.",
        "RESUME": "As a usable crossing gap or forward space opens, gain speed to enter or continue through the junction while monitoring priority traffic, whether leaving a wait or already moving.",
        "KEEP": "Continue the current speed stage while approaching or traversing the junction and assessing priority traffic; a completed stop need not be repeated solely because the junction event continues."
    },
    "RAMP_MERGE_EXIT": {
        "DECELERATE": "As traffic limits the joining or exit gap, slow to match its motion and preserve distance to vehicles ahead while preparing the transition.",
        "STOP": "When congestion or an unavailable entry gap blocks progress, stop or keep waiting to let traffic move through and open space for the merge or exit.",
        "RESUME": "As a usable gap or forward space opens, gain speed to match traffic or progress along the merge or exit corridor, preparing a possible lane transition.",
        "KEEP": "Continue in the current merge or exit lane and speed stage while monitoring parallel traffic for a remaining transition, including before a later crossing or after one is complete.",
        "LANE_CHANGE_LEFT": "When a usable gap supports joining, merging or exiting, cross the left boundary into the required lane while matching nearby traffic and preserving separation.",
        "LANE_CHANGE_RIGHT": "When a usable gap supports joining, merging or exiting, cross the right boundary into the required lane while matching nearby traffic and preserving separation."
    }
}


def action_description(context_id, action):
    """动作目的按场景给出，保持的题域边界紧接正向描述；不暴露未来数值时间窗。"""
    text = CONTEXT_ACTION_DESCRIPTIONS[context_id][action]
    if action == KEEP_ACTION:
        scope = (" KEEP allows speed adjustments within the current stage, with no new lane crossing or distinct speed-stage change."
                 if CONTEXT_BY_ID[context_id].question_domain == DOMAIN_MANEUVER else
                 " KEEP allows speed adjustments within the current stage; this speed-only answer makes no claim about lane changes.")
        text += scope + " Brief braking can occur within this stage; KEEP does not imply that traffic risks have cleared."
    return text
