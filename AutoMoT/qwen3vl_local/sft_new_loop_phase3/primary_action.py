"""Phase3 单选与 action_prior 共用的主要动作投影；不读取未来数据或猜测事件。"""

from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS

PRIMARY_ACTION_VERSION = "primary_action_v1_stop_then_crossing_then_speed_or_none"
NONE_ACTION = "NONE"
PRIMARY_ACTION_RULES = (
    "Primary-action priority: qualifying STOP first; otherwise a listed first lane crossing overrides "
    "accompanying speed changes; otherwise the speed action. "
    "Choose NONE if no listed action qualifies; this does not negate the scene."
)


def primary_action(answers, allowed_actions=ACTION_KEYS):
    """先投影当前问题域，再取 STOP > 首跨 > 纵向；空集为 NONE，冲突拒绝。"""
    allowed = tuple(allowed_actions)
    if any(key not in ACTION_KEYS for key in allowed):
        raise ValueError("unknown primary action domain")
    selected = {key for key in allowed if answers.get(key) is True}
    if len(selected & {"DECELERATE", "STOP", "RESUME"}) > 1 or len(
        selected & {"LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"}
    ) > 1:
        raise ValueError("conflicting primary action evidence")
    return next((key for key in ("STOP", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT",
                                 "DECELERATE", "RESUME") if key in selected), NONE_ACTION)


def primary_answers(answers, allowed_actions=ACTION_KEYS):
    """保留原字段，以互斥主要动作替换动作位；原始证据由调用者另存。"""
    chosen = primary_action(answers, allowed_actions)
    return {**answers, **{key: key == chosen for key in ACTION_KEYS}}


def count_none_prediction(counts, *, gt_none, predicted_none):
    """记录 NONE 的支持、误触发与漏检；格式错不能算预测 NONE。"""
    counts["NONE/gt_yes"] += int(gt_none)
    counts["NONE/pred_yes"] += int(predicted_none)
    counts["NONE/recall_hit"] += int(gt_none and predicted_none)
    counts["NONE/precision_hit"] += int(gt_none and predicted_none)
