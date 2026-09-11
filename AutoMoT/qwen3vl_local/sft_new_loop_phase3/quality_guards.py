"""Phase3 自由生成能力守卫；训练和独立评测共用，无模型依赖。"""
from typing import Any, Dict, Mapping, Tuple
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS


def choice_generation_guards(
    metrics: Mapping[str, float], *, min_format_valid_rate: float, min_exact_accuracy: float = 0.50
) -> Dict[str, Any]:
    """严格单选合同的上线守卫。

    choice 已明确排除 all-NO、invalid 和多动作标签，所以不复用它们的 binary 守卫。
    仍要求整串 parser 格式、整体单选准确率，以及五个 high-level 的实际支持、精确率和召回。
    """

    values: Dict[str, float] = {
        "strict_format_valid_rate": float(metrics.get("format_valid_rate", 0.0)),
        "choice_exact_accuracy": float(metrics.get("exact_accuracy", 0.0)),
    }
    floors: Dict[str, float] = {
        "strict_format_valid_rate": float(min_format_valid_rate),
        "choice_exact_accuracy": float(min_exact_accuracy),
    }
    for action in ACTION_KEYS:
        prefix = f"action/{action.lower()}"
        values[f"{action}_support"] = float(metrics.get(prefix + "_gt_yes", 0.0))
        values[f"{action}_precision"] = float(metrics.get(prefix + "_precision", 0.0))
        values[f"{action}_recall"] = float(metrics.get(prefix + "_recall", 0.0))
        floors[f"{action}_support"] = 1.0
        floors[f"{action}_precision"] = 0.50
        floors[f"{action}_recall"] = 0.50
    passed = {key: values[key] >= floors[key] for key in values}
    return {"all_ok": all(passed.values()), "values": values, "floors": floors, "passed": passed}


def choice_generation_score(metrics: Mapping[str, float], *, min_format_valid_rate: float) -> Tuple[float, float, float, float, float]:
    """choice checkpoint 先按守卫，再按单选 exact 和覆盖比例排序。"""

    report = choice_generation_guards(metrics, min_format_valid_rate=min_format_valid_rate)
    ratios = [
        1.0 if report["floors"][key] <= 0 else report["values"][key] / report["floors"][key]
        for key in report["values"]
    ]
    passed = list(report["passed"].values())
    exact = float(metrics.get("exact_accuracy", 0.0))
    fmt = float(metrics.get("format_valid_rate", 0.0))
    if report["all_ok"]:
        return (1.0, exact, min(ratios), fmt, float(sum(passed)))
    return (0.0, float(sum(passed)) / max(1.0, float(len(passed))), min(ratios), exact, fmt)

def generation_checkpoint_guards(
    metrics: Mapping[str, float],
    *,
    min_invalid_exact: float,
    min_lane_change_recall: float,
    min_stop_recall: float,
    min_no_action_exact: float,
) -> Dict[str, Any]:
    """返回 checkpoint 多指标门槛的逐项审计结果。"""

    lane_change_recall = min(
        float(metrics.get("action/lane_change_left_recall", 0.0)),
        float(metrics.get("action/lane_change_right_recall", 0.0)),
    )
    values = {
        "invalid_exact": float(metrics.get("slice/invalid_exact", 0.0)),
        "lane_change_recall": lane_change_recall,
        "stop_recall": float(metrics.get("action/stop_recall", 0.0)),
        "no_action_context_exact": float(metrics.get("slice/no_action_exact", 0.0)),
        "valid_exact": float(metrics.get("slice/valid_exact", 0.0)),
        "same_rs_unique_routes": float(metrics.get("same_rs_unique_routes", 0.0)),
        "same_rs_exact": float(metrics.get("invalid_subgroup/reason/same_rs_wrong_event_exact", 0.0)),
    }
    floors = {
        "invalid_exact": float(min_invalid_exact),
        "lane_change_recall": float(min_lane_change_recall),
        "stop_recall": float(min_stop_recall),
        "no_action_context_exact": float(min_no_action_exact),
        "valid_exact": 0.50,
        "same_rs_unique_routes": 2.0,
        "same_rs_exact": 0.50,
    }
    # 固定最低能力策略，不由此次 test 成绩拟合；无正例/无NONE证据不能通过。
    for action in ACTION_KEYS:
        prefix = f"action/{action.lower()}"
        values[action + "_support"] = float(metrics.get(prefix + "_gt_yes", 0))
        floors[action + "_support"] = 1.0
        values[action + "_precision"] = float(metrics.get(prefix + "_precision", 0))
        floors[action + "_precision"] = 0.50
        if action in ("DECELERATE", "RESUME"):
            values[action + "_recall"] = float(metrics.get(prefix + "_recall", 0))
            floors[action + "_recall"] = 0.50
    values["no_action_support"] = float(metrics.get("slice/no_action_samples", 0))
    floors["no_action_support"] = 1.0
    passed = {key: values[key] >= floors[key] for key in values}
    return {"all_ok": all(passed.values()), "values": values, "floors": floors, "passed": passed}


def generation_checkpoint_score(
    metrics: Mapping[str, float],
    *,
    min_invalid_exact: float,
    min_lane_change_recall: float,
    min_stop_recall: float,
    min_no_action_exact: float,
) -> Tuple[float, float, float, float, float]:
    """构造自由生成 checkpoint 选优分数；达标候选再按总 exact 选优。"""

    report = generation_checkpoint_guards(
        metrics,
        min_invalid_exact=min_invalid_exact,
        min_lane_change_recall=min_lane_change_recall,
        min_stop_recall=min_stop_recall,
        min_no_action_exact=min_no_action_exact,
    )
    exact = float(metrics.get("exact_accuracy", 0.0))
    pattern_exact = float(metrics.get("pattern/pattern_exact", 0.0))
    ratios = [
        1.0 if report["floors"][key] <= 0.0 else report["values"][key] / report["floors"][key]
        for key in report["values"]
    ]
    passed = list(report["passed"].values())
    if report["all_ok"]:
        return (1.0, exact, min(ratios), pattern_exact, float(sum(passed)))
    return (0.0, float(sum(passed)) / float(len(passed)), min(ratios), exact, pattern_exact)
