"""轨迹指标按条件/缺失/简述来源分组；模型预测的事件分组不冒充 GT。"""

from collections import Counter


def sample_groups(audit, sample):
    """组可以重叠；每组各自计数，不能把多个组相加当总样本数。"""
    groups = (
        []
        if audit.get("condition_mode") == "base"
        else ["invalid" if audit.get("invalid") else "accepted"]
    )
    if audit.get("condition_mode") != "base":
        reasons = list(audit.get("invalid", {}).values())
        groups.append(
            "confirmation/unconfirmed"
            if any(r != "domain_inapplicable" for r in reasons)
            else (
                "confirmation/expected_domain_only"
                if reasons
                else "confirmation/all_confirmed"
            )
        )
    groups.append(
        "baseline"
        if audit.get("condition_mode") == "base"
        else (
            "summary_fallback"
            if audit.get("analysis_fallback")
            else "summary_model_accepted"
        )
    )
    for key, value in audit.get("conditions", {}).items():
        if key in (
            "UE1",
            "UE3",
            "UE5",
            "UE6",
            "STATIC_OBSTACLE",
            "VULNERABLE",
            "TRAFFIC_LIGHT_ABNORMAL",
            "ROAD_STRUCTURE",
        ):
            groups.append(f'condition/{key}/{value or "UNKNOWN"}')
    for phase, exposure in sample.get(
        "upstream_exposure", {"combined": "unknown"}
    ).items():
        groups.append(f"upstream/{phase}/{exposure}")
    return groups


def grouped_counts(audit, sample, metrics):
    """保存和各组样本数配套的 loss/ADE/FDE 累加值，兼容全 rank Counter 汇总。"""
    counts = Counter()
    for group in sample_groups(audit, sample):
        counts[f"group/{group}/samples"] += 1
        for key, value in metrics.items():
            counts[f"group/{group}/{key}"] += value
    return counts
