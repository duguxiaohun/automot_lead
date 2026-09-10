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
    # 仅在显式接入全帧 event map 时存在。它是离线审计标签，不是模型预测事件，
    # 也不参与总样本分母。评测必须使用 all_special_buckets：被 Phase3 动作问答
    # 过滤的特殊帧仍是特殊驾驶情境，不能因此掉进“普通”或完全不统计。
    if sample.get("event_balance_bucket"):
        groups.append(f"event_balance/{sample['event_balance_bucket']}")
    elif "event_balance_status" in sample:
        status = str(sample.get("event_balance_status") or "unconfirmed")
        groups.append(f"event_balance_status/{status}")
        buckets = tuple(sample.get("event_balance_all_special_buckets") or ())
        groups.extend(f"event_balance/{bucket}" for bucket in buckets)
        if status == "confirmed_regular":
            groups.append("event_balance/REGULAR_BACKGROUND")
        contexts = set(sample.get("event_balance_scene_contexts") or ())
        # 当前 full-map builder 没有可靠 recovery_pending 证据，故不会命中；保留
        # 该审计组只为未来接入显式帧级状态，不能由 RE2 bucket 本身触发。
        if "RE2_RECOVERY_PENDING" in contexts:
            groups.append("event_balance/UE2_TO_RE2_RECOVERY")
    return groups


def grouped_counts(audit, sample, metrics):
    """保存和各组样本数配套的 loss/ADE/FDE 累加值，兼容全 rank Counter 汇总。"""
    counts = Counter()
    for group in sample_groups(audit, sample):
        counts[f"group/{group}/samples"] += 1
        for key, value in metrics.items():
            counts[f"group/{group}/{key}"] += value
    return counts
