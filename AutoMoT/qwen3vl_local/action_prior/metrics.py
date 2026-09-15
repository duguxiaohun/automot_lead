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
            "summary_disabled"
            if audit.get("analysis_acceptance") == "disabled"
            else "summary_fallback"
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
    groups.extend(event_sample_groups(sample))
    return groups


def grouped_counts(audit, sample, metrics):
    """保存和各组样本数配套的 loss/ADE/FDE 累加值，兼容全 rank Counter 汇总。"""
    counts = Counter()
    for group in sample_groups(audit, sample):
        counts[f"group/{group}/samples"] += 1
        for key, value in metrics.items():
            counts[f"group/{group}/{key}"] += value
    return counts


def event_sample_groups(sample):
    """只按 full map 分组；消融不生成模型先验、UNKNOWN 或复核指标。"""
    groups = []
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


def event_grouped_counts(sample, metrics):
    """训练用实际采样桶，验证用完整语义桶，与主线共用分母口径。"""
    counts = Counter()
    for group in event_sample_groups(sample):
        counts[f"group/{group}/samples"] += 1
        for key, value in metrics.items():
            counts[f"group/{group}/{key}"] += value
    return counts


def metrics_from_counts(counts):
    """并列报告计数与样本均值，字段 invalid 次数可大于样本数。"""
    n = counts["samples"]
    if not n:
        raise ValueError("no successfully evaluated samples")
    result = {
        k: v / n
        for k, v in counts.items()
        if k != "samples" and not k.startswith("group/")
    }
    for key, value in counts.items():
        if key.startswith("group/"):
            prefix, metric = key.rsplit("/", 1)
            result[key] = (
                value if metric == "samples" else value / counts[prefix + "/samples"]
            )
    result["samples"] = n
    # 自然分布分数仍用于可比的总体报告；事件均衡分数则严格按 1:…:1:2
    # 聚合每个真实全帧桶。缺桶显式标为不完整，不能悄悄以总体 ADE 代替。
    from qwen3vl_local.action_prior.event_balance import (
        EVENT_BALANCE_WEIGHTS,
        REGULAR_BACKGROUND,
        SPECIAL_BUCKETS,
    )

    event_keys = (*SPECIAL_BUCKETS, REGULAR_BACKGROUND)
    has_event_map = any(key.startswith("group/event_balance/") for key in counts)
    missing = [
        key for key in event_keys
        if result.get(f"group/event_balance/{key}/samples", 0) <= 0
    ]
    if has_event_map:
        result["event_balance_bucket_coverage_complete"] = int(not missing)
        result["event_balance_bucket_coverage_missing"] = ",".join(missing)
    sampled_metrics = ("route_ade_m", "waypoint_ade_m", "route_fde_m", "waypoint_fde_m")
    has_event_sampling_metrics = has_event_map and all(
        f"group/event_balance/{bucket}/{metric}" in result
        for bucket in event_keys
        for metric in sampled_metrics
    )
    # 默认训练窗口只记录 FM MSE，不运行 Euler ODE；此时依旧报告桶覆盖，
    # 但绝不能试图从不存在的 ADE/FDE 组指标构造事件均衡分数。
    if has_event_sampling_metrics and not missing:
        weight_sum = sum(EVENT_BALANCE_WEIGHTS[key] for key in event_keys)
        for metric in sampled_metrics:
            result[f"event_balanced_{metric}"] = sum(
                EVENT_BALANCE_WEIGHTS[key]
                * result[f"group/event_balance/{key}/{metric}"]
                for key in event_keys
            ) / weight_sum
        result["event_balanced_trajectory_score"] = (
            result["event_balanced_route_ade_m"]
            + result["event_balanced_waypoint_ade_m"]
        ) / 2.0
    result.update(
        {
            f"count/{k}": v
            for k, v in counts.items()
            if k.startswith(("prior/", "recheck/"))
        }
    )
    return result
