"""Phase3 的 route-aware 采样与审计工具。

动作 span 会在同一 route 内产生连续滑窗，直接 shuffle+truncate 会让单条长 span
占据过多权重。这里按 ``(scenario, route_id)`` 轮转：先从每条 route 取一帧，再取
每条 route 的第二帧，以此类推。实现与 `sft_new_loop_phase2.sampling` 同源。
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple, TypeVar


T = TypeVar("T")

SUPPORT_BALANCE_VERSION = "capacity_return_natural_cycles_v1"


def support_diagnostic(frames: int, routes: int, presentations=None) -> Dict[str, Any]:
    """只报告复查线索；100帧/10物理路线阈值不参与配额、过滤或标签。

    缺失动作不合成为 KEEP；低频也不自动撤去条件为 UNCOND。
    """
    return dict(frames=frames, physical_routes=routes,
                review_flags=(["fewer_than_100_frames"] if frames < 100 else [])
                             + (["fewer_than_10_physical_routes"] if routes < 10 else []),
                diagnostic_only=True, presentations=presentations,
                presentations_per_available_frame=(presentations / frames
                    if presentations is not None and frames else None))


def support_aware_quota(capacities: Mapping[str, int], target: int) -> Dict[str, int]:
    """先完整覆盖自然池，再对余量做容量内均衡；稀少格子不被单独循环放大。

    整个事件不足预算时才重复完整池，单格最多 ceil(target / pool_size) 轮。
    不用 val/test 支持率决定训练取舍，也不把稀少动作重标为 KEEP/UNCOND。
    """
    if target < 0 or any(type(n) is not int or n < 0 for n in capacities.values()):
        raise ValueError("support-aware quotas require nonnegative integer capacities/target")
    size = sum(capacities.values())
    if not size:
        if target:
            raise ValueError("support-aware quota has no supported samples")
        return dict.fromkeys(capacities, 0)
    cycles, remainder = divmod(target, size)
    partial = even_quota_with_capacity(capacities, remainder)
    return {key: cycles * n + partial[key] for key, n in capacities.items()}


def primary_action_distribution(signature_counts: Mapping[str, int]) -> Dict[str, Any]:
    """按主要动作投影报告计数；STOP+跨线仍是STOP，INVALID单列分母。"""
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS
    from qwen3vl_local.sft_new_loop_phase3.choice_semantics import primary_choice
    counts = Counter()
    for signature, count in signature_counts.items():
        if type(count) is not int or count < 0:
            raise ValueError("signature count must be a nonnegative integer")
        if signature == "INVALID":
            counts["INVALID"] += count
            continue
        parts = set(signature.split("+"))
        if parts in ({"NONE"}, {"KEEP"}):
            parts = set()
        if not parts <= set(ACTION_KEYS):
            raise ValueError(f"unknown action signature: {signature}")
        counts[primary_choice({key: key in parts for key in ACTION_KEYS})] += count
    total = sum(counts.values())
    invalid = counts.get("INVALID", 0)
    valid = total - invalid
    return {"projection": "STOP_then_first_crossing_then_speed_else_KEEP",
            "counts": dict(counts), "total": total, "valid": valid, "invalid": invalid,
            "stop_fraction_all": counts.get("STOP", 0) / total if total else None,
            "stop_fraction_valid": counts.get("STOP", 0) / valid if valid else None}


def _route_key(item: Any) -> Tuple[str, str]:
    """从 dict、WorkItem 或 FrameRow 读取稳定 route 身份。"""

    row = getattr(item, "row", item)
    if isinstance(row, Mapping):
        return str(row.get("scenario", "")), str(row.get("route_id", ""))
    return str(row.scenario), str(row.route_id)


def route_diverse_sample(items: Sequence[T], *, target: int, rng: random.Random) -> List[T]:
    """按 route 轮转抽样；样本不足时再均匀循环已有结果。"""

    source = list(items)
    if not source or int(target) <= 0:
        return source if int(target) != 0 else []

    buckets: Dict[Tuple[str, str], List[T]] = defaultdict(list)
    for item in source:
        buckets[_route_key(item)].append(item)
    route_keys = sorted(buckets)
    rng.shuffle(route_keys)
    for key in route_keys:
        rng.shuffle(buckets[key])

    selected: List[T] = []
    depth = 0
    wanted_without_replacement = min(int(target), len(source))
    while len(selected) < wanted_without_replacement:
        added = False
        for key in route_keys:
            bucket = buckets[key]
            if depth >= len(bucket):
                continue
            selected.append(bucket[depth])
            added = True
            if len(selected) >= wanted_without_replacement:
                break
        if not added:
            break
        depth += 1

    if int(target) > len(selected):
        base = list(selected)
        if not base:
            return []
        selected.extend(base[idx % len(base)] for idx in range(int(target) - len(selected)))
    return selected


def even_quota_with_capacity(capacities: Mapping[str, int], target: int) -> Dict[str, int]:
    """在若干子桶间尽量均分 target，并把超出容量的份额确定性地回流给其它桶。"""

    keys = sorted(str(key) for key in capacities)
    quotas: Dict[str, int] = {key: 0 for key in keys}
    remaining = int(target)
    active = [key for key in keys if int(capacities[key]) > 0]
    while remaining > 0 and active:
        share, extra = divmod(remaining, len(active))
        if share == 0:
            for idx in range(extra):
                key = active[idx % len(active)]
                if quotas[key] < int(capacities[key]):
                    quotas[key] += 1
                    remaining -= 1
            break
        progressed = False
        for key in list(active):
            room = int(capacities[key]) - quotas[key]
            take = min(room, share)
            if take > 0:
                quotas[key] += take
                remaining -= take
                progressed = True
            if quotas[key] >= int(capacities[key]):
                active.remove(key)
        if not progressed:
            break
    return quotas


def _route_counts_report(items: Sequence[Any]) -> Dict[str, Any]:
    """汇总一组 case 的 route 集中度。"""

    counts = Counter(_route_key(item) for item in items)
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return {
        "cases": len(items),
        "unique_routes": len(counts),
        "unique_scenarios": len({scenario for scenario, _ in counts}),
        "unique_towns": len({route.split("_")[0] for _, route in counts}),
        "towns": sorted({route.split("_")[0] for _, route in counts}),
        "max_cases_per_route": max(counts.values(), default=0),
        "route_case_counts": {
            f"{scenario}/{route_id}": int(count) for (scenario, route_id), count in ordered[:200]
        },
    }


def route_diversity_report(items: Sequence[Any]) -> Dict[str, Any]:
    """汇总总体及每个 balance class 的 route 集中度。"""

    report = _route_counts_report(items)
    grouped: Dict[str, List[Any]] = defaultdict(list)
    for item in items:
        key = str(item.get("balance_key", "")) if isinstance(item, Mapping) else getattr(item, "balance_key", "")
        if key:
            grouped[str(key)].append(item)
    report["by_balance_key"] = {key: _route_counts_report(grouped[key]) for key in sorted(grouped)}
    return report
