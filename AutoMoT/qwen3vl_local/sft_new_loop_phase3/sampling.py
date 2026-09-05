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
