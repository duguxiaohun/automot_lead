"""Direct-EVENT 的 route-aware 采样与审计工具。

帧级 EVENT span 往往在同一 route 内产生连续多帧。直接打乱后截取会让一条长 span
在小规模 generation validation 中占据过多权重。这里按 ``(scenario, route_id)``
轮转：先从每条 route 取一帧，再取各 route 的第二帧，以此类推。

``route_diverse_sample`` 对所有类别使用同一规则：不放回取完原始帧后，若目标仍
不足则循环整个结果。UE3 不再有独立的 recall 导向重采样器。
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
        return source

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










def _route_counts_report(items: Sequence[Any]) -> Dict[str, Any]:
    """汇总一组 case 的 route 集中度。"""

    counts = Counter(_route_key(item) for item in items)
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return {
        "cases": len(items),
        "unique_routes": len(counts),
        "max_cases_per_route": max(counts.values(), default=0),
        "route_case_counts": {
            f"{scenario}/{route_id}": int(count)
            for (scenario, route_id), count in ordered
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
    report["by_balance_key"] = {
        key: _route_counts_report(grouped[key])
        for key in sorted(grouped)
    }
    return report
