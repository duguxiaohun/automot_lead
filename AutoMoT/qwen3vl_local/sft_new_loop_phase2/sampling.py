"""Direct-EVENT 的 route-aware 采样与审计工具。

帧级 EVENT span 往往在同一 route 内产生连续多帧。直接打乱后截取会让一条长 span
在小规模 generation validation 中占据过多权重。这里按 ``(scenario, route_id)``
轮转：先从每条 route 取一帧，再取各 route 的第二帧，以此类推。

``route_diverse_sample`` 保留旧 validation 口径：不放回取完原始帧后，若目标仍
不足则循环整个结果。``route_balanced_sample`` 仅供训练使用：先完整保留所有原始
帧一次，只有额外曝光才按 route 轮转，并限制单帧重复次数。
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple, TypeVar


T = TypeVar("T")
UE3_TRAIN_SAMPLER_VERSION = "coverage_first_route_balanced_extras_v2"


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


def route_balanced_sample(
    items: Sequence[T],
    *,
    target: int,
    rng: random.Random,
    max_frame_repeat: int = 10,
) -> List[T]:
    """先守住全部原始帧覆盖，再对额外曝光做 route 均衡。

    与 ``route_diverse_sample`` 的区别发生在 ``target > len(items)`` 时：这里继续
    保留每个原始帧一次，但不会循环整份结果；只把剩余名额按 route 轮转分配。
    route 内轮转不同帧，任一帧最多出现 ``max_frame_repeat`` 次。这样既不丢掉长
    route 中逐帧 RGB 已证明有价值的时序变化，也不让额外重复继续偏向长 span。
    容量不足时直接失败，禁止静默退化为不受控的重复采样。
    """

    source = list(items)
    wanted = int(target)
    repeat_cap = int(max_frame_repeat)
    if not source or wanted <= 0:
        return source
    if repeat_cap <= 0:
        raise ValueError(f"max_frame_repeat must be positive, got {repeat_cap}")
    unique_frame_count = len({_frame_key(item) for item in source})
    capacity = unique_frame_count * repeat_cap
    if wanted > capacity:
        raise ValueError(
            "route-balanced sampling capacity is insufficient: "
            f"target={wanted} source={len(source)} unique_frames={unique_frame_count} "
            f"max_frame_repeat={repeat_cap} "
            f"capacity={capacity}"
        )

    # target 不超过原始桶时沿用不放回的 route-diverse 截取；正式 UE3 训练目标
    # 2048 大于原始 1083，因此会进入 coverage-first 分支并完整保留所有帧。
    selected = route_diverse_sample(
        source,
        target=min(wanted, len(source)),
        rng=rng,
    )
    if wanted <= len(selected):
        return selected

    frame_repeats = Counter(_frame_key(item) for item in selected)
    if max(frame_repeats.values(), default=0) > repeat_cap:
        raise ValueError(
            "raw source already exceeds max_frame_repeat for a duplicated frame identity: "
            f"max_raw_repeat={max(frame_repeats.values())} cap={repeat_cap}"
        )

    buckets: Dict[Tuple[str, str], List[T]] = defaultdict(list)
    for item in source:
        buckets[_route_key(item)].append(item)
    route_keys = sorted(buckets)
    rng.shuffle(route_keys)
    for key in route_keys:
        rng.shuffle(buckets[key])

    route_offsets = {key: 0 for key in route_keys}
    round_index = 0
    while len(selected) < wanted:
        # 每轮旋转起始 route，确保额外曝光在 route 间的差值最多为 1。
        start = round_index % len(route_keys)
        ordered_keys = route_keys[start:] + route_keys[:start]
        added = False
        for key in ordered_keys:
            bucket = buckets[key]
            offset = route_offsets[key]
            chosen_index = None
            for delta in range(len(bucket)):
                candidate = (offset + delta) % len(bucket)
                if frame_repeats[_frame_key(bucket[candidate])] < repeat_cap:
                    chosen_index = candidate
                    break
            if chosen_index is None:
                continue
            selected.append(bucket[chosen_index])
            frame_repeats[_frame_key(bucket[chosen_index])] += 1
            route_offsets[key] = (chosen_index + 1) % len(bucket)
            added = True
            if len(selected) >= wanted:
                break
        if not added:
            raise RuntimeError(
                "route-balanced sampler exhausted all capped frames before reaching target: "
                f"selected={len(selected)} target={wanted}"
            )
        round_index += 1
    return selected


def _frame_key(item: Any) -> Tuple[str, str, int, str]:
    """返回训练重复审计使用的稳定帧身份。"""

    row = getattr(item, "row", item)
    if isinstance(row, Mapping):
        return (
            str(row.get("scenario", "")),
            str(row.get("route_id", "")),
            int(row.get("frame_id", -1)),
            str(row.get("question_domain", "")),
        )
    return (
        str(row.scenario),
        str(row.route_id),
        int(row.frame_id),
        str(getattr(row, "question_domain", "")),
    )


def frame_repetition_report(items: Sequence[Any]) -> Dict[str, Any]:
    """汇总 sampled work 中同一帧的最大重复次数。"""

    counts = Counter(_frame_key(item) for item in items)
    histogram = Counter(counts.values())
    return {
        "cases": len(items),
        "unique_frames": len(counts),
        "max_frame_repeat": max(counts.values(), default=0),
        "repeat_histogram": {
            str(repeat): int(count) for repeat, count in sorted(histogram.items())
        },
    }


def route_extra_exposure_report(
    source: Sequence[Any], sampled: Sequence[Any]
) -> Dict[str, Any]:
    """审计 sampled 相对 raw source 新增的逐 route 曝光。"""

    raw_counts = Counter(_route_key(item) for item in source)
    sampled_counts = Counter(_route_key(item) for item in sampled)
    extra_counts = {
        route: int(sampled_counts[route] - raw_counts[route]) for route in raw_counts
    }
    values = list(extra_counts.values())
    return {
        "cases": sum(values),
        "unique_routes": sum(value > 0 for value in values),
        "min_extra_cases_per_route": min(values, default=0),
        "max_extra_cases_per_route": max(values, default=0),
        "max_deviation": max(values, default=0) - min(values, default=0),
        "all_nonnegative": all(value >= 0 for value in values),
    }


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
