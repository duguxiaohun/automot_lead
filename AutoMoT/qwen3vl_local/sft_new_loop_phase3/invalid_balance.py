"""Phase3 的 INVALID 样本签名解析、分层抽样与审计统计。

数据构建器把“上下文前提与可见布局明确不相容”的 invalid 来源写成稳定签名：
``source=<context>|true_rs=<R*>|asked_context=<wrong-context>``。
训练与评测必须继续保留这三个维度，不能把所有 INVALID 行退化成不可审计的大桶。

哪些错误上下文可以安全构造，由几何硬约束决定，不靠场景名：

* 真实 R1/R2 且不在路口、且距下一个路口足够远 -> 可以问局部路口冲突、信号失效、匝道合流；
* 真实 R3 且不在路口 -> 可以问局部路口冲突、信号失效；
* 真实 R4/R5 -> 可以问匝道合流 / 驶出。
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple, TypeVar

from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS, CONTEXT_BY_ID


_SIGNATURE_KEYS = ("source", "true_rs", "asked_context")
REQUIRED_SOURCE_CLASSES: Tuple[str, ...] = CONTEXT_IDS
REQUIRED_TRUE_RS: Tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5")
REQUIRED_WRONG_CONTEXTS: Tuple[str, ...] = CONTEXT_IDS
NO_JUNCTION_MIN_DISTANCE_M = 25.0
_T = TypeVar("_T")


def mismatched_contexts(
    *, true_rs: str, is_junction: bool, distance_to_next_junction: float
) -> Tuple[str, ...]:
    """返回该帧可以安全构造成 INVALID 的错误上下文。"""

    if bool(is_junction):
        return ("RAMP_MERGE_EXIT",) if str(true_rs) in ("R4", "R5") else ()
    if str(true_rs) in ("R1", "R2"):
        if float(distance_to_next_junction) < NO_JUNCTION_MIN_DISTANCE_M:
            return ()
        return ("JUNCTION_RULE_CONFLICT", "SIGNAL_FAILURE", "RAMP_MERGE_EXIT")
    if str(true_rs) == "R3":
        return ("JUNCTION_RULE_CONFLICT", "SIGNAL_FAILURE")
    if str(true_rs) in ("R4", "R5"):
        return ("RAMP_MERGE_EXIT",)
    return ()


def mismatched_road_contexts(*, true_rs: str, is_junction: bool,
                            distance_to_next_junction: float) -> Tuple[Tuple[str, str], ...]:
    """为每个 asked context 提供可审计的错 RS 前提，不把未标注事件当作不存在。

    只交换明确相斥的道路空间；R1/R2 远离路口时才可假称局部路口。
    返回 (asked_context, prompt_rs)，后者必须实际写入 prompt，不能只改审计字段。
    """
    rs = str(true_rs)
    if rs in ("R1", "R2") and not is_junction and distance_to_next_junction >= 25.0:
        fake_candidates = ("R4", "R5", "R3")
    elif rs == "R3" and not is_junction:
        fake_candidates = ("R4", "R5")
    elif rs in ("R4", "R5"):
        fake_candidates = ("R3",)
    else:
        fake_candidates = ()
    return tuple((ctx, next(fake for fake in fake_candidates
                            if fake in CONTEXT_BY_ID[ctx].allowed_rs))
                 for ctx in CONTEXT_IDS
                 if any(fake in CONTEXT_BY_ID[ctx].allowed_rs for fake in fake_candidates))


@dataclass(frozen=True)
class InvalidSignature:
    """一条 invalid 行的规范化来源签名。"""

    source_class: str
    true_rs: str
    asked_context: str

    @property
    def canonical(self) -> str:
        """返回与 frame index 相同的稳定字符串格式。"""

        return f"source={self.source_class}|true_rs={self.true_rs}|asked_context={self.asked_context}"


def parse_invalid_source(
    raw: str,
    *,
    row_true_rs: str | None = None,
    row_asked_context: str | None = None,
) -> InvalidSignature:
    """严格解析签名，并反查它与当前行的 RS/上下文是否一致。"""

    parts = str(raw).split("|") if str(raw) else []
    parsed: Dict[str, str] = {}
    for part in parts:
        key, separator, value = part.partition("=")
        if not separator or key not in _SIGNATURE_KEYS or not value or key in parsed:
            raise ValueError(f"invalid invalid_source signature: {raw!r}")
        parsed[key] = value
    if tuple(parsed.keys()) != _SIGNATURE_KEYS:
        raise ValueError(
            f"invalid_source must use exact ordered fields source|true_rs|asked_context, got {raw!r}"
        )
    signature = InvalidSignature(
        source_class=parsed["source"],
        true_rs=parsed["true_rs"],
        asked_context=parsed["asked_context"],
    )
    if row_true_rs is not None and signature.true_rs != str(row_true_rs):
        raise ValueError(
            f"invalid_source true_rs mismatch: signature={signature.true_rs!r} row={row_true_rs!r}"
        )
    if row_asked_context is not None and signature.asked_context != str(row_asked_context):
        raise ValueError(
            "invalid_source asked_context mismatch: "
            f"signature={signature.asked_context!r} row={row_asked_context!r}"
        )
    return signature


def signature_for_row(row: Any) -> InvalidSignature:
    """从 train/eval 的同构 FrameRow 读取并校验 invalid 签名。"""

    return parse_invalid_source(
        str(getattr(row, "invalid_source", "")),
        row_true_rs=str(getattr(row, "true_rs")),
        row_asked_context=str(getattr(row, "context_id")),
    )


def _row_of(item: Any) -> Any:
    """同时接受 FrameRow 与包含 ``row`` 的 WorkItem。"""

    return getattr(item, "row", item)


def _even_quotas(keys: Sequence[str], target: int, rng: random.Random) -> Dict[str, int]:
    """把整数 target 在已有 keys 间分配到最大差 1。"""

    ordered = sorted(str(key) for key in keys)
    if not ordered:
        raise ValueError("cannot allocate INVALID quota without buckets")
    rng.shuffle(ordered)
    base, remainder = divmod(int(target), len(ordered))
    return {key: base + int(idx < remainder) for idx, key in enumerate(ordered)}


def _cycle_sample(items: Sequence[_T], count: int, rng: random.Random) -> List[_T]:
    """无放回优先、容量不足时稳定循环，返回精确 count 条。"""

    if count <= 0:
        return []
    bucket = list(items)
    if not bucket:
        raise ValueError("cannot sample a positive INVALID quota from an empty signature bucket")
    rng.shuffle(bucket)
    sampled = bucket[:count] if len(bucket) >= count else [bucket[idx % len(bucket)] for idx in range(count)]
    rng.shuffle(sampled)
    return sampled


def invalid_subgroup_keys(row: Any) -> Dict[str, str]:
    """返回一行 invalid 的三个可审计维度。"""

    signature = signature_for_row(_row_of(row))
    return {
        "source_class": signature.source_class,
        "true_rs": signature.true_rs,
        "asked_context": signature.asked_context,
        "joint": signature.canonical,
        "reason": getattr(_row_of(row), "invalid_reason", "") or "wrong_road_structure",
    }


def balanced_invalid_items(
    items: Sequence[_T],
    *,
    target: int,
    rng: random.Random,
    require_coverage: bool = True,
) -> List[_T]:
    """按 source class，再按该 source 内联合签名均衡抽样。"""

    if int(target) < 0:
        raise ValueError(f"INVALID target must be non-negative, got {target}")
    by_source: Dict[str, Dict[str, List[_T]]] = defaultdict(lambda: defaultdict(list))
    observed_true_rs: set[str] = set()
    observed_contexts: set[str] = set()
    for item in items:
        signature = signature_for_row(_row_of(item))
        by_source[signature.source_class][signature.canonical].append(item)
        observed_true_rs.add(signature.true_rs)
        observed_contexts.add(signature.asked_context)
    missing_true_rs = [key for key in REQUIRED_TRUE_RS if key not in observed_true_rs]
    missing_contexts = [key for key in REQUIRED_WRONG_CONTEXTS if key not in observed_contexts]
    missing_sources = [key for key in REQUIRED_SOURCE_CLASSES if key not in by_source]
    if require_coverage and (missing_true_rs or missing_contexts or missing_sources):
        raise ValueError(
            "INVALID balance requires complete true-RS/wrong-context coverage; "
            f"missing_true_rs={missing_true_rs} missing_wrong_contexts={missing_contexts} "
            f"missing_sources={missing_sources}"
        )
    if int(target) == 0:
        return list(items)
    source_quotas = _even_quotas(tuple(by_source), int(target), rng)
    sampled: List[_T] = []
    source_used: Counter[str] = Counter()
    signature_used: Counter[str] = Counter()

    # 同 RS 的人工负例不能在二次采样时静默消失；仅用已有样本，不造新负例。
    same_rs = [item for item in items if getattr(_row_of(item), 'invalid_reason', '') == 'same_rs_wrong_event']
    if same_rs:
        by_asked = defaultdict(list)
        for item in same_rs:
            by_asked[signature_for_row(_row_of(item)).asked_context].append(item)
        # 先照顾只有少量来源的题，再补有多个来源的题；已具备的同 RS 题覆盖必须保留。
        ordered = sorted(by_asked, key=lambda key: (
            len({signature_for_row(_row_of(x)).source_class for x in by_asked[key]}), key))
        for asked in ordered:
            eligible = [item for item in by_asked[asked]
                        if source_used[signature_for_row(_row_of(item)).source_class]
                        < source_quotas[signature_for_row(_row_of(item)).source_class]]
            if not eligible:
                raise ValueError(f'INVALID quota cannot retain reviewed same-RS context {asked}; increase balance count')
            rng.shuffle(eligible)
            chosen = min(eligible, key=lambda item: source_used[signature_for_row(_row_of(item)).source_class])
            signature = signature_for_row(_row_of(chosen))
            sampled.append(chosen)
            source_used[signature.source_class] += 1
            signature_used[signature.canonical] += 1

    def _seed_coverage(predicate) -> None:
        """从尚有 source 配额的候选里取一条，保证抽样结果而非候选池有覆盖。"""

        candidates: List[Tuple[str, str, List[_T]]] = []
        for source_class, signature_buckets in by_source.items():
            if source_used[source_class] >= source_quotas[source_class]:
                continue
            for signature, bucket in signature_buckets.items():
                if bucket and predicate(signature_for_row(_row_of(bucket[0]))):
                    candidates.append((source_class, signature, bucket))
        if not candidates:
            raise AssertionError("INVALID coverage candidate disappeared after quota allocation")
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: (source_used[item[0]], item[0], item[1]))
        source_class, _, bucket = candidates[0]
        sampled.extend(_cycle_sample(bucket, 1, rng))
        source_used[source_class] += 1
        signature_used[signature_for_row(_row_of(bucket[0])).canonical] += 1

    # 先满足真实 RS 与错误 context 的可观测覆盖，然后仍严格补齐 source 的等额配额。
    # 这修复了旧逻辑“候选池覆盖完整但小型 generation-eval 实际只抽到 R1”的问题。
    if require_coverage:
        if int(target) < len(REQUIRED_TRUE_RS) + len(REQUIRED_WRONG_CONTEXTS):
            raise ValueError(
                f"INVALID target={target} is too small for true-RS and wrong-context coverage"
            )
        for true_rs in REQUIRED_TRUE_RS:
            if not any(signature_for_row(_row_of(item)).true_rs == true_rs for item in sampled):
                _seed_coverage(lambda signature, value=true_rs: signature.true_rs == value)
        for asked_context in REQUIRED_WRONG_CONTEXTS:
            # 一个 true-RS coverage seed 可能已同时满足某个 asked-context；不要为了
            # 重复同一维度消耗最后一个 source quota。
            already_present = any(
                signature_for_row(_row_of(item)).asked_context == asked_context for item in sampled
            )
            if not already_present:
                _seed_coverage(lambda signature, value=asked_context: signature.asked_context == value)

    for source_class in sorted(by_source):
        signature_buckets = by_source[source_class]
        remaining = source_quotas[source_class] - source_used[source_class]
        # coverage seed 已占用联合签名配额；不能把 remaining 再从零均分，否则
        # 同一签名被先 seed、再分到 remainder，出现 3:1 并在 generation eval 崩溃。
        signature_order = sorted(signature_buckets)
        rng.shuffle(signature_order)
        additions: Counter[str] = Counter()
        for _ in range(remaining):
            signature = min(signature_order, key=lambda key: signature_used[key])
            signature_used[signature] += 1
            additions[signature] += 1
        for signature in signature_order:
            sampled.extend(_cycle_sample(signature_buckets[signature], additions[signature], rng))
    # 在不改变 source/RS/asked 联合配额的前提下，优先保留约 25% 同 RS 难负例。
    # 容量不足时报告实际数量，不重复一个 source 去伪造全类覆盖。
    same_by_signature = defaultdict(list)
    for item in same_rs:
        same_by_signature[signature_for_row(_row_of(item)).canonical].append(item)
    desired = max(1, round(target * .25)) if same_rs else 0
    used = sum(getattr(_row_of(item), 'invalid_reason', '') == 'same_rs_wrong_event' for item in sampled)
    indices = list(range(len(sampled)))
    rng.shuffle(indices)
    for index in indices:
        if used >= desired:
            break
        item = sampled[index]
        bucket = same_by_signature.get(signature_for_row(_row_of(item)).canonical, [])
        if bucket and getattr(_row_of(item), 'invalid_reason', '') != 'same_rs_wrong_event':
            sampled[index] = rng.choice(bucket)
            used += 1
    rng.shuffle(sampled)
    if len(sampled) != int(target):
        raise AssertionError(f"INVALID sampler built {len(sampled)} rows, expected {target}")
    report = invalid_subgroup_report(sampled)
    if require_coverage and (
        not report["guards"]["required_true_rs_present"]
        or not report["guards"]["required_wrong_contexts_present"]
        or not report["guards"]["required_source_classes_present"]
    ):
        raise AssertionError(f"INVALID sampled coverage regressed: {report}")
    if not report["guards"]["source_class_max_deviation_le_1"]:
        raise AssertionError(f"INVALID source-class balance regressed: {report}")
    if not report["guards"]["joint_signature_within_source_max_deviation_le_1"]:
        raise AssertionError(f"INVALID joint-signature balance regressed: {report}")
    return sampled


def _count_report(counter: Mapping[str, int]) -> Dict[str, Any]:
    """输出 counts 和最大最小偏差，供 JSON/TB 审计。"""

    counts = {str(key): int(value) for key, value in sorted(counter.items())}
    values = list(counts.values())
    return {
        "counts": counts,
        "min_count": min(values) if values else 0,
        "max_count": max(values) if values else 0,
        "max_min_deviation": (max(values) - min(values)) if values else 0,
    }


def invalid_subgroup_report(items: Sequence[Any]) -> Dict[str, Any]:
    """统计 source、true RS、错误上下文和三者联合签名。"""

    source_counts: Counter[str] = Counter()
    true_rs_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    signature_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    reason_context_counts: Counter[str] = Counter()
    per_source_signature_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for item in items:
        row = _row_of(item)
        if not str(getattr(row, "invalid_source", "")):
            continue
        signature = signature_for_row(row)
        reason = getattr(row, 'invalid_reason', '') or 'wrong_road_structure'
        reason_counts[reason] += 1
        reason_context_counts[f'{reason}/{signature.asked_context}'] += 1
        source_counts[signature.source_class] += 1
        true_rs_counts[signature.true_rs] += 1
        context_counts[signature.asked_context] += 1
        signature_counts[signature.canonical] += 1
        per_source_signature_counts[signature.source_class][signature.canonical] += 1
    per_source = {
        source: _count_report(counter) for source, counter in sorted(per_source_signature_counts.items())
    }
    source_report = _count_report(source_counts)
    return {
        "total": int(sum(source_counts.values())),
        "source_class": source_report,
        "true_rs": _count_report(true_rs_counts),
        "asked_context": _count_report(context_counts),
        "joint_signature": _count_report(signature_counts),
        "reason": _count_report(reason_counts),
        "reason_asked_context": _count_report(reason_context_counts),
        "same_rs_fraction": reason_counts['same_rs_wrong_event'] / max(1, sum(reason_counts.values())),
        "same_rs_missing_contexts": [key for key in CONTEXT_IDS if not reason_context_counts[f'same_rs_wrong_event/{key}']],
        "joint_signature_within_source": per_source,
        "guards": {
            "required_source_classes_present": all(key in source_counts for key in REQUIRED_SOURCE_CLASSES),
            "required_true_rs_present": all(key in true_rs_counts for key in REQUIRED_TRUE_RS),
            "required_wrong_contexts_present": all(
                key in context_counts for key in REQUIRED_WRONG_CONTEXTS
            ),
            "source_class_max_deviation_le_1": source_report["max_min_deviation"] <= 1,
            "joint_signature_within_source_max_deviation_le_1": all(
                item["max_min_deviation"] <= 1 for item in per_source.values()
            ),
        },
    }
