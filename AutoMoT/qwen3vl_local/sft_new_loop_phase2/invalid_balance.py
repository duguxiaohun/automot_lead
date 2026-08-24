"""INVALID 样本签名解析、分层抽样与审计统计。

数据构建器把跨问题域 invalid 的来源写成稳定签名：
``source=<class>|true_rs=<R*>|question_domain=<wrong-domain>``。训练与评测必须
继续保留这三个维度，不能把所有 INVALID 行退化成一个不可审计的大桶。
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple, TypeVar


_SIGNATURE_KEYS = ("source", "true_rs", "question_domain")
REQUIRED_SOURCE_CLASSES = ("UE1", "UE3", "UE5", "UE6", "RE")
REQUIRED_TRUE_RS = ("R1", "R2", "R3", "R4", "R5")
REQUIRED_WRONG_QUESTION_DOMAINS = ("ROAD_CORRIDOR", "LOCAL_JUNCTION")
_T = TypeVar("_T")


@dataclass(frozen=True)
class InvalidSignature:
    """一条 invalid 行的规范化来源签名。"""

    source_class: str
    true_rs: str
    wrong_question_domain: str

    @property
    def canonical(self) -> str:
        """返回与 frame index 相同的稳定字符串格式。"""

        return (
            f"source={self.source_class}|true_rs={self.true_rs}|"
            f"question_domain={self.wrong_question_domain}"
        )


def parse_invalid_source(
    raw: str,
    *,
    row_true_rs: str | None = None,
    row_question_domain: str | None = None,
) -> InvalidSignature:
    """严格解析签名，并反查它与当前行的 RS/问题域是否一致。"""

    parts = str(raw).split("|") if str(raw) else []
    parsed: Dict[str, str] = {}
    for part in parts:
        key, separator, value = part.partition("=")
        if not separator or key not in _SIGNATURE_KEYS or not value or key in parsed:
            raise ValueError(f"invalid invalid_source signature: {raw!r}")
        parsed[key] = value
    if tuple(parsed.keys()) != _SIGNATURE_KEYS:
        raise ValueError(
            "invalid_source must use exact ordered fields "
            f"source|true_rs|question_domain, got {raw!r}"
        )
    signature = InvalidSignature(
        source_class=parsed["source"],
        true_rs=parsed["true_rs"],
        wrong_question_domain=parsed["question_domain"],
    )
    if row_true_rs is not None and signature.true_rs != str(row_true_rs):
        raise ValueError(
            f"invalid_source true_rs mismatch: signature={signature.true_rs!r} row={row_true_rs!r}"
        )
    if row_question_domain is not None and signature.wrong_question_domain != str(row_question_domain):
        raise ValueError(
            "invalid_source question_domain mismatch: "
            f"signature={signature.wrong_question_domain!r} row={row_question_domain!r}"
        )
    return signature


def signature_for_row(row: Any) -> InvalidSignature:
    """从 train/eval 的同构 FrameRow 读取并校验 invalid 签名。"""

    return parse_invalid_source(
        str(getattr(row, "invalid_source", "")),
        row_true_rs=str(getattr(row, "true_rs")),
        row_question_domain=str(getattr(row, "question_domain")),
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


def balanced_invalid_items(items: Sequence[_T], *, target: int, rng: random.Random) -> List[_T]:
    """按 source class，再按该 source 内联合签名均衡抽样。

    这与数据构建阶段的轮转合同一致：source class 数量最大差 1，同一 source
    下各 ``source+true_rs+wrong-domain`` 联合签名数量最大差 1。容量不足只在原
    签名桶内循环，不会跨签名偷样本。
    """

    if int(target) < 0:
        raise ValueError(f"INVALID target must be non-negative, got {target}")
    by_source: Dict[str, Dict[str, List[_T]]] = defaultdict(lambda: defaultdict(list))
    observed_true_rs: set[str] = set()
    observed_domains: set[str] = set()
    for item in items:
        signature = signature_for_row(_row_of(item))
        by_source[signature.source_class][signature.canonical].append(item)
        observed_true_rs.add(signature.true_rs)
        observed_domains.add(signature.wrong_question_domain)
    missing_sources = [key for key in REQUIRED_SOURCE_CLASSES if key not in by_source]
    missing_true_rs = [key for key in REQUIRED_TRUE_RS if key not in observed_true_rs]
    missing_domains = [key for key in REQUIRED_WRONG_QUESTION_DOMAINS if key not in observed_domains]
    if missing_sources or missing_true_rs or missing_domains:
        raise ValueError(
            "INVALID balance requires complete source/true-RS/wrong-domain coverage; "
            f"missing_source_classes={missing_sources} missing_true_rs={missing_true_rs} "
            f"missing_wrong_question_domains={missing_domains}"
        )
    if int(target) == 0:
        # 全量 eval 保留每条原始 INVALID 行，但仍必须执行上面的签名一致性和覆盖守卫。
        return list(items)
    source_quotas = _even_quotas(tuple(by_source), int(target), rng)
    sampled: List[_T] = []
    for source_class in sorted(by_source):
        signature_buckets = by_source[source_class]
        signature_quotas = _even_quotas(tuple(signature_buckets), source_quotas[source_class], rng)
        for signature in sorted(signature_buckets):
            sampled.extend(
                _cycle_sample(signature_buckets[signature], signature_quotas[signature], rng)
            )
    rng.shuffle(sampled)
    if len(sampled) != int(target):
        raise AssertionError(f"INVALID sampler built {len(sampled)} rows, expected {target}")
    report = invalid_subgroup_report(sampled)
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
    """统计 source、true RS、错误问题域和三者联合签名。"""

    source_counts: Counter[str] = Counter()
    true_rs_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    signature_counts: Counter[str] = Counter()
    per_source_signature_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for item in items:
        row = _row_of(item)
        raw = str(getattr(row, "invalid_source", ""))
        if not raw:
            continue
        signature = signature_for_row(row)
        source_counts[signature.source_class] += 1
        true_rs_counts[signature.true_rs] += 1
        domain_counts[signature.wrong_question_domain] += 1
        signature_counts[signature.canonical] += 1
        per_source_signature_counts[signature.source_class][signature.canonical] += 1
    per_source = {
        source: _count_report(counter)
        for source, counter in sorted(per_source_signature_counts.items())
    }
    source_report = _count_report(source_counts)
    return {
        "total": int(sum(source_counts.values())),
        "source_class": source_report,
        "true_rs": _count_report(true_rs_counts),
        "wrong_question_domain": _count_report(domain_counts),
        "joint_signature": _count_report(signature_counts),
        "joint_signature_within_source": per_source,
        "guards": {
            "required_source_classes_present": all(
                key in source_counts for key in REQUIRED_SOURCE_CLASSES
            ),
            "required_true_rs_present": all(key in true_rs_counts for key in REQUIRED_TRUE_RS),
            "required_wrong_question_domains_present": all(
                key in domain_counts for key in REQUIRED_WRONG_QUESTION_DOMAINS
            ),
            "signatures_parsed_and_match_row_fields": True,
            "source_class_max_deviation_le_1": source_report["max_min_deviation"] <= 1,
            "joint_signature_within_source_max_deviation_le_1": all(
                report["max_min_deviation"] <= 1 for report in per_source.values()
            ),
        },
    }


def invalid_subgroup_keys(row: Any) -> Tuple[Tuple[str, str], ...]:
    """返回用于 accuracy counter 的四个 invalid 维度键。"""

    signature = signature_for_row(row)
    return (
        ("source_class", signature.source_class),
        ("true_rs", signature.true_rs),
        ("wrong_question_domain", signature.wrong_question_domain),
        ("joint_signature", signature.canonical),
    )
