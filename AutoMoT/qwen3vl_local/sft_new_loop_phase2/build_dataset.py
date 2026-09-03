#!/usr/bin/env python3
"""构建新 Phase2 的直接 EVENT 问答训练索引。

输入沿用 `keyframe_filter/collection_output/*_result.json` 的逐帧 RS/EVENT 标注。
构建流程会先剔除 noScenarios、异常时长 route、缺数据 route 和默认的视觉风险帧，
再按 split 做确定性均衡：UE1/UE3/UE5/UE6 正类 1:1:1:1，RE 默认等于单个
UE 桶，其中默认 25% 是 R3/highway hard negative；跨 ROAD_CORRIDOR / LOCAL_JUNCTION
问题域的 invalid 默认占 valid 主数据 20%。索引中的 RGB 路径相对 data_root 保存。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_THIS = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS.parents[2]
_PROJECT_ROOT = _THIS.parents[3]
for _path in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route  # noqa: E402
from qwen3vl_local.sft_loop_phase1.audit_matrix import _iter_routes_stream, _rgb_path  # noqa: E402
from qwen3vl_local.sft_new_loop_phase2 import DATASET_NAME  # noqa: E402
from qwen3vl_local.sft_new_loop_phase2.invalid_balance import (  # noqa: E402
    REQUIRED_SOURCE_CLASSES,
    REQUIRED_TRUE_RS,
    REQUIRED_WRONG_QUESTION_DOMAINS,
)
from qwen3vl_local.sft_new_loop_phase2.prompts import (  # noqa: E402
    ANSWER_KEYS,
    DOMAIN_ANSWER_KEYS,
    EVENT_KEYS,
    INVALID_KEY,
    JUNCTION_DOMAIN,
    ROAD_DOMAIN,
)
from qwen3vl_local.sft_new_loop_phase2.sampling import (  # noqa: E402
    route_diverse_sample,
    route_diversity_report,
)
from qwen3vl_local.sft_new_loop_phase2.visual_audit import (  # noqa: E402
    DEFAULT_COVERAGE_MANIFEST,
    frame_visual_risk,
    load_review_coverage,
)

RGB_HISTORY_COUNT = 4
VALID_RS = ("R1", "R2", "R4", "R5")
STRAIGHT_RS = ("R1", "R2")
JUNCTION_RS = ("R4", "R5")
TARGET_CLASSES = (*EVENT_KEYS, "RE")
UE_CODE_TO_CLASS = {"U-E1": "UE1", "U-E3": "UE3", "U-E5": "UE5", "U-E6": "UE6"}


def _stable_unit(value: str) -> float:
    """把字符串映射到稳定 [0,1) 浮点，用于 route-disjoint split。"""

    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)


def _split(scenario: str, route_id: str, seed: int, test_ratio: float, val_ratio: float) -> str:
    """按 route 做 deterministic train/val/test 切分。"""

    value = _stable_unit(f"{seed}:{scenario}:{route_id}")
    return "test" if value < test_ratio else "val" if value < test_ratio + val_ratio else "train"


def _rs_label(annotation: Mapping[str, Any]) -> str:
    """读取 canonical primary ROAD_STRUCTURE。"""

    return str(annotation.get("primary_road_structure") or (annotation.get("frame_rs_annotation") or {}).get("label") or "UNKNOWN")


def _event_codes(annotation: Mapping[str, Any]) -> Tuple[str, ...]:
    """读取本帧所有 EVENT code，兼容 primary_event 与 overlay events。"""

    out: List[str] = []
    frame_event = annotation.get("frame_event_annotation") or {}
    for value in frame_event.get("events") or []:
        if isinstance(value, str) and value not in out:
            out.append(value)
    unusual = frame_event.get("unusual_event")
    if isinstance(unusual, str) and unusual not in out:
        out.append(unusual)
    primary = annotation.get("primary_event")
    if isinstance(primary, str):
        for part in primary.replace("+", " ").replace(",", " ").split():
            if re_like_event(part) and part not in out:
                out.append(part)
    return tuple(out)


def re_like_event(value: str) -> bool:
    """判断字符串是否像 R-E*/U-E* 事件 code。"""

    return (value.startswith("U-E") or value.startswith("R-E")) and len(value) >= 4


def _target_class(rs: str, event_codes: Sequence[str]) -> Optional[str]:
    """把完整 EVENT taxonomy 折叠成四个 UE 正类与 RE。"""

    events = set(event_codes)
    if "U-E1" in events and rs in STRAIGHT_RS:
        return "UE1"
    if "U-E3" in events and rs in STRAIGHT_RS:
        return "UE3"
    if "U-E5" in events and rs in STRAIGHT_RS:
        return "UE5"
    if "U-E6" in events and rs in JUNCTION_RS:
        return "UE6"
    if rs in VALID_RS or rs == "R3":
        return "RE"
    return None


def _history(run_dir: pathlib.Path, frame_id: int) -> Optional[List[str]]:
    """读取四帧 left-pad RGB history 路径。"""

    paths: List[str] = []
    for idx in [max(0, frame_id - offset) for offset in reversed(range(RGB_HISTORY_COUNT))]:
        path = _rgb_path(run_dir, idx)
        if path is None:
            return None
        paths.append(str(path))
    return paths


def _relative_history_paths(paths: Sequence[str], data_root: pathlib.Path) -> List[str]:
    """把 RGB 保存成相对 data_root 的可迁移路径。"""

    root = data_root.expanduser().absolute()
    out: List[str] = []
    for raw in paths:
        path = pathlib.Path(raw).expanduser().absolute()
        try:
            out.append(path.relative_to(root).as_posix())
        except ValueError as exc:
            raise ValueError(f"RGB path {path} is not under data_root {root}") from exc
    return out


def _town(annotation: Mapping[str, Any], route: Mapping[str, Any]) -> str:
    """读取 town 字段，优先使用逐帧 evidence。"""

    return str((annotation.get("evidence") or {}).get("xml_town") or route.get("xml_town") or "UNKNOWN")


def _answers_for(target_class: str, question_domain: str, *, invalid: bool) -> Dict[str, bool]:
    """构造答案和内部问题域 one-hot；问题域字段不会写入模型 prompt。"""

    answers = {key: False for key in EVENT_KEYS}
    if target_class in EVENT_KEYS and not invalid:
        answers[target_class] = True
    answers[INVALID_KEY] = bool(invalid)
    for domain, key in DOMAIN_ANSWER_KEYS.items():
        answers[key] = domain == question_domain
    return answers


def _native_question_domain(true_rs: str) -> str:
    """把真实 RS 仅用于离线构造问题域，不把 RS 文本暴露给模型。"""

    if true_rs in (*STRAIGHT_RS, "R3"):
        return ROAD_DOMAIN
    if true_rs in JUNCTION_RS:
        return JUNCTION_DOMAIN
    return "UNKNOWN"


def _mismatched_question_domain(true_rs: str) -> str:
    """返回与图像真实大类相反的问题域，用于 invalid 增强。"""

    native = _native_question_domain(true_rs)
    if native == ROAD_DOMAIN:
        return JUNCTION_DOMAIN
    if native == JUNCTION_DOMAIN:
        return ROAD_DOMAIN
    return "UNKNOWN"


def _make_row(
    *,
    base: Mapping[str, Any],
    question_domain: str,
    target_class: str,
    invalid: bool,
    invalid_source: str = "",
) -> Dict[str, Any]:
    """从基础帧记录构造最终 JSONL row。"""

    event_codes = tuple(str(x) for x in base["event_codes"])
    return {
        "dataset_name": DATASET_NAME,
        "scenario": base["scenario"],
        "route_id": base["route_id"],
        "town": base["town"],
        "split": base["split"],
        "frame_id": int(base["frame_id"]),
        "true_rs": base["rs"],
        "question_domain": question_domain,
        "event": base["primary_event"],
        "event_codes": list(event_codes),
        "target_event_class": target_class,
        "is_regular": target_class == "RE",
        "invalid_event_context": bool(invalid),
        "invalid_source": str(invalid_source),
        "answers": _answers_for(target_class, question_domain, invalid=bool(invalid)),
        "visual_label_risk": bool(base["visual_label_risk"]),
        "visual_label_risk_reasons": list(base["visual_label_risk_reasons"]),
        "history_rgb_paths": list(base["history_rgb_paths"]),
        "latest_rgb_path": base["latest_rgb_path"],
    }


def _balanced_invalid_rows(
    bases: Sequence[Mapping[str, Any]],
    *,
    split: str,
    target: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """按 source class、true RS 和错误问题域均衡构造 invalid 样本。"""

    if target <= 0:
        return [], {"candidate_buckets": {}, "sampled_signature_counts": {}}
    candidate_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    source_to_signatures: Dict[str, List[str]] = defaultdict(list)
    for base in bases:
        source_class = str(base["target_event_class"])
        true_rs = str(base["rs"])
        question_domain = _mismatched_question_domain(true_rs)
        if question_domain == "UNKNOWN":
            continue
        signature = f"source={source_class}|true_rs={true_rs}|question_domain={question_domain}"
        if signature not in source_to_signatures[source_class]:
            source_to_signatures[source_class].append(signature)
        candidate_buckets[signature].append(
            _make_row(
                base=base,
                question_domain=question_domain,
                target_class="INVALID",
                invalid=True,
                invalid_source=signature,
            )
        )
    if not candidate_buckets:
        return [], {"candidate_buckets": {}, "sampled_signature_counts": {}}

    observed_sources = set(source_to_signatures)
    observed_true_rs = {
        part.removeprefix("true_rs=")
        for signature in candidate_buckets
        for part in signature.split("|")
        if part.startswith("true_rs=")
    }
    observed_domains = {
        part.removeprefix("question_domain=")
        for signature in candidate_buckets
        for part in signature.split("|")
        if part.startswith("question_domain=")
    }
    missing_sources = [key for key in REQUIRED_SOURCE_CLASSES if key not in observed_sources]
    missing_true_rs = [key for key in REQUIRED_TRUE_RS if key not in observed_true_rs]
    missing_domains = [key for key in REQUIRED_WRONG_QUESTION_DOMAINS if key not in observed_domains]
    if missing_sources or missing_true_rs or missing_domains:
        raise ValueError(
            f"split={split} invalid candidates lack required subgroup coverage: "
            f"missing_source_classes={missing_sources} missing_true_rs={missing_true_rs} "
            f"missing_wrong_question_domains={missing_domains}"
        )

    keys = sorted(candidate_buckets)
    preferred_sources = list(EVENT_KEYS) + ["RE"]
    source_keys = [key for key in preferred_sources if key in source_to_signatures]
    source_keys.extend(key for key in sorted(source_to_signatures) if key not in set(source_keys))
    for key in keys:
        rng.shuffle(candidate_buckets[key])
    for source in source_keys:
        source_to_signatures[source] = sorted(source_to_signatures[source])
    sampled: List[Dict[str, Any]] = []
    cursors: Counter[str] = Counter()
    source_cursors: Counter[str] = Counter()
    for idx in range(int(target)):
        source = source_keys[idx % len(source_keys)]
        signatures = source_to_signatures[source]
        key = signatures[source_cursors[source] % len(signatures)]
        source_cursors[source] += 1
        bucket = candidate_buckets[key]
        sampled.append(bucket[cursors[key] % len(bucket)])
        cursors[key] += 1
    rng.shuffle(sampled)
    return sampled, {
        "candidate_buckets": {key: len(value) for key, value in sorted(candidate_buckets.items())},
        "sampled_signature_counts": dict(Counter(str(row["invalid_source"]) for row in sampled)),
        "sampled_source_class_counts": dict(
            Counter(str(row["invalid_source"]).split("|", 1)[0].removeprefix("source=") for row in sampled)
        ),
        "sampled_true_rs_question_domain_counts": dict(
            Counter(
                "|".join(
                    part
                    for part in str(row["invalid_source"]).split("|")
                    if part.startswith("true_rs=") or part.startswith("question_domain=")
                )
                for row in sampled
            )
        ),
        "sampled_true_rs_counts": dict(Counter(str(row["true_rs"]) for row in sampled)),
        "sampled_wrong_question_domain_counts": dict(
            Counter(str(row["question_domain"]) for row in sampled)
        ),
        "coverage_guards": {
            "required_source_classes_present": not missing_sources,
            "required_true_rs_present": not missing_true_rs,
            "required_wrong_question_domains_present": not missing_domains,
        },
    }


def iter_base_frames(
    args: argparse.Namespace,
    risk_stats: Optional[Counter[str]] = None,
    observed_scenario_town_pairs: Optional[set[Tuple[str, str]]] = None,
) -> Iterable[Dict[str, Any]]:
    """流式遍历可用基础帧，并可记录本次实际扫描到的 route-level scenario/Town。"""

    collection_dir = pathlib.Path(args.collection_dir)
    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    selected = None if args.scenarios == "all" else {item.strip() for item in args.scenarios.split(",") if item.strip()}
    seen = 0
    for result_path in sorted(collection_dir.glob("*_result.json")):
        scenario = result_path.stem.removesuffix("_result")
        if scenario == "noScenarios" or (selected is not None and scenario not in selected):
            continue
        for route in _iter_routes_stream(result_path):
            route_id = str(route.get("route_id") or "")
            if not route_id or str(route.get("status")) == "data_missing_skip":
                continue
            run_dir = data_root / scenario / route_id
            abnormal, _ = is_abnormal_lead_route(run_dir, scenario)
            if abnormal or not run_dir.is_dir():
                continue
            annotations = list(route.get("annotations", []) or [])
            if annotations and observed_scenario_town_pairs is not None:
                observed_scenario_town_pairs.add((scenario, _town(annotations[0], route)))
            seen += 1
            if int(args.progress_every_routes) > 0 and seen % int(args.progress_every_routes) == 0:
                print(f"[new-phase2-build] routes={seen} last={scenario}/{route_id}", flush=True)
            if args.max_routes > 0 and seen > args.max_routes:
                return
            split = _split(scenario, route_id, int(args.split_seed), float(args.test_ratio), float(args.val_ratio))
            for ann in annotations:
                try:
                    frame_id = int(ann.get("frame_id"))
                except (TypeError, ValueError):
                    continue
                rs = _rs_label(ann)
                event_codes = _event_codes(ann)
                target_class = _target_class(rs, event_codes)
                if target_class is None:
                    continue
                risk, reasons = frame_visual_risk(ann)
                if risk and risk_stats is not None:
                    risk_stats["risk_frames_seen"] += 1
                    for reason in reasons:
                        risk_stats[f"reason/{reason}"] += 1
                if risk and not args.include_visual_risk:
                    if risk_stats is not None:
                        risk_stats["risk_frames_excluded"] += 1
                    continue
                if risk and risk_stats is not None:
                    risk_stats["risk_frames_retained"] += 1
                history_abs = _history(run_dir, frame_id)
                if history_abs is None:
                    continue
                history = _relative_history_paths(history_abs, data_root)
                yield {
                    "scenario": scenario,
                    "route_id": route_id,
                    "town": _town(ann, route),
                    "split": split,
                    "frame_id": frame_id,
                    "rs": rs,
                    "primary_event": str(ann.get("primary_event") or "UNKNOWN"),
                    "event_codes": event_codes,
                    "target_event_class": target_class,
                    "visual_label_risk": risk,
                    "visual_label_risk_reasons": reasons,
                    "history_rgb_paths": history,
                    "latest_rgb_path": history[-1],
                }


def _sample_bucket(
    bucket: Sequence[Mapping[str, Any]],
    *,
    target: int,
    rng: random.Random,
    route_diverse: bool = True,
) -> List[Mapping[str, Any]]:
    """确定性抽样；train 可按 route 轮转，val/test 保持 frozen 身份。

    EVENT span 会在同一 route 中产生连续滑窗。逐帧 RGB 审计已经确认，普通帧级
    shuffle+truncate 会让单条长 span（包括 PRE/POST 边界噪声）占据过多权重。
    这里先覆盖不同 ``(scenario, route_id)``，再取每条 route 的第二帧。
    """

    if target <= 0:
        return []
    if not bucket:
        return []
    items = list(bucket)
    if route_diverse:
        return route_diverse_sample(items, target=target, rng=rng)
    rng.shuffle(items)
    if len(items) >= target:
        return items[:target]
    repeated = [items[index % len(items)] for index in range(target)]
    rng.shuffle(repeated)
    return repeated


def _balanced_rows_by_split(args: argparse.Namespace, risk_stats: Counter[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """读取候选帧并按 split 生成均衡 rows。"""

    buckets: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    invalid_sources: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    raw_counts: Counter[str] = Counter()
    actual_scenario_town_pairs: set[Tuple[str, str]] = set()
    for base in iter_base_frames(
        args,
        risk_stats,
        observed_scenario_town_pairs=actual_scenario_town_pairs,
    ):
        split = str(base["split"])
        target_class = str(base["target_event_class"])
        actual_scenario_town_pairs.add((str(base["scenario"]), str(base["town"])))
        buckets[split][target_class].append(base)
        raw_counts[f"{split}/{target_class}"] += 1
        rs = str(base["rs"])
        if rs in VALID_RS or rs == "R3":
            invalid_sources[split].append(base)

    rows: List[Dict[str, Any]] = []
    balance_report: Dict[str, Any] = {}
    required_splits = ["train"]
    if float(args.val_ratio) > 0.0:
        required_splits.append("val")
    if float(args.test_ratio) > 0.0:
        required_splits.append("test")
    for split in required_splits:
        split_buckets = buckets.get(split, {})
        ue_counts = {key: len(split_buckets.get(key, [])) for key in EVENT_KEYS}
        missing = [key for key, value in ue_counts.items() if value <= 0]
        if missing:
            raise ValueError(f"split={split} lacks direct-event UE buckets after filtering: {missing}; counts={ue_counts}")
        per_ue = int(args.target_per_ue) if int(args.target_per_ue) > 0 else min(ue_counts.values())
        re_target = max(1, int(round(float(args.regular_multiplier) * float(per_ue))))
        main_target = per_ue * len(EVENT_KEYS) + re_target
        invalid_target = max(1, int(round(float(args.invalid_ratio) * float(main_target))))
        rng = random.Random(f"{args.split_seed}:new_phase2_balance:{split}:{per_ue}:{re_target}:{invalid_target}")

        sampled: List[Dict[str, Any]] = []
        for target_class in EVENT_KEYS:
            for base in _sample_bucket(
                split_buckets[target_class],
                target=per_ue,
                rng=rng,
                route_diverse=split == "train",
            ):
                sampled.append(
                    _make_row(
                        base=base,
                        question_domain=_native_question_domain(str(base["rs"])),
                        target_class=target_class,
                        invalid=False,
                    )
                )

        regular_candidates = list(split_buckets.get("RE", []))
        highway_regular = [base for base in regular_candidates if str(base["rs"]) == "R3"]
        local_regular = [base for base in regular_candidates if str(base["rs"]) != "R3"]
        highway_target = min(
            re_target,
            max(0, int(round(float(args.highway_regular_fraction) * float(re_target)))),
        )
        local_target = re_target - highway_target
        if highway_target > 0 and not highway_regular:
            raise ValueError(f"split={split} lacks R3/highway regular negatives")
        if local_target > 0 and not local_regular:
            raise ValueError(f"split={split} lacks non-highway regular negatives")
        sampled_regular = [
            *_sample_bucket(
                highway_regular,
                target=highway_target,
                rng=rng,
                route_diverse=split == "train",
            ),
            *_sample_bucket(
                local_regular,
                target=local_target,
                rng=rng,
                route_diverse=split == "train",
            ),
        ]
        rng.shuffle(sampled_regular)
        for base in sampled_regular:
            sampled.append(
                _make_row(
                    base=base,
                    question_domain=_native_question_domain(str(base["rs"])),
                    target_class="RE",
                    invalid=False,
                )
            )

        invalid_rows, invalid_balance = _balanced_invalid_rows(
            invalid_sources.get(split, []),
            split=split,
            target=invalid_target,
            rng=random.Random(f"{args.split_seed}:new_phase2_invalid_balance:{split}:{invalid_target}"),
        )
        if len(invalid_rows) != invalid_target:
            raise ValueError(
                f"split={split} cannot construct required INVALID bucket: "
                f"target={invalid_target} built={len(invalid_rows)}"
            )
        sampled.extend(invalid_rows)
        rng.shuffle(sampled)
        rows.extend(sampled)
        sampled_by_class: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in sampled:
            sampled_by_class[str(row["target_event_class"])].append(row)
        raw_by_class = {
            key: route_diversity_report(split_buckets.get(key, []))
            for key in TARGET_CLASSES
        }
        balance_report[split] = {
            "raw_ue_counts": ue_counts,
            "raw_re_count": len(split_buckets.get("RE", [])),
            "raw_invalid_source_count": len(invalid_sources.get(split, [])),
            "target_per_ue": per_ue,
            "target_regular": re_target,
            "target_invalid": invalid_target,
            "sampled_counts": dict(Counter(row["target_event_class"] for row in sampled)),
            "sampled_question_domain_counts": dict(Counter(row["question_domain"] for row in sampled)),
            "sampled_true_rs_counts": dict(Counter(row["true_rs"] for row in sampled)),
            "route_diverse_sampling": split == "train",
            "raw_route_diversity": raw_by_class,
            "sampled_route_diversity": {
                key: route_diversity_report(sampled_by_class.get(key, []))
                for key in (*TARGET_CLASSES, "INVALID")
            },
            "regular_hard_negative_counts": {
                "highway_r3": sum(
                    1 for row in sampled if row["target_event_class"] == "RE" and row["true_rs"] == "R3"
                ),
                "applicable_local": sum(
                    1 for row in sampled if row["target_event_class"] == "RE" and row["true_rs"] != "R3"
                ),
            },
            "invalid_balance": invalid_balance,
        }
    return rows, {
        "raw_counts": dict(raw_counts),
        "balance": balance_report,
        "actual_scenario_town_pairs": [
            {"scenario": scenario, "town": town}
            for scenario, town in sorted(actual_scenario_town_pairs)
        ],
    }


def _assert_actual_review_coverage(
    actual_pairs: Iterable[Tuple[str, str]],
    review_coverage: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    """确保本次实际进入数据构建候选池的每个 scenario/Town 都已有 RGB review。"""

    reviewed_pairs = {
        (str(scenario), str(town))
        for scenario, towns in review_coverage.items()
        for town, item in towns.items()
        if int(item.get("completed_routes", 0)) >= 1
    }
    actual = {(str(scenario), str(town)) for scenario, town in actual_pairs}
    missing = sorted(actual - reviewed_pairs)
    if missing:
        rendered = [f"{scenario}/{town}" for scenario, town in missing]
        raise ValueError(
            "new Phase2 actual dataset contains scenario/Town pairs without completed full-frame RGB review: "
            f"{rendered[:50]}"
        )


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    """构建 frame_index.jsonl 和 manifest.json。"""

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    review_coverage, coverage_source = load_review_coverage(
        review_root=pathlib.Path(args.review_root),
        coverage_manifest=pathlib.Path(args.coverage_manifest),
    )
    missing_review = [
        f"{scenario}/{town}"
        for scenario, towns in sorted(review_coverage.items())
        for town, item in sorted(towns.items())
        if int(item.get("completed_routes", 0)) < 1
    ]
    if missing_review:
        raise ValueError(f"new Phase2 requires one completed full-frame RGB review per scenario/Town; missing={missing_review[:20]}")

    target = out_dir / "frame_index.jsonl"
    temporary = out_dir / ".frame_index.jsonl.tmp"
    temporary.unlink(missing_ok=True)
    risk_stats: Counter[str] = Counter()
    rows, balance = _balanced_rows_by_split(args, risk_stats)
    actual_pairs = {
        (str(item["scenario"]), str(item["town"]))
        for item in balance.get("actual_scenario_town_pairs", [])
    }
    _assert_actual_review_coverage(actual_pairs, review_coverage)
    counters: Counter[str] = Counter()
    routes: Dict[str, set[str]] = defaultdict(set)
    answer_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                split = str(row["split"])
                counters[f"frames/{split}"] += 1
                counters[f"frames/{split}/{row['target_event_class']}"] += 1
                routes[split].add(f"{row['scenario']}/{row['route_id']}")
                for key in ANSWER_KEYS:
                    answer_counts[split][f"{key}:{'YES' if row['answers'][key] else 'NO'}"] += 1
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(target)
    manifest = {
        "format": "sft_new_loop_phase2_frame_index_v1",
        "dataset_name": DATASET_NAME,
        "frame_index": str(target),
        "event_label_contract": "Targets only UE1/UE3/UE5/UE6. UE2/UE4/UE7/UE8 and all R-E codes are folded into valid RE for this phase.",
        "input_contract": "The model receives one image+text user turn and directly answers UE plus INVALID_EVENT_CONTEXT. No synthetic ROAD_STRUCTURE user turn, assistant answer, RS token, or Phase2 KV prefix is rendered. Internal question_domain only selects ROAD_CORRIDOR versus LOCAL_JUNCTION questions.",
        "invalid_contract": "Invalid rows cross the ROAD_CORRIDOR/LOCAL_JUNCTION question domains, are balanced by source class and true RS, and require all UE=NO plus INVALID_EVENT_CONTEXT=YES. Low visibility, congestion, ordinary queues, and absence of UE remain valid.",
        "balance_contract": "Within every split, UE1/UE3/UE5/UE6 positives are sampled 1:1:1:1. Train uses route-round-robin selection before taking additional frames from the same route; val/test retain the legacy deterministic frame sampler so frozen case identities remain comparable. RE defaults to one UE bucket with an explicit R3/highway fraction. Invalid defaults to 20% of valid main data.",
        "target_per_ue": int(args.target_per_ue),
        "regular_multiplier": float(args.regular_multiplier),
        "highway_regular_fraction": float(args.highway_regular_fraction),
        "invalid_ratio": float(args.invalid_ratio),
        "data_root": str(pathlib.Path(args.data_root)),
        "rgb_path_contract": "history_rgb_paths/latest_rgb_path are relative to --data-root; train/eval remap them with their own --data-root.",
        "include_visual_risk": bool(args.include_visual_risk),
        "full_frame_rgb_review_coverage": {
            "review_root": str(args.review_root),
            "coverage_manifest": str(args.coverage_manifest),
            "coverage_source": coverage_source,
            "scenarios": len(review_coverage),
            "scenario_town_pairs": sum(len(towns) for towns in review_coverage.values()),
            "completed_routes": sum(int(item.get("completed_routes", 0)) for towns in review_coverage.values() for item in towns.values()),
        },
        "split_seed": int(args.split_seed),
        "test_ratio": float(args.test_ratio),
        "val_ratio": float(args.val_ratio),
        "counts": dict(counters),
        "route_counts": {split: len(value) for split, value in sorted(routes.items())},
        "answer_counts": {split: dict(counter) for split, counter in answer_counts.items()},
        "sampling": balance,
        "visual_label_risk_counts": dict(sorted(risk_stats.items())),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数。"""

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection-dir", default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output"))
    p.add_argument("--data-root", default=str(_AUTOMOT_ROOT / "lead_data"))
    p.add_argument("--output-dir", default=str(_AUTOMOT_ROOT / "checkpoints/sft_new_loop_phase2_data"))
    p.add_argument(
        "--review-root",
        default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output/phase1_four_question_audit/full_route_rgb_label_review_20260809"),
        help="completed all-frame RGB review root; every scenario/Town must have at least one reviewed route",
    )
    p.add_argument(
        "--coverage-manifest",
        default=str(DEFAULT_COVERAGE_MANIFEST),
        help="bundled compact coverage proof used when --review-root is absent on a remote machine",
    )
    p.add_argument("--scenarios", default="all")
    p.add_argument("--split-seed", type=int, default=20260819)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--target-per-ue", type=int, default=0, help="0 uses the smallest available UE bucket per split")
    p.add_argument("--regular-multiplier", type=float, default=1.0, help="RE rows per one UE bucket")
    p.add_argument(
        "--highway-regular-fraction",
        type=float,
        default=0.25,
        help="fraction of the RE bucket reserved for valid R3/highway all-NO hard negatives",
    )
    p.add_argument("--invalid-ratio", type=float, default=0.20, help="cross-domain invalid rows as a fraction of valid main rows")
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--progress-every-routes", type=int, default=100)
    p.add_argument("--include-visual-risk", action="store_true")
    args = p.parse_args()
    if float(args.regular_multiplier) <= 0:
        raise ValueError("--regular-multiplier must be positive; RE is a required balance bucket")
    if not 0.0 <= float(args.highway_regular_fraction) <= 1.0:
        raise ValueError("--highway-regular-fraction must be in [0, 1]")
    if not 0.0 < float(args.invalid_ratio) <= 1.0:
        raise ValueError("--invalid-ratio must be in (0, 1]; INVALID is a required balance bucket")
    return args


if __name__ == "__main__":
    manifest = build_dataset(parse_args())
    print(f"sft_new_loop_phase2 dataset: frames={sum(value for key, value in manifest['counts'].items() if key.count('/') == 1)} output={manifest['frame_index']}")
