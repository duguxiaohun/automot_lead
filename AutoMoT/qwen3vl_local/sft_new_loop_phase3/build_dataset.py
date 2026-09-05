#!/usr/bin/env python3
"""构建新 Phase3 的 high-level 动作问答训练索引。

输入沿用 ``keyframe_filter/collection_output/*_result.json`` 的逐帧 RS/EVENT 标注，
再叠加同一条 run 的 ``metas/*.pkl`` 未来真实轨迹，把每一帧折叠成动作上下文，
并给出五个 high-level 动作的 YES/NO 目标。七个异常 U-E context 可由 Phase1/2
回答直接提供；R-E2/R-E3 行仅是离线训练的 transition-gate 正例，不能反向宣称
Phase1/2 的 all-NO 已经唯一确定它们。

当前合同见 MAPPING_AUDIT_20260905.md。七异常与 R-E2/R-E3/R-E5 十桶 1:1，
invalid 对每个 asked context 显式交换不相容 RS。自动候选不等于逐帧动作人工确认。
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
from qwen3vl_local.sft_new_loop_phase3 import DATASET_NAME  # noqa: E402
from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapped_contexts, context_detail, mapping_contract_hash
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import (  # noqa: E402
    ACTION_KEYS,
    CONTEXT_BY_ID,
    CONTEXT_IDS,
    POST_BYPASS_MAX_GAP_FRAMES,
    resolve_context_id,
    resolve_context_ids,
)
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import (  # noqa: E402
    REQUIRED_TRUE_RS,
    REQUIRED_WRONG_CONTEXTS,
    mismatched_contexts,
    mismatched_road_contexts,
)
from qwen3vl_local.sft_new_loop_phase3.prompts import ANSWER_KEYS, INVALID_KEY  # noqa: E402
from qwen3vl_local.sft_new_loop_phase3.sampling import (  # noqa: E402
    even_quota_with_capacity,
    route_diverse_sample,
    route_diversity_report,
)
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import (  # noqa: E402
    ACTION_RULE_VERSION,
    action_evidence,
    label_actions,
    load_route_trajectory,
)
from qwen3vl_local.sft_new_loop_phase3.visual_audit import (  # noqa: E402
    DEFAULT_COVERAGE_MANIFEST,
    frame_visual_risk,
    load_review_coverage,
)

RGB_HISTORY_COUNT = 4
NO_ACTION_SIGNATURE = "NONE"


def _source_routes(collection_dir, scenario, args):
    """默认读正式结果；显式审计模式复用已有缓存，manifest 标明范围。"""
    if getattr(args, "use_review_cache", False):
        for path in sorted(pathlib.Path(args.review_root).glob(f"{scenario}/*/*/route_annotations.json")):
            run = pathlib.Path(args.data_root) / scenario / path.parent.name
            if run.is_dir() and not is_abnormal_lead_route(run, scenario)[0]:
                yield json.loads(path.read_text())
    else:
        yield from _iter_routes_stream(collection_dir / f"{scenario}_result.json")


def _stable_unit(value: str) -> float:
    """把字符串映射到稳定 [0,1) 浮点，用于 route-disjoint split。"""

    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)


def _split(scenario: str, route_id: str, seed: int, test_ratio: float, val_ratio: float) -> str:
    """按 route 做 deterministic train/val/test 切分。"""

    value = _stable_unit(f"{seed}:{scenario}:{route_id}")
    return "test" if value < test_ratio else "val" if value < test_ratio + val_ratio else "train"


def _rs_label(annotation: Mapping[str, Any]) -> str:
    """读取 canonical primary ROAD_STRUCTURE。"""

    return str(
        annotation.get("primary_road_structure")
        or (annotation.get("frame_rs_annotation") or {}).get("label")
        or "UNKNOWN"
    )


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
            if (part.startswith("U-E") or part.startswith("R-E")) and part not in out:
                out.append(part)
    return tuple(out)


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


def action_signature(labels: Mapping[str, bool], *, context_id: Optional[str] = None) -> str:
    """把本 context 实际监督的动作折叠成稳定签名。

    纵向 context 不问变道。它们的未来轨迹即使恰好包含 lane switch，也不能作为
    signature 桶来过采样，否则训练会为了不可见、未提问的横向副动作破坏真正的
    DECELERATE/STOP/RESUME 分布。
    """

    keys = ACTION_KEYS if context_id is None else CONTEXT_BY_ID[str(context_id)].action_keys
    positives = [key for key in keys if bool(labels.get(key, False))]
    return "+".join(positives) if positives else NO_ACTION_SIGNATURE


def _answers_for(context_id: str, labels: Optional[Mapping[str, bool]], *, invalid: bool) -> Dict[str, bool]:
    """构造该行的完整答案字典。"""

    answers = {key: False for key in ACTION_KEYS}
    if not invalid and labels is not None:
        asked = set(CONTEXT_BY_ID[context_id].action_keys)
        for key in ACTION_KEYS:
            answers[key] = bool(labels.get(key, False)) and key in asked
    answers[INVALID_KEY] = bool(invalid)
    return answers


def _make_row(
    *,
    base: Mapping[str, Any],
    context_id: str,
    invalid: bool,
    invalid_source: str = "",
) -> Dict[str, Any]:
    """从基础帧记录构造最终 JSONL row。"""

    context = CONTEXT_BY_ID[context_id]
    labels = None if invalid else base["action_labels"]
    road_structure = str(base["rs"]) if not invalid else str(base["invalid_prompt_rs"])
    answers = _answers_for(context_id, labels, invalid=invalid)
    return {
        "dataset_name": DATASET_NAME,
        "mapping_contract_hash": mapping_contract_hash(),
        "scenario": base["scenario"],
        "route_id": base["route_id"],
        "town": base["town"],
        "split": base["split"],
        "frame_id": int(base["frame_id"]),
        "true_rs": base["rs"],
        "prompt_road_structure": road_structure,
        "context_id": context_id,
        "question_domain": context.question_domain,
        "source_event": context.source_event,
        "event": base["primary_event"],
        "event_codes": list(str(x) for x in base["event_codes"]),
        "balance_key": "INVALID" if invalid else context_id,
        "action_signature": (
            "INVALID" if invalid else action_signature(base["action_labels"], context_id=context_id)
        ),
        "invalid_action_context": bool(invalid),
        "invalid_source": str(invalid_source),
        "invalid_reason": "wrong_road_structure" if invalid else "",
        "mapping_evidence": dict(base.get("mapping_evidence", {})),
        "context_detail": str(base.get("context_detail", "")) if not invalid else "",
        "answers": answers,
        "goal_ego_xy": [round(float(base["goal_x"]), 3), round(float(base["goal_y"]), 3)],
        "action_evidence": base["action_evidence"],
        "visual_label_risk": bool(base["visual_label_risk"]),
        "visual_label_risk_reasons": list(base["visual_label_risk_reasons"]),
        "history_rgb_paths": list(base["history_rgb_paths"]),
        "latest_rgb_path": base["latest_rgb_path"],
    }


def _last_bypass_frame(annotations: Sequence[Mapping[str, Any]]) -> Dict[int, Optional[int]]:
    """返回每帧“距离上一段 U-E2 结束多少帧”，用于识别绕障后的回归变道。"""

    out: Dict[int, Optional[int]] = {}
    last_bypass: Optional[int] = None
    for ann in annotations:
        try:
            frame_id = int(ann.get("frame_id"))
        except (TypeError, ValueError):
            continue
        codes = _event_codes(ann)
        out[frame_id] = None if last_bypass is None else frame_id - last_bypass
        if "U-E2" in codes:
            last_bypass = frame_id
            out[frame_id] = 0
    return out


def iter_base_frames(
    args: argparse.Namespace,
    risk_stats: Optional[Counter] = None,
    observed_scenario_town_pairs: Optional[set] = None,
) -> Iterable[Dict[str, Any]]:
    """流式遍历可用基础帧，并记录本次实际扫描到的 route-level scenario/Town。"""

    collection_dir = pathlib.Path(args.collection_dir)
    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    if getattr(args, "candidate_cache", ""):
        # 审计专用：复用已标定的轨迹，再执行最新语义适配。禁止混入旧动作规则。
        with pathlib.Path(args.candidate_cache).open() as handle:
            for line in handle:
                base = json.loads(line)
                if base["action_evidence"].get("rule_version") != ACTION_RULE_VERSION:
                    raise ValueError("candidate cache action rule mismatch; rebuild from meta")
                from qwen3vl_local.sft_new_loop_phase3.lateral_rgb_audit import lateral_uncertainty
                lateral_review = lateral_uncertainty(base["scenario"], base["route_id"], base["frame_id"])
                if lateral_review:
                    if CONTEXT_BY_ID[base["context_id"]].question_domain == "FULL_MANEUVER":
                        continue
                    base["action_labels"]["LANE_CHANGE_LEFT"] = False
                    base["action_labels"]["LANE_CHANGE_RIGHT"] = False
                    base["action_evidence"].update(lateral_observation_complete=False,
                        lane_change_direction="", lateral_rgb_uncertainty=lateral_review)
                run = data_root / base["scenario"] / base["route_id"]
                if not run.is_dir() or is_abnormal_lead_route(run, base["scenario"])[0]:
                    continue
                contexts, evidence = mapped_contexts(base["scenario"], base["route_id"], base["frame_id"],
                    base["rs"], base["primary_event"], base["event_codes"])
                if base["context_id"] not in contexts:
                    continue
                base["mapping_evidence"] = evidence
                base["split"] = _split(base["scenario"], base["route_id"], args.split_seed,
                                       args.test_ratio, args.val_ratio)
                if observed_scenario_town_pairs is not None:
                    observed_scenario_town_pairs.add((base["scenario"], base["town"]))
                if base["visual_label_risk"] and not args.include_visual_risk:
                    continue
                yield base
        return
    selected = (
        None
        if args.scenarios == "all"
        else {item.strip() for item in str(args.scenarios).split(",") if item.strip()}
    )
    seen = 0
    for result_path in sorted(collection_dir.glob("*_result.json")):
        scenario = result_path.stem.removesuffix("_result")
        if scenario == "noScenarios" or (selected is not None and scenario not in selected):
            continue
        scenario_seen = 0
        for route in _source_routes(collection_dir, scenario, args):
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
            scenario_seen += 1
            if int(args.progress_every_routes) > 0 and seen % int(args.progress_every_routes) == 0:
                print(f"[new-phase3-build] routes={seen} last={scenario}/{route_id}", flush=True)
            if args.max_routes > 0 and seen > args.max_routes:
                return
            if int(args.max_routes_per_scenario) > 0 and scenario_seen > int(args.max_routes_per_scenario):
                break
            split = _split(
                scenario, route_id, int(args.split_seed), float(args.test_ratio), float(args.val_ratio)
            )
            gaps = _last_bypass_frame(annotations)
            wanted = []
            for ann in annotations:
                try:
                    frame_id = int(ann.get("frame_id"))
                except (TypeError, ValueError):
                    continue
                contexts, mapping = mapped_contexts(
                    scenario, route_id, frame_id, _rs_label(ann),
                    str(ann.get("primary_event") or "UNKNOWN"), _event_codes(ann))
                for context_id in contexts:
                    wanted.append((frame_id, ann, context_id, mapping))
            if not wanted:
                continue
            trajectory = load_route_trajectory(run_dir)
            if trajectory is None:
                continue
            for frame_id, ann, context_id, mapping in wanted:
                signals = trajectory.signals(frame_id)
                if signals is None or not signals["goal_available"]:
                    continue
                labels = label_actions(signals)
                if labels is None:
                    continue
                if (CONTEXT_BY_ID[context_id].question_domain == "FULL_MANEUVER"
                        and not signals["lateral_observation_complete"]):
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
                    "rs": _rs_label(ann),
                    "primary_event": str(ann.get("primary_event") or "UNKNOWN"),
                    "event_codes": _event_codes(ann),
                    "context_id": context_id,
                    "context_detail": " ".join(filter(None, [
                        context_detail(context_id, gaps.get(frame_id)),
                        *["Concurrent observed condition: " + CONTEXT_BY_ID[other].situation_text + "."
                          for other in mapping.get("active_context_ids", []) if other != context_id]])),
                    "mapping_evidence": mapping,
                    "action_labels": labels,
                    "action_evidence": action_evidence(signals),
                    "goal_x": float(signals["goal_x"]),
                    "goal_y": float(signals["goal_y"]),
                    "is_junction": bool(signals["is_junction"]),
                    "distance_to_next_junction": float(signals["distance_to_next_junction"]),
                    "visual_label_risk": risk,
                    "visual_label_risk_reasons": reasons,
                    "history_rgb_paths": history,
                    "latest_rgb_path": history[-1],
                }


def _sample_context_bucket(
    bucket: Sequence[Mapping[str, Any]],
    *,
    context_id: str,
    target: int,
    rng: random.Random,
    route_diverse: bool,
) -> Tuple[List[Mapping[str, Any]], Dict[str, Any]]:
    """在一个上下文桶内按动作签名尽量均分，再在签名内做 route 轮转抽样。"""

    if target <= 0 or not bucket:
        return [], {"signature_capacity": {}, "signature_quota": {}}
    by_signature: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for base in bucket:
        by_signature[action_signature(base["action_labels"], context_id=context_id)].append(base)
    capacities = {key: len(value) for key, value in by_signature.items()}
    quotas = even_quota_with_capacity(capacities, int(target))
    selected: List[Mapping[str, Any]] = []
    for key in sorted(quotas):
        count = int(quotas[key])
        if count <= 0:
            continue
        selected.extend(
            route_diverse_sample(by_signature[key], target=count, rng=rng)
            if route_diverse
            else _plain_sample(by_signature[key], count, rng)
        )
    shortfall = int(target) - len(selected)
    if shortfall > 0:
        pool = list(bucket)
        rng.shuffle(pool)
        selected.extend(pool[idx % len(pool)] for idx in range(shortfall))
    rng.shuffle(selected)
    return selected, {"signature_capacity": capacities, "signature_quota": dict(quotas)}


def _plain_sample(bucket: Sequence[Mapping[str, Any]], target: int, rng: random.Random) -> List[Mapping[str, Any]]:
    """val/test 用的确定性抽样，不做 route 轮转。"""

    items = list(bucket)
    rng.shuffle(items)
    if len(items) >= target:
        return items[:target]
    return [items[idx % len(items)] for idx in range(target)]


def _balanced_invalid_rows(
    bases: Sequence[Mapping[str, Any]],
    *,
    split: str,
    target: int,
    rng: random.Random,
    require_true_rs_coverage: bool = True,
    same_rs_rows: Sequence[Mapping[str, Any]] = (),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """按 source 上下文、true RS 和被问的错误上下文均衡构造 invalid 样本。"""

    if target <= 0:
        return [], {"candidate_buckets": {}, "sampled_signature_counts": {}}
    candidate_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    source_to_signatures: Dict[str, List[str]] = defaultdict(list)
    for base in bases:
        source_class = str(base["context_id"])
        true_rs = str(base["rs"])
        for asked, fake_rs in mismatched_road_contexts(
            true_rs=true_rs,
            is_junction=bool(base["is_junction"]),
            distance_to_next_junction=float(base["distance_to_next_junction"]),
        ):
            signature = f"source={source_class}|true_rs={true_rs}|asked_context={asked}"
            if signature not in source_to_signatures[source_class]:
                source_to_signatures[source_class].append(signature)
            candidate_buckets[signature].append(
                _make_row(base={**base, "invalid_prompt_rs": fake_rs}, context_id=asked,
                          invalid=True, invalid_source=signature)
            )
    # 构建与 train/eval 使用同一个配额实现；避免 index 与运行时口径漂移。
    from types import SimpleNamespace
    from qwen3vl_local.sft_new_loop_phase3.invalid_balance import balanced_invalid_items, invalid_subgroup_report
    pool = [SimpleNamespace(**r) for bucket in candidate_buckets.values() for r in bucket]
    pool.extend(SimpleNamespace(**r) for r in same_rs_rows)
    if not pool:
        return [], {"candidate_buckets": {}, "sampled_signature_counts": {}}
    selected = balanced_invalid_items(pool, target=target, rng=rng,
                                     require_coverage=require_true_rs_coverage)
    report = invalid_subgroup_report(selected)
    report['candidate_buckets'] = {k: len(v) for k, v in sorted(candidate_buckets.items())}
    report['same_rs_candidate_rows'] = len(same_rs_rows)
    return [vars(row) for row in selected], report


def _balanced_rows_by_split(
    args: argparse.Namespace, risk_stats: Counter
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """读取候选帧并按 split 生成均衡 rows。"""

    buckets: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    invalid_sources: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    raw_counts: Counter = Counter()
    actual_scenario_town_pairs: set = set()
    for base in iter_base_frames(args, risk_stats, observed_scenario_town_pairs=actual_scenario_town_pairs):
        split = str(base["split"])
        context_id = str(base["context_id"])
        actual_scenario_town_pairs.add((str(base["scenario"]), str(base["town"])))
        buckets[split][context_id].append(base)
        raw_counts[f"{split}/{context_id}"] += 1
        raw_counts[f"{split}/{context_id}/{action_signature(base['action_labels'], context_id=context_id)}"] += 1
        invalid_sources[split].append(base)

    from qwen3vl_local.sft_new_loop_phase3.same_rs_invalid import reviewed_invalid_rows
    same_rs_pool = reviewed_invalid_rows(args, {
        (b['scenario'], b['route_id']) for bs in invalid_sources.values() for b in bs})
    same_rs_by_split = defaultdict(list)
    for row in same_rs_pool:
        same_rs_by_split[row['split']].append(row)

    # 即使均衡容量检查失败也保留候选和缺口，方便继续逐帧审计而非重复解压 meta。
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "candidate_frames.jsonl").open("w") as handle:
        for bases in invalid_sources.values():
            for base in bases:
                handle.write(json.dumps(base, ensure_ascii=False) + "\n")
    (out_dir / "candidate_counts.json").write_text(json.dumps(dict(raw_counts), indent=2) + "\n")

    (out_dir / "same_rs_invalid_candidates.jsonl").write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + "\n" for row in same_rs_pool))
    rows: List[Dict[str, Any]] = []
    balance_report: Dict[str, Any] = {}
    required_splits = ["train"]
    if float(args.val_ratio) > 0.0:
        required_splits.append("val")
    if float(args.test_ratio) > 0.0:
        required_splits.append("test")
    for split in required_splits:
        split_buckets = buckets.get(split, {})
        context_counts = {key: len(split_buckets.get(key, [])) for key in CONTEXT_IDS}
        missing = [key for key, value in context_counts.items() if value <= 0]
        if missing:
            raise ValueError(
                f"split={split} lacks action-context buckets after filtering: {missing}; counts={context_counts}"
            )
        per_context = (
            int(args.target_per_context)
            if int(args.target_per_context) > 0
            else min(context_counts.values())
        )
        main_target = per_context * len(CONTEXT_IDS)
        invalid_target = max(1, int(round(float(args.invalid_ratio) * float(main_target))))
        rng = random.Random(f"{args.split_seed}:new_phase3_balance:{split}:{per_context}:{invalid_target}")

        sampled: List[Dict[str, Any]] = []
        signature_reports: Dict[str, Any] = {}
        for context_id in CONTEXT_IDS:
            selected, report = _sample_context_bucket(
                split_buckets[context_id],
                context_id=context_id,
                target=per_context,
                rng=rng,
                route_diverse=split == "train",
            )
            signature_reports[context_id] = report
            for base in selected:
                sampled.append(_make_row(base=base, context_id=context_id, invalid=False))

        invalid_rows, invalid_balance = _balanced_invalid_rows(
            invalid_sources.get(split, []),
            split=split,
            target=invalid_target,
            rng=random.Random(f"{args.split_seed}:new_phase3_invalid_balance:{split}:{invalid_target}"),
            require_true_rs_coverage=bool(args.require_invalid_true_rs_coverage),
            same_rs_rows=same_rs_by_split.get(split, []),
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
            sampled_by_class[str(row["balance_key"])].append(row)
        balance_report[split] = {
            "raw_context_counts": context_counts,
            "target_per_context": per_context,
            "target_invalid": invalid_target,
            "sampled_counts": dict(Counter(row["balance_key"] for row in sampled)),
            "sampled_true_rs_counts": dict(Counter(row["true_rs"] for row in sampled)),
            "sampled_action_signature_counts": dict(Counter(row["action_signature"] for row in sampled)),
            "sampled_yes_counts": {
                key: sum(1 for row in sampled if bool(row["answers"][key])) for key in ANSWER_KEYS
            },
            "context_action_signature_balance": signature_reports,
            "route_diverse_sampling": split == "train",
            "sampled_route_diversity": {
                key: route_diversity_report(sampled_by_class.get(key, []))
                for key in (*CONTEXT_IDS, "INVALID")
            },
            "invalid_balance": invalid_balance,
        }
    return rows, {
        "raw_counts": dict(raw_counts),
        "balance": balance_report,
        "actual_scenario_town_pairs": [
            {"scenario": scenario, "town": town} for scenario, town in sorted(actual_scenario_town_pairs)
        ],
    }


def _assert_actual_review_coverage(
    actual_pairs: Iterable[Tuple[str, str]],
    review_coverage: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    """确保本次实际进入构建候选池的每个 scenario/Town 都已有 RGB review。"""

    reviewed_pairs = {
        (str(scenario), str(town))
        for scenario, towns in review_coverage.items()
        for town, item in towns.items()
        if int(item.get("completed_routes", 0)) >= 1
    }
    missing = sorted({(str(scenario), str(town)) for scenario, town in actual_pairs} - reviewed_pairs)
    if missing:
        rendered = [f"{scenario}/{town}" for scenario, town in missing]
        raise ValueError(
            "new Phase3 actual dataset contains scenario/Town pairs without completed full-frame RGB review: "
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
        raise ValueError(
            f"new Phase3 requires one completed full-frame RGB review per scenario/Town; missing={missing_review[:20]}"
        )

    target = out_dir / "frame_index.jsonl"
    temporary = out_dir / ".frame_index.jsonl.tmp"
    temporary.unlink(missing_ok=True)
    risk_stats: Counter = Counter()
    rows, balance = _balanced_rows_by_split(args, risk_stats)
    _assert_actual_review_coverage(
        {(str(item["scenario"]), str(item["town"])) for item in balance.get("actual_scenario_town_pairs", [])},
        review_coverage,
    )
    counters: Counter = Counter()
    routes: Dict[str, set] = defaultdict(set)
    answer_counts: Dict[str, Counter] = defaultdict(Counter)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                split = str(row["split"])
                counters[f"frames/{split}"] += 1
                counters[f"frames/{split}/{row['balance_key']}"] += 1
                routes[split].add(f"{row['scenario']}/{row['route_id']}")
                for key in ANSWER_KEYS:
                    answer_counts[split][f"{key}:{'YES' if row['answers'][key] else 'NO'}"] += 1
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(target)

    manifest = {
        "format": "sft_new_loop_phase3_frame_index_v2_high_level_action",
        "source_scope": ("candidate_cache" if getattr(args, "candidate_cache", "") else
                         "existing_review_cache" if getattr(args, "use_review_cache", False) else "collection_results"),
        "candidate_cache": str(getattr(args, "candidate_cache", "") or ""),
        "action_review_status": "automatic_candidates_with_explicit_rgb_exclusions",
        "action_rule_version": ACTION_RULE_VERSION,
        "mapping_contract_hash": mapping_contract_hash(),
        "dataset_name": DATASET_NAME,
        "frame_index": str(target),
        "context_contract": (
            "Every explicit abnormal flag supplies a candidate; concurrent flags are retained and raw U7 is checked against the audited signal-failure answer: "
            "U-E1->LEAD_BRAKE, U-E2->STATIC_BLOCKAGE, U-E3->DYNAMIC_CUTIN, U-E4->VULNERABLE_CROSSING, "
            "U-E5->ONCOMING_INVASION, U-E6->JUNCTION_RULE_CONFLICT, U-E7->SIGNAL_FAILURE, "
            "Explicit R-E2->POST_BYPASS_RETURN (generic target/recovery lane transition), R-E3 on R3->RAMP_MERGE_EXIT, R-E5 on R5->UNSIGNALIZED_PRIORITY. "
            "The three R-E contexts are offline transition-gate positives, not Phase1/Phase2 outputs: "
            "all abnormal UE=NO cannot distinguish them from regular R-E1/R-E4/R-E5. U-E8 has no "
            "Phase1/Phase2 question and is therefore excluded."
        ),
        "action_label_contract": (
            "Longitudinal labels come from the run's own future speed curve: STOP uses a 1.5 s immediate "
            "window, DECELERATE and RESUME use a 2 s window, and the three are mutually exclusive. "
            "RESUME requires two consecutive speed samples above the gain threshold. "
            "Lateral candidates require a Driving-waypoint identity change within 3 s on the same road, with "
            "the direction resolved from the lane ordering and the lane ego entered the current continuous road visit in, "
            "so borrowing the opposing lane is LEFT and returning is RIGHT. Non-Driving or unknown waypoint "
            "windows cannot supervise lateral NO. Lane-section continuity still needs RGB/map confirmation. "
            "Scenario names never create an action label."
        ),
        "input_contract": (
            "The model receives one image+text user turn with the RGB history, the Phase1/Phase2 road "
            "structure and situation as a premise, and the route target point in ego coordinates "
            "(x forward, y negative left, y positive right)."
        ),
        "invalid_contract": (
            "Invalid includes wrong road structure and explicitly RGB-reviewed same-RS event mismatch, balanced by source context, "
            "true RS and asked wrong context, and require every action line NO plus "
            "INVALID_ACTION_CONTEXT YES. Low visibility, congestion and simply needing no action stay valid."
        ),
        "balance_contract": (
            "Within every split the ten action contexts are sampled 1:1, and each context bucket is split "
            "as evenly as capacity allows across its action signatures so every action keeps positives. "
            "Train uses route-round-robin selection; val/test use deterministic sampling. Invalid defaults "
            "to 20% of valid main data."
        ),
        "target_per_context": int(args.target_per_context),
        "invalid_ratio": float(args.invalid_ratio),
        "post_bypass_max_gap_frames": None,
        "data_root": str(pathlib.Path(args.data_root)),
        "rgb_path_contract": "history_rgb_paths/latest_rgb_path are relative to --data-root; train/eval remap them with their own --data-root.",
        "include_visual_risk": bool(args.include_visual_risk),
        "full_frame_rgb_review_coverage": {
            "review_root": str(args.review_root),
            "coverage_manifest": str(args.coverage_manifest),
            "coverage_source": coverage_source,
            "scenarios": len(review_coverage),
            "scenario_town_pairs": sum(len(towns) for towns in review_coverage.values()),
            "completed_routes": sum(
                int(item.get("completed_routes", 0))
                for towns in review_coverage.values()
                for item in towns.values()
            ),
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
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数。"""

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection-dir", default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output"))
    p.add_argument("--data-root", default=str(_AUTOMOT_ROOT / "lead_data"))
    p.add_argument("--output-dir", default=str(_AUTOMOT_ROOT / "checkpoints/sft_new_loop_phase3_data"))
    p.add_argument(
        "--review-root",
        default=str(
            _AUTOMOT_ROOT
            / "keyframe_filter/collection_output/phase1_four_question_audit/full_route_rgb_label_review_20260809"
        ),
    )
    p.add_argument("--coverage-manifest", default=str(DEFAULT_COVERAGE_MANIFEST))
    p.add_argument("--scenarios", default="all")
    p.add_argument("--use-review-cache", action="store_true",
                   help="audit/smoke only: reuse cached routes; never claim full-dataset coverage")
    p.add_argument("--candidate-cache", default="",
                   help="audit only: reuse candidate_frames.jsonl with the same trajectory rule version")
    p.add_argument("--split-seed", type=int, default=20260819)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument(
        "--target-per-context",
        type=int,
        default=0,
        help="0 uses the smallest available action-context bucket per split",
    )
    p.add_argument(
        "--invalid-ratio",
        type=float,
        default=0.20,
        help="mismatched-context invalid rows as a fraction of valid main rows",
    )
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument(
        "--require-invalid-true-rs-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require every split's INVALID candidates to cover R1-R5; disable only for route-capped smoke builds",
    )
    p.add_argument(
        "--max-routes-per-scenario",
        type=int,
        default=0,
        help="smoke only: cap usable routes per scenario so every action context still gets candidates",
    )
    p.add_argument("--progress-every-routes", type=int, default=100)
    p.add_argument("--include-visual-risk", action="store_true")
    args = p.parse_args()
    if not 0.0 < float(args.invalid_ratio) <= 1.0:
        raise ValueError("--invalid-ratio must be in (0, 1]; INVALID is a required balance bucket")
    return args


if __name__ == "__main__":
    manifest = build_dataset(parse_args())
    total = sum(value for key, value in manifest["counts"].items() if key.count("/") == 1)
    print(f"sft_new_loop_phase3 dataset: frames={total} output={manifest['frame_index']}")
