#!/usr/bin/env python3
"""构建 action-prior 均衡课程的全帧 Phase3 语义映射。

从 AutoMoT/ 目录构建（先准备当前 Phase3 candidate 和 action 三 split）：
    python qwen3vl_local/action_prior/build_event_balance_index.py --candidate-index checkpoints/sft_new_loop_phase3_data_v7/candidate_frames.jsonl --action-data-dir checkpoints/action_prior_data_event_v1 --output-dir checkpoints/action_prior_event_balance_v2
训练开关 demo 见 run.md 和 train.sh 开头；构建不读取或生成未来动作作为模型输入。

这不是 ``candidate_frames.jsonl`` 的复制品：它先扫描所有原始逐帧 RS/EVENT 标注，再用
当前 Phase3 mapping/review 合同恢复 special context；candidate 只标记该 special frame 是否
通过了 Phase3 的动作窗口、横向证据和视觉风险门。由此普通背景池只接受明确 regular，未确认
与 special-but-filtered 都不会悄悄落入背景。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.action_prior.contracts import digest, file_hash
from qwen3vl_local.action_prior.event_balance import (
    CONFIRMED_REGULAR, EVENT_BALANCE_MAPPING_POLICY, FULL_INDEX_SCHEMA, FULL_MANIFEST_SCHEMA, SPECIAL_ELIGIBLE,
    SPECIAL_FILTERED, SPECIAL_BUCKETS, UNCONFIRMED,
)
from qwen3vl_local.sft_new_loop_phase3.annotation_repair import repair_annotation
from qwen3vl_local.sft_new_loop_phase3.build_dataset import _event_codes, _last_bypass_frame, _rs_label
from qwen3vl_local.sft_new_loop_phase3.collection_reader import iter_routes
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID
from qwen3vl_local.sft_new_loop_phase3.source_mapping import context_detail, mapped_contexts, mapping_contract_hash

CONTEXT_TO_BUCKET = {
    "LEAD_BRAKE": "UE1", "STATIC_BLOCKAGE": "UE2", "DYNAMIC_CUTIN": "UE3",
    "VULNERABLE_CROSSING": "UE4", "ONCOMING_INVASION": "UE5",
    "JUNCTION_RULE_CONFLICT": "UE6", "SIGNAL_FAILURE": "UE7",
    "POST_BYPASS_RETURN": "RE2", "RAMP_MERGE_EXIT": "RE3", "UNSIGNALIZED_PRIORITY": "RE5",
}


def _identity(row):
    return str(row["scenario"]), str(row["route_id"]), int(row["frame_id"])


def _candidate_membership(path: Path, expected_hash: str):
    """只读 candidate 的 eligibility，不把其过滤后的缺席误当 full-map 常规。"""
    manifest_path = path.with_name("manifest.json")
    counts_path = path.with_name("candidate_counts.json")
    if not manifest_path.is_file() or not counts_path.is_file():
        raise ValueError("candidate must be a complete current Phase3 output directory with manifest/counts")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_index = path.with_name("frame_index.jsonl")
    if (
        manifest.get("format") != "sft_new_loop_phase3_frame_index_v3_current_phase"
        or manifest.get("mapping_contract_hash") != expected_hash
        or Path(manifest.get("frame_index", "")).name != frame_index.name
        or not frame_index.is_file()
    ):
        raise ValueError("candidate manifest is stale, incomplete, or not the current Phase3 frame-index artifact")
    expected_rows = sum(
        int(value) for key, value in json.loads(counts_path.read_text(encoding="utf-8")).items()
        if key.count("/") == 1
    )
    result = defaultdict(set)
    rows = 0
    for number, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        rows += 1
        row = json.loads(line)
        if row.get("mapping_contract_hash") != expected_hash:
            raise ValueError(f"{path}:{number}: stale Phase3 candidate mapping contract")
        if row.get("invalid_action_context") is True:
            continue
        bucket = CONTEXT_TO_BUCKET.get(str(row.get("context_id", "")))
        if bucket:
            result[_identity(row)].add(bucket)
    if not result:
        raise ValueError("candidate source has no eligible special frames")
    if rows != expected_rows:
        raise ValueError(
            f"candidate/counts coverage mismatch: candidate_rows={rows} candidate_counts={expected_rows}"
        )
    return result


def _re2_scene_state(detail: str):
    """当前标注没有“已绕过且待恢复”的可靠状态，只保留可证明的弱历史。"""
    if detail.startswith("Earlier ego encountered a static blockage"):
        return "RE2_PRIOR_OBSTACLE"
    return "RE2_NAVIGATION_TRANSITION"


def _scene_contexts(context_ids, detail):
    values = []
    for context_id in context_ids:
        bucket = CONTEXT_TO_BUCKET[context_id]
        if bucket == "RE2":
            values.append(_re2_scene_state(detail))
        else:
            values.append(bucket)
    return values


def _regular_confirmation_block_reasons(context_ids, evidence, repair, codes):
    """空 mapping 不代表确认普通：隔离、修复和不成立的 special gate 必须留在未确认池。"""
    reasons = []
    if evidence.get("rgb_quarantine"):
        reasons.append("rgb_quarantine")
    if repair.get("changes"):
        reasons.append("annotation_repair_or_gate")
    if not context_ids and any(str(code) in {"R-E2", "R-E3", "R-E5"} for code in codes):
        reasons.append("unresolved_special_regular_context")
    return reasons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection-dir", default="keyframe_filter/collection_output")
    p.add_argument("--candidate-index", required=True, help="current Phase3 candidate_frames.jsonl")
    p.add_argument(
        "--action-data-dir", required=True,
        help="the exact action_prior train/val/test index to cover and bind",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--scenarios", default="all")
    args = p.parse_args()
    out = Path(args.output_dir)
    index = out / "full_event_mapping.jsonl"
    manifest_path = out / "manifest.json"
    if index.exists() or manifest_path.exists():
        raise FileExistsError(f"{out}: full mapping is immutable; choose a new output directory")
    expected = mapping_contract_hash()
    candidate_path = Path(args.candidate_index).resolve()
    eligible = _candidate_membership(candidate_path, expected)
    action_dir = Path(args.action_data_dir).resolve()
    action_rows = []
    action_hashes = {}
    for split in ("train", "val", "test"):
        path = action_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        action_hashes[split] = file_hash(path)
        for number, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "action_prior_data_v1" or row.get("split") != split:
                raise ValueError(f"{path}:{number}: not a current action_prior split row")
            action_rows.append((str(row["scenario"]), str(row["run_id"]), int(row["anchor"]), split))
    requested = None if args.scenarios == "all" else {x.strip() for x in args.scenarios.split(",") if x.strip()}
    out.mkdir(parents=True, exist_ok=False)
    temporary = out / ".full_event_mapping.jsonl.tmp"
    counts, seen = Counter(), set()
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for result_path in sorted(Path(args.collection_dir).glob("*_result.json")):
                scenario = result_path.stem.removesuffix("_result")
                if scenario == "noScenarios" or (requested is not None and scenario not in requested):
                    continue
                for route in iter_routes(result_path):
                    route_id = str(route.get("route_id") or "")
                    if not route_id or str(route.get("status")) == "data_missing_skip":
                        continue
                    annotations = list(route.get("annotations", []) or [])
                    gaps = _last_bypass_frame(annotations)
                    for ann in annotations:
                        try:
                            frame = int(ann.get("frame_id"))
                        except (TypeError, ValueError):
                            continue
                        rs, primary, codes, repair = repair_annotation(
                            scenario, route_id, frame, ann, _rs_label(ann),
                            str(ann.get("primary_event") or "UNKNOWN"), _event_codes(ann),
                        )
                        if rs == "UNKNOWN":
                            continue
                        context_ids, evidence = mapped_contexts(scenario, route_id, frame, rs, primary, codes)
                        evidence["annotation_repair"] = repair
                        detail = " ".join(filter(None, [
                            *(context_detail(value, gaps.get(frame)) for value in context_ids),
                            *["Concurrent observed condition: " + CONTEXT_BY_ID[value].situation_text + "."
                              for value in context_ids if value != "POST_BYPASS_RETURN"],
                        ]))
                        buckets = [CONTEXT_TO_BUCKET[value] for value in context_ids]
                        key = (scenario, route_id, frame)
                        if key in seen:
                            raise ValueError(f"duplicate full-map identity: {key}")
                        seen.add(key)
                        eligible_buckets = [
                            bucket for bucket in buckets if bucket in eligible.get(key, set())
                        ]
                        # 空 context 有明确的不同原因：人工隔离、repair/gate 取消，或
                        # 真正未确认。前两类绝不能因剩余 R-E code 而掉入普通背景。
                        regular_block_reasons = _regular_confirmation_block_reasons(
                            context_ids, evidence, repair, codes
                        )
                        if buckets:
                            # 并发事实逐桶判 eligibility：某个 context 不适合 Phase3
                            # 动作问答，不能让同帧另一个通过过滤的 context 一并消失。
                            status = SPECIAL_ELIGIBLE if eligible_buckets else SPECIAL_FILTERED
                        elif (
                            codes
                            and all(str(code).startswith("R-E") for code in codes)
                            and not regular_block_reasons
                        ):
                            status = CONFIRMED_REGULAR
                        else:
                            status = UNCONFIRMED
                        row = dict(
                            schema=FULL_INDEX_SCHEMA, mapping_contract_hash=expected,
                            scenario=scenario, route_id=route_id, frame_id=frame,
                            source_split="", rs=rs, event_codes=list(codes),
                            special_buckets=buckets, eligible_buckets=eligible_buckets,
                            scene_contexts=_scene_contexts(context_ids, detail),
                            status=status, context_detail=detail,
                            regular_confirmation_blocked=regular_block_reasons,
                        )
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        counts[status] += 1
                        for value in buckets: counts[f"bucket/{value}/{status}"] += 1
            # collection annotations may not cover ordinary/unlabelled action frames. They are
            # explicitly materialized as UNCONFIRMED, never omitted or silently background.
            for scenario, route_id, frame, split in action_rows:
                key = (scenario, route_id, frame)
                if key in seen:
                    continue
                seen.add(key)
                row = dict(
                    schema=FULL_INDEX_SCHEMA, mapping_contract_hash=expected,
                    scenario=scenario, route_id=route_id, frame_id=frame,
                    source_split=split, rs="UNKNOWN", event_codes=[],
                    special_buckets=[], eligible_buckets=[], scene_contexts=[],
                    status=UNCONFIRMED, context_detail="",
                )
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[UNCONFIRMED] += 1
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(index)
    manifest = dict(
        schema=FULL_MANIFEST_SCHEMA, index_schema=FULL_INDEX_SCHEMA, index_file=index.name,
        index_sha256=file_hash(index), mapping_contract_hash=expected,
        mapping_policy=EVENT_BALANCE_MAPPING_POLICY,
        candidate_index=str(candidate_path), candidate_sha256=file_hash(candidate_path),
        action_data_dir=str(action_dir), action_dataset_hashes=action_hashes,
        rows=sum(counts[key] for key in (SPECIAL_ELIGIBLE, SPECIAL_FILTERED, CONFIRMED_REGULAR, UNCONFIRMED)),
        counts=dict(counts), special_buckets=list(SPECIAL_BUCKETS),
        regular_policy="only annotations with explicit regular R-E code and no mapped special context",
        unconfirmed_policy="never sample as REGULAR_BACKGROUND",
        re2_recovery_pending_evidence="unavailable_in_current_source_mapping; context is never emitted",
        identity=digest(dict(mapping_contract_hash=expected, candidate_sha256=file_hash(candidate_path), counts=dict(counts))),
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
