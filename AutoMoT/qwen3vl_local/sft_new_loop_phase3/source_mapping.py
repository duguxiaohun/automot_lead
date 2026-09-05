"""Phase3 独立语义适配层：读取既有人工证据，不修改 Phase1/2 标签或提示词。"""
from __future__ import annotations

from functools import lru_cache
import json
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

from qwen3vl_local.sft_new_loop_phase2.highway_ue3_audit import load_highway_ue3_decisions
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import resolve_context_ids

ROOT = Path(__file__).resolve().parents[2]
ANSWER_TABLE = ROOT / "keyframe_filter/collection_output/phase1_four_question_audit/phase1_four_question_answer_table.json"
HIGHWAY_DECISIONS = ROOT / "qwen3vl_local/sft_new_loop_phase2/highway_ue3_rgb_decisions_v1.jsonl"
REVIEW_DECISIONS = Path(__file__).with_name("mapping_rgb_decisions_v2.jsonl")
EVENT_ADDITIONS = Path(__file__).with_name("event_rgb_additions_v1.jsonl")


@lru_cache(maxsize=1)
def mapping_contract_hash():
    """训练索引绑定实际语义决定；旧索引不能绕过新隔离/同 RS 负例规则。"""
    paths = (ANSWER_TABLE, HIGHWAY_DECISIONS, REVIEW_DECISIONS, EVENT_ADDITIONS,
             Path(__file__).with_name('same_rs_invalid_review_v1.jsonl'))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_mapping_contract(row):
    if row.get('mapping_contract_hash') != mapping_contract_hash():
        raise ValueError('mapping contract mismatch; rebuild Phase3 index with current RGB decisions')


@lru_cache(maxsize=1)
def event_additions():
    """逐帧原始 RGB 明确确认的漏标；只作用于 Phase3，保留源标签用于追溯。"""
    rows = [json.loads(line) for line in EVENT_ADDITIONS.read_text().splitlines() if line.strip()]
    for row in rows:
        if row["event"] != "U-E3" or row["decision"] != "ADD_EVENT":
            raise ValueError("unsupported Phase3 visual event addition")
    return rows


@lru_cache(maxsize=1)
def review_decisions():
    """人工隔离决定仅作用于明确 route/frame，不泛化到同名场景其他路线。"""
    rows = [json.loads(line) for line in REVIEW_DECISIONS.read_text().splitlines() if line.strip()]
    return rows


@lru_cache(maxsize=1)
def evidence_tables():
    """缺失上游证据时显式失败，避免静默退回旧映射。"""
    table = json.loads(ANSWER_TABLE.read_text())
    signals = {(r["scenario"], r["rs"], r["event"]): r["answers"]["TRAFFIC_LIGHT_ABNORMAL"]
               for r in table["rows"]}
    highway, _ = load_highway_ue3_decisions(HIGHWAY_DECISIONS)
    return signals, highway


def mapped_contexts(scenario: str, route_id: str, frame_id: int, rs: str,
                    primary_event: str, events: Sequence[str]):
    """返回候选上下文及来源；原 event_codes 不改写，修正记录另外保存。"""
    signals, highway = evidence_tables()
    confirmed = signals.get((scenario, rs, primary_event))
    codes = list(events)
    evidence = {"source_event_codes": list(events), "signal_failure_answer": confirmed,
                "mapping_version": 2, "rgb_action_review": "candidate_not_frame_verified"}
    for decision in review_decisions():
        if (scenario == decision["scenario"] and route_id == decision["route_id"]
                and decision["start_frame"] <= frame_id <= decision["end_frame"]):
            evidence["rgb_quarantine"] = decision
            return (), evidence
    if (scenario, route_id, frame_id) in highway:
        codes.append("U-E3")
        evidence["ue3_rgb_override"] = highway[(scenario, route_id, frame_id)]
    for decision in event_additions():
        if (scenario == decision["scenario"] and route_id == decision["route_id"]
                and decision["start_frame"] <= frame_id <= decision["end_frame"]):
            codes.append(decision["event"])
            evidence.setdefault("phase3_rgb_event_additions", []).append(decision)
    if "U-E7" in codes and confirmed is not True:
        codes = [c for c in codes if c != "U-E7"]
        evidence["excluded_legacy_u7"] = "no_explicit_signal_failure_evidence"
        # R-E5 只取源标注已明确包含的常规事件；不从无灯/RS5凭空造事件。
    contexts = resolve_context_ids(rs, codes, signal_failure_confirmed=confirmed)
    evidence["active_context_ids"] = list(contexts)
    return contexts, evidence


def context_detail(context_id: str, frames_since_bypass: int | None) -> str:
    """只渲染已发生的历史状态，不渲染未来轨迹或动作真值。"""
    if context_id != "POST_BYPASS_RETURN":
        return ""
    if frames_since_bypass is not None and frames_since_bypass > 0:
        return ("Earlier ego encountered a static blockage in its normal path. Check from the "
                "visible history whether it actually left that lane and whether recovery is still "
                "pending. Waiting for a gap does not end a pending recovery state.")
    return ("No static-obstacle bypass history is asserted. Use visible lane geometry and the "
            "current navigation requirement to distinguish a target lane change from lane keeping.")
