"""旧 collection 的可追溯读时修复；原文件与 Phase1/2 标签保持可回查。"""
from functools import lru_cache
import json
from pathlib import Path

from keyframe_filter.evidence_guards import (
    has_signal_support, merge_trigger_supported, local_junction_supported,
    dynamic_cutin_actor_unverified)


@lru_cache(maxsize=1)
def decisions():
    return [row for name in ("annotation_repairs_20260910.json", "annotation_repairs_20260914.json")
            for row in json.loads(Path(__file__).with_name(name).read_text())]


def repair_annotation(scenario, route_id, frame_id, ann, rs, primary, events):
    """返回修正字段及证据；未知 RS 返回 UNKNOWN，不能凭次高分猜成 R3。"""
    evidence = ann.get("evidence") or {}
    event = ann.get("event_evidence") or ann.get("frame_event_annotation") or {}
    rules = set(evidence.get("rules_fired") or [])
    erules = set(event.get("rules_fired") or [])
    xodr = evidence.get("xodr") or {}
    used = (evidence.get("diagnostic_attribution") or {}).get("used_inputs") or {}
    metrics = (evidence.get("bbox_semantics") or {}).get("metrics") or {}
    independent_signal = has_signal_support(
        used.get("meta_traffic_light_valid"), metrics.get("traffic_light_affects_ego"),
        xodr.get("xodr_topology_trusted") and
        (evidence.get("diagnostic_attribution", {}).get("window_flags", {}).get("static_signal_near")))
    changes = []
    codes = list(events)
    original = dict(rs=rs, primary_event=primary, event_codes=list(events))
    # 只撤回源记录明确记为 R1 -> R4 的弱恢复，不从候选分数猜道路类别。
    weak_recovery = any(r.get("from") == "R1" and r.get("to") == "R4"
                        and r.get("reason") == "stable_meta_light_with_untrusted_xodr"
                        for r in evidence.get("r4_context_recovery", []))
    if rs == "R4" and weak_recovery and not local_junction_supported(evidence):
        rs = "R1"
        codes = [c for c in codes if c != "R-E4"]
        if primary == "R-E4":
            primary = "UNKNOWN"
        changes.append("undo_signal_recovery_without_local_junction")
    if rs == "R4" and "r4_light_hazard" in rules and not independent_signal:
        rs = "UNKNOWN"
        changes.append("quarantine_hazard_only_signalized_road")
    for row in decisions():
        if (row["scenario"] == scenario and row["route_id"] == route_id
                and row["start_frame"] <= frame_id <= row["end_frame"]):
            rs = row["road_structure"]
            # 原 R-E4 依附于错误道路，不移植成另一个 regular 事件；U-E3 由已有视觉表加入。
            codes = [c for c in codes if c != "R-E4"]
            if primary == "R-E4":
                primary = "UNKNOWN"
            changes.append(row["decision"])
    trusted_ramp = xodr.get("xodr_topology_trusted") and xodr.get("ramp_merge_split_hint")
    if ("R-E3" in codes and "event_highway_trigger_core_r3" in erules
            and not merge_trigger_supported(scenario, trusted_ramp)
            and not erules.intersection({"event_highway_actor_flow_core_r3",
                                        "event_highway_merge_approach_r3"})):
        codes.remove("R-E3")
        if primary == "R-E3":
            primary = "UNKNOWN"
        changes.append("remove_generic_trigger_only_ramp_event")
    review_reasons = (["dynamic_cutin_actor_unverified"]
        if dynamic_cutin_actor_unverified(scenario, erules, event.get("metrics") or {}) else [])
    return rs, primary, tuple(codes), dict(version="rgb_evidence_repair_20260914_v2",
        source=original, changes=changes, review_reasons=review_reasons,
        repaired_rs=rs, repaired_event_codes=codes)
