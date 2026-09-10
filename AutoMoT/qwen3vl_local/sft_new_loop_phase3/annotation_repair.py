"""旧 collection 的可追溯读时修复；原文件与 Phase1/2 标签保持可回查。"""
from functools import lru_cache
import json
from pathlib import Path

from keyframe_filter.evidence_guards import has_signal_support, merge_trigger_supported


@lru_cache(maxsize=1)
def decisions():
    return json.loads(Path(__file__).with_name("annotation_repairs_20260910.json").read_text())


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
    return rs, primary, tuple(codes), dict(version="rgb_evidence_repair_20260910_v1",
        source=original, changes=changes, repaired_rs=rs, repaired_event_codes=codes)
