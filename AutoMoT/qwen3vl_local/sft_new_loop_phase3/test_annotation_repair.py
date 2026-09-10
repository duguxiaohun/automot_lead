"""9月10日逐帧审计确认的问题与关键反例。"""
from pathlib import Path

from keyframe_filter.collector import RoadEventRuleEngine, RoadStructureRuleEngine, RoadStructure
from keyframe_filter.evidence_guards import has_signal_support
from qwen3vl_local.sft_new_loop_phase3.annotation_repair import repair_annotation
from qwen3vl_local.sft_new_loop_phase3.build_dataset import _split
from qwen3vl_local.sft_new_loop_phase3.lateral_rgb_audit import lateral_uncertainty
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import RouteTrajectory


def test_generic_static_cutin_trigger_is_not_ramp_but_real_merge_survives():
    evidence = dict(trigger_distance_m=15.21, xodr=dict(
        ramp_merge_split_hint=False, xodr_topology_trusted=False))
    assert not RoadEventRuleEngine._highway_r3_core_event_active(
        "StaticCutIn", dict(scenario_active=True), evidence)[0]
    assert RoadEventRuleEngine._highway_r3_core_event_active(
        "EnterActorFlow", dict(scenario_active=True), evidence)[0]
    evidence["xodr"].update(ramp_merge_split_hint=True)
    assert not RoadEventRuleEngine._highway_r3_core_event_active("StaticCutIn", {}, evidence)[0]
    evidence["xodr"].update(xodr_topology_trusted=True)
    assert RoadEventRuleEngine._highway_r3_core_event_active("StaticCutIn", {}, evidence)[0]


def test_hazard_alone_unknown_and_reviewed_span_exact_override():
    ann = dict(evidence=dict(rules_fired=["r4_light_hazard"]))
    run = "Town13_Rep0_1586_9_route0_01_08_19_01_08"
    assert not has_signal_support(False, False, False)
    assert has_signal_support(False, True, False)
    for f in (44, 56):
        rs, primary, codes, trace = repair_annotation("HighwayCutIn", run, f,
            ann, "R4", "R-E4", ["R-E4"])
        assert rs == "R3" and "R-E4" not in codes
        assert trace["source"]["rs"] == "R4"
    assert repair_annotation("HighwayCutIn", run, 57, ann, "R4", "R-E4", ["R-E4"])[0] == "UNKNOWN"
    ann["evidence"]["diagnostic_attribution"] = dict(used_inputs=dict(meta_traffic_light_valid=True))
    assert repair_annotation("HighwayCutIn", "another", 44, ann, "R4", "R-E4", ["R-E4"])[0] == "R4"


def test_remove_only_unsupported_regular_event_preserve_concurrent_cutin():
    ann = dict(event_evidence=dict(rules_fired=["event_highway_trigger_core_r3"]))
    result = repair_annotation("StaticCutIn", "r", 7, ann, "R3", "R-E3", ["R-E3", "U-E3"])
    assert result[2] == ("U-E3",)
    assert result[3]["source"]["event_codes"] == ["R-E3", "U-E3"]


def test_rgb_disproved_lane_id_never_becomes_negative_supervision():
    run = "Town13_Rep0_1396_11_route0_01_09_23_28_37"
    assert lateral_uncertainty("HighwayExit", run, 87)
    assert not lateral_uncertainty("HighwayExit", "another", 87)
    assert _split("HighwayExit", run, 20260819, .1, .05) == "train"
    metas = {i: dict(road_id=1, lane_id=-1, lane_type_str="Driving",
                    section_id=int(i >= 2)) for i in range(14)}
    route = RouteTrajectory(Path("s/r"), tuple(metas), metas, {})
    assert route.lateral_window_issue(0) == "lane_section_transition"
    assert route.lane_change(0) is None


def test_nonadjacent_same_side_jump_unknown_but_centerline_borrow_valid():
    metas = {i: dict(road_id=1, lane_id=-4 if i < 2 else -2,
                    lane_type_str="Driving") for i in range(14)}
    route = RouteTrajectory(Path("s/r"), tuple(metas), metas, {})
    assert route.lateral_window_issue(0) == "nonadjacent_lane_identity_jump"
    for i, meta in metas.items():
        meta["lane_id"] = -1 if i < 2 else 1
    assert route.lateral_window_issue(0) is None
    assert route.lane_change(0) == "LEFT"


def test_full_rs_engine_ignores_hazard_but_keeps_confirmed_light():
    # 与 #518 相同的触发条件：junction flag 可出现在高速连接段。
    frame = dict(is_junction=True, light_hazard=True, traffic_light_state="None",
                 active_scenarios=["HighwayCutIn"], town="Town13", pos_global=[0,0])
    engine = RoadStructureRuleEngine()
    result = engine.analyze("HighwayCutIn", 44, frame)
    assert result[0] != RoadStructure.R4
    assert "r4_light_hazard" not in result[3]["rules_fired"]
    frame["traffic_light_state"] = "Red"
    assert engine.analyze("HighwayCutIn", 44, frame)[0] == RoadStructure.R4
