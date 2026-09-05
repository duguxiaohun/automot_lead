"""RGB/真实轨迹审计发现的边界回归；不加载模型、不修改原始数据。"""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import (
    road_structure_from_answers, abnormal_events_from_answers, resolve_context_ids)
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import RouteTrajectory, label_actions
from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapped_contexts
from qwen3vl_local.sft_new_loop_phase3.transition_state import RecoveryState


def test_partial_questions_are_not_false_and_highway_fact_is_independent():
    assert road_structure_from_answers(dict(RS1=False, RS2=False, HIGHWAY=True)) == "UNKNOWN"
    assert road_structure_from_answers(dict(RS1=False, RS2=False, RS4=False, RS5=False, HIGHWAY=False)) == "R3"
    assert road_structure_from_answers(dict(RS1=True, RS_HIGHWAY=True)) == "UNKNOWN"
    assert abnormal_events_from_answers(dict(UE3=True)) == ()
    assert abnormal_events_from_answers(dict(UE3="NO", INVALID_EVENT_CONTEXT=False)) == ()


def test_no_signal_is_not_signal_failure_and_overlay_is_preserved():
    assert resolve_context_ids("R5", ["U-E7", "R-E5"]) == ("UNSIGNALIZED_PRIORITY",)
    assert set(resolve_context_ids("R4", ["U-E6", "U-E7"], signal_failure_confirmed=True)) == {
        "JUNCTION_RULE_CONFLICT", "SIGNAL_FAILURE"}
    assert resolve_context_ids("R4", ["U-E3", "R-E4"]) == ("DYNAMIC_CUTIN",)
    assert resolve_context_ids("R5", ["U-E8", "R-E5"]) == ()


def test_recovery_wait_does_not_clear_pending_state():
    state = RecoveryState()
    state.observe(blockage=True)
    assert state.candidate() is None
    state.observe(departed_lane=True)
    for _ in range(100):
        state.observe()
        assert state.candidate() == "POST_BYPASS_RETURN"
    state.observe(restored_lane=True)
    assert state.candidate() is None


def _trajectory(roads, lanes, speeds=None):
    metas = {i: dict(road_id=r, lane_id=l, lane_type_str="Driving", speed=(speeds or [5]*len(roads))[i])
             for i, (r, l) in enumerate(zip(roads, lanes))}
    return RouteTrajectory(Path("unused"), tuple(metas), metas, {roads[0]: lanes[0]})


def test_no_lane_change_from_single_frame_id_flicker_or_road_reentry():
    assert _trajectory([4]*15, [-1, -1, 1] + [-1]*12).lane_change(0) is None
    assert _trajectory([4, 7] + [4]*13, [-1, -2] + [1]*13).lane_change(0) is None
    assert _trajectory([4]*15, [-1]*4 + [1]*11).lane_change(0) == "LEFT"
    # 同一路段再次进入时，以本次进入方向为准；不能永久沿用初次的负 lane。
    t = _trajectory([4, 7] + [4]*14, [-1, -2] + [1]*4 + [2]*10)
    assert t.lane_change(2) == "RIGHT"


def test_stationary_numerical_negative_is_not_missing_future():
    t = _trajectory([4]*15, [-1]*15, [0, -.00022] + [.01]*13)
    assert len(t._future_speeds(0, 8)) == 9
    assert t._future_speeds(0, 8)[1] == 0


@pytest.mark.parametrize("speeds,expected", [
    ([6.8978, 6.3710, 3.0601, 3.3282, 1.8781, .2138, .3079, .3162, .3496], "STOP"),
    ([.6, .2, .1, .1, .5, 1, 2, 3, 4], "STOP"),
    ([0, .2, .8, 1.5, 2, 3, 4, 5, 6], "RESUME"),
    ([8, 6, 5, 5, 6, 7, 8, 9, 10], "DECELERATE"),
])
def test_ordered_longitudinal_evidence(speeds, expected):
    labels = label_actions(dict(future_speed_count=9, future_speeds=speeds))
    assert labels[expected]
    assert sum(labels[k] for k in ("STOP", "RESUME", "DECELERATE")) == 1


def test_real_legacy_u7_uses_audited_signal_answer():
    contexts, evidence = mapped_contexts("OppositeVehicleTakingPriority", "test_route", 40,
                                        "R5", "U-E7", ["U-E7", "R-E5"])
    assert contexts == ("UNSIGNALIZED_PRIORITY",)
    assert evidence["signal_failure_answer"] is False


def test_rgb_lane_identity_conflict_is_unknown_not_a_lane_change():
    from qwen3vl_local.sft_new_loop_phase3.lateral_rgb_audit import _decisions, lateral_uncertainty
    row = _decisions()[0]
    assert lateral_uncertainty(row['scenario'], row['route_id'], 86)
    assert lateral_uncertainty(row['scenario'], row['route_id'], 97)
    assert lateral_uncertainty(row['scenario'], row['route_id'], 98) is None
    t = _trajectory([355]*15, [-1]*12+[-2]*3)
    t.run_dir = Path(row['scenario'])/row['route_id']
    t.metas = {k+86:v for k,v in t.metas.items()}
    assert t.lane_change(86) is None


@pytest.mark.parametrize("lane_type", ["Shoulder", "Parking", None])
def test_non_driving_projection_cannot_become_lane_change_supervision(lane_type):
    t = _trajectory([355]*15, [-1]*4 + [-2]*11)
    t.metas[4]["lane_type_str"] = lane_type
    assert t.lateral_window_issue(0) == "non_driving_or_unknown_waypoint"
    assert t.lane_change(0) is None
    assert t.signals(0)["lateral_observation_complete"] is False


def test_past_shoulder_projection_does_not_flip_borrowed_lane_direction():
    t = _trajectory([4]*18, [-1, -2] + [1]*4 + [-1]*12)
    t.metas[1]["lane_type_str"] = "Shoulder"
    # 当前和未来窗口全是 Driving，回溯方向时跳过过去的 Shoulder 身份。
    assert t.lane_change(2) == "RIGHT"


@pytest.mark.parametrize("speeds,expected", [
    ([5.725, 5.500, 7.031, 6.729, 1.869, 1.086, 1.013, .559, .153], "DECELERATE"),
    ([6.250, 5.442, 5.808, 7.813, 6.636, 5.468, 5.826, 8.016, 6.964], None),
    ([7.407, 8.163, 9.308, 8.372, 7.196, 7.515, 8.662, 8.825, 7.885], None),
    ([6.157, 5.895, 7.166, 7.977, 6.547, 5.447, 5.686, 8.186, 6.912], None),
    ([8.145, 8.638, 8.239, 8.303, 8.564, 8.116, 8.798, 7.857, 6.357], "DECELERATE"),
    ([3.807, 2.896, 3.944, 5.453, 2.241, 4.222, 5.578, 7.461, 9.039], "DECELERATE"),
    ([0, .002, 1.522, 2.845, 4.649, 5.584, 5.063, 4.819, 4.655], "RESUME"),
    ([0, .002, 1.559, 2.927, 4.754, 5.825, 5.526, .462, 1.544], "RESUME"),
    ([0, 0, 3, .1, .1, 4, 5, 6, 7], "STOP"),
])
def test_reviewed_speed_pulses_do_not_override_real_braking(speeds, expected):
    labels = label_actions(dict(future_speed_count=9, future_speeds=speeds))
    selected = [k for k in ("DECELERATE", "STOP", "RESUME") if labels[k]]
    assert selected == ([] if expected is None else [expected])


def test_phase3_only_rgb_cutin_addition_has_exact_boundaries():
    route = "Town13_Rep0_1008_0_route0_01_08_19_33_44"
    for frame in (96, 97, 98):
        contexts, evidence = mapped_contexts("ParkingCutIn", route, frame, "R1", "R-E1", ["R-E1"])
        assert contexts == ("DYNAMIC_CUTIN",)
        assert evidence["source_event_codes"] == ["R-E1"]
        assert evidence["phase3_rgb_event_additions"]
    assert mapped_contexts("ParkingCutIn", route, 99, "R1", "R-E1", ["R-E1"])[0] == ()
    assert mapped_contexts("ParkingCutIn", "another_route", 96, "R1", "R-E1", ["R-E1"])[0] == ()


def test_dispatch_preserves_concurrency_and_does_not_invent_regular_gate():
    from qwen3vl_local.sft_new_loop_phase3.dispatch import plan_requests, action_response_status
    assert plan_requests(dict(RS1=False)).recheck == ("ROAD_STRUCTURE",)
    assert plan_requests(dict(RS1=True, UE3=True)).recheck == ("PHASE2_EVENT_CONTEXT",)
    plan = plan_requests(dict(RS4=True, TRAFFIC_LIGHT_ABNORMAL=True, UE6=True,
                             INVALID_EVENT_CONTEXT=False))
    assert set(plan.context_ids) == {"SIGNAL_FAILURE", "JUNCTION_RULE_CONFLICT"}
    plan = plan_requests(dict(RS5=True, TRAFFIC_LIGHT_ABNORMAL=False), regular_gates={"R-E2": False})
    assert plan.context_ids == ()
    assert "REGULAR_GATE:R-E5" in plan.recheck
    assert plan_requests(dict(RS5=True), regular_gates={"R-E5": True}).context_ids == ("UNSIGNALIZED_PRIORITY",)
    state = RecoveryState()
    state.observe(blockage=True, departed_lane=True)
    assert plan_requests(dict(RS1=True), recovery=state).context_ids == ("POST_BYPASS_RETURN",)
    assert action_response_status(dict(STOP=False, DECELERATE=False, RESUME=False,
                                      INVALID_ACTION_CONTEXT=False)) == "NO_LISTED_ACTION"
    assert state.recovery_pending
    assert action_response_status(dict(INVALID_ACTION_CONTEXT=True)) == "RECHECK_CONTEXT"
    with pytest.raises(ValueError):
        plan_requests(dict(RS1="NO"))


def test_old_action_index_requires_rebuild():
    from qwen3vl_local.sft_new_loop_phase3.trajectory_action import validate_action_rule
    with pytest.raises(ValueError, match="rebuild"):
        validate_action_rule(dict(action_evidence=dict(rule_version="ordered_speed_stable_lane_v3")))


def test_candidate_recheck_does_not_assert_regular_identity_or_clear_recovery():
    from qwen3vl_local.sft_new_loop_phase3.dispatch import plan_candidate_requests, candidate_response
    plan = plan_candidate_requests(dict(RS1=False, RS2=False, RS4=False, RS5=False))
    assert plan.established_contexts == ()
    assert set(plan.probe_contexts) == {'POST_BYPASS_RETURN', 'RAMP_MERGE_EXIT'}
    assert 'SIGNAL_FAILURE' not in plan_candidate_requests(dict(RS5=True)).probe_contexts
    state = RecoveryState()
    state.observe(blockage=True, departed_lane=True)
    assert 'POST_BYPASS_RETURN' in plan_candidate_requests(dict(RS2=True), recovery=state,
                                                          navigation_lane_transition=False).probe_contexts
    no = dict(DECELERATE=False, STOP=False, RESUME=False, LANE_CHANGE_LEFT=False,
              LANE_CHANGE_RIGHT=False, INVALID_ACTION_CONTEXT=False)
    assert candidate_response('POST_BYPASS_RETURN', no) == 'NOT_REJECTED_NO_ACTION'
    assert candidate_response('POST_BYPASS_RETURN', {**no, 'INVALID_ACTION_CONTEXT': True}) == 'CANDIDATE_REJECTED'
    assert state.recovery_pending
    assert candidate_response('POST_BYPASS_RETURN', {'INVALID_ACTION_CONTEXT': False}) == 'RECHECK_FORMAT'
    assert candidate_response('POST_BYPASS_RETURN', {**no, 'STOP': True, 'INVALID_ACTION_CONTEXT': True}) == 'RECHECK_FORMAT'


def test_changed_mapping_rejects_stale_index():
    from qwen3vl_local.sft_new_loop_phase3.source_mapping import validate_mapping_contract, mapping_contract_hash
    validate_mapping_contract(dict(mapping_contract_hash=mapping_contract_hash()))
    with pytest.raises(ValueError, match='rebuild'):
        validate_mapping_contract(dict(mapping_contract_hash='old_decisions'))


def test_same_rs_hard_negative_survives_runtime_sampling():
    from types import SimpleNamespace
    import random
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
    from qwen3vl_local.sft_new_loop_phase3.invalid_balance import balanced_invalid_items, invalid_subgroup_report
    rows = [SimpleNamespace(true_rs=rs, context_id=asked, invalid_reason='wrong_road_structure',
            invalid_source=f'source={src}|true_rs={rs}|asked_context={asked}')
            for src in CONTEXT_IDS for rs in ('R1','R2','R3','R4','R5') for asked in CONTEXT_IDS]
    rows.append(SimpleNamespace(**{**vars(rows[0]), 'invalid_reason': 'same_rs_wrong_event'}))
    for seed in range(20):
        selected = balanced_invalid_items(rows, target=64, rng=random.Random(seed))
        report = invalid_subgroup_report(selected)
        assert report['reason']['counts']['same_rs_wrong_event'] >= 1
        assert all(report['guards'].values())


def test_invalid_seed_coverage_counts_toward_signature_quota():
    from types import SimpleNamespace
    import random
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
    from qwen3vl_local.sft_new_loop_phase3.invalid_balance import balanced_invalid_items, invalid_subgroup_report
    rows = []
    for i, source in enumerate(CONTEXT_IDS):
        rs = ("R1", "R2", "R3", "R4", "R5")[i % 5]
        # 少量签名的源桶比全笛卡尔积更容易复现 seed 被重复分配的问题。
        for j in range(5):
            asked = CONTEXT_IDS[(i+j) % len(CONTEXT_IDS)]
            rows.append(SimpleNamespace(true_rs=rs, context_id=asked,
                invalid_source=f"source={source}|true_rs={rs}|asked_context={asked}"))
    for seed in range(20):
        sampled = balanced_invalid_items(rows, target=64, rng=random.Random(seed))
        report = invalid_subgroup_report(sampled)
        assert report["guards"]["joint_signature_within_source_max_deviation_le_1"]
    with pytest.raises(ValueError, match="missing_sources"):
        balanced_invalid_items(rows[5:], target=64, rng=random.Random(0))
