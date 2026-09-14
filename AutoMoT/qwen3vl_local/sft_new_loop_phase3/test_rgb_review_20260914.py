"""9月14日真实RGB错例对应的空间证据、未知隔离和时序合同回归。"""
import copy
import json
from pathlib import Path

from keyframe_filter.collector import ScenarioCollector
from keyframe_filter.evidence_guards import local_junction_supported, dynamic_cutin_actor_unverified
from qwen3vl_local.sft_new_loop_phase3.annotation_repair import repair_annotation
from qwen3vl_local.sft_new_loop_phase3.build_dataset import _split
from qwen3vl_local.sft_new_loop_phase3.lateral_rgb_audit import lateral_uncertainty
from qwen3vl_local.sft_new_loop_phase3.prompts import (
    SPEED_ACTION_RULES, make_prompt_spec, build_action_prompt)
from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapped_contexts


def _weak_recovery():
    return dict(evidence=dict(rules_fired=['r4_context_recovery_stable_meta_light_untrusted_xodr'],
        r4_context_recovery=[dict(from_='unused', **{'from':'R1','to':'R4',
            'reason':'stable_meta_light_with_untrusted_xodr'})],
        xodr=dict(xodr_topology_trusted=False, map_is_junction=True),
        diagnostic_attribution=dict(used_inputs=dict(meta_traffic_light_valid=True),
            window_flags=dict(strong_control_context=True, close_trigger_for_junction=True))))


def test_light_and_trigger_cannot_restore_junction_or_drop_independent_events():
    ann = _weak_recovery()
    rs, primary, codes, trace = repair_annotation('DynamicObjectCrossing','r',42,
        ann,'R4','U-E4',['U-E4','R-E4'])
    assert (rs,primary,codes) == ('R1','U-E4',('U-E4',))
    assert trace['source']['rs'] == 'R4'
    # 确认的本地几何保留；没有R1恢复来源也不能凭缺证据反推R1。
    ann['evidence']['diagnostic_attribution']['used_inputs']['bbox_junction_hint'] = True
    assert repair_annotation('s','r',42,ann,'R4','U-E4',['U-E4','R-E4'])[0] == 'R4'
    ann['evidence'].pop('r4_context_recovery')
    ann['evidence']['diagnostic_attribution']['used_inputs']['bbox_junction_hint'] = False
    assert repair_annotation('s','r',42,ann,'R4','U-E4',['U-E4','R-E4'])[0] == 'R4'


def test_collector_does_not_recover_whole_street_from_repeated_red_light():
    ann = dict(frame_id=0, primary_road_structure='R1', evidence=dict(
        traffic_light_state='Red', rules_fired=['r4_meta_tl_without_control_context_demoted_to_r1'],
        xodr=dict(xodr_topology_trusted=False), diagnostic_attribution=dict(window_flags={})))
    rows = [dict(copy.deepcopy(ann), frame_id=i) for i in range(40)]
    collector = ScenarioCollector.__new__(ScenarioCollector)
    result = collector._apply_r4_context_recovery('DynamicObjectCrossing',rows)
    assert not result['changes']
    assert all(r['primary_road_structure']=='R1' for r in rows)
    assert local_junction_supported(dict(diagnostic_attribution=dict(window_flags=dict(near_junction=True))))


def test_hazard_ambiguity_requires_review_and_does_not_fabricate_cutin_negative():
    metrics = dict(vehicle_hazard=True, brake_cutin=False, dist_to_cutin_vehicle=None)
    rules = ['event_dynamic_cutin_or_occupancy']
    assert dynamic_cutin_actor_unverified('DynamicObjectCrossing',rules,metrics)
    assert not dynamic_cutin_actor_unverified('DynamicObjectCrossing',rules,dict(metrics,brake_cutin=True))
    ann = dict(event_evidence=dict(rules_fired=rules,metrics=metrics))
    result = repair_annotation('DynamicObjectCrossing','unreviewed',15,ann,'R1','U-E3',['U-E3'])
    assert result[2] == ('U-E3',) and result[3]['review_reasons']
    run='Town02_Rep0_Town02_Scenario3_0_route0_01_10_05_05_30'
    assert not mapped_contexts('DynamicObjectCrossing',run,70,'R1','U-E3',['U-E3'])[0]
    assert mapped_contexts('DynamicObjectCrossing',run,66,'R1','U-E3',['U-E3'])[0]
    assert mapped_contexts('DynamicObjectCrossing',run,70,'R1','U-E4',['U-E4'])[0]


def test_reviewed_topology_conflict_is_unknown_only_when_transition_is_future():
    run='Town03_Rep0_route_001890_route0_01_10_02_51_06'
    assert lateral_uncertainty('ParkedObstacle',run,44)
    assert not lateral_uncertainty('ParkedObstacle',run,49)
    assert not lateral_uncertainty('ParkedObstacle','different',44)


def test_shared_speed_rules_and_all_exposed_test_routes_train_only():
    for mode in ('binary','choice'):
        spec=make_prompt_spec(variant='all_random_order',answers={},seed_key='test',
            context_id='LEAD_BRAKE',road_structure='R1',current_speed_mps=5,goal_xy=(42,0),
            action_output_mode=mode)
        assert SPEED_ACTION_RULES in build_action_prompt(spec=spec)
    rows=json.loads(Path(__file__).with_name('development_route_groups_20260914.json').read_text())
    assert rows['case_count']==552
    for group in rows['groups']:
        scenario,route=group.split('/',1)
        assert _split(scenario,route,20260914,.1,.05)=='train'
    notes=[json.loads(l) for l in Path(__file__).with_name('EVAL_RGB_REVIEW_20260914.jsonl').read_text().splitlines()]
    assert len(notes)==202
    for n in notes:
        assert _split(n['scenario'],n['route_id'].replace('_Rep0_','_Rep99_'),1,.99,.01)=='train'
