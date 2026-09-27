"""风险响应证据不能重写运动标签，也不能泄漏进任一题型的模型输入。"""
import json

import pytest

from qwen3vl_local.sft_new_loop_phase3.action_review import build_action_review, controller_snapshot
from qwen3vl_local.sft_new_loop_phase3.audit_label_boundaries import diagnose, audit
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS, CONTEXT_BY_ID
from qwen3vl_local.sft_new_loop_phase3.prompts import make_prompt_spec, build_action_prompt, build_action_target
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import RouteTrajectory, label_actions


def route(tmp_path, speeds=None):
    speeds = [8]*14 if speeds is None else speeds
    metas = {i: dict(speed=v, road_id=1, lane_id=-1, lane_type_str='Driving',
                    pos_global=[i,0], theta=0, next_target_points=[[100,0]],
                    brake=True, vehicle_hazard=True, target_speed=0,
                    speed_reduced_by_obj_id=42, scenario_actors_ids=[42])
             for i,v in enumerate(speeds)}
    return RouteTrajectory(tmp_path/'synthetic'/'r', tuple(metas), metas, {1:-1})


def test_unknown_controller_values_do_not_become_no_hazard_or_released_object():
    result = controller_snapshot(dict(brake=1, vehicle_hazard='False', target_speed=float('nan'),
                                     speed_reduced_by_obj_id=True))
    assert result['brake'] is result['vehicle_hazard'] is result['target_speed'] is None
    assert result['speed_reduced_by_obj_id'] is None
    assert result['object_membership']['scenario_actors_ids'] is None
    assert 'brake' in result['unknown_fields']
    result = controller_snapshot(dict(brake=False, target_speed=0, speed_reduced_by_obj_id=42,
                                     scenario_actors_ids=[42], scenario_obstacles_ids=[]))
    assert result['brake'] is False and result['target_speed'] == 0
    assert result['object_membership']['scenario_actors_ids'] is True
    assert result['object_membership']['scenario_obstacles_ids'] is False


def test_zero_target_with_braking_is_reviewable_keep_not_achieved_stop(tmp_path):
    trajectory = route(tmp_path)
    before = label_actions(trajectory.signals(0))
    report, after = diagnose(trajectory, 0, 'ONCOMING_INVASION')
    assert before == after == dict.fromkeys(ACTION_KEYS, False)
    assert report['primary_action'] == 'KEEP' and report['flags'] == []
    assert set(report['response_flags']) == {'keep_with_brake_command', 'keep_with_vehicle_stop_request'}
    assert report['action_review']['motion_milestones'] == []
    stopped, labels = diagnose(route(tmp_path, [0]*14), 0, 'ONCOMING_INVASION')
    assert stopped['primary_action'] == 'STOP' and labels['STOP']
    assert stopped['response_flags'] == []


def test_recorded_release_requires_known_contiguous_evidence_and_bounded_confirmation(tmp_path):
    trajectory = route(tmp_path)
    for i in range(3,14):
        trajectory.metas[i].update(target_speed=8, speed_reduced_by_obj_id=None)
    def review():
        return diagnose(trajectory,0,'ONCOMING_INVASION')[0]['action_review']
    result = review()
    assert result['positive_target_after_zero'] == dict(start_s=.75, confirmed_s=1.)
    assert result['logged_limiter_change'] == dict(at_s=.75, from_id=42, to_id=None)
    del trajectory.metas[2]['target_speed']
    del trajectory.metas[2]['speed_reduced_by_obj_id']
    result = review()
    assert result['positive_target_after_zero'] is None and result['logged_limiter_change'] is None
    assert result['target_search_known'] is False
    trajectory = route(tmp_path)
    trajectory.metas[12]['target_speed'] = trajectory.metas[13]['target_speed'] = 8
    assert review()['positive_target_after_zero'] is None


def test_motion_milestones_preserve_preparation_before_selected_crossing(tmp_path):
    trajectory = route(tmp_path,[4,5,2,2,3,4,4,4,5,5,5,5,5,5])
    for i in range(8,14):
        trajectory.metas[i]['lane_id'] = 1
    report, labels = diagnose(trajectory,0,'STATIC_BLOCKAGE')
    result = report['action_review']
    assert report['primary_action'] == 'LANE_CHANGE_LEFT'
    assert labels['DECELERATE'] and labels['LANE_CHANGE_LEFT']
    assert result['speed_action_start_s'] == .5 and result['crossing_start_s'] == 2.
    assert result['motion_milestones'][0]['action'] == 'DECELERATE'
    assert result['milestone_semantics'] == 'bounded_rule_triggers_not_complete_stage_sequence'


@pytest.mark.parametrize('before,after', [(None,3739),(42,None),(42,3739)])
def test_first_limiter_transition_includes_acquisition_release_and_replacement(tmp_path,before,after):
    trajectory = route(tmp_path)
    for i,meta in trajectory.metas.items():
        meta['speed_reduced_by_obj_id'] = before if i < 3 else after
    # 后续变化不能覆盖首次变化。
    trajectory.metas[5]['speed_reduced_by_obj_id'] = 99
    result,labels = diagnose(trajectory,0,'DYNAMIC_CUTIN')
    assert result['action_review']['logged_limiter_change'] == dict(at_s=.75,from_id=before,to_id=after)
    assert result['primary_action'] == 'KEEP' and labels == dict.fromkeys(ACTION_KEYS,False)


@pytest.mark.parametrize('unknown_frame', [0,1])
@pytest.mark.parametrize('unknown_value', ['missing',-1,True,'3739'])
def test_unknown_anchor_or_gap_never_implies_limiter_acquisition(tmp_path,unknown_frame,unknown_value):
    trajectory = route(tmp_path)
    for i,meta in trajectory.metas.items():
        meta['speed_reduced_by_obj_id'] = None if i < 3 else 3739
    if unknown_value == 'missing':
        del trajectory.metas[unknown_frame]['speed_reduced_by_obj_id']
    else:
        trajectory.metas[unknown_frame]['speed_reduced_by_obj_id'] = unknown_value
    result,_ = diagnose(trajectory,0,'DYNAMIC_CUTIN')
    assert result['action_review']['logged_limiter_change'] is None


@pytest.mark.parametrize('context_id', CONTEXT_BY_ID)
@pytest.mark.parametrize('mode', ('choice','binary'))
def test_keep_risk_meaning_shared_without_review_metadata_in_prompt(context_id,mode):
    context = CONTEXT_BY_ID[context_id]
    spec = make_prompt_spec(variant='all_random_order',answers=dict.fromkeys(ACTION_KEYS,False),
                            seed_key='response',context_id=context_id,
                            road_structure=context.allowed_rs[0],action_output_mode=mode)
    prompt = build_action_prompt(spec=spec)
    assert 'Brief braking can occur within this stage' in prompt
    assert 'KEEP does not imply that traffic risks have cleared' in prompt
    for private in ('target_speed','speed_reduced_by_obj_id','motion_milestones','review_only','1.5 s'):
        assert private not in prompt
    target = build_action_target(spec)
    assert target == 'KEEP' if mode == 'choice' else 'KEEP: YES' in target


def test_response_eval_bins_keep_original_boundary_denominator(tmp_path, monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import audit_label_boundaries as module
    monkeypatch.setattr(module, 'load_route_trajectory', lambda _: route(tmp_path))
    monkeypatch.setattr(module, 'is_abnormal_lead_route', lambda *args: (False,{}))
    row = dict(scenario='synthetic',route_id='r',frame_id=0,context_id='ONCOMING_INVASION',rs='R1')
    index = tmp_path/'index.jsonl'; index.write_text(json.dumps(row))
    cases = tmp_path/'eval.jsonl'; cases.write_text(json.dumps({**row,'all_ok':False}))
    result = audit(index,tmp_path,tmp_path/'out',cases)
    assert result['bins']['action/KEEP']['any_boundary_flag'] == 0
    assert result['response_bins']['action/KEEP']['keep_with_vehicle_stop_request'] == 1
    assert result['eval_response_bins']['keep_with_vehicle_stop_request']['exact_accuracy'] == 0
    assert result['eval_bins']['no_boundary_flag']['samples'] == 1
    assert result['labels_modified'] is False
    review = diagnose(route(tmp_path),0,'ONCOMING_INVASION')[0]['action_review']
    index.write_text(json.dumps({**row,'action_review':{**review,'applies_to_prompt_context':True}}))
    assert audit(index,tmp_path,tmp_path/'verified')['counts']['audited_unique_valid'] == 1
    review['controller']['brake'] = False
    index.write_text(json.dumps({**row,'action_review':review}))
    with pytest.raises(ValueError, match='index/raw action review mismatch'):
        audit(index,tmp_path,tmp_path/'tampered')
