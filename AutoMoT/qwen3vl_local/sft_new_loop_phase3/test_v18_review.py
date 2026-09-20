"""v18 场景隔离、主要动作语义及只报告不重标的边界审计。"""
import json
from pathlib import Path

import pytest

from qwen3vl_local.sft_new_loop_phase3.audit_label_boundaries import speed_boundaries, diagnose, audit, identity
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS, CONTEXT_BY_ID
from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapped_contexts
from qwen3vl_local.sft_new_loop_phase3.prompts import make_prompt_spec, build_action_prompt, build_action_target
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import RouteTrajectory, longitudinal_decision

ROOT = Path(__file__).parent
DECISIONS = [json.loads(l) for l in (ROOT/'mapping_rgb_decisions_v18_20260920.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('d', DECISIONS)
def test_v18_quarantine_stops_candidates_without_creating_negative(d):
    for frame in range(d['start_frame'], d['end_frame']+1):
        contexts, evidence = mapped_contexts(d['scenario'], d['route_id'], frame, 'R1', 'U-E4', ['U-E4'])
        assert contexts == () and evidence['rgb_quarantine'] == d
        assert 'answers' not in evidence
    for frame in (d['start_frame']-1, d['end_frame']+1):
        assert mapped_contexts(d['scenario'],d['route_id'],frame,'R1','U-E4',['U-E4'])[0]
    assert mapped_contexts(d['scenario'],d['route_id']+'_other',d['start_frame'],'R1','U-E4',['U-E4'])[0]


def test_margin_does_not_relabel_keep_or_use_window_tail():
    speeds = [9.1672116, 8.333, 7.3397538, 7.631, 8.277, 8.759, 8.121, 7.796, 8.732]
    copy = list(speeds)
    result = speed_boundaries(speeds)
    assert result['speed_action'] == 'NONE'
    assert result['drop_margin_mps'] == pytest.approx(0.00598452)
    assert 'near_drop_threshold' in result['flags']
    assert speeds == copy and longitudinal_decision(speeds)['action'] == 'NONE'
    tail = speed_boundaries([8]*9 + [8,5,5,5])
    assert tail['speed_action'] == 'NONE' and tail['outside_drop_s'] == 2.5
    assert 'speed_continuation_drop_just_outside_window' in tail['flags']
    assert not speed_boundaries([8]*8)['eligible']
    assert not speed_boundaries([8]*9+[float('nan')]*4)['tail_complete']


def test_gain_confirmation_and_near_stop_boundary_are_separate():
    result = speed_boundaries([8]*8+[10]+[10]*4)
    assert result['speed_action'] == 'NONE'
    assert 'gain_unconfirmed_at_2s_boundary' in result['flags']
    assert result['outside_gain_confirmed_s'] == 2.25
    stop = speed_boundaries([3]*6+[0,0,0])
    assert 'stop_pair_crosses_1_5s_boundary' in stop['flags']
    assert speed_boundaries([8,10,10,6,6,6,6,6,6])['reason'] == 'mixed_longitudinal_phase'


def synthetic_route(tmp_path):
    speeds = [4.185,5.197,.003,1.698,3.242,5.033,5.455,7.589,8.361,9.245,8.202,7.589,8.441,9.754]
    metas = {i:dict(speed=v, road_id=149, lane_id=(-1 if i < 7 else 1), lane_type_str='Driving',
                    pos_global=[i,0], theta=0, next_target_points=[[100,0]]) for i,v in enumerate(speeds)}
    return RouteTrajectory(tmp_path/'synthetic'/'r', tuple(metas), metas, {149:-1})


def test_preparatory_slowing_is_not_mistaken_for_immediate_crossing(tmp_path):
    result, labels = diagnose(synthetic_route(tmp_path),0,'STATIC_BLOCKAGE')
    assert result['primary_action']=='LANE_CHANGE_LEFT'
    assert labels['DECELERATE'] and labels['LANE_CHANGE_LEFT']
    assert result['speed_action_start_s']==.5 and result['crossing_start_s']==1.75
    assert 'speed_before_crossing' in result['flags']
    for mode in ('choice','binary'):
        spec=make_prompt_spec(variant='all_random_order',answers=labels,seed_key='v18',
            context_id='STATIC_BLOCKAGE',road_structure='R2',action_output_mode=mode)
        prompt=build_action_prompt(spec=spec)
        if mode=='choice':
            assert 'Preparatory slowing may happen before the selected crossing' in prompt
            assert build_action_target(spec)=='LANE_CHANGE_LEFT'
        else:
            assert 'may happen in sequence' in prompt
            assert 'DECELERATE: YES' in build_action_target(spec)


def test_boundary_report_deduplicates_and_scores_matching_cases_only(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import audit_label_boundaries as module
    trajectory=synthetic_route(tmp_path)
    monkeypatch.setattr(module,'load_route_trajectory',lambda _:trajectory)
    monkeypatch.setattr(module,'is_abnormal_lead_route',lambda *args:(False,{}))
    row=dict(scenario='synthetic',route_id='r',frame_id=0,context_id='STATIC_BLOCKAGE',rs='R2')
    index=tmp_path/'index.jsonl';index.write_text('\n'.join(map(json.dumps,[row,row,{**row,'invalid_action_context':True}])))
    cases=tmp_path/'eval.jsonl';cases.write_text(json.dumps({**row,'all_ok':False}))
    result=audit(index,tmp_path,tmp_path/'out',cases)
    assert result['counts']['audited_unique_valid']==1
    assert result['counts']['duplicate_rows_skipped']==1
    assert result['counts']['invalid_rows_skipped']==1
    assert result['eval_bins']['speed_before_crossing']==dict(samples=1,exact_hits=0,exact_accuracy=0)
    assert result['labels_modified'] is False


@pytest.mark.parametrize('context_id', CONTEXT_BY_ID)
def test_all_contexts_keep_conditional_purpose_without_future_diagnostics(context_id):
    context=CONTEXT_BY_ID[context_id]
    for mode in ('choice','binary'):
        spec=make_prompt_spec(variant='all_random_order',answers=dict.fromkeys(ACTION_KEYS,False),seed_key='v18',
            context_id=context_id,road_structure=context.allowed_rs[0],action_output_mode=mode)
        text=build_action_prompt(spec=spec)
        assert 'a scene alone does not establish the cause' in text
        assert 'KEEP allows speed adjustments within the current stage' in text
        if mode == 'choice' and context.question_domain != 'FULL_MANEUVER':
            assert 'Preparatory slowing' not in text and 'crossing takes priority' not in text
        for private in ('drop_margin_mps','speed_before_crossing','0.10','1.5 s','2.0 s','3.0 s'):
            assert private not in text


@pytest.mark.parametrize('mode', ('choice', 'binary'))
def test_re5_resume_covers_sustained_gain_without_waiting(mode):
    """未停车的持续增速符合原标定，两种题型不能从释义额外要求等待。"""
    from qwen3vl_local.sft_new_loop_phase3.choice_semantics import action_description
    decision = longitudinal_decision([4,4.5,5,5.5,6,6.5,7,7.5,8])
    assert decision['action'] == 'RESUME'
    labels = {key: key == decision['action'] for key in ACTION_KEYS}
    spec = make_prompt_spec(variant='all_random_order', answers=labels, seed_key='moving-re5',
                           context_id='UNSIGNALIZED_PRIORITY', road_structure='R5', action_output_mode=mode)
    description = action_description('UNSIGNALIZED_PRIORITY', 'RESUME')
    assert 'forward space opens' in description and 'already moving' in description
    assert 'enter or continue through' in description
    assert description in build_action_prompt(spec=spec)
    target = build_action_target(spec)
    assert target == 'RESUME' if mode == 'choice' else 'RESUME: YES' in target


@pytest.mark.parametrize('road_fields', (
    {'rs':'R2'}, {'prompt_road_structure':'R2'}, {'prompt_spec':{'road_structure':'R2'}},
    {'prompt_road_structure':None, 'rs':'', 'prompt_spec':{'road_structure':'R2'}},
))
def test_boundary_identity_uses_prompted_road_across_schemas(road_fields):
    row = dict(scenario='s',route_id='r',frame_id=0,context_id='STATIC_BLOCKAGE',true_rs='R5')
    assert identity({**row,**road_fields}) == ('s','r',0,'STATIC_BLOCKAGE','R2')
    with pytest.raises(ValueError, match='missing prompted road'):
        identity(row)
    with pytest.raises(ValueError, match='conflicting prompted road'):
        identity({**row,'rs':'R2','prompt_spec':{'road_structure':'R5'}})


def test_boundary_training_schema_join_counts_duplicates_and_missing_cases(tmp_path, monkeypatch):
    """真实训练记录把 RS 放在 prompt_spec；重复题去重，漏配不算模型错误。"""
    from qwen3vl_local.sft_new_loop_phase3 import audit_label_boundaries as module
    trajectory = synthetic_route(tmp_path)
    monkeypatch.setattr(module,'load_route_trajectory',lambda _:trajectory)
    monkeypatch.setattr(module,'is_abnormal_lead_route',lambda *args:(False,{}))
    base = dict(scenario='synthetic',route_id='r',frame_id=0,context_id='STATIC_BLOCKAGE')
    row = {**base,'prompt_road_structure':'R2'}
    index = tmp_path/'index.jsonl'
    index.write_text('\n'.join(map(json.dumps,[row,{**row,'context_id':'LEAD_BRAKE'}])))
    case = {**base,'step':100,'true_rs':'R5','prompt_spec':{'road_structure':'R2'},'all_ok':True}
    cases = tmp_path/'eval.jsonl'
    cases.write_text('\n'.join(map(json.dumps,[case,case,{**case,'frame_id':99}])))
    result = audit(index,tmp_path,tmp_path/'partial',cases)
    assert result['eval_bins']['all'] == dict(samples=1,exact_hits=1,exact_accuracy=1)
    assert result['eval_matching'] == dict(provided=True,input_rows=3,duplicate_rows=1,unique_cases=2,
        matched_unique_cases=1,unmatched_unique_cases=1,audited_without_case=1,status='partial')
    cases.write_text(json.dumps({**case,'frame_id':99}))
    with pytest.warns(RuntimeWarning, match='No evaluation cases matched'):
        result = audit(index,tmp_path,tmp_path/'unmatched',cases)
    assert not result['eval_bins']
    assert result['eval_matching']['status'] == 'no_matches'
    assert result['eval_matching']['unmatched_unique_cases'] == 1
    assert result['eval_matching']['audited_without_case'] == 2
    assert json.loads((tmp_path/'unmatched/summary.json').read_text())['eval_matching'] == result['eval_matching']
    cases.write_text('')
    assert audit(index,tmp_path,tmp_path/'empty',cases)['eval_matching']['status'] == 'empty_cases'
    assert audit(index,tmp_path,tmp_path/'no_eval')['eval_matching']['status'] == 'not_requested'
    cases.write_text('\n'.join(map(json.dumps,[case,{**case,'all_ok':False}])))
    with pytest.raises(ValueError, match='conflicting duplicate eval outcomes'):
        audit(index,tmp_path,tmp_path/'conflict',cases)
