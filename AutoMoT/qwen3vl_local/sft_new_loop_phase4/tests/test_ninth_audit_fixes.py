from qwen3vl_local.sft_new_loop_phase4.tests.safety_fixtures import clearances, loop_clearances
from copy import deepcopy
from collections import Counter
import json
import pytest
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.runtime import Phase4Loop
from qwen3vl_local.sft_new_loop_phase4.prompts import prompt
from qwen3vl_local.sft_new_loop_phase4.sampling import legacy_plan as plan,frame_key
from qwen3vl_local.sft_new_loop_phase4.dataset import coverage_report,groups,split_for
from qwen3vl_local.sft_new_loop_phase4.evaluate import replay
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT


def rows(n,groups=1,strata=1,shared=False):
    return [dict(id=f'{i:04}',episode={'event':'R-E3'},edge=f'edge{i%strata}',target='YES',slice='readiness',
                 scenario='S',route_id='route',physical_group=f'g{i%groups}',
                 observation={'frame_id':i//2 if shared else i}) for i in range(n)]


@pytest.mark.parametrize('n,budget,world,groups,strata,shared',[
    (4,1,1,1,1,False),(6,1,1,1,1,False),(8,2,1,1,1,False),
    (12,1,1,3,4,False),(12,4,4,3,4,False),(12,4,4,3,4,True),
    (15,1,1,4,6,True),(15,None,4,4,6,True),
])
def test_positive_budget_covers_entire_ring_despite_truncation(n,budget,world,groups,strata,shared):
    data=rows(n,groups,strata,shared);seen=set()
    for epoch in range(n):
        selected,audit=plan(data,epoch=epoch,budget=budget,world_size=world,cap=1)
        assert selected==plan(data,epoch=epoch,budget=budget,world_size=world,cap=1)[0]
        assert len(selected)%world==0 and len(selected)==len(set(selected))
        assert max(Counter(frame_key(data[i]) for i in selected).values())==1
        seen.update(selected)
        shuffled=list(reversed(data));other,_=plan(shuffled,epoch=epoch,budget=budget,world_size=world,cap=1)
        assert [data[i]['id'] for i in selected]==[shuffled[i]['id'] for i in other]
    assert seen==set(range(n))


def test_reported_four_questions_twelve_epochs_are_all_seen_three_times():
    selected=[plan(rows(4),epoch=e,budget=1)[0][0] for e in range(12)]
    assert sorted(Counter(selected).values())==[3,3,3,3]


def segments():
    return [dict(segment_id='first',source_corridor='left lane',target_corridor='middle lane',direction='RIGHT',adjacent=True),
            dict(segment_id='second',source_corridor='middle lane',target_corridor='right lane',direction='RIGHT',adjacent=True)]


def episode(**kw):
    return Episode(**dict(dict(event='R-E3',instance_id='exit',direction='RIGHT',target_corridor='middle lane',route_segments=segments()),**kw))


def observation(frame):
    return dict(frame_id=frame,history_frames=[frame-4,frame],speed_mps=4.)


def receipt(frame,seg,edges=('enter',)):
    return dict(instance_id='exit',segment_id=seg,source='causal_tracker',evidence_id=f'visible-{frame}',
                successor_observed=True,decision_frame=frame,observed_frame=frame,started_frame=frame,edges=list(edges))


def tick(loop,frame,*yes,rs=()):
    return loop.tick(observation(frame),[None]*2,lambda ep,key,*args:'YES' if key in yes else 'NO',execution_receipts=rs,maneuver_clearances=loop_clearances(loop,frame,*yes))


def test_two_right_entries_wait_for_fresh_gap_keep_stop_and_bind_receipts():
    loop=Phase4Loop(rgb_mode=2);ep=episode();loop.establish(ep,verified=True)
    assert tick(loop,10,'enter',rs=[receipt(10,'first')])['prior']['action']=='LANE_CHANGE_RIGHT'
    p=tick(loop,11,'complete','hold')
    assert p['completed_instances']==[] and p['prior']['action']=='STOP'
    assert (ep.state,ep.segment_id,ep.target_corridor)==('WAIT','second','right lane')
    assert ep.history[-1]['segment_id']=='first' and ep.history[-1]['accepted']==['complete','hold']
    before=deepcopy(loop.snapshot())
    with pytest.raises(ValueError):tick(loop,11,'enter')
    assert loop.snapshot()==before
    assert tick(loop,12)['prior']['action']=='STOP'  # overtaking vehicle still blocks next entry
    text=prompt(ep,'enter',observation(13))
    assert 'right lane' in text and 'middle lane' not in text and 'first' not in text
    assert 'only the current adjacent route segment' in text
    before=deepcopy(loop.snapshot())
    with pytest.raises(ValueError,match='segment mismatch'):tick(loop,13,'enter',rs=[receipt(13,'first')])
    assert loop.snapshot()==before
    assert tick(loop,13,'enter',rs=[receipt(13,'second')])['prior']['action']=='LANE_CHANGE_RIGHT'
    loop=Phase4Loop.restore(loop.snapshot());ep=loop.episodes['exit']
    p=tick(loop,14,'complete','hold')
    assert ep.state=='DONE' and not ep.finished and p['prior']['action']=='STOP'
    assert tick(loop,15,'release',rs=[receipt(15,'second',('release',))])['prior']['action']=='RESUME'
    assert tick(loop,16,'stable')['completed_instances']==['exit']


def test_unexecuted_segment_permission_expires_without_skipping_segment():
    ep=episode()
    ep.advance(10,{q.key:'YES' if q.key=='enter' else 'NO' for q in ep.questions()},maneuver_clearances=clearances(ep,10,'enter'))
    with pytest.raises(ValueError,match='segment mismatch'):ep.acknowledge(10)
    ep.questions()
    assert ep.state=='WAIT' and ep.segment_id=='first'
    assert ep.history[-1]['expired_start']==['enter']


def test_midsegment_release_completion_waits_for_receipt_or_expiry():
    ep=episode(state='CROSS',longitudinal='HOLD')
    ep.advance(10,{q.key:'YES' if q.key in ('complete','release') else 'NO' for q in ep.questions()})
    assert ep.segment_id=='first' and ep.state=='DONE' and not ep.finished
    ep.questions()
    assert ep.segment_id=='second' and ep.state=='WAIT' and ep.longitudinal=='HOLD'


@pytest.mark.parametrize('fault',['disconnected','duplicate_id','nonadjacent','wrong_target','wrong_event','bad_index','implicit_commit'])
def test_bad_segment_contracts_rejected(fault):
    ss=segments();kw={}
    if fault=='disconnected':ss[1]['source_corridor']='unknown'
    if fault=='duplicate_id':ss[1]['segment_id']='first'
    if fault=='nonadjacent':ss[0]['adjacent']=False
    if fault=='wrong_target':kw['target_corridor']='far exit'
    if fault=='wrong_event':kw['event']='R-E2'
    if fault=='bad_index':kw['segment_index']=2
    with pytest.raises(ValueError):
        ep=episode(route_segments=ss,**kw)
        if fault=='implicit_commit':ep.advance(10,{q.key:'YES' if q.key=='enter' else 'NO' for q in ep.questions()},execution_committed=True)


def test_segment_coverage_cannot_be_filled_by_single_target_or_catchup():
    from test_audit_fixes import all_support
    old=all_support(True)
    assert not coverage_report(old)['ready'] and len(coverage_report(old)['missing_segmented_route_support'])==12
    extra=[]
    for split in ('train','val','test'):
        for index in (0,1):
            for edge in ('enter','complete'):
                for target in ('YES','NO'):
                    extra.append(dict(split=split,episode=dict(event='R-E3',route_segments=segments(),segment_index=index),
                        edge=edge,slice='readiness',target=target,physical_group=f'{split}/{target}'))
    report=coverage_report(old+extra)
    assert not report['missing_segmented_route_support']
    assert len(report['missing_roundabout_support'])==9 and not report['ready']
    for r in extra:r['slice']='catchup'
    assert not coverage_report(old+extra)['ready']


def test_replay_truth_requires_segment_binding_and_censors_between_segments():
    init=episode(state='CROSS',longitudinal='RECOVER').to_dict()
    observations=[dict(frame_id=10,segment_id='first',truth={'stable':'YES','complete':'YES'}),
                  dict(frame_id=11,segment_id='second',truth={'stable':'YES'}),
                  dict(frame_id=12,segment_id='first',truth={'enter':'YES'})]
    result=replay(init,observations,lambda ep,key,o:'YES' if key=='complete' and o['frame_id']==10 else 'NO')
    assert any(c['reason']=='route_segment_changed' and c['segment_id']=='first' for c in result['unconfirmed_transitions'])
    assert result['steps'][1]['before']['segment_index']==1
    assert result['covered_questions']==3 and not result['completed']


def test_new_exposure_is_registered():
    for group in json.loads((ROOT/'ninth_audit_exposure_20260930.json').read_text())['train_only_groups']:
        for seed in (0,1,2026):assert split_for(group,groups(),seed)=='train'


@pytest.mark.parametrize('answer',['NO','UNKNOWN'])
def test_unconfirmed_geometry_cannot_select_next_target(answer):
    ep=episode(state='CROSS')
    ep.advance(10,{q.key:answer if q.key=='complete' else 'NO' for q in ep.questions()})
    assert ep.segment_id=='first' and ep.target_corridor=='middle lane' and ep.state=='CROSS'


def test_navigation_can_chain_left_merge_and_forward_exit_without_invented_right_action():
    ss=segments();ss[0]['direction']='LEFT';ss[1]['direction']='FORWARD'
    ep=episode(direction='LEFT',route_segments=ss,state='CROSS')
    ep.advance(10,{q.key:'YES' if q.key=='complete' else 'NO' for q in ep.questions()})
    assert ep.segment_id=='second' and ep.direction=='FORWARD' and ep.prior()['action']=='KEEP'
    p=ep.advance(11,{q.key:'YES' if q.key=='enter' else 'NO' for q in ep.questions()},maneuver_clearances=clearances(ep,11,'enter'))
    assert p['action']=='KEEP' and ep.state=='CROSS'
    ep.confirm_execution(receipt(11,'second'))


def test_segmented_demo_finishes_and_waits_between_entries():
    from qwen3vl_local.sft_new_loop_phase4.demo import segmented_exit
    trace=segmented_exit()
    assert [r['result']['prior']['action'] for r in trace]==['LANE_CHANGE_RIGHT','STOP','STOP','LANE_CHANGE_RIGHT','UNCOND']
    assert trace[-1]['result']['completed_instances']==['exit-demo']


@pytest.mark.parametrize('mode',[2,4])
def test_segment_metadata_roundtrips_builder_without_rendering_future_plan(tmp_path,mode):
    from qwen3vl_local.sft_new_loop_phase4.dataset import build,load_dataset,read_rows
    from qwen3vl_local.sft_new_loop_phase4.input_identity import model_input_key
    a=deepcopy(next(a for a in json.loads((ROOT/'reviewed_state_pairs_v4.json').read_text()) if a['episode']['event']=='R-E3' and a['edge']=='enter'))
    # Synthetic plumbing probe only, not new reviewed segment labels.
    a['evidence_id']='SYNTHETIC_segment_plumbing';a['episode']=dict(event='R-E3',instance_id='synthetic',state='WAIT',direction='RIGHT',target_corridor='middle lane',route_segments=segments())
    build([a],ROOT.parents[1]/'lead_data',tmp_path/'data',rgb_mode=mode)
    data,_=load_dataset(tmp_path/'data');row=(sum(data.values(),[])+read_rows(tmp_path/'data/review_queue.jsonl'))[0]
    text=prompt(Episode(**row['episode']),row['edge'],row['observation'])
    assert 'middle lane' in text and 'right lane' not in text
    same=deepcopy(row);same['episode']['route_segments'][1]['target_corridor']='different future goal'
    assert model_input_key(same)==model_input_key(row)
