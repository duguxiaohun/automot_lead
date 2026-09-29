import copy
import json
from pathlib import Path
import pytest
from qwen3vl_local.sft_new_loop_phase4.controller import Episode,combine_priors
from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS,template_for,get_edge
from qwen3vl_local.sft_new_loop_phase4.prompts import prompt,parse_answer
from qwen3vl_local.sft_new_loop_phase4.calibration import label_condition,interval_facts,catchup_label
from qwen3vl_local.sft_new_loop_phase4.runtime import Phase4Loop
from qwen3vl_local.sft_new_loop_phase4.sampling import plan,frame_key
from qwen3vl_local.sft_new_loop_phase4.evaluate import replay,metrics
from qwen3vl_local.sft_new_loop_phase4.train import accumulation_size
from qwen3vl_local.sft_new_loop_phase4.dataset import split_for,coverage_report


def step(ep,frame,*yes,committed=True):
    answers={e.key:('YES' if e.key in yes else 'NO') for e in ep.questions()}
    assert set(yes)<=set(answers)
    return ep.advance(frame,answers,execution_committed=committed)


def bypass(**kw):
    return Episode('U-E2','x',longitudinal='HOLD',direction='LEFT',return_direction='RIGHT',
                   target_corridor='left corridor',return_corridor='original lane',**kw)


@pytest.mark.parametrize('event',EVENTS)
def test_all_events_finish(event):
    ep=Episode(event,'x',direction='LEFT',return_direction='RIGHT',target_corridor='left',return_corridor='original')
    template=template_for(event)
    chain=['depart','pass','return','complete'] if template=='bypass' else ['enter','complete'] if template in ('lane','merge') else ['proceed','complete']
    for f,key in enumerate(chain,1):
        step(ep,f,key)
    assert ep.state=='DONE' and ep.prior()['action']=='UNCOND' and not ep.questions()


@pytest.mark.parametrize('event,branch', [('U-E4','cyclist_bypass'),('U-E4','cyclist_follow'),('U-E2','in_lane_pass')])
def test_actual_route_branches(event,branch):
    ep=Episode(event,'x',branch=branch,return_required=False,direction='RIGHT',target_corridor='right')
    chain=['depart','pass','complete'] if branch=='cyclist_bypass' else ['proceed','complete']
    for f,key in enumerate(chain,1):
        step(ep,f,key)
    assert ep.state=='DONE'


def test_no_return_and_right_bypass():
    ep=Episode('U-E2','x',return_required=False,direction='RIGHT',target_corridor='right')
    assert step(ep,1,'depart')['action']=='LANE_CHANGE_RIGHT'
    assert step(ep,2,'pass')['lateral']=='KEEP'
    assert 'return' not in [e.key for e in ep.questions()]
    assert step(ep,3,'complete')['status']=='DONE'


def test_opportunity_expires_but_committed_milestone_remains():
    ep=bypass()
    assert step(ep,1,'depart',committed=False)['action']=='LANE_CHANGE_LEFT'
    assert 'depart' in [e.key for e in ep.questions()]
    assert ep.state=='WAIT' and ep.longitudinal=='HOLD'
    step(ep,2,'depart',committed=False)
    ep.acknowledge(2)
    assert 'pass' in [e.key for e in ep.questions()]
    step(ep,3,'pass')
    step(ep,4)
    assert ep.state=='PASS'


def test_no_does_not_erase_lateral_motion():
    ep=bypass()
    step(ep,1,'depart')
    assert step(ep,2)['action']=='LANE_CHANGE_LEFT'


def test_unknown_not_no_and_can_recover():
    ep=bypass()
    answers={e.key:'NO' for e in ep.questions()}
    answers['depart']='UNKNOWN'
    assert ep.advance(1,answers)['status']=='UNKNOWN'
    assert ep.state=='WAIT'
    assert step(ep,2,'depart')['action']=='LANE_CHANGE_LEFT'


def test_missing_answer_does_not_advance():
    ep=Episode('U-E2','x',direction='LEFT',target_corridor='left')
    assert ep.advance(1,{'depart':'YES'})['action'] is None
    assert ep.state=='WAIT'


def test_atomic_contradiction_and_missing_target():
    ep=Episode('U-E2','x')
    before=ep.state,ep.longitudinal
    result=step(ep,1,'depart','restrict')
    assert result['status']=='RECHECK' and before==(ep.state,ep.longitudinal)
    ep=Episode('U-E2','x',direction='LEFT')
    assert step(ep,1,'depart')['status']=='RECHECK'
    assert ep.state=='WAIT'


def test_stronger_hold_wins_over_restrict():
    ep=Episode('U-E1','x')
    assert step(ep,1,'restrict','hold')['action']=='STOP'


def test_new_conflict_does_not_finish():
    ep=Episode('U-E6','x',state='PROCEED')
    assert step(ep,1,'re_yield')['action']=='DECELERATE'
    assert ep.state=='YIELD'


@pytest.mark.parametrize('value',[1,'true',[],{}])
def test_invalid_context_types(value):
    ep=bypass()
    with pytest.raises(ValueError):
        ep.advance(1,{},context_valid=value)


def test_normal_wait_does_not_timeout():
    ep=Episode('U-E4','x',longitudinal='HOLD',stall_limit=2)
    for frame in range(1,130):
        assert step(ep,frame)['action']=='STOP'
    assert not ep.finished and not ep.needs_recheck


def test_no_same_image_chain():
    ep=bypass()
    step(ep,10,'depart')
    with pytest.raises(ValueError):
        step(ep,10,'pass')


def test_concurrent_constraints_survive_primary_projection():
    ep=bypass();step(ep,1,'depart');ep.longitudinal='APPROACH'
    assert ep.prior()['action']=='LANE_CHANGE_LEFT'
    assert combine_priors([ep])['action']=='DECELERATE'
    other=Episode('R-E2','y',state='CROSS',direction='RIGHT',target_corridor='right')
    assert combine_priors([ep,other])['status']=='RECHECK'


def test_snapshot_and_loop_avoids_upstream_on_ordinary_progress():
    loop=Phase4Loop();loop.establish(bypass(),verified=True)
    obs=dict(frame_id=10,history_frames=[4,6,8,10],speed_mps=0.)
    result=loop.tick(obs,[None]*4,lambda ep,key,obs,images:'YES' if key=='depart' else 'NO')
    assert not result['recheck_instances']
    loop.acknowledge('x',10)
    recovered=Phase4Loop.restore(loop.snapshot())
    assert recovered.episodes['x'].state=='DEPART'
    assert [e.key for e in recovered.episodes['x'].questions()]==[e.key for e in loop.episodes['x'].questions()]


def test_condition_positive_interval_before_motion():
    ep=Episode('U-E4','x',longitudinal='HOLD')
    facts=interval_facts(dict(release_ready=True,corridor_clear=True,priority_satisfied=True),10,14,'review')
    assert [label_condition(ep,'proceed',f,facts) for f in range(10,15)]==['YES']*5
    assert label_condition(ep,'proceed',15,facts)=='UNKNOWN'
    facts['corridor_clear']['value']=False
    assert label_condition(ep,'proceed',12,facts)=='NO'


def test_future_or_action_timing_not_permission():
    ep=bypass()
    facts=interval_facts({'entry_gap_clear':True},10,15,'e')
    facts['entry_gap_clear']['observed_until']=13
    with pytest.raises(ValueError,match='future'):
        label_condition(ep,'depart',12,facts)
    facts['entry_gap_clear']['source']='future_ego_lane_change'
    with pytest.raises(ValueError,match='unreviewed'):
        label_condition(ep,'depart',14,facts)


def test_interval_only_catchup_requires_per_frame_review():
    ep=bypass()
    kw=dict(completed_at=10,valid_until=13,subsequent_stage_confirmed=True,evidence_id='review')
    with pytest.raises(ValueError,match='reviewed_transition_band'):
        catchup_label(ep,'depart',12,**kw)
    with pytest.raises(ValueError):
        catchup_label(ep,'depart',14,**kw)


@pytest.mark.parametrize('bad_field',['future_waypoints','throttle','brake','facts','target','town','scenario'])
def test_prompt_does_not_accept_label_or_future_fields(bad_field):
    obs=dict(frame_id=10,history_frames=[4,6,8,10],speed_mps=0.)
    obs[bad_field]='secret'
    with pytest.raises(ValueError):
        prompt(bypass(),'depart',obs)


@pytest.mark.parametrize('frames',[[4,6,8,11],[4,6,10,8],[4,4,8,10],[0,6,8,10]])
def test_causal_history(frames):
    with pytest.raises(ValueError):
        prompt(bypass(),'depart',dict(frame_id=10,history_frames=frames,speed_mps=1))


def test_prompt_not_action_choice():
    text=prompt(bypass(),'depart',dict(frame_id=10,history_frames=[6,10],speed_mps=0))
    assert 'entry gap permits entry' in text and 'LANE_CHANGE' not in text
    assert 'recorded onset' not in text
    with pytest.raises(ValueError):
        parse_answer('YES because')


def rows_fixture():
    rows=[]
    for frame in range(12):
        for edge in ('depart','hold','restrict'):
            rows.append(dict(id=f'{frame}_{edge}',episode={'event':'U-E2'},edge=edge,target='YES' if frame%2 else 'NO',
                             slice='readiness',scenario='S',route_id='Town_1',physical_group=f'S/{frame%3}',observation={'frame_id':frame}))
    return rows


def test_global_frame_cap_and_deterministic_epoch_rotation():
    rows=rows_fixture()
    a,report=plan(rows,epoch=0,cap=1,world_size=4)
    assert len(a)==12 and report['max_frame_repeat']==1
    assert a==plan(rows,epoch=0,cap=1,world_size=4)[0]
    b,_=plan(rows,epoch=1,cap=1,world_size=4)
    assert a!=b
    assert len(set(frame_key(rows[i]) for i in a))==12
    with pytest.raises(ValueError,match='capacity'):
        plan(rows,epoch=0,cap=1,budget=16,world_size=4)


def test_exposed_groups_never_holdout():
    for seed in range(10):
        assert split_for('group',{'group'},seed)=='train'
    assert not coverage_report([])['ready']


def test_accumulation_partial_tail():
    assert [accumulation_size(i,10,4) for i in range(10)]==[4]*8+[2]*2


def test_replay_queries_predicted_state_not_teacher_state():
    initial=bypass(return_required=False).to_dict()
    calls=[]
    def predictor(ep,key,obs):
        calls.append((obs['frame_id'],ep.state,key))
        want={1:None,2:'depart',3:'pass',4:'stable',5:'complete'}[obs['frame_id']]
        return 'YES' if key==want else 'NO'
    result=replay(initial,[dict(frame_id=f,execution_committed=True,truth={'depart':'YES'}) for f in range(1,6)],predictor)
    assert result['completed']
    assert (2,'WAIT','depart') in calls and (3,'DEPART','pass') in calls
    assert result['uncovered_questions']>0


def test_metrics_do_not_mix_catchup_with_permission_recall():
    r=[dict(target='YES',prediction='NO',event='U-E2',edge='depart',slice='readiness'),
       dict(target='YES',prediction='YES',event='U-E2',edge='depart',slice='catchup')]
    assert metrics(r)['readiness_recall']==0


@pytest.mark.parametrize('lon',['HOLD','APPROACH','RECOVER'])
def test_completion_does_not_release_unresolved_longitudinal_constraint(lon):
    ep=Episode('R-E2','x',state='CROSS',longitudinal=lon,direction='LEFT',target_corridor='target')
    assert 'complete' in [e.key for e in ep.questions()]
    prior=step(ep,1,'complete')
    assert ep.state=='DONE' and not ep.finished
    assert prior['lateral']=='KEEP' and prior['longitudinal'] in ('STOP','DECELERATE','RESUME')
    assert ep.questions()
