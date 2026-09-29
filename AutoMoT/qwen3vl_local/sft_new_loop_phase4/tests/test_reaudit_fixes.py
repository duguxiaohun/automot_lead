import copy
import pytest
from qwen3vl_local.sft_new_loop_phase4.controller import Episode,combine_priors
from qwen3vl_local.sft_new_loop_phase4.runtime import Phase4Loop
from qwen3vl_local.sft_new_loop_phase4.action_prior import export
from qwen3vl_local.sft_new_loop_phase4.prompts import prompt


def step(ep,frame,*keys):
    return ep.advance(frame,{e.key:'YES' if e.key in keys else 'NO' for e in ep.questions()})


def conflict(lon='HOLD',same_direction=False):
    a=Episode('R-E2','a',state='CROSS',longitudinal=lon,direction='LEFT',target_corridor='left_lane')
    b=Episode('U-E2','b',state='DEPART',direction='LEFT' if same_direction else 'RIGHT',target_corridor='right_lane')
    return a,b


@pytest.mark.parametrize('lon,expected',[('HOLD','STOP'),('APPROACH','DECELERATE'),('STABLE',None)])
@pytest.mark.parametrize('same_direction',[False,True])
def test_conflict_retains_braking_and_names_instances(lon,expected,same_direction):
    a,b=conflict(lon,same_direction)
    before=[copy.deepcopy(e.to_dict()) for e in (a,b)]
    prior=combine_priors([a,b])
    assert prior['status']=='RECHECK' and prior['action']==expected
    assert prior['recheck_instances']==prior['conflict_instances']==['a','b']
    out=export([a,b]);assert out['token_name']==expected and out['recheck_instances']==['a','b']
    assert [e.to_dict() for e in (a,b)]==before  # Pure export does not silently mutate state.


def test_public_tick_recheck_restore_and_revalidate_path():
    a,b=conflict();loop=Phase4Loop(rgb_mode=2)
    for ep in (a,b):loop.establish(ep,verified=True)
    obs=dict(frame_id=10,history_frames=[6,10],speed_mps=0)
    result=loop.tick(obs,[None]*2,lambda *a:'NO')
    assert result['prior']['action']=='STOP' and result['recheck_instances']==['a','b']
    loop=Phase4Loop.restore(loop.snapshot())
    assert loop.aggregate()['action']=='STOP'
    # A replacement that still disagrees is not falsely released by verified=True.
    loop.revalidate('a',Episode('R-E2','a',state='CROSS',longitudinal='HOLD',
                              direction='LEFT',target_corridor='left_lane'),verified=True)
    assert loop.episodes['a'].needs_recheck
    aa=Episode('R-E2','a',state='CROSS',longitudinal='HOLD',direction='RIGHT',target_corridor='right_lane')
    loop.revalidate('a',aa,verified=True)
    assert loop.aggregate()['recheck_instances']==['b']
    bb=Episode('U-E2','b',state='DEPART',direction='RIGHT',target_corridor='right_lane')
    loop.revalidate('b',bb,verified=True)
    out=loop.aggregate();assert out['status']=='ACTIVE' and out['action']=='STOP'
    assert out['recheck_instances']==[]


def test_public_revalidate_can_discover_exported_conflict_without_tick():
    loop=Phase4Loop();a,b=conflict()
    for ep in (a,b):loop.establish(ep,verified=True)
    assert export([a,b])['recheck_instances']==['a','b']
    loop.revalidate('b',Episode('U-E2','b',state='DEPART',direction='LEFT',target_corridor='left_lane'),verified=True)
    assert not loop.episodes['b'].needs_recheck and loop.episodes['a'].needs_recheck


def test_new_conflicting_proposals_revoke_pending_permission_but_keep_stop():
    loop=Phase4Loop(rgb_mode=2)
    for ident,direction in [('a','LEFT'),('b','RIGHT')]:
        loop.establish(Episode('U-E2',ident,longitudinal='HOLD',direction=direction,target_corridor=direction),verified=True)
    result=loop.tick(dict(frame_id=10,history_frames=[6,10],speed_mps=0),[None]*2,
                     lambda ep,key,*args:'YES' if key=='depart' else 'NO')
    assert result['prior']['action']=='STOP' and result['recheck_instances']==['a','b']
    assert all(ep.state=='WAIT' and ep.longitudinal=='HOLD' for ep in loop.episodes.values())
    with pytest.raises(ValueError):loop.acknowledge('a',10)


def test_matching_targets_do_not_force_recheck_or_lose_unrelated_stop():
    a,b=conflict();b.direction='LEFT';b.target_corridor='left_lane'
    prior=combine_priors([a,b]);assert prior['status']=='ACTIVE' and prior['action']=='STOP'
    c=Episode('U-E1','queue',longitudinal='HOLD');a.longitudinal='STABLE';b.target_corridor='different_left_lane'
    prior=combine_priors([a,b,c]);assert prior['action']=='STOP'
    assert prior['recheck_instances']==['a','b']


@pytest.mark.parametrize('event',['U-E1','U-E3','U-E4','U-E5','U-E6','U-E7','R-E5'])
def test_reyield_and_stable_cannot_turn_into_keep(event):
    ep=Episode(event,'x',state='PROCEED',longitudinal='RECOVER')
    result=step(ep,10,'re_yield','stable')
    assert ep.needs_recheck and result['status']=='RECHECK' and result['action']!='KEEP'
    assert (ep.state,ep.longitudinal)==('PROCEED','RECOVER')
    assert not ep.history  # No contradictory fact was accepted.
    ep=Episode(event,'x',state='PROCEED',longitudinal='RECOVER')
    assert step(ep,10,'re_yield')['action']=='DECELERATE'
    ep=Episode(event,'x',state='PROCEED',longitudinal='RECOVER')
    assert step(ep,10,'re_yield','hold')['action']=='STOP'


@pytest.mark.parametrize('event,branch',[('U-E2','default'),('U-E4','cyclist_bypass')])
@pytest.mark.parametrize('lon',['HOLD','APPROACH','RECOVER'])
def test_return_completion_keeps_route_corridor_through_remaining_longitudinal_work(event,branch,lon):
    ep=Episode(event,'x',branch=branch,state='RETURN',longitudinal=lon,direction='LEFT',
               return_direction='RIGHT',target_corridor='borrowed_lane',return_corridor='original_lane')
    before=prompt(ep,'complete',dict(frame_id=10,history_frames=[6,10],speed_mps=0))
    assert 'return to the established route corridor is complete' in before
    assert 'remaining longitudinal restrictions stay in effect' in before
    assert 'this event instance is complete' not in before
    prior=step(ep,10,'complete')
    assert prior['target_corridor']=='original_lane' and prior['lateral']=='KEEP' and not ep.finished
    ep=Episode(**ep.to_dict())
    assert export([ep])['instances'][0]['target_corridor']=='original_lane'
    if lon in ('HOLD','APPROACH'):
        step(ep,11,'release');ep.acknowledge(11)
    step(ep,12,'stable');assert ep.finished and ep.prior()['target_corridor']=='original_lane'


@pytest.mark.parametrize('event,branch,state,ret',[
    ('U-E2','default','PASS',False),('U-E4','cyclist_bypass','PASS',False),
    ('R-E2','default','CROSS',True),('R-E3','default','CROSS',True),
    ('U-E2','in_lane_pass','PROCEED',True),('U-E4','cyclist_follow','PROCEED',True)])
def test_other_completion_branches_keep_their_own_target(event,branch,state,ret):
    ep=Episode(event,'x',branch=branch,state=state,return_required=ret,direction='LEFT',
               target_corridor='route_target',return_corridor='unused_history')
    assert step(ep,10,'complete')['target_corridor']=='route_target'


@pytest.mark.parametrize('edge,state,lon',[('proceed','YIELD','HOLD'),('release','PROCEED','HOLD')])
def test_ue3_state_pair_covers_following_and_passing_with_vetoes(edge,state,lon):
    ep=Episode('U-E3','x',state=state,longitudinal=lon)
    text=prompt(ep,edge,dict(frame_id=10,history_frames=[6,10],speed_mps=0))
    assert len(text.splitlines())==3
    for required in ('usable following gap','passing alongside','continued lateral intrusion',
                     'conflicting approaching traffic','priority obligations are satisfied'):
        assert required in text
    assert 'left' not in text and 'StaticCutIn' not in text and 'Town13' not in text


def test_reported_development_exposure_stays_out_of_holdout():
    import json
    from qwen3vl_local.sft_new_loop_phase4.identity import ROOT
    from qwen3vl_local.sft_new_loop_phase4.dataset import groups,split_for
    report=json.loads((ROOT/'reaudit_exposure_20260929.json').read_text())
    assert len(report['train_only_groups'])==23
    assert not report['reported_coverage']['all_requested_rgb_audit_complete']
    exposed=groups()
    for group in report['train_only_groups']:
        assert group in exposed
        for seed in (1,2026,20260929):assert split_for(group,exposed,seed)=='train'


def test_pure_export_of_conflicting_uncommitted_permissions_preserves_prior_hold():
    episodes=[]
    for ident,direction in [('a','LEFT'),('b','RIGHT')]:
        ep=Episode('U-E2',ident,longitudinal='HOLD',direction=direction,target_corridor=direction)
        step(ep,10,'depart');episodes.append(ep)
    assert export(episodes)['token_name']=='STOP'
    assert all(ep.state=='DEPART' for ep in episodes)  # export remains pure


def test_temporary_unknown_does_not_claim_unavailable_revalidation():
    ep=Episode('U-E1','x',uncertain=True,longitudinal='HOLD')
    p=combine_priors([ep])
    assert p['status']=='UNKNOWN' and p['action']=='STOP' and p['recheck_instances']==[]
