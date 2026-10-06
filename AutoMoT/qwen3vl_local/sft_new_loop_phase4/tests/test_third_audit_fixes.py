from qwen3vl_local.sft_new_loop_phase4.tests.safety_fixtures import clearances, loop_clearances
import copy
import json
import pytest
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.runtime import Phase4Loop
from qwen3vl_local.sft_new_loop_phase4.calibration import reviewed_band_labels
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT


def obs(frame=10):
    return dict(frame_id=frame,history_frames=[frame-4,frame],speed_mps=0)


def predict(*yes):
    return lambda ep,key,*args:'YES' if key in yes else 'NO'


def setup_loop(two=False):
    loop=Phase4Loop(rgb_mode=2)
    for ident in (['a','b'] if two else ['a']):
        loop.establish(Episode('U-E2',ident,longitudinal='HOLD',direction='LEFT',
                               target_corridor='bypass'),verified=True)
    return loop


def receipt(ident='a',**changes):
    return dict(dict(instance_id=ident,source='causal_tracker',evidence_id='observed-start',
                     successor_observed=True,decision_frame=10,observed_frame=10,
                     started_frame=9,edges=['depart']),**changes)


@pytest.mark.parametrize('event',['U-E1','U-E3','U-E4','U-E5','U-E6','U-E7','R-E5'])
def test_reachable_hold_then_reyield_retains_stop_until_explicit_permission(event):
    loop=Phase4Loop(rgb_mode=2)
    ep=Episode(event,'a',state='PROCEED',longitudinal='RECOVER')
    loop.establish(ep,verified=True)
    assert loop.tick(obs(),[None]*2,predict('hold'))['prior']['action']=='STOP'
    assert loop.tick(obs(11),[None]*2,predict('re_yield'))['prior']['action']=='STOP'
    assert (ep.state,ep.longitudinal)==('YIELD','HOLD')
    assert loop.tick(obs(12),[None]*2,predict())['prior']['action']=='STOP'
    # Only a fresh proceed permission can release the established stop.
    assert loop.tick(obs(13),[None]*2,predict('proceed'))['prior']['action']=='RESUME'
    loop.acknowledge('a',13)
    assert ep.longitudinal=='RECOVER'


@pytest.mark.parametrize('mode',['duplicate','stale','bad_context','bad_fault','bad_images'])
def test_rejected_observation_preserves_permission_and_can_still_acknowledge(mode):
    loop=setup_loop();ep=loop.episodes['a']
    loop.tick(obs(),[None]*2,predict('depart'),maneuver_clearances=loop_clearances(loop,10,'depart'))
    before=copy.deepcopy(loop.snapshot());calls=[]
    o=obs(10 if mode=='duplicate' else 9 if mode=='stale' else 11)
    kw={'context_valid':{'a':1}} if mode=='bad_context' else {'progress_faults':{'missing':True}} if mode=='bad_fault' else {}
    with pytest.raises(ValueError):
        loop.tick(o,[None]*(4 if mode=='bad_images' else 2),lambda *args:calls.append(args),**kw)
    assert not calls and loop.snapshot()==before and loop.episodes['a'] is ep
    loop.acknowledge('a',10)
    assert ep.history[-1]['committed']


@pytest.mark.parametrize('failure',['bad_answer','exception','bad_receipt','inactive_receipt'])
def test_multiepisode_failure_rolls_back_entire_batch_and_valid_retry_succeeds(failure):
    loop=setup_loop(two=True);refs=dict(loop.episodes)
    if failure=='inactive_receipt':loop.episodes['b'].suspended=True
    before=copy.deepcopy(loop.snapshot())
    seen=[]
    def predictor(ep,key,*args):
        seen.append(ep.instance_id)
        if ep.instance_id=='b' and failure=='exception':raise RuntimeError('inference failed')
        if ep.instance_id=='b' and failure=='bad_answer':return 'MALFORMED'
        return 'YES' if key=='depart' else 'NO'
    rs=[receipt('a')]
    if failure in ('bad_receipt','inactive_receipt'):rs.append(receipt('b',edges=['enter']))
    with pytest.raises((ValueError,RuntimeError)):
        loop.tick(obs(),[None]*2,predictor,execution_receipts=rs,maneuver_clearances=loop_clearances(loop,10,'depart'))
    assert 'a' in seen and loop.snapshot()==before
    assert all(loop.episodes[k] is ref for k,ref in refs.items())
    loop.episodes['b'].suspended=False
    loop.tick(obs(),[None]*2,predict('depart'),execution_receipts=[receipt('a'),receipt('b')],maneuver_clearances=loop_clearances(loop,10,'depart'))
    assert all(ep.state=='DEPART' and ep.history[-1]['committed'] for ep in refs.values())
    assert Phase4Loop.restore(loop.snapshot()).snapshot()==loop.snapshot()


def test_new_observation_expires_permission_once_only_after_success():
    loop=setup_loop();loop.tick(obs(),[None]*2,predict('depart'),maneuver_clearances=loop_clearances(loop,10,'depart'))
    before=copy.deepcopy(loop.snapshot())
    with pytest.raises(ValueError):loop.tick(obs(11),[None]*2,lambda *a:'bad')
    assert loop.snapshot()==before
    loop.tick(obs(11),[None]*2,predict())
    ep=loop.episodes['a']
    assert (ep.state,ep.longitudinal)==('WAIT','HOLD')
    assert [r.get('expired_start') for r in ep.history]==[['depart'],None]
    assert ep.unexecuted_observations==0  # Fresh NO is ordinary waiting, not a failed execution.


def test_direct_advance_rejects_bad_answer_without_expiring_permission():
    ep=setup_loop().episodes['a']
    ep.advance(10,{e.key:'YES' if e.key=='depart' else 'NO' for e in ep.questions()},maneuver_clearances=clearances(ep,10,'depart'))
    before=copy.deepcopy(ep.to_dict())
    with pytest.raises(ValueError):ep.advance(11,{'not_an_edge':'YES'})
    assert ep.to_dict()==before
    ep.acknowledge(10)


@pytest.mark.parametrize('operation',['revalidate','acknowledge'])
def test_rejected_public_operation_does_not_commit_aggregate_conflict(operation):
    loop=setup_loop(two=True)
    for ep in loop.episodes.values():ep.state='DEPART'
    loop.episodes['b'].direction='RIGHT'
    before=copy.deepcopy(loop.snapshot())
    replacement=Episode('U-E2','a');replacement_before=copy.deepcopy(replacement.to_dict())
    with pytest.raises(ValueError):
        if operation=='revalidate':loop.revalidate('a',replacement,verified=False)
        else:loop.acknowledge('a',10)
    assert loop.snapshot()==before and replacement.to_dict()==replacement_before


def geometric_band():
    ep=Episode('U-E2','a',state='RETURN',longitudinal='HOLD',return_direction='RIGHT',return_corridor='original')
    band=dict(review_start=10,review_end=11,reference_frame=10,reference_kind='milestone_observed',
              reference_observation='return completed while lead car requires stop',start_reason='review milestone',
              stop=dict(frame=12,kind='review_end',reason='right censored'),frames={})
    for f in (10,11):
        band['frames'][str(f)]=dict(phase='readiness' if f==10 else 'catchup',observed_until=f,
            observation='returned into original corridor; independent lead-car stationary wait remains',
            facts=dict(route_corridor_reached=True,lateral_entry_complete=True),successor_confirmed=True,
            current_conflict=True,transition_blocking_conflict=False)
    return ep,band


def test_geometry_readiness_and_catchup_agree_despite_independent_stop():
    ep,band=geometric_band()
    labels=reviewed_band_labels(ep,'complete',band)
    assert labels=={10:('YES','readiness'),11:('YES','catchup')}
    for frame in (10,11):
        old=copy.deepcopy(ep)
        p=old.advance(frame,{q.key:labels[frame][0] if q.key=='complete' else 'NO' for q in old.questions()})
        assert p['event_stage_completed'] and p['action']=='STOP' and not old.finished


@pytest.mark.parametrize('bad',[None,1,'false',True])
def test_conflict_scope_missing_malformed_or_inconsistent_requires_review(bad):
    ep,b=geometric_band();item=b['frames']['11']
    item['transition_blocking_conflict']=bad
    if bad is True:item['current_conflict']=False
    with pytest.raises(ValueError,match='transition-specific'):reviewed_band_labels(ep,'complete',b)


def test_scoped_conflict_does_not_override_false_geometry_or_invalid_context():
    ep,b=geometric_band()
    b['frames']['11']['facts']['lateral_entry_complete']=False
    assert reviewed_band_labels(ep,'complete',b)[11][0]=='NO'
    assert reviewed_band_labels(ep,'complete',b,context_valid=False)[11][0]=='INVALID'
    assert reviewed_band_labels(ep,'complete',b,context_valid=None)[11][0]=='UNKNOWN'


@pytest.mark.parametrize('stop',['new_conflict','instance_boundary','calibration_anomaly'])
def test_audit_stop_excludes_old_question_instead_of_inventing_no(stop):
    ep,b=geometric_band()
    b['stop']=dict(frame=11,kind=stop,reason='reviewed cutoff, not a negative label')
    b['frames']['11']['phase']='excluded'
    assert reviewed_band_labels(ep,'complete',b)=={10:('YES','readiness')}


def test_third_audit_exposure_registered_without_labels():
    from qwen3vl_local.sft_new_loop_phase4.dataset import groups,split_for
    audit=json.loads((ROOT/'third_audit_exposure_20260930.json').read_text())
    assert len(audit['train_only_groups'])==11
    for group in audit['train_only_groups']:
        for seed in (1,2026,20260930):assert split_for(group,groups(),seed)=='train'
    assert not audit['reported_coverage']['all_requested_rgb_audit_complete']
