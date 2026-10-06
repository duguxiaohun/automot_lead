"""Regression for reachable following, visible-only supervision and planner gating."""
from copy import deepcopy
import json
import pytest
from qwen3vl_local.sft_new_loop_phase4 import dataset, visible_scope, risk_review
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.runtime import Phase4Loop
from qwen3vl_local.sft_new_loop_phase4.route_prompts import messages, prompt
from qwen3vl_local.sft_new_loop_phase4.route_calibration import label_condition, interval_facts
from qwen3vl_local.sft_new_loop_phase4.maneuver_safety import binding
from qwen3vl_local.sft_new_loop_phase4.evaluate import replay
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT, file_sha, contract
from qwen3vl_local.sft_new_loop_phase4.tests.safety_fixtures import clearances


def answers(ep, *yes):
    return {e.key:'YES' if e.key in yes else 'NO' for e in ep.questions()}


def maneuver(edge='depart', **kw):
    args = dict(event='U-E2',instance_id='hidden/maneuver/1',state='PASS' if edge=='return' else 'WAIT',
        longitudinal='HOLD',direction='LEFT',return_direction='RIGHT',target_corridor='left lane',return_corridor='original lane')
    if edge=='enter':args['event']='R-E3'
    args.update(kw)
    return Episode(**args)


@pytest.mark.parametrize('event',['U-E1','U-E3','U-E4','U-E5','U-E6','U-E7','R-E5'])
def test_actual_following_exit_needs_no_invented_restriction(event):
    ep=Episode(event,'follow',longitudinal='HOLD')
    ep.advance(10,answers(ep,'proceed'),execution_committed=True)
    assert (ep.state,ep.longitudinal)==('PROCEED','RECOVER')
    assert 'complete' not in {q.key for q in ep.questions()}
    text=prompt(ep,'recover_follow',dict(frame_id=11,history_frames=[7,11],speed_mps=1.))
    assert 'ordinary following' in text
    assert label_condition(ep,'recover_follow',11,interval_facts({'restricted_progress_established':True},11,11,'review'))=='YES'
    ep.advance(11,answers(ep,'recover_follow'))
    assert (ep.state,ep.longitudinal)==('PROCEED','FOLLOW')
    assert not ep.finished
    ep.advance(12,answers(ep,'complete'))
    assert ep.finished and all('renewed_restriction' not in r['accepted'] for r in ep.history)


@pytest.mark.parametrize('other',['hold','renewed_restriction','re_yield','stable'])
def test_following_milestone_cannot_erase_conflict_or_compete_with_stable(other):
    ep=Episode('U-E1','follow',state='PROCEED',longitudinal='RECOVER')
    ep.advance(10,answers(ep,'recover_follow',other))
    assert ep.needs_recheck and ep.longitudinal=='RECOVER'


def test_done_with_retained_restriction_can_exit_through_following():
    ep=Episode('R-E2','done',state='DONE',longitudinal='RECOVER')
    ep.advance(10,answers(ep,'recover_follow'))
    assert ep.finished and not ep.questions()


@pytest.mark.parametrize('edge',['depart','enter','return'])
@pytest.mark.parametrize('proof',['missing','denied'])
def test_rgb_yes_and_execution_flag_cannot_bypass_external_safety(edge,proof):
    ep=maneuver(edge);before=(ep.state,ep.longitudinal)
    records=clearances(ep,10,edge) if proof=='denied' else []
    for r in records:r['clear']=False
    p=ep.advance(10,answers(ep,edge),execution_committed=True,maneuver_clearances=records)
    assert (ep.state,ep.longitudinal)==before and p['action']=='STOP'
    if proof=='missing':
        assert not ep.history and ep.wait_reason.startswith('maneuver_safety_unconfirmed')
    else:
        assert ep.history[-1]['safety_denied']==[edge] and not ep.history[-1]['pending_start']
        assert not ep.uncertain and ep.wait_reason.startswith('maneuver_safety_denied')
    with pytest.raises(ValueError):ep.acknowledge(10)
    # Fresh planner evidence permits the new decision without Phase1/2 reset.
    ep.advance(11,answers(ep,edge),maneuver_clearances=clearances(ep,11,edge))
    assert ep.state!=before[0] and ep.history[-1]['pending_start']==[edge]


@pytest.mark.parametrize('field,value',[('frame_id',9),('frame_id',10.0),('instance_id','wrong'),
    ('edge','return'),('target_corridor','different'),('direction','RIGHT'),('segment_id','wrong'),
    ('route_context_id','wrong'),('source','rgb_model'),('evidence_id',''),('clear',1),
    ('rear_side_coverage_confirmed',False)])
def test_malformed_or_stale_safety_rejected_atomically(field,value):
    ep=maneuver();records=clearances(ep,10,'depart');records[0][field]=value
    before=deepcopy(ep.to_dict())
    with pytest.raises(ValueError):ep.advance(10,answers(ep,'depart'),maneuver_clearances=records)
    assert ep.to_dict()==before


def test_clearance_and_execution_are_separate_and_pending_permission_expires():
    ep=maneuver()
    ep.advance(10,answers(ep,'depart'),maneuver_clearances=clearances(ep,10,'depart'))
    assert not ep.history[-1]['committed']
    ep.advance(11,answers(ep,'depart'))
    assert ep.state=='WAIT' and ep.longitudinal=='HOLD'
    assert ep.history[-1]['expired_start']==['depart']


def test_full_loop_bad_clearance_rolls_back_every_instance_and_receipt():
    loop=Phase4Loop(rgb_mode=2)
    a=maneuver(instance_id='a');b=maneuver(instance_id='b')
    loop.establish(a,verified=True);loop.establish(b,verified=True)
    obs=dict(frame_id=10,history_frames=[6,10],speed_mps=0.)
    records=clearances(a,10,'depart')+clearances(b,10,'depart');records[-1]['target_corridor']='wrong'
    before=deepcopy(loop.snapshot())
    predictor=lambda ep,key,*args:'YES' if key=='depart' else 'NO'
    with pytest.raises(ValueError):loop.tick(obs,[None]*2,predictor,maneuver_clearances=records)
    assert loop.snapshot()==before
    records[-1]['target_corridor']='left lane'
    loop.tick(obs,[None]*2,predictor,maneuver_clearances=records)
    restored=Phase4Loop.restore(loop.snapshot())
    assert restored.snapshot()==loop.snapshot()
    old=loop.snapshot();old['version']='phase4_loop_v9'
    with pytest.raises(ValueError,match='protocol'):Phase4Loop.restore(old)


def test_replay_distinguishes_visible_yes_from_accepted_permission():
    ep=maneuver()
    r=replay(ep.to_dict(),[dict(visible_truth_scope='visible_maneuver_conditions_v1',frame_id=10,truth={'depart':'YES'},execution_committed=True)],
             lambda ep,key,obs:'YES' if key=='depart' else 'NO')
    assert r['answer_delay_frames']==[0] and r['accepted_delay_frames']==r['execution_delay_frames']==[]
    assert r['steps'][0]['prior']['action']=='STOP'


@pytest.mark.parametrize('edge',['depart','enter','return'])
def test_actual_model_message_limits_visibility_without_injecting_safety_evidence(edge):
    ep=maneuver(edge);obs=dict(frame_id=10,history_frames=[6,10],speed_mps=0.)
    msg=messages(ep,edge,obs,['one','two'])
    text=msg[1]['content'][-1]['text']
    assert 'observed camera coverage' in text and 'unseen rear or side traffic' in text
    assert 'Transition now?' in text and len(text.splitlines())==3
    assert 'evidence_id' not in text and ep.instance_id not in text


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    root=ROOT.parents[1]/'lead_data'
    if not root.exists():pytest.skip('external RGB unavailable')
    outputs={}
    for mode in (2,4):
        out=tmp_path_factory.mktemp(f'v20_{mode}')/'data'
        dataset.build(json.loads((ROOT/'reviewed_state_pairs_v7.json').read_text()),root,out,rgb_mode=mode)
        outputs[mode]=(out,*dataset.load_dataset(out,require_trainable=True))
    return outputs


@pytest.mark.parametrize('mode',[2,4])
def test_new_build_quarantines_old_gap_answers_preserves_holdout_and_exposes_missing_following(built,mode):
    out,data,m=built[mode]
    assert m['counts']==dict(train=435,val=273,test=186) and m['review_queue_count']==533
    assert not any(r['edge'] in ('depart','enter','return') for r in data['train'])
    review=dataset.read_rows(out/'review_queue.jsonl')
    quarantined=[r for r in review if r.get('review_reason')=='visible_scope_reaudit_required']
    assert len(quarantined)>128 and all(r['target']=='UNKNOWN' for r in quarantined)
    assert all(v==dict(YES=0,NO=0) for v in m['training_admission']['train_recovery_state_support'].values())
    assert len(m['training_admission']['missing_evaluation_events']['val'])==6
    assert len(m['coverage']['missing_transition_edges'])==415
    assert all(f'train/{e}/default/recover_follow' in m['coverage']['missing_transition_edges'] for e in ('U-E1','U-E5','U-E7'))
    # The frozen evaluator still binds original source; new semantics cannot claim its approval.
    plan=json.loads((ROOT/dataset.HOLDOUT_PLAN).read_text())
    assert all(file_sha(ROOT/name)==sha for name,sha in plan['frozen_rubric_sha256'].items())
    row=deepcopy(data['val'][0]);row['edge']='recover_follow'
    with pytest.raises(ValueError,match='new prospective'):dataset.check_evaluation_protocol(row,dataset.holdout_reservations())
    bad=deepcopy(quarantined[0]);bad['target']='YES'
    with pytest.raises(ValueError,match='re-audit'):visible_scope.validate_rows([bad])


def test_visible_review_covers_all_causal_rgb_and_rejects_future_or_wrong_sha(built):
    out,_,_=built[2]
    row=next(r for r in dataset.read_rows(out/'review_queue.jsonl') if r.get('review_reason'))
    proof=dict(policy=visible_scope.POLICY,reviewer='test-only',evidence_id='synthetic',frames={
        str(f):dict(rgb_sha256=h,observed_until=f,visible_conditions_reviewed=True,observation='synthetic',target='UNKNOWN')
        for f,h in zip(row['observation']['history_frames'],row['image_sha256'])})
    row['visible_scope_review']=proof
    assert visible_scope.reviewed(row)
    f=str(row['observation']['history_frames'][0]);before=deepcopy(proof)
    for field,value in [('rgb_sha256','0'*64),('observed_until',999),('visible_conditions_reviewed',False)]:
        row['visible_scope_review']=deepcopy(before);row['visible_scope_review']['frames'][f][field]=value
        with pytest.raises(ValueError):visible_scope.reviewed(row)


def test_twentieth_exposure_and_risks_are_bound_without_invented_labels():
    name='twentieth_audit_exposure_20261001.json';ledger=json.loads((ROOT/name).read_text())
    assert len(ledger['train_only_groups'])==15
    assert ledger['training_labels_added']==0
    assert contract()['calibration_assets'][name]==file_sha(ROOT/name)
    for group in ledger['train_only_groups']:
        assert all(dataset.split_for(group,dataset.groups(),seed)=='train' for seed in (1,2,2026))
    reg=risk_review.registry()
    for r in ledger['calibration_risks']:
        assert any(item['ledger']==name for item in reg[(r['scenario'],r['route_id'])])
        assert r['training_label_approved'] is False


def test_old_unscoped_lateral_truth_is_censored_instead_of_scored_as_visible_permission():
    ep=maneuver()
    r=replay(ep.to_dict(),[dict(frame_id=10,truth={'depart':'YES'},execution_committed=True,
                               maneuver_clearances=clearances(ep,10,'depart'))],
             lambda ep,key,obs:'YES' if key=='depart' else 'NO')
    assert r['answer_delay_frames']==r['accepted_delay_frames']==r['execution_delay_frames']==[]
    assert r['steps'][0]['execution_confirmed']==['depart']


@pytest.mark.parametrize('fault',['duplicate','unknown_instance','inactive_instance'])
def test_loop_rejects_unconsumed_safety_evidence(fault):
    ep=maneuver();loop=Phase4Loop(rgb_mode=2);loop.establish(ep,verified=True)
    records=clearances(ep,10,'depart')
    if fault=='duplicate':records+=deepcopy(records)
    elif fault=='unknown_instance':records[0]['instance_id']='other'
    else:ep.suspended=True
    before=deepcopy(loop.snapshot())
    with pytest.raises(ValueError):
        loop.tick(dict(frame_id=10,history_frames=[6,10],speed_mps=0.),[None]*2,
                  lambda ep,key,*a:'YES' if key=='depart' else 'NO',maneuver_clearances=records)
    assert loop.snapshot()==before
