from collections import Counter
from copy import deepcopy
import json
import pytest
from qwen3vl_local.sft_new_loop_phase4 import dataset,risk_review
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT,contract,file_sha

NAME='sixteenth_audit_exposure_20260930.json'
DATA=ROOT.parents[1]/'lead_data'


def ledger():return json.loads((ROOT/NAME).read_text())


def test_evidence_and_kinds_registered_without_label_or_split_changes():
    audit=ledger();records=audit['calibration_risks'];known=risk_review.registry()
    assert contract()['calibration_assets'][NAME]==file_sha(ROOT/NAME)
    assert len(records)==12 and len({(r['scenario'],r['route_id']) for r in records})==11
    assert Counter(r['kind'] for r in records)==dict(visual_actor_discontinuity=9,
        ego_camera_roll_near_bypass=1,cyclist_elevated_or_overlapping_car=1,cyclist_pose_and_participant_identity=1)
    assert audit['new_train_only_groups']==[] and len(audit['train_only_groups'])==12
    assert audit['local_review']==dict(new_labels=0,new_complete_sequences=0,new_visual_frames=0,rgb_hash_checks=29)
    assert not audit['reported_coverage']['all_requested_rgb_audit_complete']
    assert set(audit['train_only_groups'])<=dataset.groups()
    assert not set(audit['train_only_groups']) & set(dataset.holdout_reservations())
    used={(a['scenario'],a['route_id']) for a in json.loads((ROOT/'reviewed_state_pairs_v7.json').read_text())}
    for r in records:
        key=r['scenario'],r['route_id'];assert key not in used
        assert any(v['ledger']==NAME and v['record']==r for v in known[key])
        assert not r['training_label_approved']
        for f in r['rgb_evidence']:assert file_sha(DATA/f['path'])==f['sha256']
    # The natural exit at 131->132 must not be relabelled as discontinuity.
    pose=next(r for r in records if r['kind']=='cyclist_pose_and_participant_identity')
    assert [f['frame'] for f in pose['rgb_evidence']]==[99,100,101,105]
    assert '不是突变消失' in pose['reported_observations']
    assert not any('Scenario4_47_' in r['route_id'] for r in records)


@pytest.mark.parametrize('mode',[2,4])
def test_every_new_risk_route_requires_full_current_registry_review(mode):
    known=risk_review.registry()
    for r in ledger()['calibration_risks']:
        f=max(10,r['rgb_evidence'][0]['frame'])
        a=dict(scenario=r['scenario'],route_id=r['route_id'],label_basis='reviewed_transition_band',
               transition_band=dict(review_start=f,review_end=f,stop={'frame':f+1}))
        with pytest.raises(ValueError,match='explicit per-frame'):
            risk_review.annotation_review(a,DATA,mode,known)
    # This route has BOTH disappearance and camera-roll records. Reviewing just
    # one must not authorize either the current image or its historical inputs.
    r=next(r for r in ledger()['calibration_risks'] if r['kind']=='ego_camera_roll_near_bypass')
    risks=known[(r['scenario'],r['route_id'])]
    assert len(risks)==2
    incomplete=dict(policy=risk_review.POLICY,risk_ids=[risks[0]['risk_id']])
    with pytest.raises(ValueError,match='current registered evidence'):
        risk_review.validate_review(incomplete,risks,[],[])


@pytest.mark.parametrize('mode',[2,4])
@pytest.mark.parametrize('state',['uncertain','excluded'])
def test_pose_risk_in_history_keeps_plausible_current_yes_pending(tmp_path,mode,state):
    # Synthetic routing test only, not new supervision or approved visual facts.
    from qwen3vl_local.sft_new_loop_phase4.tests.test_risk_admission import annotation
    a=annotation();r=next(r for r in ledger()['calibration_risks'] if r['kind']=='cyclist_elevated_or_overlapping_car')
    a.update(scenario=r['scenario'],route_id=r['route_id'])
    a['frame_sha256']={str(f):file_sha(DATA/r['scenario']/r['route_id']/'rgb'/f'{f:04d}.jpg') for f in range(89,92)}
    proof=risk_review.templates([a],DATA,mode)['annotations'][0]['risk_review']
    proof.update(reviewer='SYNTHETIC test',evidence_id='SYNTHETIC no label approval')
    for item in proof['frames'].values():
        item.update(status='usable',reason='SYNTHETIC test',**{k:True for k in risk_review.CHECKS})
    proof['frames']['85']['status']=state;a['risk_review']=proof
    out=tmp_path/'data';dataset.build([a],DATA,out,rgb_mode=mode)
    data,_=dataset.load_dataset(out)
    queued=dataset.read_rows(out/'review_queue.jsonl')
    row=next(r for r in queued if r['observation']['frame_id']==89)
    assert row['target']=='UNKNOWN' and row['risk_review']['disposition']==state
    assert not any(r['observation']['frame_id']==89 for rows in data.values() for r in rows)
