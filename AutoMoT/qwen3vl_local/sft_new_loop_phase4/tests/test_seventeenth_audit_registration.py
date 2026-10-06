import json
import pytest
from qwen3vl_local.sft_new_loop_phase4 import dataset,risk_review
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT,contract,file_sha,digest

NAME='seventeenth_audit_exposure_20261001.json'
DATA=ROOT.parents[1]/'lead_data'


def ledger():return json.loads((ROOT/NAME).read_text())


def test_registered_evidence_preserves_corrections_and_uncertainty():
    audit=ledger();records=audit['calibration_risks'];known=risk_review.registry()
    assert contract()['calibration_assets'][NAME]==file_sha(ROOT/NAME)
    assert len(records)==20 and len({(r['scenario'],r['route_id']) for r in records})==14
    assert len(audit['train_only_groups'])==17 and len(audit['new_train_only_groups'])==2
    assert audit['local_review']==dict(new_labels=0,new_complete_sequences=0,new_visual_frames=0,rgb_hash_checks=40)
    assert not audit['reported_coverage']['all_requested_rgb_audit_complete']
    used={(a['scenario'],a['route_id']) for a in json.loads((ROOT/'reviewed_state_pairs_v7.json').read_text())}
    for r in records:
        key=r['scenario'],r['route_id'];assert key not in used
        assert any(v['ledger']==NAME and v['record']==r for v in known[key])
        assert not r['training_label_approved']
        for f in r['rgb_evidence']:assert file_sha(DATA/f['path'])==f['sha256']
    locations={(r['audit_id'],tuple(f['frame'] for f in r['rgb_evidence'])) for r in records}
    assert ('015',(127,128)) in locations and ('015',(128,129)) not in locations
    for candidate in [('002',(7,8)),('006',(67,68)),('007',(17,18))]:assert candidate not in locations
    assert ('008',(41,42)) in locations and ('010',(51,52)) in locations


@pytest.mark.parametrize('group',[
    'NonSignalizedJunctionLeftTurnEnterFlow/Town03_route_001040',
    'NonSignalizedJunctionLeftTurnEnterFlow/Town03_route_001041'])
def test_new_development_exposure_overrides_seed_and_rejects_old_frozen_pool(tmp_path,group):
    from qwen3vl_local.sft_new_loop_phase4.candidate_pool import scan,validate
    audit=ledger();exposed=dataset.groups();previous=exposed-set(audit['new_train_only_groups'])
    assert dataset.split_for(group,previous)==audit['new_group_previous_split'][group]
    assert all(dataset.split_for(group,exposed,seed)=='train' for seed in range(100))
    assert not set(audit['train_only_groups'])&set(dataset.holdout_reservations())
    # Exercise the actual candidate scanner on a route from the newly isolated group.
    r=next(r for r in audit['calibration_risks'] if r['physical_group']==group)
    rgb=tmp_path/'source'/r['scenario']/r['route_id']/'rgb';rgb.mkdir(parents=True)
    (rgb/'0010.jpg').write_bytes(b'inventory-only synthetic fixture')
    (rgb.parent/'metas').mkdir()
    (rgb.parent/'metas'/'0010.pkl').write_bytes(b'inventory-only synthetic fixture')
    pool=scan(tmp_path/'source',tmp_path/'pool.json')
    assert pool['routes'][0]['physical_group']==group and pool['routes'][0]['split']=='train'
    # Even a rehashed inventory with stale exposure ownership cannot be reused.
    pool['exposure_sha256']=digest(sorted(previous))
    pool['routes'][0]['split']=audit['new_group_previous_split'][group]
    pool['sha256']=digest({k:v for k,v in pool.items() if k!='sha256'})
    with pytest.raises(ValueError,match='exposure'):validate(pool)


@pytest.mark.parametrize('mode',[2,4])
def test_all_new_risk_routes_require_review_and_multi_incidents_cannot_be_omitted(mode):
    known=risk_review.registry();records=ledger()['calibration_risks']
    for r in records:
        f=max(10,r['rgb_evidence'][0]['frame'])
        a=dict(scenario=r['scenario'],route_id=r['route_id'],label_basis='reviewed_transition_band',
               transition_band=dict(review_start=f,review_end=f,stop={'frame':f+1}))
        with pytest.raises(ValueError,match='explicit per-frame'):
            risk_review.annotation_review(a,DATA,mode,known)
    r=next(r for r in records if r['audit_id']=='011')
    risks=known[(r['scenario'],r['route_id'])];assert len(risks)>1
    with pytest.raises(ValueError,match='current registered evidence'):
        risk_review.validate_review(dict(policy=risk_review.POLICY,risk_ids=[risks[0]['risk_id']]),risks,[],[])
