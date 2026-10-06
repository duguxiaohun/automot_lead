import json
import pytest
from qwen3vl_local.sft_new_loop_phase4 import dataset,risk_review
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT,contract,file_sha

NAME='twelfth_thirteenth_audit_exposure_20260930.json'


def test_latest_risk_sources_and_rgb_identity_are_bound():
    audit=json.loads((ROOT/NAME).read_text())
    assert contract()['calibration_assets'][NAME]==file_sha(ROOT/NAME)
    assert len(audit['calibration_risks'])==33
    assert len({(r['scenario'],r['route_id']) for r in audit['calibration_risks']})==32
    assert audit['local_review']['new_labels']==0
    assert not audit['reported_coverage']['all_requested_rgb_audit_complete']
    known=risk_review.registry();annotations=json.loads((ROOT/'reviewed_state_pairs_v7.json').read_text())
    used={(a['scenario'],a['route_id']) for a in annotations}
    for record in audit['calibration_risks']:
        key=record['scenario'],record['route_id']
        assert key not in used
        assert any(r['ledger']==NAME and r['record']==record for r in known[key])
        for f in record['rgb_evidence']:
            assert file_sha(ROOT.parents[1]/'lead_data'/f['path'])==f['sha256']
    thirteen=[r for r in audit['calibration_risks'] if r['source_audit']=='thirteenth']
    corrected={r['audit_id']:[f['frame'] for f in r['rgb_evidence']] for r in thirteen}
    assert corrected['008']==[86,87] and corrected['017']==[32,33]
    assert len([r for r in thirteen if r['audit_id']=='009'])==2


def test_latest_exposure_never_becomes_holdout():
    audit=json.loads((ROOT/NAME).read_text());exposed=dataset.groups()
    assert set(audit['train_only_groups'])<=exposed
    group='AccidentTwoWays/Town02_route_001609'
    assert any(dataset.split_for(group,set(),s)!='train' for s in range(100))
    assert all(dataset.split_for(group,exposed,s)=='train' for s in range(100))
    assert not set(audit['train_only_groups']) & set(dataset.holdout_reservations())


@pytest.mark.parametrize('mode',[2,4])
def test_new_disappearance_and_identity_risks_require_causal_review(mode):
    # Exercise every new registered route, not only the old risk fixtures.
    audit=json.loads((ROOT/NAME).read_text());known=risk_review.registry()
    for r in audit['calibration_risks']:
        f=max(10,r['rgb_evidence'][0]['frame'])
        annotation=dict(scenario=r['scenario'],route_id=r['route_id'],
                        label_basis='reviewed_transition_band',
                        transition_band=dict(review_start=f,review_end=f,stop={'frame':f+1}))
        with pytest.raises(ValueError,match='explicit per-frame risk_review'):
            risk_review.annotation_review(annotation,ROOT.parents[1]/'lead_data',mode,known)
