import json
import pytest
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT,contract,file_sha
from qwen3vl_local.sft_new_loop_phase4.dataset import groups,split_for


@pytest.mark.parametrize('group',[
    'noScenarios/Town05_Town05_rl_10',
    'NonSignalizedJunctionLeftTurn/Town07_route_000913',
    'noScenarios/Town04_route_000531',
    'AccidentTwoWays/Town07_route_001446',
])
def test_tenth_audit_exposed_routes_never_enter_holdout(group):
    audit=json.loads((ROOT/'tenth_audit_exposure_20260930.json').read_text())
    assert group in audit['train_only_groups']
    for seed in range(100):assert split_for(group,groups(),seed)=='train'


def test_exposure_provenance_bound_without_changing_training_labels():
    name='tenth_audit_exposure_20260930.json';audit=json.loads((ROOT/name).read_text())
    assert contract()['calibration_assets'][name]==file_sha(ROOT/name)
    assert len(audit['train_only_groups'])==4
    assert not audit['reported_coverage']['all_requested_rgb_audit_complete']
    assert audit['local_review']['new_complete_sequences']==audit['local_review']['new_labels']==0
    assert audit['local_review']['existing_rgb_frames']==72
    annotations=json.loads((ROOT/'reviewed_state_pairs_v4.json').read_text())
    used={(a['scenario'],a['route_id']) for a in annotations}
    assert len(audit['calibration_risks'])==5
    for risk in audit['calibration_risks']:
        assert (risk['scenario'],risk['route_id']) not in used
        assert risk['status']=='requires_per_frame_calibration_review_not_training_labels'
        assert [e['frame'] for e in risk['rgb_evidence']]==risk['frames']
