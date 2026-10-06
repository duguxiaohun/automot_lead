import json
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT,contract,file_sha
from qwen3vl_local.sft_new_loop_phase4.dataset import groups,split_for


def test_new_exposure_overrides_seeds_that_would_have_been_holdout():
    group='ParkedObstacleTwoWays/Town12_4608_0'
    heldout=[seed for seed in range(100) if split_for(group,set(),seed)!='train']
    assert heldout
    for seed in range(100):assert split_for(group,groups(),seed)=='train'


def test_report_evidence_bound_without_promoting_unreviewed_labels():
    name='eleventh_audit_exposure_20260930.json';audit=json.loads((ROOT/name).read_text())
    assert contract()['calibration_assets'][name]==file_sha(ROOT/name)
    assert audit['train_only_groups']==['ParkedObstacleTwoWays/Town12_4608_0']
    assert audit['local_review']['existing_rgb_frames']==34
    assert audit['local_review']['new_labels']==audit['local_review']['new_complete_sequences']==0
    assert not audit['reported_coverage']['all_requested_rgb_audit_complete']
    used={(a['scenario'],a['route_id']) for a in json.loads((ROOT/'reviewed_state_pairs_v4.json').read_text())}
    for r in audit['calibration_risks']:
        assert (r['scenario'],r['route_id']) not in used
        assert r['status']=='requires_per_frame_calibration_review_not_training_labels'
    corrected=next(r for r in audit['calibration_risks'] if r['manifest_id']=='569')
    assert [e['frame'] for e in corrected['rgb_evidence']]==[74,75]
