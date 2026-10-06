import json
import pytest
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT,file_sha
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.calibration import reviewed_band_labels
from qwen3vl_local.sft_new_loop_phase4.dataset import build,load_dataset,groups,split_for


def annotations():
    return json.loads((ROOT/'reviewed_state_pairs_v4.json').read_text())


def test_increment_preserves_previous_annotations_and_binds_original_review_evidence():
    anns=annotations();old=json.loads((ROOT/'reviewed_state_pairs_v3.json').read_text())
    audit=json.loads((ROOT/'rgb_boundary_review_v4.json').read_text())
    assert len(anns)==61 and anns[:57]==old
    assert audit['annotation_sha256']==file_sha(ROOT/'reviewed_state_pairs_v4.json')
    assert audit['parent_annotation_sha256']==file_sha(ROOT/'reviewed_state_pairs_v3.json')
    originals={r['rgb']:r['sha256'] for r in audit['original_resolution_review']}
    for a in anns[57:]:
        for frame,sha in a['frame_sha256'].items():
            assert originals[f"{a['scenario']}/{a['route_id']}/rgb/{int(frame):04d}.jpg"]==sha
    assert audit['local_review']['unique_existing_rgb_frames']==169
    assert audit['local_review']['original_resolution_frames']==43
    assert audit['local_review']['new_complete_sequences']==0
    assert audit['newly_exposed_groups']==[]
    assert not audit['reported_full_coverage']['all_requested_rgb_audit_complete']
    exposed=groups()
    for group in audit['train_only_groups']:
        for seed in (1,2026,20260930):assert split_for(group,exposed,seed)=='train'


@pytest.mark.parametrize('evidence,no,unknown,ready,catchup,stop',[
    ('sixth_audit_649_proceed',range(32,37),[37],[38],[39,40],41),
    ('sixth_audit_293_proceed',[],range(114,117),[117],[118,119,120],121),
    ('seventh_audit_406_proceed',[48,49,50],range(51,56),[56,57],[],58),
    ('seventh_audit_378_proceed',[48,49,50],[51,52,53],[54],[55,56,57],58),
])
def test_reviewed_route_boundaries_preserve_uncertainty_and_cutoffs(evidence,no,unknown,ready,catchup,stop):
    a=next(a for a in annotations() if a['evidence_id']==evidence)
    labels=reviewed_band_labels(Episode(**a['episode']),a['edge'],a['transition_band'])
    expected={f:('NO','readiness') for f in no}
    expected.update({f:('UNKNOWN','uncertain') for f in unknown})
    expected.update({f:('YES','readiness') for f in ready})
    expected.update({f:('YES','catchup') for f in catchup})
    assert labels==expected and stop not in labels


@pytest.mark.parametrize('mode',[2,4])
def test_new_bands_build_only_reviewed_questions_and_do_not_fill_missing_categories(tmp_path,mode):
    data_root=ROOT.parents[1]/'lead_data'
    if not data_root.exists():pytest.skip('external RGB unavailable')
    output=tmp_path/'increment'
    m=build(annotations()[57:],data_root,output,rgb_mode=mode)
    data,_=load_dataset(output)
    rows=data['train']
    assert m['counts']=={'train':24} and m['review_queue_count']==12 and not m['ready']
    assert sum(r['target']=='YES' for r in rows)==13
    assert sum(r['slice']=='catchup' for r in rows)==8
    assert {r['edge'] for r in rows}=={'proceed'}
    assert not data['val'] and not data['test']
    with pytest.raises(ValueError,match='support'):load_dataset(output,require_complete_coverage=True)
    for r in rows:
        assert max(r['observation']['history_frames'])==r['observation']['frame_id']
        assert len(r['images'])==mode
        assert all(r['observation']['frame_id']<a['transition_band']['stop']['frame']
                   for a in annotations()[57:] if a['evidence_id']==r['evidence_id'])


def test_ue4_visible_clearance_licenses_motion_without_inventing_execution_or_completion():
    a=next(a for a in annotations() if a['evidence_id']=='seventh_audit_406_proceed')
    assert a['transition_band']['reference_kind']=='condition_onset'
    assert a['transition_band']['stop']['kind']=='calibration_anomaly'
    for frame in (56,57):
        ep=Episode(**a['episode'])
        target=reviewed_band_labels(ep,'proceed',a['transition_band'])[frame]
        assert target==('YES','readiness')
        prior=ep.advance(frame,{q.key:target[0] if q.key=='proceed' else 'NO' for q in ep.questions()})
        assert prior['action']=='RESUME' and not ep.finished
        assert ep.history[-1]['pending_start']==['proceed'] and not ep.history[-1]['committed']
        ep.refresh_uncommitted()
        assert ep.state=='YIELD' and ep.longitudinal=='HOLD'
