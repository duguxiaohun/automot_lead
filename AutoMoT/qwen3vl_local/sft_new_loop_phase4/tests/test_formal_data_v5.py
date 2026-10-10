import json
import pytest
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT, file_sha, contract
from qwen3vl_local.sft_new_loop_phase4.dataset import build, load_dataset, groups, split_for
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.calibration import reviewed_band_labels


def read(name):
    return json.loads((ROOT / name).read_text())


def test_v5_preserves_labels_and_binds_separate_review_evidence():
    old = read('reviewed_state_pairs_v4.json')
    new = read('reviewed_state_pairs_v5.json')
    heldout = read('reviewed_holdout_state_pairs_v1.json')
    dev = read('rgb_boundary_review_v5.json')
    review = read('rgb_holdout_review_v1.json')
    assert len(old) == 61 and new[:61] == old
    assert len(new) == 93 and new[82:] == heldout
    assert dev['annotation_sha256'] == file_sha(ROOT/'reviewed_state_pairs_v5.json')
    assert review['annotation_sha256'] == file_sha(ROOT/'reviewed_holdout_state_pairs_v1.json')
    originals = {r['rgb']: r['sha256'] for r in dev['original_resolution_review']}
    for sequence in review['sequences']:
        r = sequence['sequence']
        originals.update({f"{r['scenario']}/{r['route_id']}/rgb/{int(f):04d}.jpg": h
                          for f, h in sequence['native_frames'].items()})
    for annotation in new[61:]:
        for frame, sha in annotation['frame_sha256'].items():
            assert originals[f"{annotation['scenario']}/{annotation['route_id']}/rgb/{int(frame):04d}.jpg"] == sha
    assets = contract()['calibration_assets']
    for name in ('reviewed_state_pairs_v5.json', 'rgb_boundary_review_v5.json',
                 'reviewed_holdout_state_pairs_v1.json', 'rgb_holdout_review_v1.json',
                 'formal_holdout_plan_20260930.json'):
        assert assets[name] == file_sha(ROOT/name)


def test_prospective_evaluation_reservations_do_not_become_training_with_another_seed():
    plan = read('formal_holdout_plan_20260930.json')
    review = read('rgb_holdout_review_v1.json')
    exposed = groups()
    assert review['plan_sha256'] == file_sha(ROOT/'formal_holdout_plan_20260930.json')
    for name, sha in plan['frozen_rubric_sha256'].items():
        assert file_sha(ROOT/name) == sha
    owners = {r['physical_group']: r['split'] for r in plan['reservations']}
    assert len(owners) == 51
    for group, owner in owners.items():
        assert group not in exposed and owner in ('val', 'test')
        for seed in (1, 2026, 20260929):
            assert split_for(group, exposed, seed) == owner
        with pytest.raises(ValueError, match='development exposure'):
            split_for(group, exposed | {group})


def test_first_visible_execution_does_not_fill_readiness_release_gap():
    a = next(a for a in read('reviewed_state_pairs_v5.json')
             if a['evidence_id'] == 'formal_data_294_U-E1_release_68')
    labels = reviewed_band_labels(Episode(**a['episode']), a['edge'], a['transition_band'])
    assert labels[70] == ('NO', 'readiness')
    assert labels[71] == labels[72] == ('UNKNOWN', 'uncertain')
    assert labels[73] == ('YES', 'catchup')
    assert not any(label == ('YES', 'readiness') for label in labels.values())


def test_dark_evaluation_window_does_not_manufacture_a_positive():
    a = next(a for a in read('reviewed_holdout_state_pairs_v1.json')
             if 'Town12_Rep0_2325_' in a['route_id'] and a['edge'] == 'proceed')
    labels = reviewed_band_labels(Episode(**a['episode']), a['edge'], a['transition_band'])
    assert all(labels[f] == ('UNKNOWN', 'uncertain') for f in range(83, 91))
    assert not any(label[0] == 'YES' for label in labels.values())
    assert 91 not in labels


@pytest.mark.parametrize('mode', [2, 4])
def test_increment_builds_three_isolated_splits_without_relaxing_readiness(tmp_path, mode):
    data_root = ROOT.parents[1]/'lead_data'
    if not data_root.exists():
        pytest.skip('external RGB unavailable')
    manifest = build(read('reviewed_state_pairs_v5.json'), data_root, tmp_path/'data', rgb_mode=mode)
    data, _ = load_dataset(tmp_path/'data')
    assert manifest['counts'] == {'train': 423, 'val': 33, 'test': 25}
    assert manifest['review_queue_count'] == 370
    assert len(manifest['coverage']['missing_segmented_route_support']) == 12
    assert manifest['trainable'] and not manifest['complete_coverage']
    with pytest.raises(ValueError, match='support'):
        load_dataset(tmp_path/'data', require_complete_coverage=True)
    owners = {}
    for split, rows in data.items():
        for row in rows:
            group = row['physical_group']
            assert owners.setdefault(group, split) == split
            assert len(row['images']) == mode
            assert max(row['observation']['history_frames']) == row['observation']['frame_id']
            assert row['evidence_id'].startswith('holdout_v1_') == (split != 'train')
