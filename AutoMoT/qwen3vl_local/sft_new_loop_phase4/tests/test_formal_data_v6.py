import copy
import json
import shutil

import pytest

from qwen3vl_local.sft_new_loop_phase4 import dataset
from qwen3vl_local.sft_new_loop_phase4.calibration import reviewed_band_labels
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT, contract, file_sha


def read(name):
    return json.loads((ROOT/name).read_text())


def test_v6_preserves_prior_annotations_and_full_causal_review():
    old = read('reviewed_state_pairs_v5.json')
    new = read('reviewed_holdout_state_pairs_v2.json')
    assert read('reviewed_state_pairs_v6.json') == old + new
    assert len(new) == 36
    review = read('rgb_holdout_review_v2.json')
    assert review['annotation_sha256'] == file_sha(ROOT/'reviewed_holdout_state_pairs_v2.json')
    assert review['plan_sha256'] == file_sha(ROOT/dataset.HOLDOUT_PLAN)
    assert len(review['sequences']) == 12
    assert sum(len(p['frames']) for s in review['sequences'] for p in s['pages']) == 662
    assert sum(len(s['native_frames']) for s in review['sequences']) == 244
    for a in new:
        evidence = next(s for s in review['sequences'] if a['route_id'] == s['sequence']['route_id'])
        assert a['evidence_id'] in evidence['evidence_ids']
        all_frames = {f['frame'] for p in evidence['pages'] for f in p['frames']}
        for frame, sha in a['frame_sha256'].items():
            assert evidence['native_frames'][frame] == sha
            assert {int(frame)+offset for offset in (-6, -4, -2, 0)} <= all_frames
    for name in ('reviewed_state_pairs_v6.json', 'reviewed_holdout_state_pairs_v2.json', 'rgb_holdout_review_v2.json'):
        assert contract()['calibration_assets'][name] == file_sha(ROOT/name)


@pytest.mark.parametrize('index,onset,last_wait,first_motion,stop', [
    (5,43,48,49,53), (6,34,37,38,41), (10,47,49,50,53), (11,76,81,82,86),
])
def test_clear_corridor_before_ego_motion_is_readiness(index,onset,last_wait,first_motion,stop):
    a = next(a for a in read('reviewed_holdout_state_pairs_v2.json')
             if a['evidence_id'].startswith(f'holdout_v2_formal_data_{index}_') and a['edge'] == 'proceed')
    labels = reviewed_band_labels(Episode(**a['episode']), a['edge'], a['transition_band'])
    assert all(labels[f] == ('YES','readiness') for f in range(onset,last_wait+1))
    assert labels[first_motion] == ('YES','catchup')
    assert stop not in labels


def test_ue7_crossing_despite_green_does_not_become_permission():
    a = next(a for a in read('reviewed_holdout_state_pairs_v2.json')
             if a['evidence_id'].startswith('holdout_v2_formal_data_7_') and a['edge'] == 'proceed')
    labels = reviewed_band_labels(Episode(**a['episode']), a['edge'], a['transition_band'])
    assert all(labels[f] == ('NO','readiness') for f in range(26,32))
    assert labels[32] == labels[33] == ('UNKNOWN','uncertain')
    assert labels[34] == labels[35] == ('YES','catchup')
    assert ('YES','readiness') not in labels.values()


@pytest.mark.parametrize('index', [0, 3])
def test_unconfirmed_event_context_cannot_supervise_completion(index):
    a = next(a for a in read('reviewed_holdout_state_pairs_v2.json')
             if a['evidence_id'].startswith(f'holdout_v2_formal_data_{index}_'))
    assert a['context_valid'] is None
    labels = reviewed_band_labels(Episode(**a['episode']), a['edge'], a['transition_band'],
                                  context_valid=a['context_valid'])
    assert labels and set(labels.values()) == {('UNKNOWN', 'uncertain')}
    evidence = read('rgb_holdout_review_v2.json')['sequences'][index]
    assert len(evidence['causal_context_check']) == evidence['sequence']['frame_count']
    assert all(x['current_active_scenario_type'] is None and
               x['previous_active_scenario_type'] is None for x in evidence['causal_context_check'])


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    data_root = ROOT.parents[1]/'lead_data'
    if not data_root.exists():
        pytest.skip('external RGB unavailable')
    folder = tmp_path_factory.mktemp('p4_v6')/'data'
    dataset.build(read('reviewed_state_pairs_v6.json'),data_root,folder,rgb_mode=4)
    return folder


def test_new_data_keeps_train_pool_and_formal_support_gate(built):
    rows, manifest = dataset.load_dataset(built)
    assert manifest['counts'] == {'train':423,'val':163,'test':149}
    assert manifest['review_queue_count'] == 492
    assert len(manifest['coverage']['missing_transition_edges']) == 416
    assert len(manifest['coverage']['missing_segmented_route_support']) == 12
    assert all(not r['evidence_id'].startswith('holdout_') for r in rows['train'])
    with pytest.raises(ValueError,match='support'):
        dataset.load_dataset(built,require_complete_coverage=True)


@pytest.mark.parametrize('target_split', ['train','test'])
def test_loader_rejects_reassigned_reserved_route_even_with_updated_hashes(built,tmp_path,target_split):
    folder = tmp_path/'data'
    shutil.copytree(built,folder)
    rows = {s:dataset.read_rows(folder/f'{s}.jsonl') for s in ('train','val','test')}
    group = rows['val'][0]['physical_group']
    moved = [r for r in rows['val'] if r['physical_group'] == group]
    rows['val'] = [r for r in rows['val'] if r['physical_group'] != group]
    for row in moved:
        row['split'] = target_split
    rows[target_split] += moved
    manifest = json.loads((folder/'manifest.json').read_text())
    for split, data in rows.items():
        dataset.dump_rows(folder/f'{split}.jsonl',data)
        manifest['files'][f'{split}.jsonl'] = file_sha(folder/f'{split}.jsonl')
        manifest['counts'][split] = len(data)
    (folder/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='ownership'):
        dataset.load_dataset(folder)


def test_review_queue_also_obeys_reserved_ownership(built,tmp_path):
    folder = tmp_path/'data'
    shutil.copytree(built,folder)
    rows = dataset.read_rows(folder/'review_queue.jsonl')
    next(r for r in rows if r['split'] == 'val')['split'] = 'train'
    dataset.dump_rows(folder/'review_queue.jsonl',rows)
    manifest = json.loads((folder/'manifest.json').read_text())
    manifest['review_queue_sha256'] = file_sha(folder/'review_queue.jsonl')
    (folder/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='ownership'):
        dataset.load_dataset(folder)


def test_reserved_annotation_requires_label_only_protocol(tmp_path):
    a = copy.deepcopy(read('reviewed_holdout_state_pairs_v2.json')[0])
    del a['evaluation_annotation_protocol']
    with pytest.raises(ValueError,match='label-only protocol'):
        dataset.build([a],ROOT.parents[1]/'lead_data',tmp_path/'data')


def test_loader_rechecks_frozen_evaluation_rubric(built,monkeypatch):
    monkeypatch.setattr(dataset,'file_sha',lambda p: '0'*64 if p.name == 'prompts.py' else file_sha(p))
    with pytest.raises(ValueError,match='frozen evaluation rubric changed'):
        dataset.load_dataset(built)


def test_builder_rechecks_frozen_evaluation_rubric(tmp_path,monkeypatch):
    monkeypatch.setattr(dataset,'file_sha',lambda p: '0'*64 if p.name == 'calibration.py' else file_sha(p))
    with pytest.raises(ValueError,match='frozen evaluation rubric changed'):
        dataset.build([],ROOT.parents[1]/'lead_data',tmp_path/'data')
