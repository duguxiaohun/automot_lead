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


def test_review_binds_all_nine_sequences_and_native_boundaries():
    anns = read('reviewed_holdout_state_pairs_v3.json')
    assert len(anns) == 23
    assert read('reviewed_state_pairs_v7.json') == read('reviewed_state_pairs_v6.json') + anns
    review = read('rgb_holdout_review_v3.json')
    assert review['annotation_sha256'] == file_sha(ROOT/'reviewed_holdout_state_pairs_v3.json')
    assert review['plan_sha256'] == file_sha(ROOT/dataset.HOLDOUT_PLAN)
    assert len(review['sequences']) == 9
    assert sum(len(p['frames']) for s in review['sequences'] for p in s['pages']) == 1084
    assert sum(len(s['native_frames']) for s in review['sequences']) == 207
    for a in anns:
        s = next(s for s in review['sequences'] if s['sequence']['route_id'] == a['route_id'])
        assert a['evidence_id'] in s['evidence_ids']
        frames = {f['frame'] for p in s['pages'] for f in p['frames']}
        for f, sha in a['frame_sha256'].items():
            assert s['native_frames'][f] == sha
            assert {int(f)+o for o in (-6,-4,-2,0)} <= frames
        assert len(s['causal_context_check']) == s['sequence']['frame_count']
    for name in ('reviewed_state_pairs_v7.json','reviewed_holdout_state_pairs_v3.json','rgb_holdout_review_v3.json'):
        assert contract()['calibration_assets'][name] == file_sha(ROOT/name)


def test_invading_turn_motion_never_fills_readiness_permission_gap():
    anns = [a for a in read('reviewed_holdout_state_pairs_v3.json') if a['edge'] in ('proceed','release')]
    assert len(anns) == 14
    for a in anns:
        labels = reviewed_band_labels(Episode(**a['episode']),a['edge'],a['transition_band'])
        assert ('YES','readiness') not in labels.values()
        assert ('NO','readiness') in labels.values()
        if not a['evidence_id'].startswith('holdout_v3_formal_data_0_'):
            assert ('YES','catchup') in labels.values()


@pytest.mark.parametrize('index,stop', [(0,130),(1,68),(2,96),(6,113),(7,73),(8,184)])
def test_spawn_anomaly_stops_old_completion_question(index,stop):
    a = next(a for a in read('reviewed_holdout_state_pairs_v3.json')
             if a['evidence_id'].startswith(f'holdout_v3_formal_data_{index}_') and a['edge']=='complete')
    assert a['transition_band']['stop']['kind'] == 'calibration_anomaly'
    labels = reviewed_band_labels(Episode(**a['episode']),a['edge'],a['transition_band'])
    assert all(f < stop for f in labels)
    if index != 8:
        assert not any(answer == 'YES' for answer, _ in labels.values())
    else:
        assert labels[181] == ('YES','readiness')  # causal completion before later anomaly


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    data_root = ROOT.parents[1]/'lead_data'
    if not data_root.exists():
        pytest.skip('external RGB unavailable')
    folder = tmp_path_factory.mktemp('p4_v7')/'data'
    dataset.build(read('reviewed_state_pairs_v7.json'),data_root,folder,rgb_mode=4)
    return folder


def resign(folder,name):
    m = json.loads((folder/'manifest.json').read_text())
    if name=='review_queue.jsonl':
        m['review_queue_sha256'] = file_sha(folder/name)
    else:
        m['files'][name] = file_sha(folder/name)
        m['counts'][name[:-6]] = len(dataset.read_rows(folder/name))
    (folder/'manifest.json').write_text(json.dumps(m))


@pytest.mark.parametrize('name', ['val.jsonl','review_queue.jsonl'])
@pytest.mark.parametrize('change', ['physical_group','route_id','history_route','history_frame','absolute_image','parent_route','missing_hash'])
def test_loader_rejects_false_source_identity_even_after_resigning(built,tmp_path,name,change):
    folder=tmp_path/'data';shutil.copytree(built,folder)
    rows=dataset.read_rows(folder/name);r=next(r for r in rows if r['split']=='val')
    if change=='physical_group': r['physical_group']='InvadingTurn/Town12_fake'
    elif change=='route_id': r['route_id']='Town12_Rep0_fake_0_route0'
    elif change=='history_route': r['images'][0]=r['images'][0].replace(r['route_id'],'Town12_Rep0_other_0_route0')
    elif change=='history_frame': r['images'][0]=r['images'][-1]
    elif change=='absolute_image': r['images'][0]='/'+r['images'][0]
    elif change=='parent_route': r['route_id']='../'+r['route_id']
    else: r['image_sha256'].pop()
    dataset.dump_rows(folder/name,rows);resign(folder,name)
    with pytest.raises(ValueError,match='source'):
        dataset.load_dataset(folder)


def test_forged_group_cannot_move_reserved_images_into_training(built,tmp_path):
    folder=tmp_path/'data';shutil.copytree(built,folder)
    val=dataset.read_rows(folder/'val.jsonl');train=dataset.read_rows(folder/'train.jsonl')
    group=val[0]['physical_group'];moved=[r for r in val if r['physical_group']==group]
    val=[r for r in val if r['physical_group']!=group]
    fake=next(f'HardBreakRoute/Town13_unreserved_{i}' for i in range(100)
              if dataset.split_for(f'HardBreakRoute/Town13_unreserved_{i}',dataset.groups())=='train')
    for r in moved:
        r.update(split='train',physical_group=fake);r.pop('evaluation_annotation_protocol',None)
    for name,rows in [('train.jsonl',train+moved),('val.jsonl',val)]:
        dataset.dump_rows(folder/name,rows);resign(folder,name)
    with pytest.raises(ValueError,match='source route physical_group'):
        dataset.load_dataset(folder)


@pytest.mark.parametrize('component,value', [('scenario','../InvadingTurn'),('route_id','/absolute'),('route_id','a/b'),('route_id','a\\b')])
def test_builder_rejects_ambiguous_source_components(tmp_path,component,value):
    a=copy.deepcopy(read('reviewed_holdout_state_pairs_v3.json')[0]);a[component]=value
    with pytest.raises(ValueError,match='source route component'):
        dataset.build([a],ROOT.parents[1]/'lead_data',tmp_path/'data')
