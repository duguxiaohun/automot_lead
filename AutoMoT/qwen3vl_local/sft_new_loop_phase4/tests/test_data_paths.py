"""External LEAD mounts retain logical identities throughout production/loading."""
import copy
import json

import pytest
from PIL import Image
from qwen3vl_local.sft_new_loop_phase4 import candidate_pool, dataset
from qwen3vl_local.sft_new_loop_phase4 import privileged_geometry as geometry
from qwen3vl_local.sft_new_loop_phase4.data_paths import data_path
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT, file_sha
from qwen3vl_local.sft_new_loop_phase4.model import load_images
from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import write_source
from qwen3vl_local.sft_new_loop_phase4.tests.test_condition_builder import record, producer
from qwen3vl_local.sft_new_loop_phase4 import condition_builder


@pytest.mark.parametrize('layout', ['root', 'scenario', 'route', 'assets', 'files'])
def test_external_links_scan_replay_and_load_with_identical_provenance(tmp_path, layout):
    disk=tmp_path/'disk';s,r=write_source(disk,20)
    direct=geometry.load_frame(disk,s,r,20)
    root=tmp_path/'lead_data'
    if layout=='root':root.symlink_to(disk,target_is_directory=True)
    elif layout=='scenario':
        root.mkdir();(root/s).symlink_to(disk/s,target_is_directory=True)
    elif layout=='route':
        (root/s).mkdir(parents=True);(root/s/r).symlink_to(disk/s/r,target_is_directory=True)
    else:
        for kind in ('rgb','metas','bboxes'):
            dest=root/s/r/kind;dest.parent.mkdir(parents=True,exist_ok=True)
            if layout=='assets':dest.symlink_to(disk/s/r/kind,target_is_directory=True)
            else:
                dest.mkdir()
                for path in (disk/s/r/kind).iterdir():(dest/path.name).symlink_to(path)
    pool=candidate_pool.scan(root,tmp_path/'pool.json');candidate_pool.validate(pool)
    assert len(pool['routes'])==1 and pool['routes'][0]['route_id']==r
    linked=geometry.load_frame(root,s,r,20)
    assert linked==direct
    assert all(x['path'].startswith(f'{s}/{r}/') for x in linked['sources'])
    rgb=linked['sources'][-1]
    row=dict(images=[rgb['path']],image_sha256=[rgb['sha256']],observation=dict(history_frames=[20]))
    assert load_images(row,root)[0].tobytes()==load_images(row,disk)[0].tobytes()
    rec=record();rec['sources'][0]['sha256']=rgb['sha256']
    condition_builder.validate_record(rec,producer(),root)
    Image.new('RGB',(384,384),'red').save(disk/rgb['path'])
    with pytest.raises(ValueError,match='RGB source mismatch'):load_images(row,root)
    with pytest.raises(ValueError,match='content mismatch'):condition_builder.validate_record(rec,producer(),root)


@pytest.mark.parametrize('relative', ['', '.', '..', '../outside', 'a/../b', '/tmp/asset',
                                      'a//b', './a', 'a/./b', 'a\\b', 'a\x00b'])
def test_lexical_escape_is_rejected(tmp_path,relative):
    with pytest.raises(ValueError,match='invalid relative dataset path'):data_path(tmp_path,relative)


def test_real_reviewed_build_via_external_routes_preserves_labels_and_inputs(tmp_path):
    original=ROOT.parents[1]/'lead_data'
    if not original.exists():pytest.skip('external LEAD unavailable')
    annotations=json.loads((ROOT/'reviewed_state_pairs_v9.json').read_text())[:1]
    linked=tmp_path/'lead_data'
    for ann in annotations:
        parent=linked/ann['scenario'];parent.mkdir(parents=True,exist_ok=True)
        (parent/ann['route_id']).symlink_to((original/ann['scenario']/ann['route_id']).resolve(),target_is_directory=True)
    for mode in (2,4):
        direct_dir=tmp_path/f'direct{mode}';link_dir=tmp_path/f'linked{mode}'
        dataset.build(copy.deepcopy(annotations),original,direct_dir,rgb_mode=mode)
        dataset.build(copy.deepcopy(annotations),linked,link_dir,rgb_mode=mode)
        data,_=dataset.load_dataset(link_dir)
        for split in ('train','val','test'):
            assert (direct_dir/f'{split}.jsonl').read_bytes()==(link_dir/f'{split}.jsonl').read_bytes()
            for row in data[split]:assert len(load_images(row,linked))==mode


def test_new_protocol_preserves_all_independent_reservations():
    old=json.loads((ROOT/'producer_manual_check_plan_v18_20261007.json').read_text())
    new=json.loads((ROOT/dataset.PRODUCER_CHECK_PLAN).read_text())
    assert old['reservations']==new['reservations']
    assert old['excluded_reservations']==new['excluded_reservations']
    assert dataset.producer_check_reservations()
    assert new['frozen_sources']['data_paths.py']==file_sha(ROOT/'data_paths.py')
