"""Regression: geometry survives genuine native action evidence failures."""
import copy
import json
import pickle
from pathlib import Path
import pytest
from PIL import Image
from .. import dataset, regions, objective, admission, planning
from ...geometry_study import six
from ...geometry_v1 import dataset as native
from ...geometry_v1.tests.test_geometry import Lanes, meta
from qwen3vl_local.sft_new_loop_phase3 import trajectory_action as rule


def test_route_end_geometry_survives_native_six_rejection(tmp_path, monkeypatch):
    root=tmp_path/'raw'; route='Town00_End'; run=root/'Synthetic'/route
    (run/'metas').mkdir(parents=True);(run/'rgb').mkdir();maps=tmp_path/'maps';maps.mkdir()
    # The current metadata still contains future 20 Hz positions. No later
    # saved 4 Hz observations exist to recognize native upcoming intentions.
    for frame in range(9):
        m=meta(frame);m.update(pos_global=[frame*.75,0,0],theta=0,road_id=1,lane_id=-1,lane_type_str='Driving')
        (run/'metas'/f'{frame:04d}.pkl').write_bytes(pickle.dumps(m))
    Image.new('RGB',(32,32)).save(run/'rgb/0008.jpg')
    entry=dict(scenario='Synthetic',route_id=route,town='Town00',split='train',
        anchors=[dict(frame=8,event='LEAD_BRAKE',road_structure='R1')])
    parent,_=native.build([entry],root,Lanes(),tmp_path/'native',require_coverage=False)
    row=parent['records'][0]
    action=six.build(row,rule.load_route_trajectory(run),[],Lanes())
    assert action['anchor_action'] is None and not any(t['valid'] for t in action['targets'])
    def forbidden(*a,**k):pytest.fail('geometry must never invoke native six labeling')
    monkeypatch.setattr(six,'build',forbidden)
    monkeypatch.setattr(rule,'label_actions',forbidden)
    derived,receipt=dataset.build(tmp_path/'native/dataset.json',maps,tmp_path/'abc')
    assert receipt['anchors']==1 and not receipt['legacy_action_filter_used']
    assert not receipt['rejected']
    for v in regions.VARIANTS:
        loaded=dataset.load(tmp_path/'abc/dataset.json',variant=v)
        r=loaded['records'][0]
        assert all(t['valid'] for t in r['targets'])
        assert all(h['origin_frame']==8 and h['reference_frame']=='current' for h in r['observation']['history'])
        assert objective.allowed_indices(r)==list(range(9))
        assert loaded['admission']['status']=='rejected' # train-only smoke, not formal ready

    # Exercise the actual CPU coordinator: finished audit is distinct from
    # formal admission, and a train-only smoke must never launch a GPU.
    from .. import cpu
    study_root=tmp_path/'cpu_run'
    monkeypatch.setattr('sys.argv',['cpu','--geometry',str(tmp_path/'native/dataset.json'),
        '--maps',str(maps),'--study-root',str(study_root)])
    assert cpu.main()==2
    status=json.loads((study_root/'cpu_status.json').read_text())
    assert status['status']=='completed' and status['ready_status']=='rejected'
    report=json.loads(Path(status['report']).read_text())
    assert not report['formal_conditions_passed'] and not report['legacy_action_filter_used']
    assert report['capacity']['status']=='rejected'


def test_distinct_region_labels_shared_trajectory_and_masks():
    from .test_study import record
    from ...geometry_v1 import codec
    r=record();r['road_structure']='R3'
    r['targets']=[codec.classify([15.,0.],2,7.5,0)]*4
    r['observation']['history']=[codec.classify([-15.,0.],2,7.5,0)]*4
    before=copy.deepcopy(r);out=dataset.combine(r)
    assert r==before
    assert set(out['representations'])=={'a','b','c'}
    assert [out['representations'][v]['targets'][0]['name'] for v in regions.VARIANTS]==[
        'CENTER_REFERENCE','CENTER_NEAR','CENTER_FAR']
    assert [out['representations'][v]['history'][0]['name'] for v in regions.VARIANTS]==[
        'CENTER_REFERENCE','CENTER_NEAR','CENTER_FAR']
    assert not any(k.startswith('six/') for k in out['sampling_mass'])
    for v in regions.VARIANTS:
        assert [t['valid'] for t in out['representations'][v]['targets']]==[True]*4
    r['targets']=[dict(valid=False,reason='missing_endpoint')]+r['targets'][1:]
    assert all(not view['targets'][0]['valid'] for view in dataset.combine(r)['representations'].values())


def test_old_matched_receipt_and_six_are_not_accepted(tmp_path):
    from ...geometry_study import regions as old_regions
    assert regions.contract('a')!=old_regions.contract('a')
    with pytest.raises(ValueError,match='unknown_representation'):regions.contract('six')
    path=tmp_path/'dataset.json'
    dataset.write(path,dict(contract={v:old_regions.contract(v) for v in old_regions.VARIANTS},code=dataset.code_identity()))
    dataset.write(path.with_name('receipt.json'),dict(dataset_sha256=dataset.sources.file_sha(path)))
    with pytest.raises(ValueError,match='contract_or_code_changed'):dataset.load(path,check_sources=False)


def test_loader_parses_exact_bytes_it_hashed(tmp_path, monkeypatch):
    path=tmp_path/'dataset.json'
    original=dict(contract={v:regions.contract(v) for v in regions.VARIANTS},
        code=dataset.code_identity(),records=[],marker='original')
    dataset.write(path,original)
    dataset.write(path.with_name('receipt.json'),dict(dataset_sha256=dataset.sources.file_sha(path)))
    read_bytes=Path.read_bytes
    def replace_after_read(p):
        raw=read_bytes(p)
        if p==path:
            dataset.write(p,dict(original,marker='changed after read'))
        return raw
    monkeypatch.setattr(Path,'read_bytes',replace_after_read)
    assert dataset.load(path,check_sources=False)['marker']=='original'
    with pytest.raises(ValueError,match='dataset_changed'):dataset.load(path,check_sources=False)
