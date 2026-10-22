"""Candidate restoration and mixed-generation discovery regressions."""
import json
import pickle
import shutil
from pathlib import Path

import pytest
from PIL import Image

from qwen3vl_local.audit_joint import data_prepare,phase3_common_data as common
from qwen3vl_local.audit_joint import phase3_base_trial as trial
from qwen3vl_local.audit_joint.geometry_v1 import dataset,sources
from qwen3vl_local.audit_joint.geometry_v1.tests.test_geometry import meta,Lanes
from qwen3vl_local.audit_joint.tests.test_data_prepare_review_fixes import prepared_common


@pytest.fixture
def geometry_inputs(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import build_dataset as old,annotation_repair
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
    raw=tmp_path/'raw';collection=tmp_path/'collection';collection.mkdir()
    maps=tmp_path/'maps';maps.mkdir();checkpoints=tmp_path/'checkpoints'
    ids=[f'Town00_geometry_{i}' for i in range(50)
         if old._split('Synthetic',f'Town00_geometry_{i}',20260920,.10,.05)=='train'][:4]
    routes=[]
    for i,rid in enumerate(ids):
        run=raw/'Synthetic'/rid
        (run/'metas').mkdir(parents=True);(run/'rgb').mkdir()
        for f in range(18):(run/'metas'/f'{f:04d}.pkl').write_bytes(pickle.dumps(meta(f)))
        if i<3:Image.new('RGB',(32,32)).save(run/'rgb/0008.jpg')
        routes.append(dict(route_id=rid,xml_town='Town00',annotations=[dict(frame_id=8),dict(frame_id=9)]))
    (collection/'Synthetic_result.json').write_text(json.dumps(dict(routes=routes)))
    monkeypatch.setattr(old,'is_abnormal_lead_route',lambda run,*a:(not (run/'rgb/0008.jpg').is_file(),None))
    monkeypatch.setattr(annotation_repair,'repair_annotation',lambda *a:('STRAIGHT','U-E1',['U-E1'],{}))
    monkeypatch.setattr(old,'mapped_contexts',lambda *a:(list(CONTEXT_IDS),{}))
    monkeypatch.setattr(sources,'LaneMap',lambda *a:Lanes())
    return raw,collection,maps,checkpoints,ids


def snapshot(root):
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('previous',['legacy','complete','bad_rgb'])
def test_restored_candidate_generates_60_instead_of_reusing_30(geometry_inputs,monkeypatch,previous):
    raw,coll,maps,checkpoints,ids=geometry_inputs
    if previous=='bad_rgb':
        for rid in ids[:3]:(raw/'Synthetic'/rid/'rgb/0009.jpg').write_bytes(b'not a JPEG')
    if previous=='legacy':
        olddir=checkpoints/'legacy';routes=list(dataset.annotation_routes(coll,raw))
        dataset.build(routes,raw,Lanes(),olddir,input_paths=list(coll.glob('*_result.json')))
        oldpath=olddir/'dataset.json'
    else:
        oldpath=Path(data_prepare.geometry(raw,coll,maps,checkpoints)['dataset']);olddir=oldpath.parent
    assert len(dataset.load(oldpath)['records'])==30
    before=snapshot(olddir)
    for rid in ids[:3]:Image.new('RGB',(32,32),(10,20,30)).save(raw/'Synthetic'/rid/'rgb/0009.jpg')
    result=data_prepare.geometry(raw,coll,maps,checkpoints)
    assert result['action']=='build' and Path(result['dataset'])!=oldpath
    got=dataset.load(result['dataset']);assert len(got['records'])==60
    expected,_=dataset.build(list(dataset.annotation_routes(coll,raw)),raw,Lanes(),raw.parent/'native_reference',
                             input_paths=list(coll.glob('*_result.json')))
    assert got['records']==expected['records']
    assert snapshot(olddir)==before
    assert result['discovery_skipped'][0]['reason']==(
        'legacy_missing_complete_candidate_proof' if previous=='legacy' else 'candidate_dependencies_changed')
    def forbidden(*a,**kw):pytest.fail('matching candidate dependencies rebuilt')
    monkeypatch.setattr(dataset,'build',forbidden)
    again=data_prepare.geometry(raw,coll,maps,checkpoints)
    assert again['dataset']==result['dataset'] and again['action']=='reuse_existing'


def test_previously_rejected_route_restored(geometry_inputs):
    raw,coll,maps,checkpoints,ids=geometry_inputs
    first=data_prepare.geometry(raw,coll,maps,checkpoints)
    proof=json.loads(Path(first['dataset']).with_name('generation_inputs.json').read_text())
    assert len(proof['dependencies']['route_members'])==4
    assert len(proof['dependencies']['routes'])==3
    Image.new('RGB',(32,32)).save(raw/'Synthetic'/ids[-1]/'rgb/0008.jpg')
    second=data_prepare.geometry(raw,coll,maps,checkpoints)
    assert len(dataset.load(second['dataset'])['records'])==40 and second['dataset']!=first['dataset']


def test_geometry_corrupt_candidate_proof_not_skipped(geometry_inputs):
    raw,coll,maps,checkpoints,_=geometry_inputs
    result=data_prepare.geometry(raw,coll,maps,checkpoints)
    p=Path(result['dataset']).with_name('generation_inputs.json');p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='candidate input proof changed'):
        data_prepare.geometry(raw,coll,maps,checkpoints)


@pytest.mark.parametrize('legacy_kind',['no_binding','old_binding'])
def test_unverifiable_legacy_does_not_block_valid_v2(tmp_path,monkeypatch,capsys,legacy_kind):
    raw,root,coll,_,calls,run=prepared_common(tmp_path,monkeypatch)
    legacy=root.parent/'a_legacy';shutil.copytree(root,legacy)
    receipt=trial.read(legacy/'common_receipt.json')
    if legacy_kind=='no_binding':receipt.pop('input_bindings_sha256')
    else:
        trial.write(legacy/'input_bindings.json',dict(collection={},routes=30))
        receipt['input_bindings_sha256']=trial.file_sha(legacy/'input_bindings.json')
    trial.write(legacy/'common_receipt.json',receipt)
    for p in coll.iterdir():p.unlink()
    before=snapshot(legacy)
    index,_=common.ensure(root.parent/'default',raw,coll,run,discover_existing=True)
    assert index==root/'native/frame_index.jsonl' and not calls
    assert str(legacy) in capsys.readouterr().out and snapshot(legacy)==before
    with pytest.raises(common.UnverifiableLegacyInputs):
        common.ensure(legacy,raw,coll,run,discover_existing=True)


@pytest.mark.parametrize('damage',['index','binding','receipt'])
def test_corrupt_legacy_still_blocks_discovery(tmp_path,monkeypatch,damage):
    raw,root,coll,_,calls,run=prepared_common(tmp_path,monkeypatch)
    legacy=root.parent/'a_legacy';shutil.copytree(root,legacy)
    if damage=='index':
        receipt=trial.read(legacy/'common_receipt.json');receipt.pop('input_bindings_sha256')
        trial.write(legacy/'common_receipt.json',receipt)
        p=legacy/'native/frame_index.jsonl';p.write_text(p.read_text()+' ')
    elif damage=='binding':(legacy/'input_bindings.json').write_text('{}')
    else:(legacy/'common_receipt.json').write_text('{')
    for p in coll.iterdir():p.unlink()
    with pytest.raises(ValueError):common.ensure(root.parent/'default',raw,coll,run,discover_existing=True)
    assert not calls


def test_incomplete_geometry_cache_cannot_bypass_completion_via_discovery(geometry_inputs):
    raw,coll,maps,checkpoints,_=geometry_inputs
    result=data_prepare.geometry(raw,coll,maps,checkpoints)
    Path(result['dataset']).parent.parent.joinpath('complete.json').unlink()
    with pytest.raises(ValueError,match='Incomplete geometry cache'):
        data_prepare.geometry(raw,coll,maps,checkpoints)


def test_explicit_geometry_selection_reports_fixed_snapshot_scope(geometry_inputs):
    raw,coll,maps,checkpoints,_=geometry_inputs
    olddir=checkpoints/'legacy'
    dataset.build(list(dataset.annotation_routes(coll,raw)),raw,Lanes(),olddir,
                  input_paths=list(coll.glob('*_result.json')))
    result=data_prepare.geometry(raw,coll,None,checkpoints,existing=olddir/'dataset.json')
    assert result['selection']=='explicit_fixed_artifact'
    assert result['candidate_completeness_verified'] is False
