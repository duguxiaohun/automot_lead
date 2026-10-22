"""Exercise actual cache misses and raw-input invalidation, not just cache hits."""
import copy
import json
import pickle
from pathlib import Path

from PIL import Image
import pytest

from qwen3vl_local.audit_joint import data_prepare, phase3_common_data as common
from qwen3vl_local.audit_joint import phase3_base_trial as trial
from qwen3vl_local.audit_joint.tests.test_phase3_common_data import common_fixture


def test_geometry_first_native_build_generator_reload_and_reuse(tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint.geometry_v1 import dataset,sources
    from qwen3vl_local.audit_joint.geometry_v1.tests.test_geometry import meta,Lanes
    from qwen3vl_local.sft_new_loop_phase3 import build_dataset as old,annotation_repair
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
    raw=tmp_path/'raw';collection=tmp_path/'collection';collection.mkdir()
    maps=tmp_path/'maps';maps.mkdir();checkpoints=tmp_path/'checkpoints'
    routes=[]
    ids=[f'Town00_geometry_{i}' for i in range(30)
         if old._split('Synthetic',f'Town00_geometry_{i}',20260920,.10,.05)=='train'][:3]
    for rid in ids:
        run=raw/'Synthetic'/rid
        (run/'metas').mkdir(parents=True);(run/'rgb').mkdir()
        for f in range(17):(run/'metas'/f'{f:04d}.pkl').write_bytes(pickle.dumps(meta(f)))
        Image.new('RGB',(32,32)).save(run/'rgb/0008.jpg')
        routes.append(dict(route_id=rid,xml_town='Town00',annotations=[dict(frame_id=8)]))
    (collection/'Synthetic_result.json').write_text(json.dumps(dict(routes=routes)))
    monkeypatch.setattr(old,'is_abnormal_lead_route',lambda *a:(False,None))
    monkeypatch.setattr(annotation_repair,'repair_annotation',lambda *a:('STRAIGHT','U-E1',['U-E1'],{}))
    monkeypatch.setattr(old,'mapped_contexts',lambda *a:(list(CONTEXT_IDS),{}))
    monkeypatch.setattr(sources,'LaneMap',lambda *a:Lanes())
    a=data_prepare.geometry(raw,collection,maps,checkpoints)
    artifact=dataset.load(a['dataset'])
    assert len(artifact['records'])==30 and not artifact['scoped_smoke_only']
    assert {r['split'] for r in artifact['records']}=={'train','val','test'}
    def forbidden(*a,**kw):pytest.fail('Matching geometry artifact rebuilt')
    monkeypatch.setattr(dataset,'build',forbidden)
    b=data_prepare.geometry(raw,collection,maps,checkpoints)
    assert a['dataset']==b['dataset'] and b['action'] in ('reuse_existing','reuse')


def prepared_common(tmp_path,monkeypatch):
    job,root,_=common_fixture(tmp_path,monkeypatch);data=Path(job['data_root'])
    keys=trial.read(root/'input_bindings.json')['route_keys']
    collection=tmp_path/'collection';collection.mkdir()
    grouped={}
    for scenario,rid in keys:grouped.setdefault(scenario,[]).append(dict(route_id=rid))
    for scenario,routes in grouped.items():
        (collection/(scenario+'_result.json')).write_text(json.dumps(dict(routes=routes)))
    inputs=common.source_inputs(collection,data)
    trial.write(root/'input_bindings.json',inputs)
    receipt=trial.read(root/'common_receipt.json');receipt['input_bindings_sha256']=trial.file_sha(root/'input_bindings.json')
    trial.write(root/'common_receipt.json',receipt)
    calls=[]
    def run(name,cmd,log):
        # Expensive native global index selection is substituted only here;
        # dependency enumeration, checks, reference and native validation are real.
        calls.append(name);out=Path(cmd[cmd.index('--output-dir')+1]);out.mkdir()
        for name in ('frame_index.jsonl','manifest.json'):(out/name).write_bytes((root/'native'/name).read_bytes())
    return data,root,collection,inputs,calls,run


@pytest.mark.parametrize('change',['speed','add_meta','remove_meta','remove_rgb','add_rgb'])
def test_common_raw_change_new_index_version_then_reuse(tmp_path,monkeypatch,change):
    data,root,coll,inputs,calls,run=prepared_common(tmp_path,monkeypatch)
    old={n:(root/n).read_bytes() for n in ('common_receipt.json','input_bindings.json','native/frame_index.jsonl')}
    raw=data.joinpath(*inputs['route_keys'][0]);p=raw/'metas/0000.pkl'
    if change=='speed':
        value=pickle.loads(p.read_bytes());value['speed']+=7;p.write_bytes(pickle.dumps(value))
    elif change=='add_meta':(raw/'metas/9999.pkl').write_bytes(p.read_bytes())
    elif change=='remove_meta':p.unlink()
    elif change=='remove_rgb':(raw/'rgb/0000.jpg').unlink()
    else:Image.new('RGB',(32,32)).save(raw/'rgb/9999.jpg')
    assert inputs['collection']==common.source_inputs(coll,data)['collection']
    a,_=common.ensure(root,data,coll,run,discover_existing=True)
    b,_=common.ensure(root,data,coll,run,discover_existing=True)
    # Also discover correctly when several historical input versions coexist.
    c,_=common.ensure(root.parent/'another_default',data,coll,run,discover_existing=True)
    assert a==b==c and a!=root/'native/frame_index.jsonl' and calls==['build_common']
    assert old=={n:(root/n).read_bytes() for n in old}


def test_common_absent_exports_still_verify_raw_and_refuse_stale(tmp_path,monkeypatch):
    data,root,coll,inputs,calls,run=prepared_common(tmp_path,monkeypatch)
    for p in coll.iterdir():p.unlink()
    common.ensure(root,data,coll,run)
    assert not calls
    p=data.joinpath(*inputs['route_keys'][0])/'metas/0000.pkl';p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='local annotations required'):
        common.ensure(root,data,coll,run)
    assert not calls


def test_legacy_index_cannot_be_certified_without_dependency_proof(tmp_path,monkeypatch):
    data,root,coll,inputs,calls,run=prepared_common(tmp_path,monkeypatch)
    receipt=trial.read(root/'common_receipt.json');receipt.pop('input_bindings_sha256')
    trial.write(root/'common_receipt.json',receipt)
    for p in coll.iterdir():p.unlink()
    with pytest.raises(ValueError,match='lacks raw dependency proof'):
        common.ensure(root,data,coll,run,discover_existing=True)
    assert not calls


def test_common_change_during_build_never_publishes_complete_receipt(tmp_path,monkeypatch):
    data,root,coll,inputs,calls,run=prepared_common(tmp_path,monkeypatch)
    p=data.joinpath(*inputs['route_keys'][0])/'metas/0000.pkl'
    def changing(*args):
        run(*args);p.write_bytes(p.read_bytes()+b'changed')
    out=tmp_path/'new'
    with pytest.raises(ValueError,match='dependencies changed during build'):
        common.ensure(out,data,coll,changing)
    assert (out/'build_failure.json').is_file() and not (out/'common_receipt.json').exists()
