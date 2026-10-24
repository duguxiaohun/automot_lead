import copy
import json
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from qwen3vl_local.audit_joint.abc_delivery import sha,digest,write
from qwen3vl_local.audit_joint.geometry_v1 import dataset as native
from qwen3vl_local.audit_joint.geometry_v1.tests.test_geometry import meta,Lanes
from qwen3vl_local.audit_joint.geometry_abc_v2.dataset import combine
from qwen3vl_local.action_expert_ablation.geometry_abc import data,loader,runtime,contracts


def fixture(tmp_path,monkeypatch):
    import laspy
    from qwen3vl_local.action_expert_ablation import common
    root=tmp_path/'raw';route=root/'Synthetic/Town00_Test'
    for kind in ('metas','rgb','lidar'):(route/kind).mkdir(parents=True)
    metas={i:meta(i) for i in range(9)};metas[8]['route']=np.arange(20).reshape(10,2).tolist()
    meta_path=route/'metas/0008.pkl';meta_path.write_bytes(pickle.dumps(metas[8]))
    Image.new('RGB',(16,16),'red').save(route/'rgb/0008.jpg')
    cloud=laspy.LasData(laspy.LasHeader(point_format=3,version='1.2'))
    cloud.x=[1.,2.];cloud.y=[0.,1.];cloud.z=[0.,.2];cloud.write(route/'lidar/0008.laz')
    r=native.record(dict(scenario='Synthetic',route_id='Town00_Test',town='Town00',split='train'),
        dict(frame=8,event='LEAD_BRAKE',road_structure='R1'),metas,Lanes())
    r.update(physical_group='Synthetic/Town00_Test',image_sha256=sha(route/'rgb/0008.jpg'),
             waypoint_xy=data.validate_continuous(metas[8]).tolist())
    r=data.compact_record(combine(r));directory=tmp_path/'data'
    key=digest(['Synthetic','Town00_Test']);receipt=directory/'routes'/key/'receipt.json'
    write(receipt,dict(bindings={str(p.relative_to(root)):sha(p) for p in route.rglob('*') if p.is_file()}))
    write(directory/'manifest.json',dict(identity='fixture',code=contracts.source_identity(data_only=True),routes={key:sha(receipt)}))
    # Only the unrelated legacy route smoothing helper is substituted. Actual
    # metadata/RGB/LAZ decoding and all native target checks run below.
    monkeypatch.setattr(common,'leadmot_train',lambda:SimpleNamespace(_pad_rows=lambda rows,n:rows))
    args=SimpleNamespace(data_root=str(root),data_dir=str(directory),abc_dataset_identity='fixture',
                         abc_manifest_sha256=sha(directory/'manifest.json'),smooth_route=False,route_points=10)
    return r,args,route


@pytest.mark.parametrize('changed',['rgb','lidar','metas','route_target','root'])
def test_exact_consumption_rejects_same_path_and_root_drift(tmp_path,monkeypatch,changed):
    r,args,route=fixture(tmp_path,monkeypatch)
    ds=loader.Dataset([runtime.sample(r,args.data_root)],args)
    first=ds[0];assert '_error' not in first and set(first['sample']['consumed_sources'])=={'rgb','lidar','metas'}
    if changed=='root':
        alternative=tmp_path/'other';shutil.copytree(args.data_root,alternative)
        Image.new('RGB',(16,16),'blue').save(alternative/'Synthetic/Town00_Test/rgb/0008.jpg')
        changed_args=copy.copy(args);changed_args.data_root=str(alternative)
        result=loader.Dataset([runtime.sample(r,alternative)],changed_args)[0]
    else:
        if changed=='rgb':Image.new('RGB',(16,16),'blue').save(route/'rgb/0008.jpg')
        elif changed=='lidar':(route/'lidar/0008.laz').write_bytes(b'changed lidar')
        else:
            path=route/'metas/0008.pkl';value=pickle.loads(path.read_bytes())
            if changed=='route_target':value['route'][0][0]+=1
            else:value['speed']+=.1
            path.write_bytes(pickle.dumps(value))
        result=ds[0] # reused Dataset/receipt cache must still check new input bytes
    assert 'consumed_source_changed' in result['_error']


def test_decode_uses_verified_bytes_even_if_path_changes_after_read(tmp_path,monkeypatch):
    from qwen3vl_local.action_expert_ablation.geometry_abc import consumption
    r,args,route=fixture(tmp_path,monkeypatch);sample=runtime.sample(r,args.data_root)
    baseline=loader.Dataset([sample],args)[0]
    original=consumption.checked_bytes
    def swap_after_hash(path,expected):
        raw=original(path,expected)
        if str(path).endswith('.jpg'):Image.new('RGB',(16,16),'blue').save(path)
        return raw
    monkeypatch.setattr(consumption,'checked_bytes',swap_after_hash)
    ds=loader.Dataset([sample],args);result=ds[0]
    assert '_error' not in result and np.array_equal(result['clip']['rgb'],baseline['clip']['rgb'])
    assert 'consumed_source_changed' in ds[0]['_error']


def test_route_path_cannot_override_bound_root(tmp_path,monkeypatch):
    r,args,route=fixture(tmp_path,monkeypatch);sample=runtime.sample(r,args.data_root)
    sample['route_dir']=str(tmp_path/'other_route')
    assert 'consumed_route_root_changed' in loader.Dataset([sample],args)[0]['_error']


def test_manifest_cannot_be_replaced_after_admission(tmp_path,monkeypatch):
    r,args,route=fixture(tmp_path,monkeypatch)
    path=Path(args.data_dir)/'manifest.json';value=json.loads(path.read_text());value['routes']={};write(path,value)
    with pytest.raises(ValueError,match='consumed_source_changed'):
        loader.Dataset([runtime.sample(r,args.data_root)],args)


def test_changed_receipt_cannot_rebind_changed_sensor(tmp_path,monkeypatch):
    r,args,route=fixture(tmp_path,monkeypatch)
    image=route/'rgb/0008.jpg';Image.new('RGB',(16,16),'blue').save(image)
    path=next((Path(args.data_dir)/'routes').glob('*/receipt.json'))
    value=json.loads(path.read_text());value['bindings'][r['image']]=sha(image);write(path,value)
    assert 'consumed_source_changed' in loader.Dataset([runtime.sample(r,args.data_root)],args)[0]['_error']
