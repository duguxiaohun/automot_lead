import json,pickle,copy
from pathlib import Path
import pytest
from PIL import Image
from .. import dataset,prepare,admission
from ...geometry_v1 import dataset as native
from ...geometry_v1.tests.test_geometry import Lanes,meta
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS

def parent_dataset(tmp_path,monkeypatch):
    root=tmp_path/'raw';maps=tmp_path/'maps';maps.mkdir();routes=[]
    for split in ('train','val','test'):
        for event in CONTEXT_IDS:
            for i in range(2 if split=='train' and event=='LEAD_BRAKE' else 1):
                route=f'Town00_{split}_{event}_{i}';run=root/'Synthetic'/route
                (run/'metas').mkdir(parents=True);(run/'rgb').mkdir()
                for frame in range(9 if split=='test' and event=='LEAD_BRAKE' else 40):
                    m=meta(frame);m.update(pos_global=[frame*.75,0,0],theta=0,road_id=1,lane_id=-1,lane_type_str='Driving')
                    (run/'metas'/f'{frame:04d}.pkl').write_bytes(pickle.dumps(m))
                Image.new('RGB',(32,32)).save(run/'rgb/0008.jpg')
                routes.append(dict(scenario='Synthetic',route_id=route,town='Town00',split=split,
                    anchors=[dict(frame=8,event=event,road_structure='R1')]))
    parent=tmp_path/'parent';native.build(routes,root,Lanes(),parent)
    monkeypatch.setattr(dataset.sources,'LaneMap',lambda _:Lanes())
    return parent/'dataset.json',maps

def test_real_native_invalid_test_repaired_in_new_artifact_cache_and_proof(tmp_path,monkeypatch):
    parent,maps=parent_dataset(tmp_path,monkeypatch);before=parent.read_bytes()
    first=prepare.prepare(parent,maps,tmp_path/'cache',per_event=1,world_size=1)
    second=prepare.prepare(parent,maps,tmp_path/'cache',per_event=1,world_size=1)
    assert first['status']=='ready' and first['generation']['action']=='build' and second['generation']['action']=='reuse'
    assert first['dataset_sha256']==second['dataset_sha256'] and parent.read_bytes()==before
    a=dataset.load(first['dataset'],variant='six');report=a['split_report']
    assert report['before']['test']['LEAD_BRAKE']['anchors']==0
    assert report['after']['test']['LEAD_BRAKE']['anchors']==1 and len(report['moves'])==1
    assert a['admission']['split_members_modified'] and a['admission']['split_report_sha256']==report['sha256']
    assert 'LEAD_BRAKE' in a['admission']['weak_event_support']['test']
    assert prepare.load_ready(tmp_path/'cache/ready_1.json',per_event=1,world_size=1)==second
    proof=Path(first['dataset']).with_name('split_coverage.json');saved=proof.read_text();proof.write_text('{}')
    with pytest.raises(ValueError,match='split_receipt'):dataset.load(first['dataset'],check_sources=False)
    proof.write_text(saved)
    file=tmp_path/'raw/Synthetic/Town00_train_LEAD_BRAKE_0/metas/0000.pkl'
    file.write_bytes(file.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='source_content_changed'):dataset.load(first['dataset'])

def test_additional_exposure_blocks_donors_and_is_bound(tmp_path,monkeypatch):
    parent,maps=parent_dataset(tmp_path,monkeypatch)
    path=tmp_path/'exposure.json';path.write_text(json.dumps(dict(additional_development=dict(groups=[
        'Synthetic/Town00_train_LEAD_BRAKE_0','Synthetic/Town00_train_LEAD_BRAKE_1']))))
    with pytest.raises(ValueError,match='native_split_coverage_infeasible'):
        prepare.prepare(parent,maps,tmp_path/'cache',per_event=1,world_size=1,development_manifests=[path])
    ready=json.loads((tmp_path/'cache/ready_1.json').read_text());assert ready['status']=='rejected'
    a=dataset.load(ready['dataset']);assert not a['split_report']['moves']
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='exposure_sources_changed'):dataset.load(ready['dataset'],check_sources=False)

@pytest.mark.parametrize('explicit', [False, True])
def test_real_cpu_coordinator_resolves_maps_and_keeps_rejection_report(tmp_path,monkeypatch,explicit):
    from .. import cpu
    from ... import phase3_maps
    parent,maps=parent_dataset(tmp_path,monkeypatch)
    app=tmp_path/'app'
    source=maps if explicit else app/'CARLA_0915/CarlaUE4/Content/Carla/Maps'
    source.mkdir(parents=True,exist_ok=True)
    for town in phase3_maps.TOWNS:(source/(town+'.xodr')).write_text(town)
    monkeypatch.setattr(cpu,'APP',app)
    # Real resolver and output protection together; only LaneMap geometry uses
    # the synthetic parent fixture. Mocking resolve used to hide root/maps overlap.
    monkeypatch.setattr('sys.argv',['cpu','--geometry',str(parent),
        '--study-root',str(tmp_path/'cpu')]+(['--maps',str(maps)] if explicit else []))
    assert cpu.main()==2 # one train anchor/event cannot satisfy formal 1280
    status=json.loads((tmp_path/'cpu/cpu_status.json').read_text())
    assert status['ready_status']=='rejected' and status['status']=='completed'
    report=json.loads(Path(status['report']).read_text())
    assert report['split_report']['status']=='infeasible' and not report['formal_conditions_passed']
    assert report['admission']['coverage']['test']['missing_events']==['LEAD_BRAKE']
    assert not (tmp_path/'cpu/maps').exists()
    assert len(list((tmp_path/'cpu_maps').glob('*.xodr')))==12
