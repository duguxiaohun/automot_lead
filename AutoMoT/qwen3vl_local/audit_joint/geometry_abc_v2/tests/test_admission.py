import copy
import json
from pathlib import Path
import pytest
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
from .. import admission, dataset, planning, regions, train
from .. import prepare as preparation
from .test_study import record


def cohort(val_per_event=1):
    rows=[]
    for split in ('train','val','test'):
        for event in CONTEXT_IDS:
            for frame in range(8,8+(val_per_event if split=='val' else 1)):
                r=record(frame,event,route=f'Town00_{split}_{event}')
                r['split']=split
                r['representations']={v:dict(targets=record(variant=v)['targets']) for v in regions.VARIANTS}
                rows.append(r)
    return dict(records=rows,scoped_smoke_only=False)


def fake_cache(monkeypatch,tmp_path,artifact):
    geometry=tmp_path/'parent'/'dataset.json';geometry.parent.mkdir();geometry.write_text('{}')
    maps=tmp_path/'maps';maps.mkdir()
    data=tmp_path/'common'/'dataset.json';data.parent.mkdir();data.write_text('{}')
    monkeypatch.setattr(preparation.native,'load',lambda *a,**k:dict(data_root=str(tmp_path/'raw'),map_sha256={}))
    monkeypatch.setattr(preparation.cache,'ensure',lambda *a,**k:dict(directory=str(data.parent),action='reuse'))
    monkeypatch.setattr(dataset,'load',lambda *a,**k:copy.deepcopy(artifact))
    return geometry,maps,data


def test_missing_test_events_reject_cpu_before_capacity_and_invalidate_ready(tmp_path,monkeypatch):
    a=cohort();a['records']=[r for r in a['records'] if not (r['split']=='test' and r['event'] in ('LEAD_BRAKE','RAMP_MERGE_EXIT'))]
    before=copy.deepcopy(a)
    g,m,d=fake_cache(monkeypatch,tmp_path,a)
    def forbidden(*a,**k):raise AssertionError('capacity must not run before coverage admission')
    monkeypatch.setattr(planning,'project',forbidden)
    root=tmp_path/'cache';root.mkdir();(root/'ready_1.json').write_text('{"status":"ready"}')
    with pytest.raises(ValueError,match='test_event_coverage_incomplete'):
        preparation.prepare(g,m,root,per_event=1,world_size=1)
    state=json.loads((root/'ready_1.json').read_text())
    assert state['status']=='rejected' and a==before
    report=json.loads(Path(state['admission']).read_text())
    assert report['coverage']['test']['missing_events']==['LEAD_BRAKE','RAMP_MERGE_EXIT']
    assert report['validation']['a']['macro_recall_defined']
    with pytest.raises(ValueError,match='not_ready'):
        preparation.load_ready(root/'ready_1.json',per_event=1,world_size=1)




def test_admitted_members_support_ready_binding_and_plan_failure(tmp_path,monkeypatch):
    a=cohort();check=admission.require(admission.inspect(a))
    assert len(check['fixed_validation_ids'])==10
    assert all(v['macro_recall_defined'] and v['queries']['pure']==40 for v in check['validation'].values())
    assert [r['id'] for r in train.training_members(a)]==check['fixed_validation_ids']
    g,m,d=fake_cache(monkeypatch,tmp_path,a);root=tmp_path/'cache'
    ready=preparation.prepare(g,m,root,per_event=1,world_size=1)
    assert ready['status']=='ready'
    assert preparation.load_ready(root/'ready_1.json',per_event=1,world_size=1)==ready
    assert preparation.prepare(g,m,root,per_event=1,world_size=1)==ready
    with pytest.raises(ValueError,match='not_ready'):
        preparation.load_ready(root/'ready_1.json',per_event=2,world_size=1)
    plan=Path(ready['plan']);saved=plan.read_text();plan.write_text('{}')
    with pytest.raises(ValueError,match='plan_changed'):
        preparation.load_ready(root/'ready_1.json',per_event=1,world_size=1)
    plan.write_text(saved)
    ad=Path(ready['admission']);saved=ad.read_text();ad.write_text('{}')
    with pytest.raises(ValueError,match='admission_changed'):
        preparation.load_ready(root/'ready_1.json',per_event=1,world_size=1)
    ad.write_text(saved)
    def fail(*a,**k):raise ValueError('capacity_failure')
    monkeypatch.setattr(planning,'project',fail)
    with pytest.raises(ValueError,match='capacity_failure'):
        preparation.prepare(g,m,root,per_event=1,world_size=1)
    assert json.loads((root/'ready_1.json').read_text())['status']=='rejected'


def test_common_masks_must_be_shared():
    a=cohort();a['records'][0]['representations']['a']['targets'][0]['valid']=False
    with pytest.raises(ValueError,match='common_query_mask_mismatch'):admission.inspect(a)


def test_native_full_preparation_freezes_all_arm_validation_and_reuses(tmp_path,monkeypatch):
    import pickle
    from PIL import Image
    from ...geometry_v1 import dataset as native
    from ...geometry_v1.tests.test_geometry import Lanes, meta
    root=tmp_path/'raw';maps=tmp_path/'maps';maps.mkdir();routes=[]
    for split in ('train','val','test'):
        for event in CONTEXT_IDS:
            route=f'Town00_{split}_{event}';run=root/'Synthetic'/route
            (run/'metas').mkdir(parents=True);(run/'rgb').mkdir()
            for frame in range(40):
                m=meta(frame);m.update(pos_global=[frame*.75,0,0],theta=0,road_id=1,lane_id=-1,lane_type_str='Driving')
                (run/'metas'/f'{frame:04d}.pkl').write_bytes(pickle.dumps(m))
            Image.new('RGB',(32,32)).save(run/'rgb/0008.jpg')
            routes.append(dict(scenario='Synthetic',route_id=route,town='Town00',split=split,
                anchors=[dict(frame=8,event=event,road_structure='R1')]))
    parent=tmp_path/'native';native.build(routes,root,Lanes(),parent)
    monkeypatch.setattr(dataset.sources,'LaneMap',lambda _:Lanes())
    cache=tmp_path/'cache'
    first=preparation.prepare(parent/'dataset.json',maps,cache,per_event=1,world_size=1)
    second=preparation.prepare(parent/'dataset.json',maps,cache,per_event=1,world_size=1)
    assert first['generation']['action']=='build' and second['generation']['action']=='reuse'
    assert first['dataset_sha256']==second['dataset_sha256']
    assert preparation.load_ready(cache/'ready_1.json',per_event=1,world_size=1)==second
    ids=[]
    for v in regions.VARIANTS:
        a=dataset.load(second['dataset'],variant=v)
        members=train.training_members(a);ids.append([r['id'] for r in members])
        assert a['admission']['sha256']==second['admission_sha256']
        assert all('representations' not in r for r in members)
        binding=dict(admission_sha256=a['admission']['sha256'],fixed_validation_ids=ids[-1])
        train.verify_admission_binding(a,binding)
        with pytest.raises(ValueError,match='checkpoint_admission_mismatch'):
            train.verify_admission_binding(a,dict(binding,admission_sha256='wrong'))
    assert all(x==ids[0] for x in ids) and len(ids[0])==10
