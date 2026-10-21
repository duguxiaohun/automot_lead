"""Regression counterexamples from the geometry implementation review."""
import copy
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import pytest
import torch

from qwen3vl_local.audit_joint.geometry_v1 import codec, dataset, expert, objective, planning, sources, splits, startup, train
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
from qwen3vl_local.sft_new_loop_phase3.build_dataset import physical_route_group
from .test_geometry import Lanes, meta, record


def test_history_past_minus_current_with_current_speed_and_rotated_lane():
    class Rotated:
        def at(self, town, position, yaw):
            return codec.reference(math.pi/2, yaw)
    metas={i:meta(i,speed=1+i,yaw=math.pi/2) for i in range(9)}
    # World displacement to past is (-6,-10). Current lane +y => lateral +6.
    for i,m in metas.items(): m['ego_matrix'][:2,3]=[i*.75,i*1.25]
    observed=sources.history(metas,8,'Town00',Rotated())
    expected=codec.classify([-6,-10],2,9,math.pi/2)
    assert observed[0]['name']==expected['name']=='RIGHT_NEAR'
    assert all(h['reference_speed_mps']==9 and h['origin_frame']==8 for h in observed)
    assert [h['endpoint_frame'] for h in observed]==[0,2,4,6]
    # Old speeds, lanes and future values must not affect current-reference labels.
    for i,m in metas.items():
        m['future_positions']='poison'
        if i<8: m['speed']='poison'
    assert sources.history(metas,8,'Town00',Rotated())==observed
    metas[9]={'ego_matrix':'poison'}
    assert sources.history(metas,8,'Town00',Rotated())==observed
    del metas[1]
    assert [h['valid'] for h in sources.history(metas,8,'Town00',Rotated())]==[False,True,True,True]


def test_prompt_rejects_old_history_reference_and_never_reads_targets():
    r=record();expected=objective.messages(r,'image')
    r['targets']='poison';assert objective.messages(r,'image')==expected
    del r['observation']['history'][0]['reference_frame']
    with pytest.raises(ValueError,match='history_current_reference'): objective.messages(r,'image')


@pytest.mark.parametrize('field',['id','frame','speed','history','navigation','event','image','route'])
def test_prediction_binding_rejects_wrong_sample(field):
    r=record()
    p=dict(source='predicted',token_ids=[4]*4,reference=copy.deepcopy(r['observation']['reference']),
           contract=codec.contract(),sample_binding=objective.observation_binding(r))
    wrong=copy.deepcopy(r)
    if field=='speed':wrong['observation']['speed_mps']+=1
    elif field=='history':wrong['observation']['history'][0]['token_id']=0
    elif field=='navigation':wrong['observation']['navigation_xy'][0][0]+=1
    elif field=='route':wrong['route_id']='Town00_Other'
    elif field=='frame':wrong['frame']+=1
    else:wrong[field]+='_other'
    with pytest.raises(ValueError,match='prediction_sample_mismatch'):expert.condition(wrong,prediction=p)
    r['targets']='poison'
    assert expert.condition(r,prediction=p)['geometry_token_ids'].tolist()==[[4]*4]
    del p['sample_binding']
    with pytest.raises(ValueError,match='prediction_sample_mismatch'):expert.condition(r,prediction=p)


def test_inherits_real_additional_development_and_manifest_union(tmp_path):
    known='Accident/Town06_route_001856'
    contract=splits.development_contract();assert known in contract['groups']
    r=record();r.update(scenario='Accident',route_id='Town06_route_001856',split='test',physical_group=known)
    with pytest.raises(ValueError,match='development_group_in_holdout'):dataset.validate_splits([r])
    r['route_id']+='_10_10_10_10_10'
    with pytest.raises(ValueError,match='development_group_in_holdout'):dataset.validate_splits([r])
    manifest=tmp_path/'manifest.json'
    dataset.write(manifest,dict(additional_development=dict(groups=['Synthetic/Town00_Test'],source_sha256='existing_contract')))
    extra=splits.development_contract([manifest])
    assert known in extra['groups'] and 'Synthetic/Town00_Test' in extra['groups']
    assert str(manifest) in extra['sources']
    with pytest.raises(ValueError,match='development_group_in_holdout'):
        dataset.validate_splits([record(split='val')],extra['groups'])


def support_rows(n=4):
    return [dict(id=f'{g}:{event}',scenario='Synthetic',route_id=f'Town00_{g}',frame=8,
                 physical_group=f'Synthetic/Town00_{g}',event=event,split='train',targets=[dict(valid=True)])
            for g in range(n) for event in CONTEXT_IDS]


def test_coverage_after_geometry_filtering_moves_entire_physical_groups():
    rows=support_rows()
    # A candidate route loses every target; it cannot repair holdout coverage.
    for r in rows[:len(CONTEXT_IDS)]:r['targets']=[dict(valid=False)]
    duplicate=dict(rows[len(CONTEXT_IDS)],id='repeat',route_id='Town00_1_10_10_10_10_10')
    rows.append(duplicate)
    report=splits.complete(rows,development=[])
    assert not report['unresolved']
    assert all(count>=1 for values in report['after'].values() for count in values.values())
    dataset.validate_splits(rows)
    assert duplicate['split']==rows[len(CONTEXT_IDS)]['split']
    assert rows[0]['split']=='train'
    missing=support_rows(2)
    assert splits.complete(missing,[])['unresolved']
    protected=support_rows(2)
    assert splits.complete(protected,['Synthetic/Town00_0'])['unresolved']
    assert all(r['split']=='train' for r in protected if r['physical_group']=='Synthetic/Town00_0')


def build_fixture(tmp_path, *, formal=True, extra=()):
    root=tmp_path/'raw';routes=[]
    for i in range(3 if formal else 1):
        rid=f'Town00_{i}';run=root/'Synthetic'/rid
        (run/'metas').mkdir(parents=True);(run/'rgb').mkdir()
        for frame in range(9):(run/'metas'/f'{frame:04d}.pkl').write_bytes(pickle.dumps(meta(frame)))
        Image.new('RGB',(32,32)).save(run/'rgb/0008.jpg')
        routes.append(dict(scenario='Synthetic',route_id=rid,town='Town00',split='train',
            anchors=[dict(frame=8,event=e,road_structure='STRAIGHT') for e in CONTEXT_IDS]))
    a,_=dataset.build(routes,root,Lanes(),tmp_path/'new',require_coverage=formal,development_manifests=extra)
    return a,tmp_path/'new/dataset.json'


def test_formal_builder_repairs_splits_and_reloads_before_training(tmp_path):
    a,path=build_fixture(tmp_path)
    assert a==dataset.load(path) and not a['scoped_smoke_only']
    assert len(train.training_members(a))==10
    assert not a['split_coverage']['unresolved']
    assert (path.parent/'candidate_split_coverage.json').is_file()


def test_extra_manifest_is_bound_and_scoped_smoke_cannot_train(tmp_path):
    manifest=tmp_path/'manifest.json'
    dataset.write(manifest,dict(additional_development=dict(groups=['Synthetic/Town00_0'])))
    a,path=build_fixture(tmp_path,formal=False,extra=[manifest])
    assert str(manifest) in a['proof'] and a==dataset.load(path)
    with pytest.raises(ValueError,match='scoped_smoke'):train.training_members(a)
    dataset.write(manifest,dict(additional_development=dict(groups=['Synthetic/Town00_other'])))
    with pytest.raises(ValueError,match='development_contract_changed'):dataset.load(path)


def matching_rows(options):
    return [record(frame=8+frame,event=chr(65+event)) for event,frames in enumerate(options) for frame in frames]


def test_shared_capacity_counterexample_and_chain_reassignment():
    for options in (([0,1],[0]),([0,1],[1,2],[0])):
        rows=matching_rows(options)
        work,report=planning.plan(rows,per_event=10,world_size=1,max_repeat=10)
        counts=Counter((r['frame']) for r in work)
        assert max(counts.values())<=10 and set(report['event_counts'].values())=={10}
        assert report['capacity_reassignments']>0
    work,_=planning.plan(matching_rows(([0,1],[0])),per_event=10,world_size=1,max_repeat=10)
    assert {r['frame'] for r in work if r['event']=='A'}=={9}
    with pytest.raises(ValueError,match='capacity'):
        planning.plan(matching_rows(([0],[0])),per_event=10,world_size=1,max_repeat=10)


def test_capacity_matches_exhaustive_feasibility_on_small_graphs():
    subsets=[tuple(i for i in range(3) if mask&(1<<i)) for mask in range(1,8)]
    # All 343 three-event/three-frame graphs, checked against exhaustive assignments.
    for options in itertools.product(subsets,repeat=3):
        feasible=any(len(set(assignment))==3 for assignment in itertools.product(*options))
        try:work,_=planning.plan(matching_rows(options),per_event=1,world_size=1,max_repeat=1)
        except ValueError as exc:
            assert not feasible and 'capacity' in str(exc),options
        else:assert feasible and len({r['frame'] for r in work})==3,options


def test_startup_probe_bound_and_budget_covers_source_hash(tmp_path,monkeypatch):
    rows=support_rows(50)
    assert len(startup.probe_records(rows,rows))==20
    now=[0.]
    monkeypatch.setattr(startup.time,'monotonic',lambda:now[0])
    budget=startup.Budget(1/3600)
    def slow():now[0]=2.
    with pytest.raises(TimeoutError,match='startup_time_budget'):budget.run('source_verification',slow)
    p=tmp_path/'file';p.write_bytes(b'source')
    with pytest.raises(TimeoutError):sources.file_sha(p,check_budget=budget.check)


def _startup_worker(rank, init, source, output, corrupt):
    import torch.distributed as dist
    dist.init_process_group('gloo',init_method=init,rank=rank,world_size=2)
    calls=[];original=dataset.sources.file_sha
    def sha(path,**kwargs):
        calls.append(str(path));return original(path,**kwargs)
    dataset.sources.file_sha=sha
    try:
        try:
            a,checksum=startup.load_dataset(source,rank=rank,world=2,budget=startup.Budget(1))
            result=dict(records=len(a['records']),sha=checksum,error=None)
        except RuntimeError as exc:result=dict(error=str(exc))
        result['raw_meta_hashes']=sum('/metas/' in p for p in calls)
        Path(output,f'worker{rank}.json').write_text(json.dumps(result))
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('corrupt',[False,True])
def test_two_gloo_ranks_share_source_verification_and_failure(tmp_path,corrupt):
    a,path=build_fixture(tmp_path,formal=False)
    if corrupt:
        meta_path=next(Path(p) for p in a['proof'] if '/metas/' in p)
        meta_path.write_bytes(b'changed')
    torch.multiprocessing.spawn(_startup_worker,args=('file://'+str(tmp_path/'init'),str(path),str(tmp_path),corrupt),nprocs=2,join=True)
    values=[json.loads((tmp_path/f'worker{rank}.json').read_text()) for rank in range(2)]
    assert values[0]['raw_meta_hashes']>0 and values[1]['raw_meta_hashes']==0
    if corrupt:
        assert all('source_content_changed' in v['error'] for v in values)
    else:
        assert values[0]['sha']==values[1]['sha'] and all(v['records']==10 and v['error'] is None for v in values)
