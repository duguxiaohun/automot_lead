import copy
import itertools
import json
import math
import pickle
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from PIL import Image

from ...geometry_v1 import codec, dataset as native
from ...geometry_v1.tests.test_geometry import record as old_record, meta, Lanes
from ...geometry_v1.tests.test_pipeline import Tiny, packed
from .. import regions, dataset, planning, objective, metrics, expert, train


def record(frame=8, event='LEAD_BRAKE', variant='a', route='Town00_Test'):
    r=old_record(8,event)
    r.update(frame=frame,road_structure='R3',route_id=route,physical_group='Synthetic/'+route,
             id=native.digest([route,frame,event]),variant=variant)
    if variant!='a':
        r['targets']=[regions.relabel(t,variant,'R3') for t in r['targets']]
        r['observation']['history']=[regions.relabel(t,variant,'R3') for t in r['observation']['history']]
    r['sampling_mass']={f'a/{i}/4':1. for i in range(4)}
    return r


@pytest.mark.parametrize('variant,road,lo,hi',[('b','R3',20,24),('b','R2',3,13),('c','R3',3,13)])
def test_fixed_boundaries_shared_by_all_durations_and_history(variant,road,lo,hi):
    for t in (.5,1,1.5,2):
        for radius,expected in ((0,3),(lo-1e-6,3),(lo,4),(hi,4),(hi+1e-6,5),(10000,5)):
            item=codec.classify([radius,0],t,8,0)
            out=regions.relabel(item,variant,road)
            assert out['token_id']==expected
            assert (out['inner_radius_m'],out['outer_radius_m'])==(lo,hi)
            assert 'q_mps' not in out
    with pytest.raises(ValueError):regions.bounds(variant,'UNKNOWN')


def test_a_is_byte_equivalent_and_left_near_exists_for_fixed_three_meters():
    item=codec.classify([0,-2.5],1,5,0)
    assert regions.relabel(item,'a','R3')==item
    assert regions.relabel(item,'c','R3')['token_id']==0
    assert regions.relabel(dict(valid=False,reason='missing'),'b','R3')==dict(valid=False,reason='missing')






def test_same_order_all_arms_three_epochs_no_frame_repeat():
    rows=[record(f,e,route=f'Town00_R{f%4}') for e in ('E1','E2') for f in range(8,40)]
    p=planning.project(rows,per_event=12,world_size=4,epochs=3)
    total={}
    for plan in p['plans']:
        assert plan['max_frame_repeat']==1 and plan['optimizer_steps']==6
        for k in plan['ordered_ids']:total[k]=total.get(k,0)+1
    assert p['actual_exposure_advanced'] is False
    for variant in regions.VARIANTS:
        other=copy.deepcopy(rows)
        for r in other:r['variant']=variant;r['targets']=[dict(valid=True,name='DIFFERENT')]*4
        assert planning.project(other,per_event=12,world_size=4,epochs=3)==p
    assert max(x['max_run_frame_repeat'] for x in p['plans'])<=3


def test_shared_frame_capacity_matches_exhaustive_assignment():
    # Every event has an individually adequate pool; shared capacities can still fail.
    for masks in itertools.product(range(1,4),repeat=4):
        rows=[record(8+f,f'E{e}') for f,m in enumerate(masks) for e in range(2) if m&(1<<e)]
        if {r['event'] for r in rows}!={'E0','E1'}:continue
        possible=any(sum(x==0 for x in choice)==2 and sum(x==1 for x in choice)==2
            for choice in itertools.product(*[[e for e in range(2) if m&(1<<e)] for m in masks]))
        try:
            work,_=planning.plan(rows,per_event=2,world_size=1,epochs=3)
        except ValueError:
            assert not possible
        else:
            assert possible and len({(r['route_id'],r['frame']) for r in work})==4
    with pytest.raises(ValueError):planning.plan([record()],per_event=2,world_size=1)


def test_nine_region_soft_loss_and_no_target_leakage():
    for variant in regions.VARIANTS:
        r=record(variant=variant);before=objective.messages(r,'image')
        r['targets']='POISON';r['sampling_mass']='POISON';r['six_audit']='POISON'
        assert objective.messages(r,'image')==before
    r=record();r['targets'][0].update(target=[.5,0,0,.5,0,0,0,0,0])
    r['targets'][1]['valid']=False
    logits=torch.randn(4,15,requires_grad=True)
    objective.loss_from_logits(logits,r['targets'],list(range(9))).backward()
    assert logits.grad[1].abs().sum()==0 and logits.grad[0,12]==0


def prediction(r,ids):
    return dict(token_ids=ids,format_valid=[True]*4,contract=regions.contract(r['variant']),
                sample_binding=objective.observation_binding(r),source='predicted')


def test_history_copy_transition_metrics_and_binding():
    r=record();r['targets'][2]['token_id']=3;r['targets'][3]['token_id']=3
    case=dict(record=r,prediction=prediction(r,[4]*4))
    report=metrics.evaluate([case])
    assert report['history']['conflicts_copy']==2
    assert report['transitions']['adjacent_future']['recall']==0
    assert report['macro_recall']==.5
    wrong=copy.deepcopy(r);wrong['variant']='b'
    with pytest.raises(ValueError):expert.condition(wrong,expected_variant='b',prediction=case['prediction'])
    assert expert.condition(r,expected_variant='a',mode='none')['geometry_token_ids'].tolist()==[[9]*4]
    unknown=metrics.intervention(r,'history_unknown')
    assert all(not h['valid'] for h in unknown['observation']['history'])
    assert r['observation']['history'][-1]['valid']


def test_native_raw_build_common_mask_source_reverification(tmp_path,monkeypatch):
    root=tmp_path/'raw';run=root/'Synthetic'/'Town00_Test';maps=tmp_path/'maps';maps.mkdir()
    (run/'metas').mkdir(parents=True);(run/'rgb').mkdir()
    for i in range(40):
        m=meta(i);m.update(pos_global=[i*.75,0,0],theta=0,road_id=1,lane_id=-1,lane_type_str='Driving')
        (run/'metas'/f'{i:04d}.pkl').write_bytes(pickle.dumps(m))
    for i in (8,9):Image.new('RGB',(32,32)).save(run/'rgb'/f'{i:04d}.jpg')
    route=dict(scenario='Synthetic',route_id='Town00_Test',town='Town00',split='train',
               anchors=[dict(frame=i,event='LEAD_BRAKE',road_structure='R1') for i in (8,9)])
    parent=tmp_path/'native'
    native.build([route],root,Lanes(),parent,require_coverage=False)
    monkeypatch.setattr(dataset.sources,'LaneMap',lambda _:Lanes())
    study,report=dataset.build(parent/'dataset.json',maps,tmp_path/'study')
    assert report['anchors']==2
    for v in regions.VARIANTS:
        loaded=dataset.load(tmp_path/'study/dataset.json',variant=v)
        assert loaded['contract']['variant']==v
        assert all(any(t['valid'] for t in r['targets']) for r in loaded['records'])
    with pytest.raises(FileExistsError):dataset.build(parent/'dataset.json',maps,tmp_path/'study')
    from ..prepare import prepare
    states=[]
    for _ in range(2):
        with pytest.raises(ValueError,match='scoped_smoke_dataset_not_for_formal_training'):
            prepare(parent/'dataset.json',maps,tmp_path/'cache',per_event=1,world_size=1)
        states.append(json.loads((tmp_path/'cache/ready_1.json').read_text()))
    assert [s['generation']['action'] for s in states]==['build','reuse']
    assert states[0]['dataset_sha256']==states[1]['dataset_sha256']
    assert all(s['status']=='rejected' for s in states)
    p=run/'metas/0000.pkl';p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='source_content_changed'):dataset.load(tmp_path/'study/dataset.json')






def test_budget_stop_counts_only_completed_optimizer_updates():
    model=Tiny();bundle=SimpleNamespace(model=model,device='cpu')
    opt=torch.optim.AdamW(model.parameters(),lr=.01);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    exposure={};result=train.train_epoch(bundle,[record(i) for i in range(8,12)],opt,sched,packed,
                                         exposure=exposure,max_epoch_hours=1e-12,timing_steps=1)
    assert result['status']=='budget_stop' and result['steps']==1
    assert sum(result['actual_exposure'].values())==1 and exposure=={}


def _gloo_worker(rank,init,out):
    import torch.distributed as dist
    dist.init_process_group('gloo',init_method=init,rank=rank,world_size=2)
    try:
        torch.manual_seed(2);model=torch.nn.parallel.DistributedDataParallel(Tiny())
        bundle=SimpleNamespace(model=model,device=torch.device('cpu'))
        opt=torch.optim.AdamW(model.parameters(),lr=.01);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        result=train.train_epoch(bundle,[record(i) for i in range(8,12)],opt,sched,packed,rank=rank,world=2)
        Path(out,f'rank{rank}.json').write_text(json.dumps(result))
    finally:dist.destroy_process_group()


def test_real_two_rank_optimizer_exposure(tmp_path):
    torch.multiprocessing.spawn(_gloo_worker,args=('file://'+str(tmp_path/'init'),str(tmp_path)),nprocs=2,join=True)
    a=json.loads((tmp_path/'rank0.json').read_text());b=json.loads((tmp_path/'rank1.json').read_text())
    assert a['actual_exposure']==b['actual_exposure'] and sum(a['actual_exposure'].values())==4
    assert a['steps']==b['steps']==2
