import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from qwen3vl_local.audit_joint.geometry_v1 import dataset, codec, objective, train, metrics
from .test_geometry import meta, record, Lanes


def test_candidate_collection_does_not_call_old_action_filter(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import build_dataset as old
    from qwen3vl_local.sft_new_loop_phase3 import annotation_repair
    root=tmp_path/'raw';(root/'Synthetic'/'Town00_Test').mkdir(parents=True)
    collection=tmp_path/'collection';collection.mkdir()
    dataset.write(collection/'Synthetic_result.json',dict(routes=[dict(route_id='Town00_Test',annotations=[dict(frame_id=0)])]))
    monkeypatch.setattr(old,'is_abnormal_lead_route',lambda *a:(False,None))
    monkeypatch.setattr(annotation_repair,'repair_annotation',lambda *a:('STRAIGHT','U-E1',['U-E1'],{}))
    monkeypatch.setattr(old,'mapped_contexts',lambda *a:(['LEAD_BRAKE'],{}))
    def forbidden(*a,**kw):raise AssertionError('legacy action eligibility invoked')
    monkeypatch.setattr(old,'label_actions',forbidden)
    monkeypatch.setattr(old,'load_route_trajectory',forbidden)
    result=list(dataset.annotation_routes(collection,root))
    assert len(result)==1 and result[0]['anchors']==[dict(frame=0,event='LEAD_BRAKE',road_structure='STRAIGHT')]


def test_real_builder_reload_rejects_mutated_source_and_preserves_old(tmp_path):
    root=tmp_path/'data';run=root/'Synthetic'/'Town00_Test'
    (run/'metas').mkdir(parents=True);(run/'rgb').mkdir()
    for i in range(17):(run/'metas'/f'{i:04d}.pkl').write_bytes(pickle.dumps(meta(i)))
    for i in (8,9):Image.new('RGB',(32,32)).save(run/'rgb'/f'{i:04d}.jpg')
    route=dict(scenario='Synthetic',route_id='Town00_Test',town='Town00',split='train',
        anchors=[dict(frame=i,event='UE',road_structure='STRAIGHT') for i in (8,9)])
    a,receipt=dataset.build([route],root,Lanes(),tmp_path/'new',require_coverage=False)
    assert receipt['anchors']==2 and receipt['legacy_action_filter_used'] is False
    assert dataset.load(tmp_path/'new/dataset.json')==a
    with pytest.raises(FileExistsError):dataset.build([route],root,Lanes(),tmp_path/'new',require_coverage=False)
    p=run/'metas/0000.pkl';p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='source_content_changed'):dataset.load(tmp_path/'new/dataset.json')


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(20,12)
    def forward(self,input_ids,logits_to_keep,**kwargs):
        return SimpleNamespace(logits=self.embedding(input_ids)[:,logits_to_keep])


def packed(r):
    return dict(inputs=dict(input_ids=torch.tensor([[0,1,2,3]])),positions=[0,1,2,3],choices=list(range(9)))


def test_actual_optimizer_exposure_save_reload_and_budget_stop(tmp_path):
    torch.manual_seed(2);model=Tiny();bundle=SimpleNamespace(model=model,device=torch.device('cpu'))
    opt=torch.optim.AdamW(model.parameters(),lr=.01);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    work=[record(i) for i in (8,9,10,11)]
    before=model.embedding.weight.detach().clone()
    result=train.train_epoch(bundle,work,opt,sched,packed)
    assert result['steps']==4 and sum(result['actual_exposure'].values())==4
    assert not torch.equal(before,model.embedding.weight)
    source=tmp_path/'dataset.json';source.write_text('{}')
    artifact=dict(contract=codec.contract(),code={'test':'tiny'})
    binding=dict(contract=artifact['contract'],code=artifact['code'],dataset_sha256=dataset.sources.file_sha(source))
    train.save_checkpoint(tmp_path/'ckpt',bundle,opt,sched,binding=binding,epoch=1,step=4,
        exposure=result['actual_exposure'],status='epoch_complete')
    other=SimpleNamespace(model=Tiny(),device=torch.device('cpu'))
    state=train.restore(other,tmp_path/'ckpt',artifact=artifact,dataset_path=source)
    assert torch.equal(other.model.embedding.weight,model.embedding.weight)
    assert state['exposure']==result['actual_exposure']
    stop=train.train_epoch(bundle,work,opt,sched,packed,max_epoch_hours=1e-12,timing_steps=1)
    assert stop['status']=='budget_stop' and sum(stop['actual_exposure'].values())==1
    wrong=dict(artifact,contract={'version':'old_six_actions'})
    with pytest.raises(ValueError,match='contract'):train.restore(other,tmp_path/'ckpt',artifact=wrong,dataset_path=source)


def test_metrics_include_missing_cells_and_format_failures():
    r=record();r['targets'][1]=codec.unavailable('missing',1)
    pred=dict(token_ids=[4]*4,format_valid=[True,True,False,True])
    report=metrics.evaluate([dict(record=r,prediction=pred)])
    assert report['counts']['all']==3 and report['accuracy']['all']==1
    assert report['full_vocabulary_accuracy']['all']==pytest.approx(2/3)
    assert report['missing_targets']=={'1':1} and report['unsupported_cells']


def _gloo_worker(rank, init, out):
    import torch.distributed as dist
    dist.init_process_group('gloo',init_method=init,rank=rank,world_size=2)
    try:
        torch.manual_seed(2)
        model=torch.nn.parallel.DistributedDataParallel(Tiny())
        bundle=SimpleNamespace(model=model,device=torch.device('cpu'))
        opt=torch.optim.AdamW(model.parameters(),lr=.01);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        result=train.train_epoch(bundle,[record(i) for i in (8,9,10,11)],opt,sched,packed,rank=rank,world=2)
        Path(out,f'rank{rank}.json').write_text(json.dumps(result))
    finally:dist.destroy_process_group()


def test_real_two_rank_exposure(tmp_path):
    torch.multiprocessing.spawn(_gloo_worker,args=('file://'+str(tmp_path/'init'),str(tmp_path)),nprocs=2,join=True)
    a=json.loads((tmp_path/'rank0.json').read_text());b=json.loads((tmp_path/'rank1.json').read_text())
    assert a['actual_exposure']==b['actual_exposure'] and sum(a['actual_exposure'].values())==4
    assert a['steps']==b['steps']==2
