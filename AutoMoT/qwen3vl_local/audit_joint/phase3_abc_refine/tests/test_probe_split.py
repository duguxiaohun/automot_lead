import copy
from types import SimpleNamespace
import pytest
import torch
from qwen3vl_local.audit_joint.phase3_abc_refine import preflight,probe_inputs,bindings


def fixture():
    records=[dict(id=f'train{i}',split='train') for i in range(6)]+[dict(id=s,split=s) for s in ('val','test')]
    extrema={}
    for v in 'abc':
        for i,r in enumerate(records):
            packed=dict(tokens=100+i,inputs={'image_grid_thw':torch.tensor([[1,20 if i==4 else 2,4]])})
            probe_inputs.update(extrema,r,v,packed)
    return records,dict(extrema=extrema)


@pytest.mark.parametrize('variant',list('abc'))
def test_real_probe_control_flow_passes_only_train_extrema_to_train_epoch(tmp_path,monkeypatch,variant):
    from qwen3vl_local.audit_joint.geometry_abc_v2 import dataset,train
    from qwen3vl_local.sft_new_loop_phase3 import train as native
    records,processor=fixture();captured=[]
    class Reached(Exception):pass
    def intercept(bundle,work,*a,**kw):captured.extend(work);raise Reached()
    monkeypatch.setattr(native,'setup_distributed',lambda *a:(0,0,4))
    monkeypatch.setattr(native,'cleanup_distributed',lambda:None)
    monkeypatch.setattr(native,'load_model_with_lora',lambda *a,**k:SimpleNamespace(model=torch.nn.Linear(2,2)))
    monkeypatch.setattr(bindings,'model_identity',lambda *a:{'fixture':'model'})
    monkeypatch.setattr(bindings,'verify_processor',lambda *a:processor)
    monkeypatch.setattr(dataset,'load',lambda *a,**k:{'records':records})
    monkeypatch.setattr(torch.nn.parallel,'DistributedDataParallel',lambda m,**k:m)
    monkeypatch.setattr(torch.distributed,'barrier',lambda:None)
    monkeypatch.setattr(train,'train_epoch',intercept)
    args=SimpleNamespace(model_dir='/fixture',processor_report='/fixture',dataset='/fixture',variant=variant,output=tmp_path/'probe')
    with pytest.raises(Reached):preflight.probe(args)
    assert [r['id'] for r in captured]==['train5','train4','train0','train1']
    assert all(r['split']=='train' for r in captured)


def test_holdout_forward_has_no_grad_no_updates_and_restores_mode():
    records,processor=fixture();selected=probe_inputs.selection(records,processor,'a');m=torch.nn.Linear(2,3)
    before=copy.deepcopy(m.state_dict());seen=[]
    def pack(r):
        item=next(x for x in selected['holdout_forward_only'] if x['id']==r['id'])
        return dict(tokens=item['tokens'],inputs={'image_grid_thw':torch.tensor(item['image_grid_thw'])})
    def read(bundle,packed):
        assert not torch.is_grad_enabled() and not bundle.model.training
        out=bundle.model(torch.ones(1,2));assert not out.requires_grad;seen.append(packed['tokens']);return out
    proofs=probe_inputs.holdout_forward(SimpleNamespace(model=m),records,selected['holdout_forward_only'],pack,read)
    assert len(seen)==2 and m.training and all(p.grad is None for p in m.parameters())
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert all(x['updates']==0 for x in proofs)
    with pytest.raises(ValueError,match='split_changed'):
        probe_inputs.holdout_forward(SimpleNamespace(model=m),records,selected['backward'][:1],pack,read)


def test_actual_packed_maximum_must_match_report():
    records,processor=fixture();item=probe_inputs.selection(records,processor,'a')['backward'][0]
    packed=dict(tokens=item['tokens']-1,inputs={'image_grid_thw':torch.tensor(item['image_grid_thw'])})
    with pytest.raises(ValueError,match='shape_changed'):probe_inputs.verify_shape(packed,item)
    processor['extrema']['a']['train']['tokens']['id']='test'
    with pytest.raises(ValueError,match='split_or_member'):probe_inputs.selection(records,processor,'a')
