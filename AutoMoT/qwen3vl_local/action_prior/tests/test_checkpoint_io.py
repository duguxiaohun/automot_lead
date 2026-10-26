"""Real save/continue and failure propagation with four CPU distributed ranks."""
from collections import OrderedDict
from datetime import timedelta
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from qwen3vl_local.action_prior import checkpoint_io, training_core


def test_cpu_payload_preserves_metadata_values_and_rng(tmp_path):
    state = OrderedDict(weight=torch.randn(2,3,requires_grad=True))
    state._metadata = {'': {'version': 1}}
    rng = torch.get_rng_state().clone()
    payload = dict(decoder=state, optimizer={'state': {0: {'step': torch.tensor(1)}}},
                   nested=[(torch.tensor(2.),'keep')], extra={'variant':'a'})
    out = tmp_path/'state.pt'
    result = checkpoint_io.save(out,lambda:payload,0,1)
    saved = torch.load(out,weights_only=False)
    assert saved['decoder']._metadata == state._metadata
    assert torch.equal(saved['decoder']['weight'],state['weight'])
    assert saved['decoder']['weight'].device.type == 'cpu'
    assert not saved['decoder']['weight'].requires_grad and state['weight'].requires_grad
    assert torch.equal(torch.get_rng_state(),rng)
    assert result['bytes']==out.stat().st_size and not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('failure',['factory','write','rename'])
def test_write_failures_preserve_previous_checkpoint(tmp_path,monkeypatch,failure):
    path=tmp_path/'latest.pt';path.write_bytes(b'previous complete checkpoint')
    def factory():
        if failure=='factory':raise ValueError('fixture payload failure')
        return {'value':torch.tensor(3)}
    if failure=='write':
        def failed(*args,**kwargs):raise OSError('fixture disk full')
        monkeypatch.setattr(torch,'save',failed)
    if failure=='rename':
        def failed(*args,**kwargs):raise OSError('fixture rename failure')
        monkeypatch.setattr(Path,'replace',failed)
    with pytest.raises(RuntimeError,match='checkpoint write failed'):
        checkpoint_io.save(path,factory,0,1)
    assert path.read_bytes()==b'previous complete checkpoint'


def test_nccl_uses_cached_cpu_group_not_default_gpu_group(monkeypatch):
    parent=object();group=object();created=[]
    monkeypatch.setattr(dist,'is_initialized',lambda:True)
    monkeypatch.setattr(dist,'get_world_size',lambda:4)
    monkeypatch.setattr(dist,'get_backend',lambda:'nccl')
    monkeypatch.setattr(dist,'group',SimpleNamespace(WORLD=parent))
    monkeypatch.setattr(dist,'new_group',lambda **kw:created.append(kw) or group)
    assert checkpoint_io.control_group(4) is group
    assert checkpoint_io.control_group(4) is group
    assert created==[dict(backend='gloo',timeout=timedelta(seconds=600))]
    checkpoint_io._CONTROL.pop(parent)


def _distributed_save(rank, rendezvous, directory, fail):
    from qwen3vl_local.action_expert_ablation.geometry_abc.tests.test_core import tiny,inputs,record
    from qwen3vl_local.action_expert_ablation.geometry_abc import worker
    from qwen3vl_local.action_prior.flow_matching import make_training_flow,flow_matching_loss
    from qwen3vl_local.action_expert_ablation.common import leadmot_train
    torch.set_num_threads(1);torch.manual_seed(321)
    root=Path(directory)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=4,
                            timeout=timedelta(seconds=20))
    try:
        model=tiny();ddp=torch.nn.parallel.DistributedDataParallel(model)
        opt=torch.optim.AdamW(model.parameters(),lr=.001);opt.action_contract={'test':'native_save'}
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        ema=leadmot_train()._DecoderEMA(model)
        r=record(rank);r['split']='train'
        flow=make_training_flow(torch.ones(1,10,2),torch.ones(1,8,2),model.flow_config)
        output=ddp(**inputs(model,r),flow_state=flow['flow_state'],flow_time=flow['flow_time'],sample_trajectory=False)
        loss=flow_matching_loss(output,flow['flow_target_velocity'],10,.5,1.)[0]
        loss.backward();opt.step();scheduler.step();ema.update(model);opt.zero_grad(set_to_none=True)
        original=torch.save
        def delayed(payload,stream,*args,**kwargs):
            if not isinstance(payload,dict) or payload.get('schema')!='abc_trajectory_checkpoint_v1':
                return original(payload,stream,*args,**kwargs)
            time.sleep(.4)
            assert not list(root.glob('continued_rank*')),'a peer ran ahead of checkpoint publication'
            if fail:raise OSError('fixture rank0 disk failure')
            original(payload,stream)
        if rank==0:torch.save=delayed
        args=SimpleNamespace(best_selection_metric='natural_ade',route_loss_weight=.5,waypoint_loss_weight=1.)
        try:
            training_core.save_training_checkpoint(root/'latest.pt',model,opt,scheduler,ema,args,{'variant':'a'},
                {'train':'fixture'},{'epoch':0,'micro':1},1,1.,rank,4,
                schema='abc_trajectory_checkpoint_v1',contract_key='condition_contract')
        except RuntimeError as exc:
            assert fail and 'fixture rank0 disk failure' in str(exc)
            (root/f'error_rank{rank}.json').write_text(json.dumps({'error':str(exc)}))
            return
        finally:
            torch.save=original
        assert not fail and (root/'latest.pt').is_file()
        (root/f'continued_rank{rank}').write_text('published')
        saved=torch.load(root/'latest.pt',weights_only=False)
        assert saved['step']==1 and len(saved['rng_by_rank'])==4
        class Runtime:
            device=torch.device('cpu')
            def forward_sample(self,sample,decoder,config,dtype,clip=None,**kw):
                return decoder(**inputs(model,sample['abc_record']),**kw)
        prepared=dict(gt_route=torch.ones(10,2),gt_waypoints=torch.ones(8,2),clip={})
        before={k:v.clone() for k,v in model.named_parameters()}
        proof=worker.post_checkpoint_backward(Runtime(),{'abc_record':r},prepared,ddp,model,
            model.config,torch.float32,model.flow_config,args,opt)
        assert proof['optimizer_updates']==0 and proof['split']=='train'
        assert all(torch.equal(before[k],v) and v.grad is None for k,v in model.named_parameters())
        (root/f'passed_rank{rank}.json').write_text(json.dumps(proof))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('fail',[False,True])
def test_four_ranks_native_save_then_real_abc_backward_or_shared_failure(tmp_path,fail):
    import torch.multiprocessing as mp
    context=mp.start_processes(_distributed_save,args=(str(tmp_path/'rendezvous'),str(tmp_path),fail),
                               nprocs=4,join=False,start_method='spawn')
    until=time.monotonic()+40
    try:
        while time.monotonic()<until:
            if context.join(timeout=.2):break
        else:pytest.fail('checkpoint coordination hung')
    finally:
        for process in context.processes:
            if process.is_alive():process.terminate()
        for process in context.processes:process.join(timeout=5)
    if fail:
        errors=[json.loads(p.read_text())['error'] for p in tmp_path.glob('error_rank*.json')]
        assert len(errors)==4 and len(set(errors))==1
        assert not list(tmp_path.glob('continued_rank*')) and not (tmp_path/'latest.pt').exists()
    else:
        assert len(list(tmp_path.glob('passed_rank*.json')))==4
