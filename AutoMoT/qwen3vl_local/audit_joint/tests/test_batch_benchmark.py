from copy import deepcopy
from collections import Counter
from pathlib import Path
import json
import sys
import pytest
from qwen3vl_local.audit_joint import batch_benchmark as b


def records():
    return [dict(id=f'{e}/{f}',event=str(e),sequence=f'route{e}',frame=f,
                 action='STOP' if f<24 else 'KEEP',split='train',group=f'route{e}')
            for e in range(10) for f in range(40)]


def test_fixed_membership_update_groups_and_temporal_stride():
    source=records();baseline=None
    for mode in (2,4):
        for world in (1,4):
            for batch in (1,2,4):
                updates,a=b.schedule(source,per_event=16,update_samples=16,world=world,batch=batch,mode=mode,seed=7)
                binding=(a['sample_membership_sha256'],a['update_membership_sha256'])
                if baseline is None:baseline=binding
                assert binding==baseline
                consumed=[]
                for block in updates:
                    for rank in range(world):
                        micros=b.rank_microbatches(block,rank,world,batch)
                        assert len(micros)==16//(world*batch)
                        consumed.extend(i for m in micros for i in m)
                        for micro in micros:
                            assert len(micro)==batch
                            assert all(source[y]['frame']-source[x]['frame']==(2 if mode==2 else 1)
                                       for x,y in zip(micro,micro[1:]))
                assert len(consumed)==len(set(consumed))==160
                assert Counter(source[i]['event'] for i in consumed)==dict.fromkeys(map(str,range(10)),16)


def test_invalid_budget_or_holdout_rejected():
    with pytest.raises(ValueError,match='divide'):b.schedule(records(),per_event=16,update_samples=16,world=4,batch=3,mode=2,seed=7)
    source=records();source[0]['split']='val'
    with pytest.raises(ValueError,match='training'):b.schedule(source,per_event=16,update_samples=16,world=1,batch=1,mode=2,seed=7)


def measured_fixture(folder,world=4,batch=2):
    source=records();updates,a=b.schedule(source,per_event=16,update_samples=16,world=world,batch=batch,mode=2,seed=7)
    for rank in range(world):
        steps=[]
        for u,block in enumerate(updates):
            for m,indices in enumerate(b.rank_microbatches(block,rank,world,batch)):
                steps.append(dict(update=u,micro=m,rank=rank,ids=[source[i]['id'] for i in indices],
                    events=[source[i]['event'] for i in indices],optimizer_update=m+1==a['accumulation']))
        (folder/f'rank{rank}_steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in steps))
        b.write(folder/f'rank{rank}.json',dict(status='measured',rank=rank,selection=a,batch_size=batch,
            presentations=160//world,optimizer_updates=10,seconds=10+rank,stage_seconds={'prepare_cpu':1,'backward_with_ddp':7},
            peak_allocated_bytes=100,peak_reserved_bytes=150,fits_with_headroom=True))
    return a


def test_collect_all_ranks_consumption_and_slowest_wall_time(tmp_path):
    measured_fixture(tmp_path)
    result=b.collect(tmp_path,4,0)
    assert result['status']=='measured' and result['samples_per_second']==160/13
    assert result['largest_measured_stage']=='backward_with_ddp'
    path=tmp_path/'rank2_steps.jsonl';steps=[json.loads(s) for s in path.read_text().splitlines()]
    steps[0]['ids'].reverse()
    path.write_text(''.join(json.dumps(r)+'\n' for r in steps))
    with pytest.raises(ValueError,match='consumption'):b.collect(tmp_path,4,0)


def test_failed_or_missing_rank_never_recommends(tmp_path):
    measured_fixture(tmp_path)
    assert b.collect(tmp_path,4,1)['status']=='error'
    (tmp_path/'rank1.json').unlink()
    assert b.collect(tmp_path,4,0)['status']=='error'


def test_summary_requires_same_inputs_all_repeats_and_headroom(tmp_path):
    measured_fixture(tmp_path);row=b.collect(tmp_path,4,0)
    trials=[dict(row,rgb_mode=2,batch_size=2,repeat=i) for i in range(2)]
    trials+=[dict(row,rgb_mode=2,batch_size=4,repeat=i,samples_per_second=99,fits_with_headroom=False) for i in range(2)]
    assert b.summarize(trials,2)['fastest_measured_candidates']['2']['batch_size']==2
    assert b.summarize(trials[:1],2)['fastest_measured_candidates']=={}
    trials[1]=deepcopy(trials[1]);trials[1]['selection']['update_membership_sha256']='changed'
    with pytest.raises(ValueError,match='different'):b.summarize(trials,2)


def training_fixture():
    import torch
    model=torch.nn.Linear(2,1,bias=False)
    with torch.no_grad():model.weight.copy_(torch.tensor([[.2,.3]]))
    x=torch.arange(32,dtype=torch.float32).reshape(16,2)/32
    y=x.sum(1,keepdim=True)*.6
    def prepare(indices):return dict(x=x[indices],y=y[indices]),{}
    def loss(model,inputs,indices):return ((model(inputs['x'])-inputs['y'])**2).mean()
    return model,prepare,loss


def test_phase4_uses_training_encoder_on_original_device(monkeypatch):
    import torch
    from types import SimpleNamespace
    bundle=SimpleNamespace(device='cuda:3',tokenizer=SimpleNamespace(pad_token_id=0))
    calls=[]
    def encode_batch(actual,rows,root,**kwargs):
        calls.append((actual,rows,root,kwargs))
        return dict(input_ids=torch.ones((2,3),dtype=torch.long))
    monkeypatch.setitem(sys.modules,'qwen3vl_local.sft_new_loop_phase4.rgb_short.model',
                        SimpleNamespace(encode_batch=encode_batch))
    args=SimpleNamespace(phase='4',data_root=Path('data'),max_length=99)
    prepare,_=b.callbacks(args,bundle,['a','b','c'],4,{'observation_contract':'current'})
    inputs,meta=prepare([2,0])
    assert calls==[(bundle,['c','a'],Path('data'),{'max_length':99})]
    assert bundle.device=='cuda:3' and bundle.observation_contract=='current'
    assert inputs['input_ids'].shape==(2,3) and meta['padded_tokens']==6


def test_execute_device_encoder_has_no_second_transfer_or_cpu_label():
    import torch
    model,prepare,loss=training_fixture();rows=[]
    result=b.execute(model,torch.optim.AdamW(model.parameters()),list(model.parameters()),[list(range(16))],
        rank=0,world=1,batch=2,prepare=prepare,loss_for=loss,device='invalid-device',
        timing='detailed',synchronize=lambda:None,barrier=lambda:None,emit=rows.append,inputs_on_device=True)
    assert result['optimizer_updates']==1 and result['presentations']==16
    assert 'encode_batch_device' in result['stage_seconds']
    assert 'prepare_cpu' not in result['stage_seconds'] and 'host_to_device' not in result['stage_seconds']


def test_execute_fixed_effective_batch_matches_optimizer_result():
    import torch
    states=[]
    for batch in (1,2,4):
        model,prepare,loss=training_fixture();optimizer=torch.optim.SGD(model.parameters(),lr=.1);log=[]
        result=b.execute(model,optimizer,list(model.parameters()),[list(range(8)),list(range(8,16))],
            rank=0,world=1,batch=batch,prepare=prepare,loss_for=loss,device='cpu',timing='detailed',
            synchronize=lambda:None,barrier=lambda:None,emit=log.append,memory=lambda:{'allocated':123})
        states.append(model.weight.detach().clone())
        assert result['optimizer_updates']==2 and result['presentations']==16
        assert len(log)==16//batch and sum(r['optimizer_update'] for r in log)==2
        assert log[0]['stage_cuda_bytes']['forward_loss']['allocated']==123
        assert all(k in result['stage_seconds'] for k in ('prepare_cpu','host_to_device','forward_loss','backward','clip_optimizer'))
    for state in states[1:]:torch.testing.assert_close(states[0],state)


def ddp_child(rank,init,output):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=init,rank=rank,world_size=2)
    model,prepare,loss=training_fixture();model=DistributedDataParallel(model)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1);log=[]
    result=b.execute(model,optimizer,list(model.parameters()),[list(range(8)),list(range(8,16))],
        rank=rank,world=2,batch=2,prepare=prepare,loss_for=loss,device='cpu',timing='end_to_end',
        synchronize=lambda:None,barrier=dist.barrier,emit=log.append)
    assert 'stage_seconds' not in result and 'host_submission_seconds' in result
    b.write(Path(output)/f'ddp{rank}.json',dict(weight=model.module.weight.tolist(),result=result,log=log))
    dist.destroy_process_group()


def test_actual_two_rank_gloo_accumulation_matches_full_batch(tmp_path):
    import torch
    import torch.multiprocessing as mp
    mp.spawn(ddp_child,args=('file://'+str(tmp_path/'rendezvous'),str(tmp_path)),nprocs=2,join=True)
    model,prepare,loss=training_fixture();optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    for indices in (list(range(8)),list(range(8,16))):
        optimizer.zero_grad();inputs,_=prepare(indices);loss(model,inputs,indices).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    for rank in range(2):
        result=json.loads((tmp_path/f'ddp{rank}.json').read_text())
        torch.testing.assert_close(torch.tensor(result['weight']),model.weight)
        assert result['result']['presentations']==8 and len(result['log'])==4


def test_parent_uses_simultaneous_torchrun_and_alternates_order(tmp_path,monkeypatch):
    (tmp_path/'index').write_text('{}');(tmp_path/'manifest.json').write_text('{}');(tmp_path/'config.json').write_text('{}')
    commands=[]
    monkeypatch.setattr(b,'run_owned',lambda cmd, log, env, timeout: commands.append((cmd,dict(env=env))) or 0)
    monkeypatch.setattr(b,'collect',lambda folder,world,code:dict(status='error',returncode=9))
    monkeypatch.setattr(sys,'argv',['bench','--phase','3','--index',str(tmp_path/'index'),
        '--data-root',str(tmp_path),'--model-dir',str(tmp_path),'--output',str(tmp_path/'out'),
        '--gpus','0,1,2,3','--modes','2','--batches','1','2','4','--continue-on-error'])
    with pytest.raises(SystemExit) as exc:b.main()
    assert exc.value.code==2
    assert [int(c[c.index('--batches')+1]) for c,_ in commands]==[1,2,4,4,2,1]
    assert all('--nproc_per_node=4' in c and kw['env']['CUDA_VISIBLE_DEVICES']=='0,1,2,3' for c,kw in commands)
    assert json.loads((tmp_path/'out/summary.json').read_text())['fastest_measured_candidates']=={}


@pytest.mark.parametrize('phase',['3','4'])
def test_batch_preparation_preserves_encoder_device_and_loss(tmp_path,monkeypatch,phase):
    import torch
    from types import SimpleNamespace
    from qwen3vl_local.sft_new_loop_phase3 import train as t
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import model as p4
    spec=SimpleNamespace(target_output_keys=['ACTION'],action_output_mode='choice')
    items=[SimpleNamespace(spec=spec,row=SimpleNamespace(history_rgb_paths=['a','b','c','d'])) for _ in range(2)]
    packed=[]
    for ids in ([1,2,3],[4,5,6,7]):
        x=torch.tensor(ids)
        packed.append(dict(input_ids=x,attention_mask=torch.ones_like(x),labels=x,
            loss_weights=torch.tensor([0.]*(len(x)-2)+[1.,.25]),
            component_ids=torch.tensor([0]*(len(x)-2)+[1,-1]),vision={}))
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.emb=torch.nn.Embedding(11,11)
        def forward(self,input_ids,labels=None,**kw):
            logits=self.emb(input_ids)
            loss=None if labels is None else torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,11),labels[:,1:].reshape(-1))
            return SimpleNamespace(logits=logits,loss=loss)
    model=Model();bundle=SimpleNamespace(model=model,device='cuda:99',tokenizer=SimpleNamespace(pad_token_id=0))
    args=SimpleNamespace(phase=phase,data_root=tmp_path,max_length=8192,prompt_variant='baseline')
    if phase=='3':
        counter=iter(packed)
        monkeypatch.setattr(t,'_load_images',lambda *a,**kw:[])
        monkeypatch.setattr(t,'build_action_target',lambda _: '')
        monkeypatch.setattr(t,'_build_inputs',lambda *a,**kw:next(counter))
    else:
        def encoder(actual,row,*a,**kw):
            assert actual is bundle and str(actual.device)=='cuda:99'
            value=packed[items.index(row)]
            return {k:value[k].unsqueeze(0) for k in ('input_ids','attention_mask','labels')}
        # Distinct metadata avoids equality collapsing the two mock rows.
        items[0].index=0;items[1].index=1
        monkeypatch.setattr(p4,'encode',encoder)
    prepare,loss=b.callbacks(args,bundle,items,2,{'observation_contract':{}})
    inputs,meta=prepare([0,1])
    assert all(v.device.type=='cpu' for v in inputs.values())
    assert meta['padded_tokens']==8
    if phase=='3':assert meta['tokens']==7
    actual=loss(model,inputs,[0,1]);actual.backward();grad=model.emb.weight.grad.clone();model.zero_grad()
    if phase=='3':
        cpu=SimpleNamespace(model=model,device='cpu',tokenizer=bundle.tokenizer)
        expected=t._loss_batch(cpu,packed,[spec,spec])[0]
    else:
        expected=torch.stack([model(input_ids=v['input_ids'][None],labels=v['labels'][None]).loss*(len(v['input_ids'])-1) for v in packed]).sum()/5
    expected.backward()
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(grad,model.emb.weight.grad)


def test_warmup_is_also_ten_event_equal():
    source=records()
    updates,_=b.schedule(source,per_event=32,update_samples=16,world=4,batch=2,mode=2,seed=7)
    warm=b.warmup_schedule(updates,source,1)
    assert len(warm)==10
    assert Counter(source[i]['event'] for block in warm for i in block)==dict.fromkeys(map(str,range(10)),16)


def test_parent_timeout_terminates_worker_process_group(tmp_path,monkeypatch):
    (tmp_path/'index').write_text('{}');(tmp_path/'manifest.json').write_text('{}');(tmp_path/'config.json').write_text('{}')
    calls=[]
    monkeypatch.setattr(b,'run_owned',lambda *args: calls.append(args) or 124)
    monkeypatch.setattr(sys,'argv',['bench','--phase','3','--index',str(tmp_path/'index'),
        '--data-root',str(tmp_path),'--model-dir',str(tmp_path),'--output',str(tmp_path/'out'),
        '--gpus','0','--modes','2','--batches','1','--repeats','1'])
    with pytest.raises(SystemExit) as exc:b.main()
    assert exc.value.code==2
    assert len(calls)==1
    result=json.loads((tmp_path/'out/summary.json').read_text())
    assert result['trials'][0]['returncode']==124 and result['fastest_measured_candidates']=={}
