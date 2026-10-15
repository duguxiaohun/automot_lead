from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
import pytest
from qwen3vl_local.sequence_batch import plan, rank_batches, collate


def pool(events=10,n=80):
    return [dict(id=f'{e}/{i}',event=str(e),sequence=f'{e}/route',frame=i,
        action='STOP' if i<60 else 'RESUME',split='train',group=f'{e}/route')
        for e in range(events) for i in range(n)]


def test_equal_contiguous_four_rank_batches_and_history_first():
    rows=pool();history=None;seen=set()
    for epoch in range(5):
        order,a=plan(rows,per_event=16,epoch=epoch,world_size=4,batch_size=2,history=history)
        assert Counter(rows[i]['event'] for i in order)==dict.fromkeys(map(str,range(10)),16)
        assert not seen.intersection(order)
        seen.update(order);history=a['next_history']
        for start in range(0,len(order),8):
            frames=[rows[i]['frame'] for i in order[start:start+8]]
            assert frames==list(range(frames[0],frames[0]+16,2))
        for rank in range(4):
            batches=rank_batches(order,rank=rank,world_size=4,batch_size=2)
            assert all(len(batch)==2 for batch in batches)
            assert all(rows[b[1]]['frame']-rows[b[0]]['frame']==2 for b in batches)
    assert len(seen)==len(rows)
    order,b=plan(rows,per_event=16,epoch=4,world_size=4,batch_size=2)
    assert b['next_history']==history


def test_short_sequences_stitch_without_cross_route_history_and_repeat_after_coverage():
    rows=pool(n=11)
    for r in rows:r['sequence']+=f'/{r["frame"]//3}'
    order,a=plan(rows,per_event=16,world_size=4,batch_size=2)
    for e in range(10):
        chosen=[i for i in order if rows[i]['event']==str(e)]
        assert len(set(chosen[:11]))==11
        assert max(Counter(chosen).values())==2
    assert any(s['length']<8 for segments in a['segments'].values() for s in segments)


def test_pairing_order_and_resume_tamper_and_split_leak():
    rows=pool();other=list(reversed(rows))
    a,report=plan(rows,per_event=16,world_size=4,batch_size=2)
    b,_=plan(other,per_event=16,world_size=4,batch_size=2)
    assert [rows[i]['id'] for i in a]==[other[i]['id'] for i in b]
    history=deepcopy(report['next_history']);history['counts']['0/0']+=1
    with pytest.raises(ValueError,match='exposure'):
        plan(rows,per_event=16,epoch=1,world_size=4,batch_size=2,history=history)
    with pytest.raises(ValueError,match='divisible'):
        plan(rows,per_event=15,world_size=4,batch_size=2)
    rows[0]['split']='val'
    with pytest.raises(ValueError,match='split'):
        plan(rows,per_event=16)


def test_transition_overlapping_history_is_allowed_and_actions_compensate():
    rows=pool(events=1,n=80)
    _,a=plan(rows,per_event=16,world_size=4,batch_size=2)
    assert a['action_balance']['0']['actual']=={'STOP':8,'RESUME':8}
    # Overlapping physical frames across distinct events are allowed in training.
    other=[dict(r,id='other/'+r['id'],event='other') for r in rows]
    assert len(plan(rows+other,per_event=16)[0])==32


def test_collation_padding_and_visual_order():
    import torch
    a=dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones(1,3,dtype=torch.long),
        labels=torch.tensor([[-100,2,3]]),pixel_values=torch.full((4,8),11.),image_grid_thw=torch.tensor([[1,2,2]]))
    b={k:v.clone() for k,v in a.items()};b['input_ids']=torch.tensor([[7,8]])
    b['attention_mask']=torch.ones(1,2,dtype=torch.long);b['labels']=torch.tensor([[-100,8]])
    b['pixel_values'].fill_(22)
    out=collate([a,b],0)
    assert out['input_ids'].tolist()==[[1,2,3],[7,8,0]]
    assert out['attention_mask'].tolist()==[[1,1,1],[1,1,0]]
    assert out['labels'].tolist()==[[-100,2,3],[-100,8,-100]]
    assert out['pixel_values'][:,0].tolist()==[11]*4+[22]*4
    assert out['image_grid_thw'].shape==(2,3)


def test_phase3_one_forward_and_gradient_equal_to_independent_samples():
    import torch
    from qwen3vl_local.sft_new_loop_phase3.train import _loss_batch, _loss_one
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.emb=torch.nn.Embedding(11,11);self.calls=0
        def forward(self,input_ids,**kw):
            self.calls+=1
            return SimpleNamespace(logits=self.emb(input_ids))
    model=Model();bundle=SimpleNamespace(model=model,device='cpu',tokenizer=SimpleNamespace(pad_token_id=0))
    specs=[SimpleNamespace(target_output_keys=['ACTION'],action_output_mode='choice')]*2
    packed=[]
    for ids in ([1,2,3],[4,5,6,7]):
        x=torch.tensor(ids)
        packed.append(dict(input_ids=x,attention_mask=torch.ones_like(x),labels=x,
            loss_weights=torch.tensor([0.]*(len(x)-2)+[1.,.25]),
            component_ids=torch.tensor([0]*(len(x)-2)+[1,-1]),vision={}))
    loss,stats=_loss_batch(bundle,packed,specs);loss.backward()
    assert model.calls==1 and len(stats)==2
    grad=model.emb.weight.grad.clone();model.zero_grad()
    scalar=sum(_loss_one(bundle,p,s)[0] for p,s in zip(packed,specs))/2
    scalar.backward()
    torch.testing.assert_close(loss,scalar)
    torch.testing.assert_close(grad,model.emb.weight.grad)


@pytest.mark.parametrize('frames',[2,4])
def test_real_qwen_multimodal_batch_logits_and_gradients(frames):
    import torch
    from qwen3vl_local.qwen35.tests.test_backend import model as fixture
    model=fixture.__wrapped__()
    inputs=[]
    for length in (2,5):
        ids=torch.tensor([[10]+[x for _ in range(frames) for x in (2,4,4,4,4,3)]+list(range(11,11+length))])
        labels=torch.full_like(ids,-100);labels[:,-2:]=ids[:,-2:]
        inputs.append(dict(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,
            pixel_values=torch.randn(frames*16,1536),image_grid_thw=torch.tensor([[1,4,4]]*frames)))
    batch=collate(inputs,0)
    out=model(**batch,use_cache=False);out.loss.backward()
    grad=model.lm_head.weight.grad.clone();model.zero_grad()
    losses=[]
    for i,row in enumerate(inputs):
        one=model(**row,use_cache=False)
        torch.testing.assert_close(out.logits[i,:row['input_ids'].shape[1]],one.logits[0],atol=2e-5,rtol=2e-4)
        losses.append(one.loss)
    mean=torch.stack(losses).mean();mean.backward()
    torch.testing.assert_close(out.loss,mean,atol=2e-5,rtol=2e-4)
    torch.testing.assert_close(grad,model.lm_head.weight.grad,atol=2e-5,rtol=2e-4)


def test_fixed_evaluation_membership_equal_across_modes_and_label_blind():
    from qwen3vl_local.sft_new_loop_phase3.sequence_evaluation import select
    rows=pool()
    one,a=select(rows,[str(e) for e in range(10)],limit=16,stride=1)
    two,b=select(rows,[str(e) for e in range(10)],limit=16,stride=2)
    assert set(one)==set(two) and a['membership_sha256']==b['membership_sha256']
    assert a['per_event']==b['per_event']==16
    for r in rows:r['action']='changed test label'
    assert select(rows,[str(e) for e in range(10)],limit=16,stride=1)[0]==one


def test_batched_consumption_receipt_checks_every_member_and_update():
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.diagnostics import sample_records,sample_binding
    from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
    from qwen3vl_local.audit_joint.phase4_epoch_handoff import check_consumption
    rows=[dict(id=str(i),target='YES',label_basis='weak_rule_teacher') for i in range(10240)]
    samples=sample_records(rows,list(range(len(rows))),4,2)
    sampling=dict(event_counts=dict.fromkeys(EVENTS,1024),ordered_samples=sample_binding(samples,4),batch_size=2,accumulation=8)
    ranks=[]
    for rank in range(4):
        local=[r for r in samples if r['rank']==rank];steps=[]
        for j in range(len(local)//2):
            batch=local[j*2:j*2+2]
            steps.append(dict(epoch=0,rank=rank,local_index=j,batch_size=2,ids=[r['id'] for r in batch],
                positions=[r['position'] for r in batch],loss=.2,optimizer_update=(j+1)%8==0,optimizer_steps=(j+1)//8))
        ranks.append(steps)
    assert check_consumption(samples,sampling,ranks)['status']=='matched_recorded_plan'
    ranks[2][4]['ids'].reverse()
    assert check_consumption(samples,sampling,ranks)['status']=='incomplete_or_inconsistent'


def test_probe_isolates_trials_and_retains_errors(tmp_path,monkeypatch):
    import sys,json
    from qwen3vl_local.audit_joint import batch_probe
    def child(cmd,**kw):
        output=cmd[cmd.index('--output')+1];batch=int(cmd[cmd.index('--batches')+1])
        from pathlib import Path
        Path(output).write_text(json.dumps(dict(status='measured' if batch==1 else 'oom',fits_with_headroom=batch==1)))
        return 0
    monkeypatch.setattr(batch_probe.subprocess,'call',child)
    monkeypatch.setattr(sys,'argv',['probe','--phase','3','--index','index','--data-root','rgb','--model-dir','model',
        '--gpus','0,1','--batches','1','2','4','--output',str(tmp_path/'probe')])
    batch_probe.main()
    result=json.loads((tmp_path/'probe/summary.json').read_text())
    assert len(result['trials'])==8
    assert result['conservative_measured_candidates']=={'2':1,'4':1}


def test_unseen_anchors_before_another_question_on_seen_anchor():
    rows=pool(events=1,n=8)
    rows.extend(dict(rows[0],id=f'extra/{i}') for i in range(20))
    order,a=plan(rows,per_event=8,world_size=1,batch_size=2)
    assert len({rows[i]['frame'] for i in order})==8


@pytest.mark.parametrize('mode',[2,4])
def test_phase4_actual_training_loop_consumes_batches_and_persists_exposure(tmp_path,monkeypatch,mode):
    import torch,json,sys
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import train,evaluation_policy,contract
    from qwen3vl_local.qwen35 import adapters
    from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
    rows=[dict(id=f'{event}/{i}',scenario='S',route_id=event,physical_group=event,split='train',
        episode=dict(event=event,instance_id=event),observation=dict(frame_id=i),edge='hold',
        target='YES',label_basis='weak_rule_teacher') for event in EVENTS for i in range(6)]
    manifest=dict(training_admission={},coverage={'ready':False},student_view={'selected_ids':{}},
        observation_contract=contract.observation_contract(mode))
    monkeypatch.setattr(train,'load_dataset',lambda *a,**k:({'train':rows,'val':[],'test':[]},manifest))
    monkeypatch.setattr(evaluation_policy,'prepare',lambda data,**kw:(data,dict(training_admission={'evaluation_scope':'fixture'},
        selection_metric='fixture',plans={'val':{'selected_ids':[]}})))
    monkeypatch.setattr(evaluation_policy,'score_cases',lambda *a,**kw:dict(selection_agreement=.5))
    monkeypatch.setattr(train,'evaluate_rows',lambda *a,**kw:({},[]))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.));self.shapes=[]
        def forward(self,input_ids,**kw):
            self.shapes.append(tuple(input_ids.shape))
            return SimpleNamespace(loss=(self.weight*input_ids.float()).square().mean())
    model=Model()
    class Processor:
        def save_pretrained(self,path):(path/'tokenizer.json').write_text('{}')
    bundle=SimpleNamespace(model=model,processor=Processor(),device='cpu')
    monkeypatch.setattr(train,'load_bundle',lambda *a,**k:bundle)
    monkeypatch.setattr(train,'encode_batch',lambda b,rows,*a,**k:dict(input_ids=torch.ones(len(rows),3,dtype=torch.long)))
    def save(model,path):path.mkdir(parents=True);(path/'adapter_config.json').write_text('{}')
    monkeypatch.setattr(adapters,'save_adapter',save)
    output=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['train','--dataset','fixture','--rgb-mode',str(mode),'--output-dir',str(output),
        '--epochs','2','--epoch-samples','40','--batch-size','2','--accumulation','3'])
    train.main()
    assert model.shapes==[(2,3)]*40
    for epoch in range(2):
        audit=json.loads((output/f'epoch_{epoch:03d}_sampling.json').read_text())
        assert audit['anchor_stride']==(2 if mode==2 else 1)
        assert audit['event_counts']==dict.fromkeys(EVENTS,4)
        runtime=json.loads((output/f'epoch_{epoch:03d}_runtime.json').read_text())['ranks'][0]
        assert runtime['microbatches']==20 and runtime['presentations']==40 and runtime['optimizer_steps']==7
    state=torch.load(output/'epoch_001/training_state.pt',weights_only=True)
    assert state['sampling_history']['next_epoch']==2
    assert sum(state['sampling_history']['counts'].values())==80
    assert all(n>0 for n in state['sampling_history']['counts'].values())
