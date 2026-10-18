from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
import pytest
from qwen3vl_local.sequence_batch import plan, rank_batches, collate


def critical_rows():
    rows=[dict(id=f'common/{i}',event='U-E2',sequence='common',frame=i,
               action='depart/NO',split='train',group='common') for i in range(40)]
    rows += [dict(id=f'rare/{i}',event='U-E2',sequence=f'rare/{i}',frame=100+i,
                  action='return/YES',split='train',group=f'rare/{i}') for i in range(2)]
    return rows


def test_critical_training_policy_agrees_with_admission():
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.batch_sampling import CRITICAL_EDGES
    from qwen3vl_local.audit_joint.rule_baseline import CRITICAL_EDGES as admission_edges
    assert CRITICAL_EDGES == admission_edges


@pytest.mark.parametrize('stride',[1,2])
@pytest.mark.parametrize('world',[1,4])
def test_critical_minimum_keeps_isolated_positive_with_exact_quota(stride,world):
    rows=critical_rows();required={'U-E2':{'return/YES':1}}
    before=deepcopy(rows);history=None;rare=[]
    for epoch in range(3):
        order,a=plan(rows,per_event=8,world_size=world,batch_size=2,stride=stride,
                     epoch=epoch,history=history,action_minimums=required,max_question_repeat=1)
        assert len(order)==len(set(order))==8
        assert Counter(rows[i]['action'] for i in order)['return/YES']>=1
        assert a['action_minimums']['U-E2']['return/YES']['reserved']==1
        assert a['action_minimums']['U-E2']['return/YES']['actual']>=1
        assert a['action_minimums']['U-E2']['return/YES']['unique_questions']>=1
        assert a['action_minimums']['U-E2']['return/YES']['physical_groups']>=1
        rare.append([rows[i]['id'] for i in order if rows[i]['action']=='return/YES'])
        history=a['next_history']
        reverse=list(reversed(rows))
        reverse_order,_=plan(reverse,per_event=8,world_size=world,batch_size=2,stride=stride,
                             epoch=epoch,action_minimums=required,max_question_repeat=1)
        assert [rows[i]['id'] for i in order]==[reverse[i]['id'] for i in reverse_order]
    assert rare[0]!=rare[1] and rows==before


def test_critical_minimum_binding_capacity_and_eligible_evidence():
    rows=critical_rows();required={'U-E2':{'return/YES':1}}
    rows[-1]['reserve_eligible']=False
    order,a=plan(rows,per_event=8,action_minimums=required)
    assert rows[-2]['id'] in [rows[i]['id'] for i in order]
    for changed in (None,{}, {'U-E2':{'return/YES':2}}):
        with pytest.raises(ValueError):
            plan(rows,per_event=8,epoch=1,history=a['next_history'],action_minimums=changed)
    for invalid in ({'U-E2':{'return/YES':2}}, {'U-E2':{'missing':1}},
                    {'U-E2':{'return/YES':True}}, {'unknown':{'x':1}}):
        with pytest.raises(ValueError):plan(rows,per_event=8,action_minimums=invalid)
    with pytest.raises(ValueError):
        plan(critical_rows(),per_event=1,action_minimums={'U-E2':{'return/YES':2}})


@pytest.mark.parametrize('stride',[1,2])
def test_phase4_reserves_native_readiness_not_catchup_or_reviewed(stride):
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.batch_sampling import plan as p4_plan
    rows=[]
    for r in critical_rows():
        edge,target=r['action'].split('/')
        rows.append(dict(id=r['id'],episode={'event':r['event'],'instance_id':r['sequence']},
            scenario='scenario',route_id=r['sequence'],physical_group=r['group'],split='train',
            observation={'frame_id':r['frame']},edge=edge,target=target,slice='readiness',label_basis='weak_rule_teacher'))
    rows[-1]['slice']='catchup'
    _,a=p4_plan(rows,epoch=0,budget=8,world_size=4,batch_size=2,anchor_stride=stride)
    cell=a['action_minimums']['U-E2']['return/YES']
    assert cell['available']==1 and cell['reserved']==1
    assert 'U-E2/return/NO' in a['unsupported_critical_cells']
    rows[-2]['label_basis']='reviewed_rgb'
    _,a=p4_plan(rows,epoch=0,budget=8,world_size=4,batch_size=2,anchor_stride=stride)
    assert 'U-E2/return/YES' in a['unsupported_critical_cells']
    assert not a['action_minimums'].get('U-E2',{})


def pool(events=10,n=80):
    return [dict(id=f'{e}/{i}',event=str(e),sequence=f'{e}/route',frame=i,
        action='STOP' if i<60 else 'RESUME',split='train',group=f'{e}/route')
        for e in range(events) for i in range(n)]


@pytest.mark.parametrize('stride',[1,2])
@pytest.mark.parametrize('lengths,expected',[
    ([6,6,2,2],[6,2,6,2]), ([6,6,1,1,1,1],[6,1,1,6,1,1])])
def test_batch_tail_preserves_complete_long_pieces(stride,lengths,expected):
    rows=[dict(id=f'{route}/{frame}',event='event',sequence=str(route),frame=frame*stride,
               action='same',split='train',group=str(route))
          for route,length in enumerate(lengths) for frame in range(length)]
    order,report=plan(rows,per_event=16,world_size=4,batch_size=2,stride=stride)
    assert [s['length'] for s in report['segments']['event']]==expected
    assert report['tail_packing']['event']['truncated_pieces']==0
    assert report['unique_questions']==report['new_questions']==16
    assert report['event_counts']=={'event':16} and report['action_balance']['event']['actual']=={'same':16}
    assert all(rows[b[1]]['frame']-rows[b[0]]['frame']==stride
               for b in rank_batches(order,rank=0,world_size=4,batch_size=2))
    reversed_rows=list(reversed(rows))
    reordered,_=plan(reversed_rows,per_event=16,world_size=4,batch_size=2,stride=stride)
    assert [rows[i]['id'] for i in order]==[reversed_rows[i]['id'] for i in reordered]


def test_tail_packing_is_bounded_minimizes_joins_and_has_no_false_fit():
    from qwen3vl_local.sequence_batch import tail_piece_length
    # A single 1 cannot be used twice to fill 2; fall back without cutting
    # a long piece into even more fragments after consuming that singleton.
    assert tail_piece_length({6:[0,1],1:[0]},2)==6
    assert tail_piece_length({6:[0,1],2:[0],1:[0,1]},2)==2
    assert tail_piece_length({8:[0],3:[0],2:[0,1,2]},6)==2
    assert tail_piece_length({8:[0],3:[0,1],2:[0,1,2]},6)==3


def test_tail_packing_against_exhaustive_bounded_combinations():
    from itertools import product
    from qwen3vl_local.sequence_batch import tail_piece_length
    for counts in product(range(3),repeat=4):
        if not any(counts):continue
        available={i+1:list(range(n)) for i,n in enumerate(counts) if n}
        for remaining in range(1,9):
            feasible=[]
            for usage in product(*(range(n+1) for n in counts)):
                combo=tuple(sorted((i+1 for i,n in enumerate(usage) for _ in range(n)),reverse=True))
                if sum(combo)==remaining:feasible.append(combo)
            if feasible:
                optimum=min(feasible,key=lambda c:(len(c),tuple(-n for n in c)))
                assert tail_piece_length(available,remaining)==optimum[0]
            else:
                assert tail_piece_length(available,remaining)==max(available)


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


@pytest.mark.parametrize('stride',[1,2])
def test_same_anchor_alternatives_compensate_without_changing_continuity(stride):
    rows=pool(events=1,n=16)
    for r in rows:
        r['frame']*=stride;r['action']='proceed/NO'
    # ID order deliberately favors proceed. Rare alternatives are only at the
    # first two anchors; postponing selection would lose both opportunities.
    rows += [dict(r,id='zz/'+r['id'],action='hold/YES') for r in rows[:2]]
    order,a=plan(rows,per_event=16,world_size=4,batch_size=2,stride=stride)
    assert a['action_balance']['0']['target']=={'hold/YES':2,'proceed/NO':14}
    assert a['action_balance']['0']['actual']==a['action_balance']['0']['target']
    assert len(set(rows[i]['frame'] for i in order))==16
    assert a['new_questions']==16 and a['max_question_repeat']==1
    assert all(s['length']==8 and s['last_frame']-s['first_frame']==7*stride for s in a['segments']['0'])
    reverse=list(reversed(rows))
    reverse_order,_=plan(reverse,per_event=16,world_size=4,batch_size=2,stride=stride)
    assert [rows[i]['id'] for i in order]==[reverse[i]['id'] for i in reverse_order]
    next_order,b=plan(rows,per_event=16,epoch=1,world_size=4,batch_size=2,stride=stride,history=a['next_history'])
    assert b['new_questions']==2 and len(set(next_order[:2])-set(order))==2


@pytest.mark.parametrize('version',[1,2,3])
def test_previous_sampler_history_is_rejected(version):
    rows=pool(events=1,n=16)
    _,a=plan(rows,per_event=8)
    a['next_history']['policy']=f'event_equal_contiguous_unseen_first_v{version}'
    with pytest.raises(ValueError,match='history'):
        plan(rows,per_event=8,epoch=1,history=a['next_history'])


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


def test_fixed_evaluation_equal_quota_mode_stride_and_label_blind():
    from qwen3vl_local.sft_new_loop_phase3.sequence_evaluation import select
    rows=pool()
    one,a=select(rows,[str(e) for e in range(10)],limit=16,stride=1)
    two,b=select(rows,[str(e) for e in range(10)],limit=16,stride=2)
    assert len(set(one))==len(set(two))==160
    assert a['per_event']==b['per_event']==16
    for r in rows:r['action']='changed test label'
    assert select(rows,[str(e) for e in range(10)],limit=16,stride=1)[0]==one
    assert select(list(reversed(rows)),[str(e) for e in range(10)],limit=16,stride=2)[0]==two


@pytest.mark.parametrize('stride',[1,2])
def test_evaluation_prefers_longest_route_and_keeps_same_anchor_alternatives(stride):
    from qwen3vl_local.sft_new_loop_phase3.sequence_evaluation import select
    rows=[dict(id=f'long/{i}/{j}',event='e',sequence='long',frame=i*stride)
          for i in range(8) for j in range(2)]
    rows.append(dict(id='isolated',event='e',sequence='short',frame=0))
    by_id={r['id']:r for r in rows}
    for seed in range(10):
        ids,report=select(rows,['e'],seed=seed,limit=8,stride=stride)
        assert [by_id[i]['frame'] for i in ids]==list(range(0,8*stride,stride))
        assert all(by_id[i]['sequence']=='long' for i in ids)
        assert report['per_event']==8
    ids,_=select(rows,['e'],limit=17,stride=stride)
    assert len(ids)==len(set(ids))==17


@pytest.mark.parametrize('stride',[1,2])
def test_contiguous_repeat_cap_is_hard_and_bound_to_history(stride):
    rows=pool(events=10,n=2)
    with pytest.raises(ValueError,match='capacity'):
        plan(rows,per_event=16,world_size=4,batch_size=2,stride=stride,max_question_repeat=1)
    indices,report=plan(rows,per_event=16,world_size=4,batch_size=2,stride=stride,max_question_repeat=8)
    assert max(Counter(indices).values())==8 and report['max_question_repeat_limit']==8
    next_indices,second=plan(rows,per_event=16,epoch=1,world_size=4,batch_size=2,stride=stride,
                            max_question_repeat=8,history=report['next_history'])
    assert max(Counter(next_indices).values())==8
    assert plan(rows,per_event=16,epoch=1,world_size=4,batch_size=2,stride=stride,max_question_repeat=8)[0]==next_indices
    with pytest.raises(ValueError,match='history'):
        plan(rows,per_event=16,epoch=1,world_size=4,batch_size=2,stride=stride,
             max_question_repeat=9,history=report['next_history'])
    with pytest.raises(ValueError,match='history'):
        plan(rows,per_event=16,epoch=1,world_size=4,batch_size=2,stride=stride,
             history=report['next_history'])


def test_phase4_repeat_limit_reaches_shared_sampler():
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.batch_sampling import plan as p4_plan
    from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
    rows=[dict(id=f'{event}/{i}',episode=dict(event=event,instance_id=event),scenario='s',route_id=event,
               observation=dict(frame_id=4+i*2),edge='hold',target='YES',split='train',physical_group=event)
          for event in EVENTS for i in range(2)]
    with pytest.raises(ValueError,match='capacity'):
        p4_plan(rows,epoch=0,budget=160,expected_events=EVENTS,world_size=4,batch_size=2,max_question_repeat=1)
    ids,report=p4_plan(rows,epoch=0,budget=160,expected_events=EVENTS,world_size=4,batch_size=2)
    assert report['max_question_repeat_limit'] is None and max(Counter(ids).values())==8


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
    (tmp_path/'index').write_text('{}')
    (tmp_path/'manifest.json').write_text('{}')
    (tmp_path/'rgb').mkdir();(tmp_path/'model').mkdir()
    (tmp_path/'model/config.json').write_text('{}')
    monkeypatch.chdir(tmp_path)
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


@pytest.mark.parametrize('option', ['--index','--data-root','--model-dir','--output'])
def test_probe_empty_shell_variable_rejected_before_workers(tmp_path, monkeypatch, capsys, option):
    import sys
    from qwen3vl_local.audit_joint import batch_probe
    argv=['probe','--phase','3','--index','index','--data-root','rgb',
          '--model-dir','model','--output',str(tmp_path/'out')]
    argv[argv.index(option)+1]=''
    monkeypatch.setattr(sys,'argv',argv)
    monkeypatch.setattr(batch_probe.subprocess,'call',lambda *a,**kw:pytest.fail('worker launched'))
    with pytest.raises(SystemExit) as exc:batch_probe.main()
    assert exc.value.code==2
    assert option in capsys.readouterr().err
    assert not (tmp_path/'out').exists()


def test_probe_directory_index_rejected_before_workers(tmp_path,monkeypatch,capsys):
    import sys
    from qwen3vl_local.audit_joint import batch_probe
    (tmp_path/'config.json').write_text('{}')
    monkeypatch.setattr(sys,'argv',['probe','--phase','3','--index','.',
        '--data-root',str(tmp_path),'--model-dir',str(tmp_path),'--output',str(tmp_path/'out')])
    monkeypatch.setattr(batch_probe.subprocess,'call',lambda *a,**kw:pytest.fail('worker launched'))
    with pytest.raises(SystemExit) as exc:batch_probe.main()
    assert exc.value.code==2
    assert '--index must be an existing index file' in capsys.readouterr().err
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('stride',[1,2])
def test_action_segment_score_decrements_deficit_without_sacrificing_continuity(stride):
    rows=[dict(id=f'{route}/{i}',event='event',sequence=str(route),frame=i*stride,
               action=action,split='train',group=str(route))
          for route,actions in enumerate(['AAAABBBB','AAAABBBB','AAAAAAAA','BBBBBBBB'])
          for i,action in enumerate(actions)]
    for seed in range(20):
        order,report=plan(rows,per_event=16,world_size=4,batch_size=2,stride=stride,seed=seed)
        assert report['action_balance']['event']==dict(target={'A':8,'B':8},actual={'A':8,'B':8})
        assert [s['length'] for s in report['segments']['event']]==[8,8]
        assert report['unique_questions']==report['new_questions']==16
        assert report['tail_packing']['event']['truncated_pieces']==0
        second,again=plan(rows,per_event=16,world_size=4,batch_size=2,stride=stride,seed=seed,
                          epoch=1,history=report['next_history'])
        assert not set(order)&set(second)
        assert again['cumulative_unique_questions']==32
