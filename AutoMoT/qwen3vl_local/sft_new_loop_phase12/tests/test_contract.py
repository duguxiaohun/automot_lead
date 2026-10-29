import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from qwen3vl_local.sft_new_loop_phase12 import prompts, cache, plan, data, evaluate, UPDATES
from qwen3vl_local.sft_new_loop_phase2 import prompts as old
from qwen3vl_local.audit_joint.phase12_retrain.budget import profile
from qwen3vl_local.qwen35.tests.test_backend import model


def test_budgets_and_invalid_ratio():
    b=profile(2,'phase2_10k');assert b['epochs']==6 and b['quota']==1024 and b['expected_updates']==10752==UPDATES
    assert profile(2)['expected_updates']==5376
    with pytest.raises(ValueError):profile(1,'phase2_10k')


def test_independent_questions_and_targets():
    answers={k:False for k in prompts.P1.ANSWER_KEYS};answers['RS1']=True
    event={k:False for k in old.ANSWER_KEYS};event['DOMAIN_ROAD_CORRIDOR']=True;event['UE3']=True
    item=SimpleNamespace(spec=old.make_prompt_spec(variant='all_random_order',answers=event,seed_key='x'))
    a,b=prompts.specs(item,answers);targets=prompts.targets(answers,a,b);texts=prompts.texts(a,b,'4rgb')
    assert set(a.output_keys)==set(answers) and len(a.output_keys)==8
    assert b is item.spec and b.output_keys==item.spec.output_keys
    assert prompts.P1.parse_phase1_output(targets[0],spec=a)['RS1']=='YES'
    assert prompts.P2.parse_event_output(targets[1],spec=b)['UE3'] is True
    assert prompts.messages(['image'],texts[0])[0]==prompts.messages(['image'],texts[1])[0]
    assert 'GROUP:' not in targets[0] and 'DETAIL:' not in targets[0]
    assert 'RS1: YES' not in texts[1]
    orders=[prompts.P1.make_prompt_spec(variant='all_random_order',answers=answers,seed_key=str(i)).output_keys for i in range(16)]
    assert any(o[0].startswith('RS') for o in orders) and len(set(orders))>8


@pytest.mark.parametrize('frames',[2,4])
@pytest.mark.parametrize('suffix_length',[3,65])
def test_real_hybrid_multimodal_forks_equal_independent_full_forward(model,frames,suffix_length):
    prefix=[10]+[v for _ in range(frames) for v in (2,4,4,4,4,3)]
    a=torch.tensor([prefix+list(range(11,11+suffix_length))]);b=torch.tensor([prefix+[21,22,23,24,25]])
    pixels=torch.randn(frames*16,3*2*16*16);grid=torch.tensor([[1,4,4]]*frames)
    def inputs(ids):return dict(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=pixels,image_grid_thw=grid)
    a,b=inputs(a),inputs(b)
    with torch.no_grad():
        n,pref=cache.prefill_prefix(model,a,b)
        frozen=[v.clone() if v is not None else None for v in pref.past_key_values.recurrent_states]
        oa=cache.fork(model,a,n,pref);ob=cache.fork(model,b,n,pref);ob2=cache.fork(model,b,n,pref)
        fa=model(**a,use_cache=False);fb=model(**b,use_cache=False)
    torch.testing.assert_close(oa.logits,fa.logits[:,n:],rtol=2e-4,atol=2e-5)
    torch.testing.assert_close(ob.logits,fb.logits[:,n:],rtol=2e-4,atol=2e-5)
    torch.testing.assert_close(ob.logits,ob2.logits,rtol=0,atol=0)
    assert pref.past_key_values.get_seq_length()==n
    for old_state,new in zip(frozen,pref.past_key_values.recurrent_states):
        if old_state is not None:torch.testing.assert_close(old_state,new,rtol=0,atol=0)
    bad={**b,'input_ids':b['input_ids'].clone()};bad['input_ids'][0,0]=31
    with pytest.raises(ValueError):cache.prefix_length(a,bad,3)
    bad={**b,'pixel_values':pixels+1}
    with pytest.raises(ValueError):cache.prefix_length(a,bad,3)


def test_old_correct_losses_are_reported_even_if_net_accuracy_ties():
    from qwen3vl_local.sft_new_loop_phase12.compare import compare
    paths=['s/r/rgb/0000.jpg','s/r/rgb/0003.jpg'];hashes={p:'bound-image' for p in paths}
    old=[dict(case_index=0,scenario='s',route_id='r',frame_id=3,history_rgb_paths_used=paths,
              gt={'RS1':'YES','VULNERABLE':'NO'},parsed={'RS1':'YES','VULNERABLE':'YES'})]
    new=[dict(case_index=0,key=['s','r',3],rgb=paths,phase1=dict(gt={'RS1':True,'VULNERABLE':False},parsed={'RS1':False,'VULNERABLE':False}))]
    report=compare(old,new,hashes,hashes,1)
    assert report['review_required'] and report['old_correct_losses'][0]['question']=='RS1'
    assert report['questions']['VULNERABLE']['fixed']==1
    assert compare(old,new,hashes,{},1)['counts']['missing_or_different_rgb_sha']==1


@pytest.mark.parametrize('frames',[2,4])
def test_cache_forks_with_active_nonzero_lora(model,frames):
    from peft import LoraConfig,get_peft_model
    adapted=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,lora_dropout=0,
        target_modules=['q_proj','v_proj','o_proj'],task_type='CAUSAL_LM')).eval()
    with torch.no_grad():
        for name,param in adapted.named_parameters():
            if 'lora_B' in name:param.normal_(0,.03)
    test_real_hybrid_multimodal_forks_equal_independent_full_forward(adapted,frames,65)


def test_four_idle_gpu_selection_does_not_take_other_jobs(monkeypatch):
    from qwen3vl_local.sft_new_loop_phase12.launch import gpu_ids
    import qwen3vl_local.sft_new_loop_phase12.launch as launch
    monkeypatch.setattr(launch.subprocess,'check_output',lambda *a,**k:'0,0,0\n1,0,0\n2,0,0\n3,0,0\n4,40000,99\n')
    monkeypatch.delenv('GPU_IDS',raising=False);assert gpu_ids()==['0','1','2','3']
    monkeypatch.setenv('GPU_IDS','1,2,3,4')
    with pytest.raises(ValueError):gpu_ids()


def test_invalid_outputs_count_as_errors():
    c=dict(scenario='s',phase1={'gt':{'VULNERABLE':True},'parsed':{'VULNERABLE':None}},
           phase2={'gt':{'UE3':False},'parsed':{'UE3':None}})
    m=evaluate.metrics([c]);assert m['both_exact']==0
    assert m['phase1']['questions']['VULNERABLE']['fn']==1
    assert m['phase2']['questions']['UE3']['accuracy']==0


def test_real_tiny_lora_gets_gradients_from_both_independent_supervisions(model):
    from peft import LoraConfig,get_peft_model
    from qwen3vl_local.sft_new_loop_phase12 import objective
    from qwen3vl_local.qwen35.backend import assistant_header_text
    class Tokenizer:
        eos_token_id=1
        def __call__(self,text,**kwargs):
            return dict(input_ids=[1 if c=='\0' else ord(c)+10 for c in text],offset_mapping=[(i,i+1) for i in range(len(text))])
        def convert_tokens_to_ids(self,text):return 1
    tokenizer=Tokenizer();pixels=torch.randn(16,3*2*16*16)
    class Processor:
        def apply_chat_template(self,messages,**kwargs):return assistant_header_text(self)+messages[-1]['content']+'\0'
        def __call__(self,text,**kwargs):
            ids=torch.tensor([[10,2,4,4,4,4,3]+tokenizer(text[0])['input_ids']])
            return dict(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=pixels,image_grid_thw=torch.tensor([[1,4,4]]))
    adapted=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,lora_dropout=0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).train()
    bundle=SimpleNamespace(model=adapted,processor=Processor(),tokenizer=tokenizer,device=torch.device('cpu'))
    facts={k:k=='VULNERABLE' for k in prompts.P1.ANSWER_KEYS}
    event={k:False for k in prompts.P2.ANSWER_KEYS};event.update(DOMAIN_ROAD_CORRIDOR=True,UE3=True)
    second=prompts.P2.make_prompt_spec(variant='all_random_order',answers=event,seed_key='grad')
    first,_=prompts.specs(SimpleNamespace(spec=second),facts);targets=prompts.targets(facts,first,second)
    gradients=[]
    for target,spec in zip(targets,(first,second)):
        adapted.zero_grad(set_to_none=True)
        packed=objective.pack(bundle,[object()],'fixture',target,spec)
        assert (packed['loss_weights'][:7]==0).all() and (packed['component_ids']>0).sum()>=len(spec.output_keys)
        loss,_=objective.loss(bundle,packed,spec);(loss*.5).backward()
        grads={n:p.grad.clone() for n,p in adapted.named_parameters() if p.requires_grad and p.grad is not None}
        assert grads and sum(float(g.abs().sum()) for g in grads.values())>0
        gradients.append(grads)
    adapted.zero_grad(set_to_none=True)
    for target,spec in zip(targets,(first,second)):
        loss,_=objective.loss(bundle,objective.pack(bundle,[object()],'fixture',target,spec),spec);(loss*.5).backward()
    for name,param in adapted.named_parameters():
        if name in gradients[0]:torch.testing.assert_close(param.grad,gradients[0][name]+gradients[1][name])
    with pytest.raises(ValueError):objective.pack(bundle,[object()],'fixture',targets[0],first,max_length=5)
