import ast
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
import torch

from qwen3vl_local.audit_joint.phase3_base_dense import baseline, data, objectives
from qwen3vl_local.audit_joint import phase3_base_trial as trial, phase3_base_trial_worker as worker
from qwen3vl_local.sft_new_loop_phase3 import train
from qwen3vl_local.sft_new_loop_phase3.test_validation_balance import candidate_rows
from qwen3vl_local.audit_joint.tests.test_phase3_sequence_trial import job
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.tests.test_batch_runtime import row,tokenizer


def args(tmp_path):
    return worker.make_args(job(tmp_path))


def rows():
    source=[x for x in candidate_rows() if not x.invalid_source]
    result=[]
    for x in source:
        allowed=(*train.CONTEXT_BY_ID[x.context_id].action_keys,'KEEP')
        for f in range(4,24):
            action=allowed[0] if f<12 else allowed[f%len(allowed)]
            answers={key:key==action for key in x.answers}
            result.append(replace(x,split='train',frame_id=f,route_id=x.route_id+'train',
                answers=answers,action_signature=action if action!='KEEP' else 'NONE',
                history_rgb_paths=[f'/rgb/{x.context_id}/{i}.jpg' for i in range(f-3,f+1)]))
    return result


def test_historical_selector_ast_and_exact_three_epochs(tmp_path):
    old=subprocess.check_output(['git','show','71cc49636eb367c1e1c06743953f48e61fd17f4c:AutoMoT/qwen3vl_local/sft_new_loop_phase3/train.py'],text=True)
    tree=ast.parse(old)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_balanced_work')
    namespace=dict(vars(train))
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<historical selector>','exec'),namespace)
    cfg=args(tmp_path);cfg.focus_balance_count=16
    history=None;previous=None
    for epoch in range(3):
        expected=namespace['_balanced_work'](rows(),target_per_bin=16,seed=cfg.seed+epoch*1000003,
                                            route_diverse=True,action_output_mode='choice')
        actual,audit=baseline.independent_work(rows(),cfg,epoch=epoch,world_size=4,history=history)
        assert actual==expected
        assert audit['event_counts']==dict.fromkeys(train.CONTEXT_IDS,16)
        assert not audit['contiguous_priority'] and not audit['least_seen_priority']
        if previous is not None:assert actual!=previous
        previous=actual;history=audit['next_history']
    with pytest.raises(ValueError,match='sequence exposure'):
        baseline.independent_work(rows(),cfg,epoch=1,world_size=4,history={'next_epoch':1,'counts':{}})


def test_base_profile_and_strict_budget(tmp_path):
    cfg=args(tmp_path)
    assert cfg.batch_size==cfg.grad_accum==1 and cfg.num_epochs==3
    assert cfg.focus_balance_count*10//4*3==7680 and cfg.warmup_steps==2000
    assert cfg.train_route_diverse and not cfg.reuse_frozen_rgb
    cfg.batch_size=2
    with pytest.raises(ValueError,match='B1'):baseline.independent_work(rows(),cfg,epoch=0,world_size=4)


def test_future_targets_never_backfill_or_cross_context_split():
    source=rows();pool=data.Pool(source);anchor=source[0]
    rec=pool.future(anchor)
    assert rec['frames']==[4,8,12] and rec['images']==anchor.history_rgb_paths
    assert len(rec['actions'])==3
    missing=data.Pool([x for x in source if x.frame_id!=8]).future(anchor)
    assert missing['actions'][1] is None
    assert 'answers' not in rec and 'current_speed' not in rec
    with pytest.raises(ValueError,match='crosses'):
        data.Pool(source+[replace(anchor,split='test')])
    with pytest.raises(ValueError,match='Duplicate'):data.Pool(source+[anchor])


def test_exact_eight_image_windows_and_no_sparse_fill():
    source=rows();pool=data.Pool(source);anchor=source[4]
    rec=pool.windows(anchor)
    assert len(rec['images'])==8 and rec['frames']==[4,5,6,7,8]
    assert data.Pool([x for x in source if x.frame_id!=6]).windows(anchor) is None
    corrupt=[replace(x,history_rgb_paths=['bad']*4) if x is source[2] else x for x in source]
    with pytest.raises(ValueError,match='Noncontiguous'):data.Pool(corrupt).windows(anchor)


def tiny_packed(inputs,positions,targets):
    return dict(inputs=inputs,positions=positions,targets=targets,choices=list(range(20,26)),
                tokens=inputs['input_ids'].shape[1],images=len(inputs['image_grid_thw']))


def test_real_qwen_causal_five_readouts_no_future_rgb_influence(model):
    model.model.visual.requires_grad_(False)
    inputs=row(8)
    # Query positions are text tokens after each image, not target action tokens.
    ids=[];positions=[]
    for i in range(8):
        ids.extend([2,4,4,4,4,3])
        if i>=3:
            ids.append(30+i);positions.append(len(ids)-1)
    inputs['input_ids']=torch.tensor([ids]);inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
    inputs['mm_token_type_ids']=(inputs['input_ids']==4).long()
    p=tiny_packed(inputs,positions,[20,21,22,23,24])
    loss,stats=objectives.readout_loss(model,p,'cpu')
    assert stats['supervised']==5 and loss.isfinite()
    expected=model(**inputs,logits_to_keep=torch.tensor(positions),use_cache=False).logits.detach()
    changed={k:v.clone() for k,v in inputs.items()}
    changed['pixel_values'][4*16:]*=7
    actual=model(**changed,logits_to_keep=torch.tensor(positions),use_cache=False).logits
    torch.testing.assert_close(actual[:,0],expected[:,0],atol=1e-5,rtol=1e-4)
    assert not torch.allclose(actual[:,-1],expected[:,-1])
    loss.backward()
    assert model.model.language_model.layers[0].linear_attn.in_proj_qkv.weight.grad.isfinite().all()
    altered=dict(p,targets=[25]*5)
    assert torch.equal(altered['inputs']['input_ids'],p['inputs']['input_ids'])


def test_real_qwen_strict_windows_shared_vision_gradient_matches_independent(model):
    model.model.visual.requires_grad_(False)
    pixels=torch.randn(8*16,1536)
    packed=[]
    for i in range(5):
        inputs=row(4,pixels=pixels[i*16:(i+4)*16])
        packed.append(tiny_packed(inputs,[inputs['input_ids'].shape[1]-1],[20+i]))
    expected=torch.stack([objectives.readout_loss(model,p,'cpu')[0] for p in packed]).mean()
    expected.backward();grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    actual,report=objectives.window_loss(SimpleNamespace(model=model,tokenizer=tokenizer(),device='cpu'),packed)
    actual.backward()
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    for n,p in model.named_parameters():
        if n in grads:torch.testing.assert_close(p.grad,grads[n],atol=1e-5,rtol=1e-3)
    assert report['visual_reuse']['encoded_images']==8
    assert report['visual_reuse']['reused_images']==12


def test_future_missing_loss_mask_and_target_not_input(model):
    inputs=row(4);p=tiny_packed(inputs,[1,8,inputs['input_ids'].shape[1]-1],[20,-100,22])
    loss,report=objectives.readout_loss(model,p,'cpu')
    assert report['supervised']==2
    logits=model(**inputs,logits_to_keep=torch.tensor(p['positions']),use_cache=False).logits[0].float()
    expected=torch.nn.functional.cross_entropy(logits[[0,2]],torch.tensor([20,22]))
    torch.testing.assert_close(loss,expected)


def test_messages_are_independent_of_labels():
    rec=dict(mode='stream',actions=['STOP']*5)
    msg=objectives.messages(rec,list(range(8)))
    assert msg==objectives.messages(dict(rec,actions=['KEEP']*5),list(range(8)))
    content=msg[1]['content'];seen=0;counts=[]
    for item in content:
        if item['type']=='image':seen+=1
        else:counts.append(seen)
    assert counts==[4,5,6,7,8]


@pytest.mark.parametrize('mode',['future','stream','windows'])
def test_real_processor_pack_and_small_qwen_backward(processor,model,mode):
    from PIL import Image
    for key,token in [('image_token_id','<|image_pad|>'),('video_token_id','<|video_pad|>'),
                      ('vision_start_token_id','<|vision_start|>'),('vision_end_token_id','<|vision_end|>')]:
        setattr(model.config,key,processor.tokenizer.convert_tokens_to_ids(token))
    model.model.visual.requires_grad_(False)
    bundle=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    rec=dict(mode=mode,actions=['STOP']*(3 if mode=='future' else 5),offsets_seconds=[0,1,2])
    images=[Image.new('RGB',(64,64),(i*20,40,60)) for i in range(4 if mode=='future' else 8)]
    packed=objectives.pack(bundle,rec,images)
    loss,stats=(objectives.window_loss(bundle,packed) if mode=='windows'
                else objectives.readout_loss(model,packed,'cpu'))
    loss.backward()
    assert stats['supervised']==(3 if mode=='future' else 5)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    model.eval()
    predictions=objectives.predict(bundle,rec,images)
    changed=dict(rec,actions=['KEEP']*len(rec['actions']))
    assert predictions==objectives.predict(bundle,changed,images)
    assert len(predictions)==len(rec['actions'])
    with pytest.raises(ValueError,match='token budget'):
        objectives.pack(bundle,rec,images,max_length=10)


def test_image_resolution_cap_is_enforced(processor):
    from PIL import Image
    bundle=SimpleNamespace(processor=processor,tokenizer=processor.tokenizer)
    rec=dict(mode='future',actions=['STOP'],offsets_seconds=[0])
    p=objectives.pack(bundle,rec,[Image.new('RGB',(1152,384))]*4,max_pixels=262144)
    assert all(int(x.prod())<=1024 for x in p['inputs']['image_grid_thw'])


def test_base_native_optimizer_resume_equals_uninterrupted(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import preflight
    from qwen3vl_local.sft_new_loop_phase3.test_entry_admission import fixture
    index=fixture(tmp_path,'valid');base=tmp_path/'base';base.mkdir();(base/'config.json').write_text('{}')
    for name in ('_contiguous_work','_loss_batch','SEQUENCE_POLICY'):
        monkeypatch.setattr(train,name,getattr(train,name))
    baseline.install_baseline()
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    def load(*a,**kw):
        m=torch.nn.Linear(1,1)
        return SimpleNamespace(model=m,unwrap=lambda:m,lora_target_modules=['weight'])
    monkeypatch.setattr(train,'load_model_with_lora',load)
    monkeypatch.setattr(train,'_load_images',lambda *a,**kw:[])
    monkeypatch.setattr(train,'_build_inputs',lambda *a,**kw:{})
    monkeypatch.setattr(train,'_loss_batch',lambda b,p,s:(b.model(torch.ones(1,1)).square().mean(),[{}]))
    def run(name,resume=None,stop=1):
        j=job(tmp_path);j.update(output=str(tmp_path/name),index=str(index),model_dir=str(base),resume=str(resume) if resume else None)
        cfg=worker.make_args(j);cfg.device='cpu';cfg.focus_balance_count=4;cfg.warmup_steps=20
        cfg.prompt_variant='baseline';cfg.eval_steps=cfg.generation_eval_steps=0
        cfg.save_final=False;cfg.stop_after_epoch=stop
        train.train(cfg)
        return Path(cfg.output_dir)
    first=run('first');checkpoint=first/'training_states/epoch_0001'
    resumed=run('resumed',checkpoint,stop=None);full=run('full',stop=None)
    def state(root):return torch.load(root/'training_states/epoch_0003/training_state.pt',weights_only=True)
    a,b=state(resumed),state(full)
    assert a['global_step']==b['global_step']==120
    for key in a['trainable']:torch.testing.assert_close(a['trainable'][key],b['trainable'][key],rtol=0,atol=0)
    assert a['sampling_history']==dict(policy=baseline.POLICY,next_epoch=3)


def test_single_loss_and_gradient_match_historical_function(processor,model):
    old=subprocess.check_output(['git','show','71cc49636eb367c1e1c06743953f48e61fd17f4c:AutoMoT/qwen3vl_local/sft_new_loop_phase3/train.py'],text=True)
    function=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name=='_loss_one')
    namespace=dict(vars(train));exec(compile(ast.Module(body=[function],type_ignores=[]),'<historical loss>','exec'),namespace)
    for key,token in [('image_token_id','<|image_pad|>'),('video_token_id','<|video_pad|>'),
                      ('vision_start_token_id','<|vision_start|>'),('vision_end_token_id','<|vision_end|>')]:
        setattr(model.config,key,processor.tokenizer.convert_tokens_to_ids(token))
    from PIL import Image
    bundle=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    spec=train._make_item(rows()[0],seed=7,action_output_mode='choice').spec
    packed=train._build_inputs(bundle,images=[Image.new('RGB',(32,32))]*4,history_rgb_mode='4rgb',
        target=train.build_action_target(spec),spec=spec,max_length=8192,format_loss_weight=.25)
    expected,stats=namespace['_loss_one'](bundle,packed,spec)
    expected.backward();grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    actual,new_stats=baseline.single_loss(bundle,[packed],[spec]);actual.backward()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert new_stats==[stats]
    for n,p in model.named_parameters():
        if n in grads:torch.testing.assert_close(p.grad,grads[n],rtol=0,atol=0)


def test_dense_real_processor_enters_native_training_and_checkpoint(tmp_path,monkeypatch,processor):
    from qwen3vl_local.audit_joint.phase3_base_dense import runtime
    from qwen3vl_local.sft_new_loop_phase3 import preflight
    from qwen3vl_local.sft_new_loop_phase3 import sequence_evaluation
    monkeypatch.setattr(sequence_evaluation,'select_work',sequence_evaluation.select_work)
    from qwen3vl_local.sft_new_loop_phase3.test_entry_admission import fixture
    index=fixture(tmp_path,'valid');base=tmp_path/'base';base.mkdir();(base/'config.json').write_text('{}')
    for name in ('_make_item','_contiguous_work','_loss_batch','SEQUENCE_POLICY','_build_inputs',
                 'load_model_with_lora','_save_adapter','_write_run_metadata'):
        monkeypatch.setattr(train,name,getattr(train,name))
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.values=torch.nn.Parameter(torch.zeros(320))
        def forward(self,input_ids,logits_to_keep,**kw):
            return SimpleNamespace(logits=self.values[None,None].expand(1,len(logits_to_keep),-1))
    def load(*a,**kw):
        m=Tiny();return SimpleNamespace(model=m,unwrap=lambda:m,lora_target_modules=['values'],
            processor=processor,tokenizer=processor.tokenizer,device='cpu')
    monkeypatch.setattr(train,'load_model_with_lora',load)
    monkeypatch.setattr(runtime,'initialize',lambda b,j:b)
    from PIL import Image
    monkeypatch.setattr(train,'_load_images',lambda paths,**kw:[Image.new('RGB',(32,32)) for _ in paths])
    j=job(tmp_path);j.update(index=str(index),model_dir=str(base),mode='future',init_adapter='fixture',init_adapter_files={})
    pool=runtime.load_pool(j);runtime.install(j,pool)
    cfg=runtime.make_args(j);cfg.device='cpu';cfg.focus_balance_count=1;cfg.warmup_steps=2
    cfg.prompt_variant='baseline';cfg.eval_steps=cfg.generation_eval_steps=0;cfg.save_final=False
    train.train(cfg)
    state=torch.load(Path(cfg.output_dir)/'training_states/epoch_0001/training_state.pt',weights_only=True)
    assert state['global_step']==10 and state['trainable']['values'].abs().sum()>0
    manifest=json.loads((Path(cfg.output_dir)/'train_run_manifest.json').read_text())
    assert manifest['dense_contract']['ground_truth_in_input'] is False
