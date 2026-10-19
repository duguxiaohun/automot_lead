"""Strict visibility and differentiable text-prefix cache sharing."""
from types import SimpleNamespace
import copy
import pytest
import torch
from PIL import Image
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.tests.test_batch_runtime import tokenizer
from qwen3vl_local.audit_joint.phase3_base_dense import objectives,shapes,worker,prefix_cache
from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_query_origins_probe import windows


def sharing(packed,length=1):
    return [dict(p,shared_prefix_tokens=length) for p in packed]


@pytest.mark.parametrize('vary',[False,True])
def test_prefix_logits_and_all_parameter_gradients_match_full_windows(model,vary):
    model.model.visual.requires_grad_(False);model.train()
    bundle=SimpleNamespace(model=model,tokenizer=tokenizer(),device='cpu')
    packed=windows(vary_positions=vary)
    reference,_=objectives.window_loss(bundle,packed)
    reference.backward();grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    actual,stats=objectives.window_loss(bundle,sharing(packed))
    actual.backward()
    torch.testing.assert_close(actual,reference,atol=2e-6,rtol=2e-5)
    for n,p in model.named_parameters():
        if n in grads:torch.testing.assert_close(p.grad,grads[n],atol=2e-5,rtol=2e-3)
    assert stats['visual_reuse']['shared_prefix_tokens']==1
    assert not stats['visual_reuse']['image_language_states_shared']
    assert stats['visual_reuse']['language_tokens_without_sharing']-stats['visual_reuse']['language_tokens_computed']==4


def test_window_outside_images_cannot_affect_readout_and_repeated_forward_is_fresh(model):
    model.model.visual.requires_grad_(False);model.eval()
    bundle=SimpleNamespace(model=model,tokenizer=tokenizer(),device='cpu')
    packed=sharing(windows())
    with torch.no_grad():
        expected,_=objectives.window_logits(bundle,packed)
        changed=copy.deepcopy(packed)
        # Frame0 appears only in window0; window4 contains frames4..7.
        changed[0]['inputs']['pixel_values'][:16]*=4
        actual,_=objectives.window_logits(bundle,changed)
        again,_=objectives.window_logits(bundle,packed)
    torch.testing.assert_close(actual[1:],expected[1:],atol=1e-6,rtol=1e-5)
    assert not torch.allclose(actual[0],expected[0])
    torch.testing.assert_close(again,expected)


def test_real_processor_prefix_boundary_and_shape(processor,model):
    for key,token in [('image_token_id','<|image_pad|>'),('video_token_id','<|video_pad|>'),
                      ('vision_start_token_id','<|vision_start|>'),('vision_end_token_id','<|vision_end|>')]:
        setattr(model.config,key,processor.tokenizer.convert_tokens_to_ids(token))
    model.model.visual.requires_grad_(False)
    bundle=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    rec=dict(mode='windows_prefix',actions=['STOP']*5,offsets_seconds=[0]*5,premises=['x'*i for i in range(5)])
    images=[Image.new('RGB',(32,32),(i*20,10,30)) for i in range(8)]
    packed=objectives.pack(bundle,rec,images,max_batch_tokens=32768)
    measured=shapes.measure(processor,rec,[i.size for i in images])
    assert all(p['shared_prefix_tokens']==measured['shared_prefix_tokens'] for p in packed)
    assert measured['model_logit_rows']==26 and measured['vision_encoding_upper_images']==8
    # Full processor prompt is tested through the real hybrid language model.
    with torch.no_grad():
        shared,_=objectives.window_logits(bundle,packed)
        full,_=objectives.window_logits(bundle,[{k:v for k,v in p.items() if k!='shared_prefix_tokens'} for p in packed])
    torch.testing.assert_close(shared,full,atol=2e-5,rtol=2e-4)


def test_eight_source_probe_does_not_collapse_equal_pixels_and_has_no_holdout_updates(model):
    model.model.visual.requires_grad_(False)
    bundle=SimpleNamespace(model=model,tokenizer=tokenizer(),device='cpu')
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    def forbidden(*a,**kw):pytest.fail('Holdout update')
    report=worker.probe_record(bundle,dict(usage='evaluation'),sharing(windows(identical=True)),
                               SimpleNamespace(step=forbidden,zero_grad=forbidden))
    assert [r['visual_reuse']['encoded_images'] for r in report]==[1,8]
    assert all(p.grad is None and torch.equal(p,before[n]) for n,p in model.named_parameters())


def test_prefix_rejects_image_state_stochastic_training_and_checkpoint_replay(model):
    model.model.visual.requires_grad_(False);model.train()
    bundle=SimpleNamespace(model=model,tokenizer=tokenizer(),device='cpu')
    with pytest.raises(ValueError,match='image-free|Image-bearing'):
        objectives.window_logits(bundle,sharing(windows(),7))
    model.gradient_checkpointing_enable()
    with pytest.raises(ValueError,match='checkpointing disabled'):
        objectives.window_logits(bundle,sharing(windows()))
    model.gradient_checkpointing_disable()
    model.extra_test_dropout=torch.nn.Dropout(.1)
    with pytest.raises(ValueError,match='deterministic language'):
        objectives.window_logits(bundle,sharing(windows()))


def test_eight_source_probe_matches_default_logits_gradients_and_rejects_wrong_overlap(model):
    model.model.visual.requires_grad_(False);model.train()
    bundle=SimpleNamespace(model=model,tokenizer=tokenizer(),device='cpu')
    packed=sharing(windows(identical=True,vary_positions=True))
    a,_=objectives.window_loss(bundle,packed);a.backward()
    grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    observed=[]
    original=model.model.visual.forward
    def actual_tower(pixels,grid_thw,**kwargs):
        observed.append(len(grid_thw));return original(pixels,grid_thw,**kwargs)
    model.model.visual.forward=actual_tower
    try:b,stats=objectives.window_loss(bundle,packed,source_slots=True)
    finally:del model.model.visual.forward
    b.backward()
    assert observed==[8] and stats['visual_reuse']['encoded_images']==8
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    for n,p in model.named_parameters():
        if n in grads:torch.testing.assert_close(p.grad,grads[n],atol=2e-5,rtol=2e-3)
    broken=copy.deepcopy(packed);broken[1]['inputs']['pixel_values']=broken[1]['inputs']['pixel_values'].clone();broken[1]['inputs']['pixel_values'][:16]+=1
    with pytest.raises(ValueError,match='Overlapping source frame differs'):
        objectives.window_logits(bundle,broken,source_slots=True)


def _ddp_prefix(rank,root,protocol):
    import torch.distributed as dist
    from pathlib import Path
    from peft import LoraConfig,get_peft_model
    from qwen3vl_local.qwen35.tests.test_backend import model as make_model
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(Path(root)/'init'),rank=rank,world_size=4)
    try:
        base=make_model.__wrapped__();base.model.visual.requires_grad_(False)
        network=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,
            target_modules=['q_proj','in_proj_qkv'],task_type='CAUSAL_LM'))
        for n,p in network.named_parameters():
            if 'lora_B' in n:torch.nn.init.normal_(p,std=.01)
        network.train();reference=copy.deepcopy(network)
        wrapped=torch.nn.parallel.DistributedDataParallel(network,find_unused_parameters=False)
        actual_bundle=SimpleNamespace(model=wrapped,tokenizer=tokenizer(),device='cpu')
        ref_bundle=SimpleNamespace(model=reference,tokenizer=tokenizer(),device='cpu')
        opt=torch.optim.SGD(network.parameters(),lr=.001);ref_opt=torch.optim.SGD(reference.parameters(),lr=.001)
        for step in range(2):
            packed=windows(vary_positions=protocol!='bank')
            for item in packed:item['inputs']['pixel_values']+=rank*.01
            if protocol=='bank':
                from qwen3vl_local.audit_joint.phase3_base_dense import frame_bank
                ref_logits,_=frame_bank.forward(ref_bundle,packed,share_frames=False)
                ref_loss=torch.nn.functional.cross_entropy(ref_logits,torch.arange(20,25))
            else:ref_loss,_=objectives.window_loss(ref_bundle,packed)
            ref_loss.backward()
            expected={}
            for n,p in reference.named_parameters():
                if p.requires_grad:
                    dist.all_reduce(p.grad);p.grad/=4;expected[n]=p.grad.clone()
            if protocol=='bank':
                logits,_=frame_bank.forward(actual_bundle,packed)
                loss=torch.nn.functional.cross_entropy(logits,torch.arange(20,25))
            else:loss,_=objectives.window_loss(actual_bundle,sharing(packed))
            loss.backward()
            for n,p in network.named_parameters():
                if p.requires_grad:torch.testing.assert_close(p.grad,expected[n],atol=3e-5,rtol=3e-3)
            opt.step();opt.zero_grad(set_to_none=True);ref_opt.step();ref_opt.zero_grad(set_to_none=True)
            for (n,p),(rn,rp) in zip(network.named_parameters(),reference.named_parameters()):
                assert n==rn;torch.testing.assert_close(p,rp,atol=3e-6,rtol=3e-4)
        (Path(root)/f'rank{rank}.ok').write_text('two optimizer updates; shared-prefix LoRA gradients match full-window mean')
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('protocol',['prefix','bank'])
def test_four_rank_lora_prefix_backward_and_next_update(tmp_path,protocol):
    torch.multiprocessing.start_processes(_ddp_prefix,args=(str(tmp_path),protocol),nprocs=4,join=True,start_method='spawn')
    assert len(list(tmp_path.glob('rank*.ok')))==4
