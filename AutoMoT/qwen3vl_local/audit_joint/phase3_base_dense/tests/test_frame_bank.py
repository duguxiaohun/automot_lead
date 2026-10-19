"""Independent protocol reference, strict visibility and shared image-language KV."""
from types import SimpleNamespace
import copy
import pytest
import torch
from PIL import Image
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.tests.test_batch_runtime import tokenizer
from qwen3vl_local.audit_joint.phase3_base_dense import objectives,frame_bank,shapes,worker
from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_query_origins_probe import windows


def bank(packed):return [dict(p,frame_bank=True) for p in packed]


@pytest.mark.parametrize('vary',[False,True])
def test_shared_eight_language_encodings_match_twenty_recomputed_reference(model,vary):
    model.model.visual.requires_grad_(False);model.train()
    bundle=SimpleNamespace(model=model,tokenizer=tokenizer(),device='cpu')
    packed=bank(windows(vary_positions=vary))
    # Varying prefixes in this tiny fixture must be kept identical, as in real prompts.
    if vary:
        packed=bank(windows())
        for i,p in enumerate(packed):
            for key in ('input_ids','attention_mask','mm_token_type_ids'):
                pad=torch.full((1,i),12 if key=='input_ids' else 1 if key=='attention_mask' else 0,dtype=torch.long)
                p['inputs'][key]=torch.cat((p['inputs'][key],pad),dim=1)
            p['tokens']+=i
    calls=[]
    hook=model.register_forward_pre_hook(lambda module,args,kw:calls.append(kw['input_ids'].shape[0]),with_kwargs=True)
    reference,ref=frame_bank.forward(bundle,packed,share_frames=False)
    assert calls==[1,20,5];calls.clear()
    reference.square().mean().backward()
    grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    actual,stats=frame_bank.forward(bundle,packed)
    hook.remove();assert calls==[1,8,5]
    actual.square().mean().backward()
    torch.testing.assert_close(actual,reference,atol=3e-5,rtol=3e-4)
    for n,p in model.named_parameters():
        if n in grads:torch.testing.assert_close(p.grad,grads[n],atol=3e-5,rtol=3e-3)
    assert stats['frame_language_encodings']==8 and ref['frame_language_encodings']==20
    assert stats['query_recurrence']=='text_prefix_only'
    assert stats['window_source_slots']==[list(range(i,i+4)) for i in range(5)]


@pytest.mark.parametrize('edge',[0,7])
def test_outside_window_has_zero_effect_and_inside_image_changes_prediction(model,edge):
    model.model.visual.requires_grad_(False);model.eval()
    bundle=SimpleNamespace(model=model,tokenizer=tokenizer(),device='cpu')
    packed=bank(windows())
    with torch.no_grad():
        expected,_=frame_bank.forward(bundle,packed)
        changed=copy.deepcopy(packed)
        window=0 if edge==0 else 4
        changed[window]['inputs']['pixel_values']=changed[window]['inputs']['pixel_values'].clone()
        indices=slice(0,16) if edge==0 else slice(-16,None)
        changed[window]['inputs']['pixel_values'][indices]*=3
        actual,_=frame_bank.forward(bundle,changed)
        repeated,_=frame_bank.forward(bundle,packed)
    others=[i for i in range(5) if i!=window]
    torch.testing.assert_close(actual[others],expected[others],atol=1e-6,rtol=1e-5)
    assert not torch.allclose(actual[window],expected[window])
    torch.testing.assert_close(repeated,expected)


def test_bank_shape_matches_real_processor_and_actual_cache_geometry(processor,model):
    for key,token in [('image_token_id','<|image_pad|>'),('video_token_id','<|video_pad|>'),
                      ('vision_start_token_id','<|vision_start|>'),('vision_end_token_id','<|vision_end|>')]:
        setattr(model.config,key,processor.tokenizer.convert_tokens_to_ids(token))
    model.model.visual.requires_grad_(False)
    bundle=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    rec=dict(mode='windows_bank',actions=['STOP']*5,offsets_seconds=[0]*5,premises=['x'*i for i in range(5)])
    images=[Image.new('RGB',(32,32),(i*20,10,30)) for i in range(8)]
    measured=shapes.measure(processor,rec,[i.size for i in images])
    # Byte-tokenizer fixture has a long text prefix; use its measured budget for
    # this model-level check, not the public CLI's fixed production upper cap.
    packed=objectives.pack(bundle,rec,images,max_batch_tokens=measured['padded_batch_tokens'])
    assert packed[0]['bank_shape']['live_cache_token_slots']==measured['padded_batch_tokens']
    with torch.no_grad():actual,stats=objectives.window_logits(bundle,packed)
    for key in ('model_logit_rows','language_tokens_computed','live_cache_token_slots','shared_prefix_tokens'):
        assert stats[key]==measured[key]
    assert actual.shape==(5,model.config.text_config.vocab_size)
    predictions=objectives.predict(bundle,rec,images,max_batch_tokens=measured['padded_batch_tokens'])
    assert predictions==objectives.predict(bundle,dict(rec,actions=['KEEP']*5),images,
        max_batch_tokens=measured['padded_batch_tokens'])
    with pytest.raises(ValueError,match='cache-token'):
        objectives.pack(bundle,rec,images,max_batch_tokens=measured['padded_batch_tokens']-1)


def test_bank_probe_encodes_eight_source_slots_and_never_trains_holdout(model):
    model.model.visual.requires_grad_(False)
    bundle=SimpleNamespace(model=model,tokenizer=tokenizer(),device='cpu')
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    def forbidden(*a,**kw):pytest.fail('Holdout update')
    reports=worker.probe_record(bundle,dict(usage='evaluation'),bank(windows(identical=True)),
        SimpleNamespace(step=forbidden,zero_grad=forbidden))
    assert [r['visual_reuse']['encoded_images'] for r in reports]==[1,8]
    assert all(r['visual_reuse']['frame_language_encodings']==8 for r in reports)
    assert all(p.grad is None and torch.equal(p,before[n]) for n,p in model.named_parameters())
