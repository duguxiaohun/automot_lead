from types import SimpleNamespace
import pytest
import torch
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.batch_runtime import left_collate, frozen_rgb_reuse, greedy_batch


def row(frames, extra=0, pixels=None):
    ids=torch.tensor([[10]*(1+extra)+[x for _ in range(frames) for x in (2,4,4,4,4,3)]+[11]])
    return dict(input_ids=ids,attention_mask=torch.ones_like(ids),
                pixel_values=torch.randn(frames*16,1536) if pixels is None else pixels,
                image_grid_thw=torch.tensor([[1,4,4]]*frames),
                mm_token_type_ids=(ids==4).long())


def tokenizer():
    return SimpleNamespace(pad_token_id=0,eos_token_id=1,convert_tokens_to_ids=lambda _:1,
        decode=lambda t,**kw:','.join(map(str,[v for v in t.tolist() if v not in (0,1)])))


@pytest.mark.parametrize('frames',[2,4])
def test_real_model_reuse_logits_and_language_gradients(model,frames):
    model.model.visual.requires_grad_(False)
    pixel=torch.randn(16,1536)
    inputs=left_collate([row(frames,0,pixel.repeat(frames,1)),row(frames,3,pixel.repeat(frames,1))],0)
    baseline=model(**inputs,use_cache=False).logits
    baseline.sum().backward()
    grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    with frozen_rgb_reuse(model) as stats:
        actual=model(**inputs,use_cache=False).logits
    actual.sum().backward()
    torch.testing.assert_close(actual,baseline,atol=2e-5,rtol=2e-4)
    for n,p in model.named_parameters():
        if n in grads:torch.testing.assert_close(p.grad,grads[n],atol=1e-4,rtol=1e-3)
    assert stats['encoded_images']==1 and stats['reused_images']==2*frames-1
    assert 'forward' not in model.model.visual.__dict__


@pytest.mark.parametrize('frames',[2,4])
@pytest.mark.parametrize('phase3',[False,True])
def test_real_model_batch_generation_matches_single(model,frames,phase3):
    model.model.visual.requires_grad_(False)
    shared=torch.randn(16,1536)
    inputs=[row(frames,0,shared.repeat(frames,1)),row(frames,3,shared.repeat(frames,1))]
    bundle=SimpleNamespace(model=model,tokenizer=tokenizer())
    singles=[greedy_batch(bundle,[r],max_new_tokens=5,phase3=phase3)[0] for r in inputs]
    batched=greedy_batch(bundle,inputs,max_new_tokens=5,phase3=phase3,reuse_rgb=True)
    assert batched==singles


def test_fallback_and_exception_restore(model):
    inputs=row(2)
    original=model.model.visual.forward
    with frozen_rgb_reuse(model) as stats:
        model(**inputs,use_cache=False)
    assert stats['reason']=='trainable_vision'
    model.model.visual.requires_grad_(False)
    with pytest.raises(RuntimeError),frozen_rgb_reuse(model):
        raise RuntimeError('forward failed')
    assert model.model.visual.forward==original


def test_left_padding_keeps_vision_order():
    inputs=[row(2),row(4,2)]
    result=left_collate(inputs,0)
    for i,r in enumerate(inputs):
        n=r['input_ids'].shape[1]
        for key in ('input_ids','attention_mask','mm_token_type_ids'):
            assert torch.equal(result[key][i,-n:],r[key][0])
            assert not result[key][i,:-n].any()
    assert torch.equal(result['pixel_values'],torch.cat([r['pixel_values'] for r in inputs]))


@pytest.mark.parametrize('frames',[2,4])
def test_phase3_batched_generation_matches_original_kv_loop(model,frames):
    from qwen3vl_local.sft_v3.train import KVState, _student_generate_kv
    tok=tokenizer()
    bundle=SimpleNamespace(model=model,tokenizer=tok,processor=SimpleNamespace(
        batch_decode=lambda ids,**kw:[tok.decode(r) for r in ids]))
    inputs=[row(frames),row(frames,3)]
    expected=[]
    with torch.no_grad():
        for r in inputs:
            out=model(**r,use_cache=True,return_dict=True)
            state=KVState(decoded_input_ids=r['input_ids'],cache_input_ids=r['input_ids'],
                          attention_mask=r['attention_mask'],past_key_values=out.past_key_values,
                          rope_deltas=out.rope_deltas,next_logits=out.logits[:,-1,:])
            expected.append(_student_generate_kv(bundle,state,5)[0])
    assert greedy_batch(bundle,inputs,max_new_tokens=5,phase3=True)==expected


def test_evaluation_batches_tail_and_retains_case_order(monkeypatch,tmp_path):
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import evaluate
    from qwen3vl_local.sft_new_loop_phase4.tests.test_short_rgb import source_row
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.data import short_row
    original=short_row(source_row(tmp_path))
    rows=[dict(original,id=f'case{i}') for i in range(3)]
    calls=[]
    def generate(bundle, selected, root, length, **kw):
        calls.append([r['id'] for r in selected])
        return [(r['target'],r['target']) for r in selected]
    monkeypatch.setattr(evaluate,'generate_batch',generate)
    report,cases=evaluate.evaluate_rows(None,rows,tmp_path,batch_size=2,reuse_rgb=True)
    assert [len(c) for c in calls]==[2,1]
    assert [c['id'] for c in cases]==[r['id'] for r in rows]
    assert all(c['prediction']==r['target'] for c,r in zip(cases,rows))
    assert report['generation_batch_size']==2


def test_distinct_images_not_reused_and_input_mutation_not_cached(model):
    model.model.visual.requires_grad_(False)
    inputs=row(2)
    with frozen_rgb_reuse(model) as stats:
        model(**inputs,use_cache=False)
        inputs['pixel_values'][16:]=inputs['pixel_values'][:16]
        model(**inputs,use_cache=False)
    assert stats['images']==4 and stats['encoded_images']==3


def test_gradient_checkpointing_train_path(model):
    model.model.visual.requires_grad_(False)
    model.train();model.gradient_checkpointing_enable();model.enable_input_require_grads()
    pixel=torch.randn(16,1536)
    inputs=left_collate([row(2,0,pixel.repeat(2,1)),row(2,3,pixel.repeat(2,1))],0)
    loss=model(**inputs,use_cache=False).logits.square().mean();loss.backward()
    expected={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    with frozen_rgb_reuse(model) as stats:
        actual=model(**inputs,use_cache=False).logits.square().mean()
    actual.backward()
    assert stats['encoded_images']==1
    torch.testing.assert_close(actual,loss)
    for n,p in model.named_parameters():
        if n in expected:torch.testing.assert_close(p.grad,expected[n],atol=1e-6,rtol=1e-3)
