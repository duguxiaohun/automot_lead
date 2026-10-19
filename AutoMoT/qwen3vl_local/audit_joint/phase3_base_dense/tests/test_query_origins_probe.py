"""Temporal-origin and memory-probe counterexamples from the third review."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from PIL import Image
from qwen3vl_local.audit_joint.phase3_base_dense import objectives, shapes, worker
from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_base_dense import tiny_packed
from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.tests.test_batch_runtime import row, tokenizer
from qwen3vl_local.sft_new_loop_phase3 import trajectory_action as rule


def windows(*,identical=False,vary_positions=False):
    pixels=torch.randn(16,1536).repeat(8,1) if identical else torch.randn(8*16,1536)
    result=[]
    for i in range(5):
        inputs=row(4,extra=i if vary_positions else 4,pixels=pixels[i*16:(i+4)*16])
        result.append(tiny_packed(inputs,[inputs['input_ids'].shape[1]-1],[20+i]))
    return result


def test_future_origins_reset_native_outward_return_and_speed_rules():
    # At 1s cross left; at 4s return right. +3s must be allowed to predict right.
    metas={i:dict(lane_id=-1 if 4<=i<16 else -2,road_id=1,section_id=0,
                  lane_type_str='Driving',speed=8.+i*.1) for i in range(40)}
    trajectory=rule.RouteTrajectory(Path('/synthetic/scenario/route'),tuple(metas),metas,{1:-2})
    assert trajectory.lane_change(0)==rule.DIRECTION_LEFT
    assert trajectory.lane_change(12)==rule.DIRECTION_RIGHT
    assert trajectory.signals(0)['speed']!=trajectory.signals(12)['speed']
    record=dict(mode='future',actions=['LANE_CHANGE_LEFT','LANE_CHANGE_RIGHT','KEEP'],offsets_seconds=[0,3,6])
    messages=objectives.messages(record,list(range(4)))
    text=str(messages)
    assert 'after the newest frame' not in text and 'relative to current speed' not in text
    assert 'speed at T' in text and 'outward crossing was before T' in text
    for offset in (0,3,6):assert f'Prediction origin T: +{offset} seconds' in text
    assert 'immediate window is 1.5s' in text and 'speed window 2s' in text and 'lane-crossing window 3s' in text
    # Targets, including future speed, never enter this prompt.
    assert messages==objectives.messages(dict(record,actions=['STOP']*3),list(range(4)))
    assert str(metas[12]['speed']) not in text


def test_equal_old_cost_selects_larger_logit_union(tmp_path,processor):
    image=tmp_path/'rgb.png';Image.new('RGB',(32,32)).save(image)
    base=dict(mode='windows',images=[str(image)]*8,actions=['STOP']*5,offsets_seconds=[0]*5)
    records={'a_aligned':dict(base,premises=['x'*100]*5),
             'b_shifted':dict(base,premises=['x'*n for n in range(96,101)])}
    audit=shapes.scan(processor,records,32768)
    a,b=(audit['shapes'][key] for key in records)
    for key in ('max_sequence_tokens','padded_batch_tokens','vision_patch_presentations'):
        assert a[key]==b[key]
    assert (a['model_logit_rows'],b['model_logit_rows'])==(5,25)
    assert audit['probe_ids']==['b_shifted']
    bundle=SimpleNamespace(processor=processor,tokenizer=processor.tokenizer)
    for key,record in records.items():
        packed=objectives.pack(bundle,record,[Image.new('RGB',(32,32))]*8,max_batch_tokens=32768)
        assert [p['positions'] for p in packed]==audit['shapes'][key]['query_positions']


def test_real_model_equal_token_cost_has_different_logit_output(model):
    model.model.visual.requires_grad_(False)
    bundle=SimpleNamespace(model=model,tokenizer=tokenizer(),device='cpu')
    outputs=[]
    handle=model.register_forward_hook(lambda module,args,out:outputs.append(tuple(out.logits.shape)))
    try:
        for varying in (False,True):
            objectives.window_logits(bundle,windows(vary_positions=varying))
    finally:handle.remove()
    assert outputs==[(5,1,320),(5,5,320)]


@pytest.mark.parametrize('identical,encoded',[(True,1),(False,8)])
def test_upper_workload_encodes_twenty_and_preserves_logits_gradients(model,identical,encoded):
    model.model.visual.requires_grad_(False)
    bundle=SimpleNamespace(model=model,tokenizer=tokenizer(),device='cpu')
    packed=windows(identical=identical,vary_positions=True)
    actual,report=objectives.window_loss(bundle,packed)
    actual.backward();grad={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    assert report['visual_reuse']['encoded_images']==encoded
    model.zero_grad(set_to_none=True)
    observed=[]
    handle=model.model.visual.register_forward_pre_hook(lambda module,args,kw:observed.append(int(kw['grid_thw'].shape[0])),with_kwargs=True)
    try:upper,stats=objectives.window_loss(bundle,packed,reuse_vision=False)
    finally:handle.remove()
    upper.backward()
    assert observed==[20] and stats['visual_reuse']['encoded_images']==20
    torch.testing.assert_close(actual,upper,atol=1e-6,rtol=1e-5)
    for n,p in model.named_parameters():
        if n in grad:torch.testing.assert_close(p.grad,grad[n],atol=1e-5,rtol=1e-3)


def test_holdout_window_probe_checks_both_vision_paths_without_updates(model):
    model.model.visual.requires_grad_(False)
    bundle=SimpleNamespace(model=model,tokenizer=tokenizer(),device='cpu')
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    def forbidden(*a,**kw):pytest.fail('holdout optimizer called')
    reports=worker.probe_record(bundle,dict(usage='evaluation'),windows(identical=True),
                                SimpleNamespace(step=forbidden,zero_grad=forbidden))
    assert [r['visual_reuse']['encoded_images'] for r in reports]==[1,20]
    assert all(r['phase']=='evaluation_forward_only' for r in reports)
    assert all(p.grad is None and torch.equal(p,before[n]) for n,p in model.named_parameters())


def test_training_window_probe_checks_both_paths_with_adam_state(model):
    model.model.visual.requires_grad_(False)
    bundle=SimpleNamespace(model=model,tokenizer=tokenizer(),device='cpu')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    reports=worker.probe_record(bundle,{},windows(identical=True),optimizer)
    assert [r['visual_reuse']['encoded_images'] for r in reports]==[1,1,20,20]
    assert all(r['phase']=='train_optimizer_step' for r in reports)
    assert all(int(state['step'])==4 for state in optimizer.state.values())
