import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from qwen3vl_local.qwen35.vendor.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from qwen3vl_local.qwen35.vendor.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration, Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb,
)
from qwen3vl_local.qwen35.backend import LocalModel
from qwen3vl_local.mrope_utils import qwen3vl_incremental_forward
from qwen3vl_local.engine import _clone_cache
from qwen3vl_local.goalgen.qwen_kv import segment_kv_for_dit

torch.set_num_threads(1)


@pytest.fixture
def model():
    torch.manual_seed(41)
    config = Qwen3_5Config(
        text_config=dict(vocab_size=320, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
                         linear_key_head_dim=8, linear_value_head_dim=8,
                         rope_parameters=dict(rope_type="default", rope_theta=10000000.,
                                              partial_rotary_factor=0.5, mrope_section=[1, 1, 2]),
                         pad_token_id=0, eos_token_id=1),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                           patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                           out_hidden_size=32, num_position_embeddings=16),
        image_token_id=4, video_token_id=5, vision_start_token_id=2, vision_end_token_id=3,
    )
    return Qwen3_5ForConditionalGeneration(config).eval()


@pytest.mark.parametrize('suffix_len', [1, 3, 7, 63, 65])
def test_cached_suffix_matches_full(model, suffix_len):
    ids = torch.arange(10, 18 + suffix_len).unsqueeze(0)
    with torch.no_grad():
        full = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        prefix = model(input_ids=ids[:, :8], attention_mask=torch.ones_like(ids[:, :8]), use_cache=True)
        branch = _clone_cache(prefix.past_key_values)
        output = qwen3vl_incremental_forward(model, feed_ids=ids[:, 8:],
            attention_mask=torch.ones_like(ids), past_key_values=branch,
            prefix_len=8, rope_deltas=prefix.rope_deltas)
    torch.testing.assert_close(output.logits, full[:, 8:], atol=2e-5, rtol=2e-4)
    assert prefix.past_key_values.get_seq_length() == 8
    assert output.past_key_values.get_seq_length() == ids.shape[1]
    assert len(segment_kv_for_dit(branch, num_segments=1)) == 1
    with pytest.raises(ValueError):
        segment_kv_for_dit(branch, num_segments=4)


@pytest.mark.parametrize('frames', [1, 2, 4])
def test_multiframe_prefill_and_decode(model, frames):
    # 64x64 -> 4 merged tokens; exercises a NONZERO multimodal rotary delta.
    ids = torch.tensor([[10] + [x for _ in range(frames) for x in (2, 4, 4, 4, 4, 3)] + [11]])
    pixels = torch.randn(frames * 16, 3 * 2 * 16 * 16)
    grids = torch.tensor([[1, 4, 4]] * frames)
    suffix = torch.tensor([[12, 13, 14]])
    with torch.no_grad():
        prefix = model(input_ids=ids, pixel_values=pixels, image_grid_thw=grids,
                       attention_mask=torch.ones_like(ids), use_cache=True)
        assert prefix.rope_deltas.item() < 0
        output = qwen3vl_incremental_forward(model, feed_ids=suffix,
            attention_mask=torch.ones(1, ids.shape[1]+3, dtype=torch.long),
            past_key_values=prefix.past_key_values, prefix_len=ids.shape[1], rope_deltas=prefix.rope_deltas)
        whole = torch.cat((ids, suffix), 1)
        full = model(input_ids=whole, pixel_values=pixels, image_grid_thw=grids,
                     attention_mask=torch.ones_like(whole), use_cache=False)
    torch.testing.assert_close(output.logits, full.logits[:, -3:], atol=2e-5, rtol=2e-4)


def test_left_padded_batch_and_branch_selection(model):
    ids=torch.tensor([[0,0,10,11,12], [10,11,12,13,14]])
    mask=(ids!=0).long()
    with torch.no_grad():
        pref=model(input_ids=ids,attention_mask=mask,use_cache=True)
        out=qwen3vl_incremental_forward(model,feed_ids=torch.tensor([[15,16],[15,16]]),
            attention_mask=torch.cat((mask,torch.ones(2,2,dtype=torch.long)),1),
            past_key_values=_clone_cache(pref.past_key_values),prefix_len=5,rope_deltas=pref.rope_deltas)
        for row in range(2):
            whole=torch.cat((ids[row][mask[row].bool()],torch.tensor([15,16]))).view(1,-1)
            full=model(input_ids=whole,attention_mask=torch.ones_like(whole),use_cache=False)
            torch.testing.assert_close(out.logits[row],full.logits[0,-2:],atol=2e-5,rtol=2e-4)
        branch=_clone_cache(pref.past_key_values)
        branch.reorder_cache(torch.tensor([1]))
        assert branch.conv_states[0].shape[0]==1
        assert branch.recurrent_states[0].shape[0]==1
        assert branch.key_cache[3].shape[0]==1


def test_partial_interleaved_rope_matches_upstream():
    from qwen3vl_local.leadmot.mot_block import apply_mrope
    config = Qwen3_5TextConfig(head_dim=256, rope_parameters=dict(
        rope_type='default', rope_theta=10000000., partial_rotary_factor=.25,
        mrope_section=[11, 11, 10]))
    q, k = torch.randn(2, 4, 5, 256), torch.randn(2, 4, 5, 256)
    positions = torch.randint(0, 200, (3, 2, 5))
    cos, sin = Qwen3_5TextRotaryEmbedding(config)(q, positions)
    expected = apply_rotary_pos_emb(q, k, cos, sin)
    actual = apply_mrope(q, k, positions, 10000000., (11, 11, 10), .25, True)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    assert torch.equal(actual[0][..., 64:], q[..., 64:])


def test_local_roundtrip_and_no_network(model, tmp_path, monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError('runtime attempted network')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    model.save_pretrained(tmp_path)
    restored = LocalModel.from_pretrained(tmp_path)
    assert restored.__class__.__module__.startswith('qwen3vl_local.qwen35.vendor.')
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    with pytest.raises(FileNotFoundError):
        LocalModel.from_pretrained(tmp_path / 'missing')
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'qwen3_vl'}))
    with pytest.raises(ValueError, match='Expected Qwen3.5'):
        LocalModel.from_pretrained(tmp_path)


def test_lora_covers_delta_layers_and_backprop(model):
    from peft import LoraConfig, get_peft_model, TaskType
    from qwen3vl_local.sft_v2.train import _lora_target_modules
    targets = _lora_target_modules(model, vision_scope='off', strict_vision_scope=True)
    assert any(t.endswith('in_proj_qkv') for t in targets)
    assert any(t.endswith('in_proj_a') for t in targets)
    assert not any('.visual.' in t for t in targets)
    adapted = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=targets,
                                              task_type=TaskType.CAUSAL_LM))
    ids = torch.tensor([[10, 11, 12, 13]])
    adapted(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    grads = [(n, p.grad) for n, p in adapted.named_parameters() if 'lora_B' in n]
    assert grads and all(g is not None and torch.isfinite(g).all() for n, g in grads)
    assert any(g.abs().sum() > 0 for n, g in grads if 'in_proj_qkv' in n)


@pytest.mark.parametrize('suffix_len', [1, 3])
def test_cached_suffix_gradient(model, suffix_len):
    with torch.no_grad():
        prefix=model(input_ids=torch.tensor([[10,11,12]]), use_cache=True)
    out=qwen3vl_incremental_forward(model,feed_ids=torch.arange(13,13+suffix_len).view(1,-1),
        attention_mask=torch.ones(1,3+suffix_len,dtype=torch.long),past_key_values=prefix.past_key_values,
        prefix_len=3,rope_deltas=prefix.rope_deltas)
    out.logits.square().mean().backward()
    assert model.model.language_model.layers[0].linear_attn.in_proj_qkv.weight.grad.isfinite().all()


def test_adapter_save_load_and_reject_legacy(model, tmp_path):
    from peft import LoraConfig, get_peft_model, TaskType
    from qwen3vl_local.qwen35.adapters import save_adapter, LocalPeftModel, validate_adapter
    model.save_pretrained(tmp_path / "base")
    model = LocalModel.from_pretrained(tmp_path / "base")
    adapted=get_peft_model(model,LoraConfig(r=2,target_modules=['q_proj'],task_type=TaskType.CAUSAL_LM))
    save_adapter(adapted,tmp_path)
    validate_adapter(tmp_path)
    loaded=LocalPeftModel.from_pretrained(adapted.unload(),tmp_path)
    assert loaded.config.model_type=='qwen3_5'
    marker=tmp_path/'qwen35_backend.json'
    content=json.loads(marker.read_text());content['enable_thinking']=True
    marker.write_text(json.dumps(content))
    with pytest.raises(ValueError,match='contract mismatch'):
        validate_adapter(tmp_path)
    marker.unlink()
    with pytest.raises(ValueError,match='lacks'):
        validate_adapter(tmp_path)


def test_prefill_matches_unmodified_upstream(model):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config as UpstreamConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration as UpstreamModel
    baseline=UpstreamModel(UpstreamConfig(**model.config.to_dict())).eval()
    baseline.load_state_dict(model.state_dict())
    ids=torch.tensor([[10,2,4,4,4,4,3,11]])
    kwargs=dict(input_ids=ids,pixel_values=torch.randn(16,1536),image_grid_thw=torch.tensor([[1,4,4]]),
                mm_token_type_ids=(ids==4).long(),attention_mask=torch.ones_like(ids),use_cache=True)
    with torch.no_grad():
        a=model(**kwargs);b=baseline(**kwargs)
    torch.testing.assert_close(a.logits,b.logits)
    torch.testing.assert_close(a.past_key_values.key_cache[3],b.past_key_values.key_cache[3])
    torch.testing.assert_close(a.past_key_values.recurrent_states[0],b.past_key_values.recurrent_states[0])


def test_native_generation_and_manual_decode_match(model):
    ids=torch.tensor([[10,11,12]])
    with torch.no_grad():
        native=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=4,do_sample=False,
                              eos_token_id=None,pad_token_id=0)
        pref=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True)
        logits=pref.logits[:,-1,:];tokens=[];cache=pref.past_key_values
        for step in range(4):
            token=logits.argmax(-1,keepdim=True);tokens.append(token)
            out=qwen3vl_incremental_forward(model,feed_ids=token,
                attention_mask=torch.ones(1,4+step,dtype=torch.long),past_key_values=cache,
                prefix_len=3+step,rope_deltas=pref.rope_deltas)
            logits=out.logits[:,-1,:];cache=out.past_key_values
    assert torch.equal(native[:,3:],torch.cat(tokens,1))


def test_planning_bridge_and_decoder_backward(model):
    from qwen3vl_local.leadmot.config import LeadMoTPlanningDecoderConfig
    from qwen3vl_local.leadmot.decoder import LeadMoTPlanningDecoder
    from qwen3vl_local.qwen35.integration import segment_for_decoder
    config=LeadMoTPlanningDecoderConfig(hidden_size=16,num_kv_heads=1,head_dim=16,num_heads=1,
        num_qwen_layers=4,qwen_full_attention_layers=(3,),num_layers=1,
        partial_rotary_factor=.5,mrope_section_dim=(1,1,2),use_bev=False)
    with torch.no_grad():
        cache=model(input_ids=torch.tensor([[10,11,12]]),use_cache=True).past_key_values
    segments=segment_for_decoder(cache,config)
    decoder=LeadMoTPlanningDecoder(config)
    # Exercise the actual prefix-attention block with the hybrid model's K/V.
    x=torch.randn(1,5,16,requires_grad=True)
    decoder.blocks[0](x,segments[0],rope_position_offset=3).square().mean().backward()
    assert x.grad.isfinite().all()
    assert decoder.blocks[0].attn.q_proj.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.parameters())


def test_fresh_text_prefill_does_not_inherit_image_delta(model):
    ids=torch.tensor([[10,11,12]])
    with torch.no_grad():
        expected=model(input_ids=ids,use_cache=False).logits
        model.model.rope_deltas=torch.tensor([[-100]])
        actual=model(input_ids=ids,use_cache=False)
    torch.testing.assert_close(actual.logits,expected)
    assert actual.rope_deltas.item()==0


@pytest.mark.parametrize('short_len', [0, 1, 3])
@pytest.mark.parametrize('scoring', [False, True])
def test_v5_ragged_q1_then_q2_matches_individual(model, short_len, scoring):
    from qwen3vl_local.sft_v5.train import (
        KVState, _pad_rollout_id_list, _append_token_ids_padded_no_logits,
        _append_token_ids_with_logits_padded_scoring)
    bundle = SimpleNamespace(model=model, tokenizer=SimpleNamespace(pad_token_id=0))
    prefix = torch.tensor([[10, 11, 12], [10, 11, 12]])
    answers = [torch.arange(20, 20+short_len).view(1, -1), torch.tensor([[20,21,22,23,24]])]
    with torch.no_grad():
        out = model(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=True)
        state = KVState(prefix, prefix, torch.ones_like(prefix), out.past_key_values,
                        out.rope_deltas, out.logits[:, -1])
    if scoring:
        ids, mask = _pad_rollout_id_list(bundle, answers)
        state, _ = _append_token_ids_with_logits_padded_scoring(bundle, state, ids, mask)
    else:
        with torch.no_grad():
            state = _append_token_ids_padded_no_logits(bundle, state, answers)
    q2 = torch.tensor([[30,31,32], [30,31,32]])
    state, predictions = _append_token_ids_with_logits_padded_scoring(
        bundle, state, q2, torch.ones_like(q2, dtype=torch.bool))
    for row in range(2):
        whole = torch.cat((prefix[row:row+1], answers[row], q2[row:row+1]), 1)
        with torch.no_grad():
            single = model(input_ids=whole, attention_mask=torch.ones_like(whole), use_cache=True)
        torch.testing.assert_close(predictions[row], single.logits[0,-4:-1], atol=2e-5, rtol=2e-4)
        for layer in (0,1,2):
            torch.testing.assert_close(state.past_key_values.conv_states[layer][row],
                single.past_key_values.conv_states[layer][0], atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(state.past_key_values.recurrent_states[layer][row],
                single.past_key_values.recurrent_states[layer][0], atol=2e-5, rtol=2e-4)
    predictions.square().mean().backward()
    assert model.model.language_model.layers[0].linear_attn.in_proj_qkv.weight.grad.isfinite().all()


def test_adapter_rejects_same_structure_changed_weights(model, tmp_path):
    from peft import LoraConfig, get_peft_model, TaskType
    from qwen3vl_local.qwen35.adapters import save_adapter, LocalPeftModel
    base, adapter = tmp_path/'base', tmp_path/'adapter'
    model.save_pretrained(base)
    loaded = LocalModel.from_pretrained(base)
    adapted = get_peft_model(loaded, LoraConfig(r=2, target_modules=['q_proj'],task_type=TaskType.CAUSAL_LM))
    save_adapter(adapted, adapter)
    with torch.no_grad():
        model.lm_head.weight.add_(0.1)
    model.save_pretrained(base)
    changed = LocalModel.from_pretrained(base)
    with pytest.raises(ValueError, match='weights'):
        LocalPeftModel.from_pretrained(changed, adapter)


@pytest.mark.parametrize('asset', ['model.safetensors', 'chat_template.jinja', 'tokenizer.json', 'preprocessor_config.json', 'adapter_model.safetensors'])
def test_planning_content_identity_and_relocation(tmp_path, asset):
    import shutil
    from qwen3vl_local.leadmot.config import build_qwen_backbone_contract, require_qwen_backbone_match
    base, adapter = tmp_path/'base', tmp_path/'adapter'
    base.mkdir(); adapter.mkdir()
    for name in ('config.json', 'tokenizer.json', 'preprocessor_config.json'):
        (base/name).write_text('{}')
    (base/'chat_template.jinja').write_text('original template')
    (base/'model.safetensors').write_bytes(b'original weights')
    (adapter/'adapter_config.json').write_text('{}')
    (adapter/'adapter_model.safetensors').write_bytes(b'adapter')
    expected = build_qwen_backbone_contract(base, adapter)
    shutil.copytree(base, tmp_path/'moved_base'); shutil.copytree(adapter, tmp_path/'moved_adapter')
    require_qwen_backbone_match(expected, build_qwen_backbone_contract(tmp_path/'moved_base', tmp_path/'moved_adapter'), 'moved')
    target = adapter/asset if asset.startswith('adapter_') else base/asset
    target.write_bytes(target.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='mismatch'):
        require_qwen_backbone_match(expected, build_qwen_backbone_contract(base, adapter), 'changed')


def test_sharded_base_identity(tmp_path):
    from qwen3vl_local.qwen35.identity import base_asset_hashes
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'part1.safetensors','b':'part2.safetensors'}}))
    for name in ('part1.safetensors','part2.safetensors'):
        (tmp_path/name).write_bytes(b'weights')
    before = base_asset_hashes(tmp_path)
    (tmp_path/'part2.safetensors').write_bytes(b'changed')
    assert before != base_asset_hashes(tmp_path)
    (tmp_path/'part1.safetensors').unlink()
    with pytest.raises(ValueError, match='shard'):
        base_asset_hashes(tmp_path)


def test_goalgen_checkpoint_save_and_eval_bind_content(tmp_path, monkeypatch):
    from qwen3vl_local.goalgen.train import save_checkpoint
    from qwen3vl_local.goalgen import eval as evaluation
    from qwen3vl_local.goalgen.dit import DiTMoTConfig
    from qwen3vl_local.leadmot.config import build_qwen_backbone_contract
    base = tmp_path/'base'; base.mkdir()
    (base/'config.json').write_text('{}')
    (base/'model.safetensors').write_bytes(b'base weights')
    (base/'chat_template.jinja').write_text('template')
    adapter = tmp_path/'adapter'; adapter.mkdir()
    (adapter/'adapter_config.json').write_text('{}')
    (adapter/'adapter_model.safetensors').write_bytes(b'original adapter')
    contract = build_qwen_backbone_contract(base, adapter)
    args = SimpleNamespace(checkpoint_dir=str(base), qwen_adapter_dir=str(adapter),
        qwen_backbone_contract=contract, allow_qwen_adapter_mismatch=True, artifact_version='v1',
        qwen_kv_segment_mode="select_last")
    module = torch.nn.Linear(2, 2)
    module.cfg = DiTMoTConfig(num_layers=1, hidden_dim=32, n_heads=2)
    module.patch_unpatch_metadata = lambda path: {'path':path}
    optimizer = torch.optim.SGD(module.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _:1.)
    ema = SimpleNamespace(state_dict=module.state_dict, decay=.9)
    save_checkpoint(tmp_path/'run', module, optimizer, scheduler, ema, {}, 1, args)
    kv = [(torch.zeros(1,2,3,16), torch.zeros(1,2,3,16))]
    def reached_model(*args, **kwargs):
        raise RuntimeError('reached model construction')
    monkeypatch.setattr(evaluation, 'DiTMoT', reached_model)
    for name in ('latest.pt', 'checkpoint-000001/goalgen_v1.pt'):
        path = tmp_path/'run'/name
        assert torch.load(path, weights_only=False)['qwen_backbone'] == contract
        with pytest.raises(RuntimeError, match='reached model construction'):
            evaluation.build_dit_from_ckpt(path, kv, args, torch.device('cpu'), torch.float32)
    path = tmp_path/'run/latest.pt'
    args.qwen_kv_segment_mode = "mean"
    with pytest.raises(ValueError, match='segment mode mismatch'):
        evaluation.build_dit_from_ckpt(path, kv, args, torch.device('cpu'), torch.float32)
    args.qwen_kv_segment_mode = "select_last"
    (adapter/'adapter_model.safetensors').write_bytes(b'replaced same path')
    with pytest.raises(ValueError, match='mismatch'):
        evaluation.build_dit_from_ckpt(path, kv, args, torch.device('cpu'), torch.float32)
    payload = torch.load(path, weights_only=False); payload.pop('qwen_backbone'); torch.save(payload, path)
    with pytest.raises(ValueError, match='legacy checkpoint'):
        evaluation.build_dit_from_ckpt(path, kv, args, torch.device('cpu'), torch.float32)


def test_system_cache_after_image_keeps_text_delta_through_decode(model):
    from qwen3vl_local.engine import LocalQwen35Engine, GenerationTrace
    engine = LocalQwen35Engine('.', device='cpu', cache_system_prompt=True, max_gen_tokens=3)
    engine.model = model
    engine.processor = SimpleNamespace(batch_decode=lambda ids, **kwargs: [str(ids.tolist())])
    engine._eos_ids = lambda: set()
    engine.apply_system_prefix_template = lambda text: 'system'
    prefix = torch.tensor([[10, 11]])
    engine.prepare_inputs = lambda text, images: dict(input_ids=prefix, attention_mask=torch.ones_like(prefix))
    trace = lambda: GenerationTrace('', {}, {}, {})
    image_ids = torch.tensor([[10,11,2,4,4,4,4,3,12]])
    image_inputs = dict(input_ids=image_ids, attention_mask=torch.ones_like(image_ids),
        pixel_values=torch.randn(16, 3*2*16*16), image_grid_thw=torch.tensor([[1,4,4]]))
    image_out = engine.prefill_with_optional_system_cache('system', image_inputs, trace())
    assert image_out.rope_deltas.item() == -2
    text_ids = torch.tensor([[10,11,13,14]])
    inputs = dict(input_ids=text_ids, attention_mask=torch.ones_like(text_ids))
    text_trace = trace()
    cached = engine.prefill_with_optional_system_cache('system', inputs, text_trace)
    assert text_trace.system_prompt_cache_used
    assert cached.rope_deltas.item() == 0
    # Do not fresh-prefill the shared model before decoding: that would hide
    # the stale model-level delta bug after the first incremental step.
    reference = copy.deepcopy(model)
    generated = engine.decode(inputs, cached, text_trace)
    assert generated.shape[1] == 3
    assert engine._last_decode_state['rope_deltas'].item() == 0
    whole = engine._last_decode_state['cache_input_ids']
    with torch.no_grad():
        expected = reference(input_ids=whole, attention_mask=torch.ones_like(whole), use_cache=False)
    torch.testing.assert_close(engine._last_decode_state['next_logits'], expected.logits[:,-1], atol=2e-5, rtol=2e-4)
    assert engine._system_prompt_cache['rope_deltas'].item() == 0


@pytest.mark.parametrize('mode', ['select_last', 'mean', 'concat_layers'])
def test_goalgen_segment_mode_contract(mode):
    from qwen3vl_local.goalgen.qwen_kv import require_kv_segment_mode
    payload = {'args': {'qwen_kv_segment_mode': mode}}
    require_kv_segment_mode(payload, mode, 'saved args')
    for other in {'select_last', 'mean', 'concat_layers'} - {mode}:
        with pytest.raises(ValueError, match='segment mode mismatch'):
            require_kv_segment_mode(payload, other, 'wrong mode')
    with pytest.raises(ValueError, match='missing/invalid'):
        require_kv_segment_mode({}, mode, 'missing mode')
