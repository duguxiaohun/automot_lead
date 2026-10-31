"""No shell exports: new production defaults, real tiny BF16 models, explicit legacy controls."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from qwen3vl_local.qwen35 import cache_policy, text_prefix as tp, online_benchmark as bench, online
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35.tests.test_online_benchmark import setup
from qwen3vl_local.qwen35.tests.test_text_prefix import inputs


@pytest.fixture
def default_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith(('AUTOMOT_TEXT_PREFIX','AUTOMOT_ONLINE_TEXT_PREFIX','AUTOMOT_QUESTION_')):
            monkeypatch.delenv(key)


def test_fresh_process_selects_text_layout_and_matching_phase3_release(default_environment):
    script='''
import json
from qwen3vl_local.qwen35 import text_prefix as t,cache_policy as p
from qwen3vl_local.audit_joint import phase3_base_trial as trial
from qwen3vl_local.sft_new_loop_phase12 import prompts
assert prompts.messages([None],"question")[1]['content'][0]['type']=='text'
assert trial.RELEASE.name=='phase3_text_prefix_sources_r14_20261031.json'
assert len(trial.verified_sources())==64
assert t.enabled() and t.cache_enabled() and t.online_enabled()
print(json.dumps(p.current()))
'''
    result=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True,check=True)
    policy=json.loads(result.stdout)
    assert policy['precision']=='bf16_base_native_fp32_v1'
    assert policy['limits']==dict(max_entries=2,max_bytes=1024**3,max_tokens=8192)


@pytest.mark.parametrize('phase',[1,2,3,12])
def test_no_exports_cross_image_production_stream(model,tmp_path,default_environment,phase):
    bundle,rows,_=setup(model,tmp_path,phase)
    assert tp.cache_enabled() and cache_policy.inference_dtype()==torch.bfloat16
    result=bench.benchmark(bundle,phase,rows,window_frames=2,max_new_tokens=3,repeats=2)
    assert result['same_partition_equivalence']
    for trial in result['trials']:
        queries=trial['cached']['frames'][1]['queries']
        assert queries and all(q['cache_class']=='full_hit' for q in queries)
    assert not result['quality_certified'] and not result['carla_certified']


@pytest.mark.parametrize('phase',[1,2,3,12])
@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
def test_online_non_bf16_rejected_before_any_asset_loading(tmp_path,default_environment,phase,dtype):
    # Paths do not exist: precision rejection must precede file or model loading.
    with pytest.raises(ValueError,match='requires BF16'):
        if phase==12:
            from qwen3vl_local.sft_new_loop_phase12 import infer
            infer.load_online_bundle(tmp_path/'missing_base',tmp_path/'missing_adapter',device='cpu',dtype=dtype)
        else:
            online.load_online_bundle(tmp_path/'missing_base',tmp_path/'missing_adapter',phase,device='cpu',dtype=dtype)


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
def test_runtime_precision_change_rejected_before_cache_reuse(model,tmp_path,default_environment,dtype):
    b,_,_=setup(model,tmp_path,1);tp.prefill(b,inputs());before=tp.bank(b).stats.copy()
    model.to(dtype=dtype)
    with pytest.raises(ValueError,match='requires BF16'):tp.prefill(b,inputs())
    assert before==tp.bank(b).stats


def test_wrong_autocast_rejected_and_native_fp32_buffers_preserved(model,tmp_path,default_environment):
    b,_,_=setup(model,tmp_path,1)
    with torch.autocast('cpu',dtype=torch.float16):
        with pytest.raises(ValueError,match='BF16 autocast'):tp.prefill(b,inputs())
    out=tp.prefill(b,inputs())
    assert out.past_key_values.recurrent_states[0].dtype==torch.float32


def test_default_training_still_backpropagates_without_persistent_cache(model,default_environment):
    from peft import LoraConfig,get_peft_model
    from qwen3vl_local.sft_new_loop_phase12 import objective
    adapted=get_peft_model(model.to(dtype=torch.bfloat16),LoraConfig(r=2,target_modules=['q_proj'],task_type='CAUSAL_LM')).train()
    b=SimpleNamespace(model=adapted,device=torch.device('cpu'));cache_policy.start_training(b)
    item=inputs();ids=item['input_ids'][0]
    packed=dict(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids.clone(),
        loss_weights=torch.ones_like(ids,dtype=torch.float32),component_ids=torch.ones_like(ids),
        vision={k:v for k,v in item.items() if k not in ('input_ids','attention_mask')})
    value,_=objective.loss(b,packed,SimpleNamespace(output_keys=['answer']));value.backward()
    grads=[p.grad for n,p in adapted.named_parameters() if 'lora_B' in n]
    assert grads and all(g is not None and g.isfinite().all() and g.abs().sum()>0 for g in grads)
    assert not hasattr(b,'_text_prefix_cache')
    with pytest.raises(ValueError,match='inference-only'):tp.prefill(b,item)


def test_default_prior_engine_loads_real_two_adapters(model,tmp_path,default_environment):
    from peft import LoraConfig,get_peft_model
    from qwen3vl_local.qwen35.backend import LocalModel
    from qwen3vl_local.qwen35.adapters import save_adapter
    from qwen3vl_local.action_prior.runtime import PriorEngine
    from qwen3vl_local.action_prior.prompt_versions import prompt_module
    model.save_pretrained(tmp_path/'base')
    base=LocalModel.from_pretrained(tmp_path/'base',dtype=torch.bfloat16)
    contract={}
    for phase in (1,2):
        _,_,cfg=setup(model,tmp_path,phase)
        module=prompt_module(phase,cfg)
        cfg.update(history_rgb_mode='4rgb',history_rgb_count=4,history_rgb_selected_indices=[0,1,2,3])
        cfg['production_prompt_sha256']=(module.phase1_prompt_sha256 if phase==1 else module.event_prompt_sha256)(history_rgb_mode='4rgb')
        adapted=get_peft_model(base,LoraConfig(r=2,target_modules=['q_proj'],task_type='CAUSAL_LM'))
        path=tmp_path/f'phase{phase}';save_adapter(adapted,path)
        (path/f'sft_new_loop_phase{phase}_adapter_config.json').write_text(json.dumps(cfg))
        base=adapted.unload();contract[f'phase{phase}']=dict(path=str(path),metadata=cfg)
    engine=SimpleNamespace(model=base,_last_decode_state=None,_system_prompt_cache=None)
    prior=PriorEngine(engine,contract)  # No online_text_prefix argument and no env exports.
    assert prior.online_text_prefix and len(prior._prior_cache_policies)==2
    prior._check_cache_policy()
    master=next(p for n,p in prior.adapters.named_parameters() if 'lora_B' in n)
    master.data=master.data.to(torch.bfloat16)
    with pytest.raises(ValueError,match='FP32 LoRA masters'):prior._check_cache_policy()


def test_explicit_legacy_and_disabled_cache_remain_distinct(default_environment,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE','0')
    assert tp.enabled() and not tp.cache_enabled() and tp.online_enabled()
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','0')
    assert not tp.enabled() and not tp.online_enabled() and cache_policy.current() is None
    assert tp.content([None],'q')[0]['type']=='image'
    monkeypatch.setenv('AUTOMOT_ONLINE_TEXT_PREFIX','1')
    with pytest.raises(ValueError,match='requires text-before-images'):tp.online_enabled()


def test_new_default_does_not_accept_legacy_metadata(default_environment):
    with pytest.raises(ValueError,match='layout'):cache_policy.from_metadata({})
