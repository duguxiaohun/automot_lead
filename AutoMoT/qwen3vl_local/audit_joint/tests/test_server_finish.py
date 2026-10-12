import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3vl_local.audit_joint import first_epoch_budget as budget, server_finish as finish, rule_baseline as rb


def test_meta_profile_matches_real_lora_parameter_count(tmp_path):
    import torch
    from peft import LoraConfig, get_peft_model
    from qwen3vl_local.qwen35.vendor.configuration_qwen3_5 import Qwen3_5Config
    from qwen3vl_local.qwen35.vendor.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    from qwen3vl_local.sft_v2.train import _lora_target_modules
    config=Qwen3_5Config(text_config=dict(vocab_size=320,hidden_size=32,intermediate_size=64,
        num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
        linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,
        rope_parameters=dict(rope_type='default',rope_theta=10000000.,partial_rotary_factor=.5,mrope_section=[1,1,2]),pad_token_id=0,eos_token_id=1),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,patch_size=16,spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=32,num_position_embeddings=16),
        image_token_id=4,video_token_id=5,vision_start_token_id=2,vision_end_token_id=3)
    config.save_pretrained(tmp_path)
    measured=budget.profile(tmp_path)  # no weight files exist
    model=Qwen3_5ForConditionalGeneration(config)
    for p in model.parameters():p.requires_grad=False
    targets=_lora_target_modules(model,vision_scope='off',strict_vision_scope=True)
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',target_modules=targets))
    assert measured['parameters']==sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert measured['target_shapes']
    request=budget.request(measured,tmp_path,stored_evidence={'fixture':'test'})
    assert all(v['upper_bound_bytes'] is not None for v in request['components'].values())
    assert request['components']['optimizer_per_checkpoint']['upper_bound_bytes'] > 8*measured['parameters']
    assert request['modes']==2 and request['retained_checkpoints_per_mode']==1


def test_missing_model_retains_stage_and_never_starts_replay(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4 import candidate_pool,teacher_replay
    pilot=tmp_path/'pilot';pilot.mkdir();(pilot/'candidates.json').write_text('{}')
    output=tmp_path/'out';output.mkdir()
    monkeypatch.setattr(candidate_pool,'validate',lambda x:x)
    monkeypatch.setattr(budget,'profile',lambda _: (_ for _ in ()).throw(FileNotFoundError('model config missing')))
    monkeypatch.setattr(teacher_replay,'produce',lambda *a,**kw:pytest.fail('replay must not start'))
    result=finish.build(pilot,tmp_path,output,tmp_path/'model')
    assert result['status']=='blocked' and result['stage']=='checkpoint_budget_profile'
    assert json.loads((output/'build_result.json').read_text())==result
    receipt=finish.package(output,2)
    assert receipt['status']=='verified'


def test_baseline_keeps_both_mode_diagnostics(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import data
    from qwen3vl_local.sft_new_loop_phase4 import evaluation_policy,full_sampling
    monkeypatch.setattr(rb,'dataset_bindings',lambda _: {})
    monkeypatch.setattr(evaluation_policy,'pack_binding',lambda _: {})
    monkeypatch.setattr(data,'load_dataset',lambda *a,**kw: ({'train':[],'val':[],'test':[]},{}))
    monkeypatch.setattr(full_sampling,'validate_dataset_scope',lambda _:None)
    monkeypatch.setattr(evaluation_policy,'prepare',lambda *a,**kw: (_ for _ in ()).throw(ValueError('missing events')))
    report=rb.check(tmp_path,tmp_path,space_path=tmp_path)
    assert set(report['modes'])=={'2','4'}
    assert all('missing events' in m['error'] for m in report['modes'].values())
    assert report['status']=='blocked'


def test_guard_rejection_still_leaves_a_completion_receipt(tmp_path):
    result=finish.package(tmp_path,2)
    assert result['status']=='verified'
    assert json.loads((tmp_path/'finish_receipt.json').read_text())==dict(
        status='blocked_or_interrupted',command_exit_code=2,build_result_exists=False)
    assert (tmp_path/'handoff/fullflow_report.zip').is_file()
    assert (tmp_path/'handoff/fullflow_package.json').is_file()
