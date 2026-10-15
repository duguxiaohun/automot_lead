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
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import evaluation_policy
    from qwen3vl_local.sft_new_loop_phase4 import full_sampling
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


def test_completed_admission_block_is_not_interruption_and_report_is_preserved(tmp_path):
    import zipfile
    original = dict(status='blocked', stage='finished_cpu', errors=['missing U-E5/proceed/YES'])
    (tmp_path/'build_result.json').write_text(json.dumps(original))
    finish.package(tmp_path, 2)
    receipt = json.loads((tmp_path/'finish_receipt.json').read_text())
    assert receipt['status'] == 'admission_blocked'
    assert receipt['build_stage'] == 'finished_cpu'
    with zipfile.ZipFile(tmp_path/'handoff/fullflow_report.zip') as z:
        assert json.loads(z.read('build_result.json')) == original
    assert json.loads((tmp_path/'build_result.json').read_text()) == original


def test_failure_after_successful_report_never_grants_admission(tmp_path):
    (tmp_path/'build_result.json').write_text(json.dumps(dict(status='admitted_rule_baseline',stage='finished_cpu')))
    finish.package(tmp_path, 9)
    receipt = json.loads((tmp_path/'finish_receipt.json').read_text())
    assert receipt['status'] == 'blocked_or_interrupted'
    assert receipt['build_status'] == 'admitted_rule_baseline'


def test_raw_gate_matches_final_cells_and_ignores_holdout_or_catchup():
    from qwen3vl_local.sft_new_loop_phase4.tests.test_production_selection_v50 import pool
    from qwen3vl_local.sft_new_loop_phase4.teacher_rules import rule_class
    data = pool('rule_teacher')
    for row in data['train']:
        row['edge'] = {'U-E2':'return', 'U-E7':'complete'}.get(row['episode']['event'], 'proceed')
    final = rb.measure(data, dict(train_ids=[], plans={}, exclusions=[]))
    rows = []
    for row in data['train']:
        key = rule_class(row['episode']['event'],row['edge'],4,'readiness')+'/'+row['target']
        rows.append(dict(split='train',physical_group=row['physical_group'],targets={key:1}))
    raw = finish.raw_training_support(dict(routes=rows))
    assert raw['critical_support'] == final['critical_support']
    assert not raw['missing_critical_support']
    target = rule_class('U-E5','proceed',4,'readiness')+'/YES'
    for row in rows:
        if target in row['targets']:row['split']='val'
    rows.append(dict(split='train',physical_group='p',targets={rule_class('U-E5','proceed',4,'catchup')+'/YES':20}))
    assert finish.raw_training_support(dict(routes=rows))['missing_critical_support'] == ['U-E5/proceed/YES']


def test_missing_raw_support_stops_before_compilation_and_still_packages(tmp_path, monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4 import candidate_pool, teacher_replay, teacher_data
    from qwen3vl_local.audit_joint.io import file_sha
    pilot = tmp_path/'pilot';(pilot/'production').mkdir(parents=True)
    (pilot/'candidates.json').write_text('{}')
    index = pilot/'production/index.json';index.write_text(json.dumps(dict(routes=[])))
    out = tmp_path/'out';out.mkdir()
    monkeypatch.setattr(candidate_pool,'validate',lambda x:x)
    monkeypatch.setattr(budget,'profile',lambda _:dict(fixture=True))
    monkeypatch.setattr(teacher_replay,'produce',lambda *a,**kw:dict(complete_causal_production=True,index_sha256=file_sha(index)))
    monkeypatch.setattr(teacher_data,'compile_bundle',lambda *a,**kw:pytest.fail('must not compile missing support'))
    result = finish.build(pilot,tmp_path,out,tmp_path/'model')
    assert result['compilation_skipped'] is True and result['blocking_stage']=='raw_training_support'
    assert result['training_started'] is False and not (out/'data4').exists()
    finish.package(out,2)
    assert json.loads((out/'finish_receipt.json').read_text())['status']=='admission_blocked'
    import zipfile
    with zipfile.ZipFile(out/'handoff/fullflow_report.zip') as z:
        assert 'raw_training_support.json' in z.namelist()


@pytest.mark.parametrize('name,payload', [
    ('build_result.json', b'{"status":'),
    ('automatic_admission.json', b'[]'),
    ('data4/manifest.json', b'null'),
    ('references4/references.json', b'\xff'),
    ('view/student_view.json', b'{'),
    ('first_epoch_budget.json', b'{}'),
    ('source_check.json', b' ' * 10_000_001),
])
def test_broken_reports_keep_a_bounded_failure_package(tmp_path, name, payload):
    import zipfile
    from qwen3vl_local.audit_joint.io import file_sha
    original = dict(status='admitted_existing_data_development', stage='finished_cpu')
    (tmp_path/'build_result.json').write_text(json.dumps(original))
    path = tmp_path/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    sha = file_sha(path)
    receipt = finish.package(tmp_path, 0)
    assert receipt['status'] == 'verified' and receipt['bytes'] < 30_000_000
    assert file_sha(path) == sha
    with zipfile.ZipFile(tmp_path/'handoff/fullflow_report.zip') as z:
        assert z.testzip() is None
        completion = json.loads(z.read('finish_receipt.json'))
        assert completion['status'] == 'blocked_or_interrupted'
        assert completion['diagnostics_complete'] is False
        errors = json.loads(z.read('collection_errors.json'))['errors']
        assert any(e.get('sha256') == sha and e['path'] == str(path) for e in errors)
        import hashlib
        assert {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()} == receipt['files']


def test_aggregate_diagnostics_budget_keeps_other_evidence(tmp_path):
    import zipfile
    for name in ('source_check.json', 'build_result.json', 'automatic_admission.json'):
        (tmp_path/name).write_text(json.dumps(dict(padding='x' * 7_000_000)))
    (tmp_path/'build.log').write_text('last stage failed\n')
    finish.package(tmp_path, 9)
    with zipfile.ZipFile(tmp_path/'handoff/fullflow_report.zip') as z:
        assert sum(i.file_size for i in z.infolist()) < 30_000_000
        assert 'remaining package budget' in z.read('collection_errors.json').decode()
        assert 'last stage failed' in z.read('log_tail.json').decode()


def test_intact_completed_result_keeps_admitted_status(tmp_path):
    import zipfile
    result = dict(status='admitted_existing_data_development', stage='finished_cpu',
                  coverage_policy='existing_data_development', unsupported_capabilities=['U-E5/proceed/YES'])
    (tmp_path/'build_result.json').write_text(json.dumps(result))
    finish.package(tmp_path, 0)
    with zipfile.ZipFile(tmp_path/'handoff/fullflow_report.zip') as z:
        assert json.loads(z.read('build_result.json')) == result
        assert json.loads(z.read('finish_receipt.json'))['status'] == result['status']
        assert 'collection_errors.json' not in z.namelist()


def test_completion_fields_cannot_duplicate_unbounded_payload(tmp_path):
    import zipfile
    result = dict(status='admitted_existing_data_development', stage='finished_cpu',
                  unsupported_capabilities=['x' * 20_000])
    (tmp_path/'build_result.json').write_text(json.dumps(result))
    finish.package(tmp_path, 0)
    with zipfile.ZipFile(tmp_path/'handoff/fullflow_report.zip') as z:
        receipt = json.loads(z.read('finish_receipt.json'))
        assert receipt['status'] == 'blocked_or_interrupted'
        assert receipt['unsupported_capabilities']['omitted'] is True
        assert json.loads(z.read('build_result.json')) == result
