"""Contract wiring and fail-closed suite orchestration; no model-accuracy claims."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from . import prompts, prompt_contract as contract, train, eval as evaluation, four_runs
from .prompt_candidate import CANDIDATE_NAME, spec_from_case
from .test_prompt_candidate import case


@pytest.mark.parametrize('rgb', ['4rgb', '2rgb_endpoints'])
@pytest.mark.parametrize('output', ['binary', 'choice'])
def test_real_input_and_generation_helpers_use_same_candidate(rgb, output, monkeypatch):
    spec = spec_from_case(case(output))
    images = [object()] * (4 if rgb == '4rgb' else 2)
    seen = []
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            seen.append(messages); return 'chat'
        def __call__(self, **kwargs):
            return {'input_ids': torch.tensor([[1, 2]])}
    # Stop before token masking, after the actual training input renderer.
    bundle = SimpleNamespace(processor=Processor())
    assert train._build_inputs(bundle, images=images, spec=spec, history_rgb_mode=rgb,
        prompt_variant=CANDIDATE_NAME, target=prompts.build_action_target(spec),
        max_length=1, format_loss_weight=.25) is None
    monkeypatch.setattr(evaluation, '_kv_start_state', lambda b,m: seen.append(m))
    monkeypatch.setattr(evaluation, '_student_generate_kv', lambda *a: ('STOP', None, None))
    evaluation._generate(bundle, images, spec=spec, audit=False, history_rgb_mode=rgb,
        max_new_tokens=64, prompt_variant=CANDIDATE_NAME)
    assert seen[0][:-1] == seen[1]
    text = seen[1][1]['content'][-1]['text']
    assert 'Start from the newest frame' in text and 'Brief braking can occur' not in text
    assert len(seen[1][1]['content']) == len(images) + 1
    assert contract.action_prompt_sha256(history_rgb_mode=rgb, action_output_mode=output) == prompts.action_prompt_sha256(history_rgb_mode=rgb, action_output_mode=output)
    assert contract.action_prompt_sha256(prompt_variant=CANDIDATE_NAME, history_rgb_mode=rgb, action_output_mode=output) != prompts.action_prompt_sha256(history_rgb_mode=rgb, action_output_mode=output)


@pytest.mark.parametrize('variant', ['baseline', CANDIDATE_NAME])
def test_saved_adapter_roundtrip_and_mismatch_rejected(tmp_path, monkeypatch, variant):
    from qwen3vl_local.qwen35 import adapters
    monkeypatch.setattr('sys.argv', ['train.py', '--prompt-variant', variant, '--model-dir', str(tmp_path/'base')])
    args = train.parse_args()
    monkeypatch.setattr(adapters, 'save_adapter', lambda *a: None)
    bundle = SimpleNamespace(unwrap=lambda: object(), lora_target_modules=['q_proj'])
    directory = train._save_adapter(bundle, tmp_path, args, step=7)
    cfg = evaluation._validate_action_adapter(directory, Path(args.model_dir))
    assert contract.resolve_prompt_variant('auto', cfg) == variant
    other = CANDIDATE_NAME if variant == 'baseline' else 'baseline'
    with pytest.raises(ValueError, match='differs'): contract.resolve_prompt_variant(other, cfg)
    if variant == 'baseline':
        legacy = deepcopy(cfg); del legacy['prompt_variant']
        assert contract.adapter_prompt_variant(legacy) == 'baseline'
    cfg['production_prompt_sha256'] = 'f'*64
    (directory/'sft_new_loop_phase3_adapter_config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='sha256 mismatch'):
        evaluation._validate_action_adapter(directory, Path(args.model_dir))


def suite_args(tmp_path, command='run'):
    index = tmp_path/'data/frame_index.jsonl'; index.parent.mkdir(); index.write_text('fixture')
    return argparse.Namespace(command=command, groups=None, output=tmp_path/'suite', index=index,
        prompt_variant=CANDIDATE_NAME, data_root=tmp_path/'lead_data', automot_root=tmp_path,
        model_dir=tmp_path/'model', audit_root=tmp_path, collection_dir=tmp_path, workers=1, gpus='0,1')


def test_plan_does_not_inherit_truncation_or_wrong_split(tmp_path, monkeypatch):
    args = suite_args(tmp_path)
    for k,v in dict(MAX_STEPS='2', MAX_FRAMES='4', EVAL_SPLIT='test', OUTPUT_DIR='old', PROMPT_VARIANT='baseline').items():
        monkeypatch.setenv(k,v)
    env = four_runs.environment(args, '4rgb', 'binary')
    assert env['MAX_STEPS'] == env['MAX_FRAMES'] == '0'
    assert env['EVAL_SPLIT'] == 'val' and env['PROMPT_VARIANT'] == CANDIDATE_NAME
    assert 'OUTPUT_DIR' not in env
    plan = four_runs.plan(args)
    assert len(plan['groups']) == 4 and len({g['train_dir'] for g in plan['groups']}) == 4
    assert len({g['prompt_sha256'] for g in plan['groups']}) == 4


def test_suite_runs_all_four_even_if_promotion_gate_fails(tmp_path, monkeypatch):
    args = suite_args(tmp_path)
    monkeypatch.setattr(four_runs, 'runtime_check', lambda a: None)
    monkeypatch.setattr(four_runs, 'prepare', lambda a: dict(index_sha256=four_runs.sha(a.index)))
    monkeypatch.setattr(four_runs, 'audit_inputs_check', lambda a: None)
    monkeypatch.setattr(four_runs, 'historical_dataset_comparison', lambda a: [])
    calls=[]
    def command(a, argv, log, **kwargs):
        calls.append(log)
        if log.startswith('train_'):
            root=Path(kwargs['env']['OUTPUT_DIR'])/'final'; root.mkdir(parents=True)
            (root/'sft_new_loop_phase3_adapter_config.json').write_text('{}')
        return 2 if log.startswith('regression_') else 0
    monkeypatch.setattr(four_runs, 'command', command)
    four_runs.run(args)
    state=json.loads((args.output/'suite.json').read_text())
    assert state['status']=='completed' and len(state['completed_groups'])==4
    assert not any(g['regression_gate_passed'] for g in state['completed_groups'])
    assert calls == [f'{stage}_{r}_{o}.log' for r,o,_ in four_runs.GROUPS for stage in ('train','eval','replay','regression')]
    with pytest.raises(FileExistsError): four_runs.run(args)


def test_failed_preflight_never_calls_training(tmp_path, monkeypatch):
    args=suite_args(tmp_path)
    def fail(a): raise RuntimeError('model missing')
    monkeypatch.setattr(four_runs, 'runtime_check', fail)
    monkeypatch.setattr(four_runs, 'command', lambda *a, **k: pytest.fail('must not run'))
    with pytest.raises(RuntimeError, match='model missing'): four_runs.run(args)
    assert json.loads((args.output/'suite.json').read_text())['status']=='failed'


def test_exposure_applies_before_holdout_repair_without_changing_default(tmp_path, monkeypatch):
    from collections import Counter
    from . import build_dataset as builder, split_coverage
    group='s/Town01_1'
    extra=tmp_path/'extra.json';extra.write_text(json.dumps({'groups':[group]}))
    before=builder.development_route_groups()
    base=dict(scenario='s', route_id='Town01_Rep0_1_route0_01_01_01_01_01',
              town='Town01', split='test', context_id='LEAD_BRAKE',
              action_labels={k:False for k in prompts.ANSWER_KEYS})
    monkeypatch.setattr(builder, 'iter_base_frames', lambda *a,**k: iter([base]))
    class ReachedRepair(Exception): pass
    def inspect(rows, **kwargs):
        assert rows[0]['split']=='train'
        assert group in kwargs['development']
        raise ReachedRepair()
    monkeypatch.setattr(split_coverage, 'complete_context_splits', inspect)
    args=SimpleNamespace(development_groups_extra=str(extra),val_ratio=.05,test_ratio=.1,split_seed=17)
    with pytest.raises(ReachedRepair): builder._balanced_rows_by_split(args, Counter())
    assert builder.development_route_groups()==before and group not in before


def test_exposure_preflight_rejects_test_leakage_and_old_manifest(tmp_path, monkeypatch):
    from . import candidate_data, preflight
    index=tmp_path/'frame_index.jsonl'
    with pytest.raises(FileNotFoundError): candidate_data.validate_candidate_index(index)
    manifest=dict(prompt_contract=dict(prompt_name=prompts.PROMPT_NAME,
        production_prompt_sha256={o:{r:prompts.action_prompt_sha256(history_rgb_mode=r,action_output_mode=o)
            for r in ('4rgb','2rgb_endpoints')} for o in ('binary','choice')}))
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='exposure-isolated'): candidate_data.validate_candidate_index(index)
    manifest['additional_development']=candidate_data.exposure_contract(candidate_data.EXPOSURE)
    path.write_text(json.dumps(manifest));candidate_data.validate_candidate_index(index)
    group=manifest['additional_development']['groups'][0];scenario,route=group.split('/',1)
    index.write_text(json.dumps(dict(scenario=scenario,route_id=route,split='test',current_speed_mps=0))+'\n')
    for name in ('validate_action_rule','validate_mapping_contract','validate_choice_row'):
        monkeypatch.setattr(preflight,name,lambda *a:None)
    with pytest.raises(ValueError,match='development route in holdout'): preflight.check_index(index)


@pytest.mark.parametrize('rgb,output', [(r,o) for r,o,_ in four_runs.GROUPS])
def test_loss_and_generation_validation_forward_candidate(rgb, output, monkeypatch):
    from .test_validation_balance import candidate_rows
    item=train._make_item(candidate_rows()[0],seed=1,action_output_mode=output)
    seen=[]
    class Processor:
        def apply_chat_template(self,messages,**kwargs): seen.append(messages); return 'chat'
        def __call__(self,**kwargs): return {'input_ids':torch.tensor([[1,2]])}
    model=torch.nn.Linear(1,1)
    bundle=SimpleNamespace(model=model,unwrap=lambda:model,processor=Processor(),tokenizer=object(),device=torch.device('cpu'))
    monkeypatch.setattr(train,'_load_images',lambda paths:[object() for _ in paths])
    train.evaluate_loss(bundle,[item],history_rgb_mode=rgb,prompt_variant=CANDIDATE_NAME,
        max_length=1,format_loss_weight=.25,device=torch.device('cpu'),world_size=1)
    monkeypatch.setattr(train,'_kv_start_state',lambda b,m:seen.append(m))
    monkeypatch.setattr(train,'_student_generate_kv',lambda *a:(prompts.build_action_target(item.spec),None,None))
    metrics=train.evaluate_generation_probe(bundle,[item],history_rgb_mode=rgb,prompt_variant=CANDIDATE_NAME,max_new_tokens=64)
    assert metrics['exact_accuracy']==1
    assert len(seen)==2
    assert seen[0][1]['content'][-1]['text']==seen[1][1]['content'][-1]['text']
    assert 'Start from the newest frame' in seen[1][1]['content'][-1]['text']


def _count_route_fixture(task):
    _, _, rows = task
    if any(r.get('bad') for r in rows): raise ValueError('bad raw fixture')
    return len(rows)


def test_parallel_raw_audit_counts_match_and_worker_errors_propagate(tmp_path, monkeypatch):
    from . import audit_raw_index as raw
    index=tmp_path/'index.jsonl'
    rows=[dict(scenario='s',route_id=str(i//2),frame_id=i) for i in range(8)]
    index.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    monkeypatch.setattr(raw,'check_index',lambda *a,**k:{})
    monkeypatch.setattr(raw,'_audit_route',_count_route_fixture)
    assert raw.audit(index,tmp_path,workers=0)==raw.audit(index,tmp_path,workers=2)
    rows[-1]['bad']=True
    index.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='bad raw fixture'): raw.audit(index,tmp_path,workers=2)
