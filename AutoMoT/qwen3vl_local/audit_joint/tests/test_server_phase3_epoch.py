import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3vl_local.audit_joint import server_phase3_epoch as tool


def test_fixed_first_epoch_resets_ambient_smoke_and_save_knobs(monkeypatch):
    for k, v in dict(MAX_STEPS='40', EVAL_SPLIT='test', SAVE_STEPS='1', NUM_EPOCHS='7',
                     SAVE_BEST_VAL='1', FOCUS_BALANCE_COUNT='2', WORLD_SIZE='8').items():
        monkeypatch.setenv(k, v)
    env = tool.training_env('index', 'data', 'model', 'out', '2rgb_short')
    assert env['MAX_STEPS'] == env['SAVE_STEPS'] == env['SAVE_BEST_VAL'] == env['SAVE_BEST_GENERATION'] == '0'
    assert env['NUM_EPOCHS'] == env['SAVE_FINAL'] == '1'
    assert env['EVAL_SPLIT'] == 'val' and env['FOCUS_BALANCE_COUNT'] == '1024'
    assert 'WORLD_SIZE' not in env
    assert env['ACTION_OUTPUT_MODE'] == 'choice' and env['HISTORY_RGB_MODE'] == '2rgb_short'


@pytest.fixture
def cases(tmp_path):
    from qwen3vl_local.sft_new_loop_phase3.test_prompt_candidate import case
    from qwen3vl_local.sft_new_loop_phase3.prompts import parse_action_output, build_action_target
    from qwen3vl_local.sft_new_loop_phase3.prompt_candidate import spec_from_case
    row = case('choice')
    spec = spec_from_case(row)
    row['raw_output'] = build_action_target(spec)
    row['parsed'] = parse_action_output(row['raw_output'], spec=spec)
    a, b = tmp_path / 'train.jsonl', tmp_path / 'reload.jsonl'
    a.write_text(json.dumps(row) + '\n')
    row['parsed'] = {k: 'YES' if v else 'NO' for k, v in row['parsed'].items()}
    b.write_text(json.dumps(row) + '\n')
    return a, b


def test_reload_reparses_and_matches_complete_case_set(cases):
    assert tool.compare_reload(*cases)['status'] == 'verified'


@pytest.mark.parametrize('defect', ['raw', 'missing', 'duplicate', 'rgb', 'spec'])
def test_reload_rejects_changed_or_unbound_results(cases, defect):
    a, b = cases
    row = json.loads(b.read_text())
    if defect == 'raw':
        row['raw_output'] = 'nonsense'
    elif defect == 'rgb':
        row['history_rgb_paths_used'] = ['other.jpg']
    elif defect == 'spec':
        row['frame_id'] += 1
    b.write_text('' if defect == 'missing' else (json.dumps(row) + '\n') * (2 if defect == 'duplicate' else 1))
    with pytest.raises(ValueError):
        tool.compare_reload(a, b)


def test_preparation_member_tampering_rejected_before_index_read(tmp_path):
    from qwen3vl_local.audit_joint.server_pilot import package
    package(tmp_path, {'prepare_result.json': {'status': 'blocked'}})
    (tmp_path / 'pilot_report.zip').rename(tmp_path / 'phase3_prepare_report.zip')
    (tmp_path / 'pilot_package.json').rename(tmp_path / 'phase3_prepare_package.json')
    receipt = json.loads((tmp_path / 'phase3_prepare_package.json').read_text())
    receipt['files']['prepare_result.json'] = 'wrong'
    (tmp_path / 'phase3_prepare_package.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='member SHA'):
        tool.verified_preparation(tmp_path)


def test_failure_package_does_not_include_weights(tmp_path):
    import zipfile
    (tmp_path / 'adapter_model.safetensors').write_bytes(b'excluded')
    (tmp_path / 'epoch.log').write_text('interrupted')
    assert tool.package(tmp_path, 75) == 2
    with zipfile.ZipFile(tmp_path / 'phase3_epoch_report.zip') as z:
        assert z.testzip() is None
        assert set(z.namelist()) == {'epoch_log_tail.json', 'handoff.json'}


@pytest.mark.parametrize('mode', ['2rgb_short', '4rgb'])
def test_real_eval_cli_reads_saved_contract_without_forbidden_rgb_override(tmp_path, monkeypatch, mode):
    from qwen3vl_local.sft_new_loop_phase3 import train, eval as evaluation
    from qwen3vl_local.sft_new_loop_phase3.prompt_candidate import CANDIDATE_NAME
    from qwen3vl_local.qwen35 import adapters
    monkeypatch.setattr('sys.argv', ['train', '--model-dir', str(tmp_path / 'base'),
        '--history-rgb-mode', mode, '--action-output-mode', 'choice', '--prompt-variant', CANDIDATE_NAME])
    args = train.parse_args()
    monkeypatch.setattr(adapters, 'save_adapter', lambda *a: None)
    adapter = train._save_adapter(SimpleNamespace(unwrap=lambda: object(), lora_target_modules=['q_proj']),
                                  tmp_path, args, step=2560)
    argv = tool.evaluation_argv('index', 'data', tmp_path / 'base', adapter, tmp_path / 'eval', 'val', 32)
    monkeypatch.setattr('sys.argv', ['eval', *argv[3:]])
    parsed = evaluation.parse_args()
    cfg = evaluation._validate_action_adapter(Path(parsed.adapter_dir), tmp_path / 'base')
    assert evaluation._resolve_history_rgb_mode(parsed.history_rgb_mode, cfg) == (mode, 'adapter_config')
    assert evaluation._resolve_action_output_mode(parsed.action_output_mode, cfg)[0] == 'choice'
    monkeypatch.setattr('sys.argv', ['eval', *argv[3:], '--history-rgb-mode', mode])
    with pytest.raises(ValueError, match='only for base-Qwen'):
        evaluation.parse_args()


def test_progress_replaces_only_mutable_state_and_preserves_error(tmp_path):
    tool.write_json(tmp_path / 'admission.json', {'immutable': True})
    for state in ({'status': 'training'}, {'status': 'evaluating'},
                  {'status': 'blocked', 'error': 'reload failed'}):
        tool.update_state(tmp_path, state)
        assert json.loads((tmp_path / 'epoch_result.json').read_text()) == state
    with pytest.raises(FileExistsError):
        tool.write_json(tmp_path / 'admission.json', {})
    assert list(tmp_path.glob('.epoch_state_*')) == []


@pytest.fixture
def reuse_fixture(tmp_path, monkeypatch):
    from qwen3vl_local.audit_joint.server_pilot import package
    from qwen3vl_local.sft_new_loop_phase4.production_storage import binding
    from qwen3vl_local.sft_new_loop_phase3.train import CONTEXT_IDS
    prior = tmp_path / 'prior'; prior.mkdir()
    train = prior / '2rgb_short/train'; (train / 'final').mkdir(parents=True)
    weight = train / 'final/adapter_model.safetensors'; weight.write_bytes(b'fixture weight')
    (train / 'balance').mkdir()
    consumed = dict(status='verified', actual={k: 1024 for k in CONTEXT_IDS})
    (train / 'balance/consumed_epoch_0001.json').write_text(json.dumps(consumed))
    evidence, accepted = {'bindings': {}}, {'modes': {'choice': {'plan': 'fixture'}}}
    members = {'admission.json': {'prepared': evidence, 'sampling': accepted},
        'source_check.json': {'files': []}, 'validation_input_bindings.json': [],
        '2rgb_short/adapter_bindings.json': [binding(weight)],
        '2rgb_short/train/balance/consumed_epoch_0001.json': consumed}
    package(prior, members)
    (prior / 'pilot_report.zip').rename(prior / 'phase3_epoch_report.zip')
    (prior / 'pilot_package.json').rename(prior / 'phase3_epoch_package.json')
    monkeypatch.setattr(tool, 'REVIEWED_EPOCH_SHA', tool.file_sha(prior / 'phase3_epoch_report.zip'))
    return prior, evidence, accepted


def test_completed_adapter_reuse_is_read_only_and_bound(reuse_fixture):
    prior, evidence, accepted = reuse_fixture
    before = {str(p): tool.file_sha(p) for p in prior.rglob('*') if p.is_file()}
    reuse = tool.verified_reuse(prior, evidence, accepted)
    assert reuse['status'] == 'verified'
    assert reuse['train_path'] == str(prior / '2rgb_short/train')
    assert before == {str(p): tool.file_sha(p) for p in prior.rglob('*') if p.is_file()}


@pytest.mark.parametrize('defect', ['weight', 'asset_set', 'report', 'prepare', 'zip', 'symlink'])
def test_reuse_rejects_changed_evidence(reuse_fixture, defect):
    prior, evidence, accepted = reuse_fixture
    weight = prior / '2rgb_short/train/final/adapter_model.safetensors'
    if defect == 'weight':
        weight.write_bytes(b'changed')
    elif defect == 'asset_set':
        (weight.parent / 'extra.json').write_text('{}')
    elif defect == 'report':
        (prior / '2rgb_short/train/balance/consumed_epoch_0001.json').write_text('{}')
    elif defect == 'prepare':
        evidence['changed'] = True
    elif defect == 'zip':
        (prior / 'phase3_epoch_report.zip').write_bytes(b'changed')
    elif defect == 'symlink':
        target = prior / 'identical'; target.write_bytes(weight.read_bytes())
        weight.unlink(); weight.symlink_to(target)
    with pytest.raises(ValueError):
        tool.verified_reuse(prior, evidence, accepted)


@pytest.mark.parametrize('reuse', [False, True])
@pytest.mark.parametrize('failure', [None, '2rgb_short_reload_val', '4rgb_test'])
def test_coordinator_completes_or_packages_failure_without_retraining_reused_mode(tmp_path, monkeypatch, cases, reuse, failure):
    from qwen3vl_local.audit_joint import server_readiness, server_phase3_prepare, first_epoch_budget, training_budget
    from qwen3vl_local.sft_new_loop_phase3 import train, preflight
    from qwen3vl_local.sft_new_loop_phase3.prompt_candidate import CANDIDATE_NAME
    from qwen3vl_local.qwen35 import adapters
    from qwen3vl_local.sft_new_loop_phase4.production_storage import binding
    index = tmp_path / 'index'; index.write_text('fixture')
    evidence = {'bindings': {'index': binding(index)}}
    plans = {'modes': {'choice': {k: {'plan': {'selected_ids': ['fixture']}} for k in ('val_generation', 'test')}},
             'prompt_sha256': {}}
    monkeypatch.setattr(server_readiness, 'source_check', lambda *a: {'status': 'matched'})
    monkeypatch.setattr(tool, 'verified_preparation', lambda *a: (evidence, plans))
    monkeypatch.setattr(server_phase3_prepare, 'sampling_plans', lambda *a: plans)
    monkeypatch.setattr(preflight, 'check_index', lambda *a, **kw: None)
    monkeypatch.setattr(train, '_read_rows', lambda *a, **kw: [])
    monkeypatch.setattr(train, '_validation_work', lambda *a, **kw: ([], {}))
    monkeypatch.setattr(first_epoch_budget, 'profile', lambda *a: {})
    monkeypatch.setattr(first_epoch_budget, 'request', lambda *a, **kw: {'components': {'phase3_index': {}}})
    monkeypatch.setattr(training_budget, 'plan', lambda v: {'sufficient': True})
    monkeypatch.setattr(adapters, 'save_adapter', lambda *a: None)
    monkeypatch.setattr(tool, 'training_env', lambda i,d,m,o,v: {'OUTPUT_DIR': str(o), 'HISTORY_RGB_MODE': v})
    trained, evaluated = [], []

    def save_train(path, mode):
        path.mkdir(parents=True)
        monkeypatch.setattr('sys.argv', ['train', '--model-dir', str(tmp_path / 'base'),
            '--history-rgb-mode', mode, '--action-output-mode', 'choice', '--prompt-variant', CANDIDATE_NAME])
        args = train.parse_args()
        train._save_adapter(SimpleNamespace(unwrap=lambda: object(), lora_target_modules=['q_proj']),
                            path, args, step=2560)
        (path / 'balance').mkdir()
        tool.write_json(path / 'balance/consumed_epoch_0001.json',
                        {'status': 'verified', 'actual': {k: 1024 for k in train.CONTEXT_IDS}})
        for n in ('balance/epoch_0001.json', 'train_run_manifest.json', 'train_balance.json', 'final_generation.json'):
            tool.write_json(path / n, {})
        (path / 'final_generation_val_cases.jsonl').write_bytes(cases[0].read_bytes())

    if reuse:
        prior = tmp_path / 'prior/train'
        save_train(prior, '2rgb_short')
        monkeypatch.setattr(tool, 'verified_reuse', lambda *a: dict(mode='2rgb_short',
            train_path=str(prior), bindings=[], status='verified'))

    def popen(argv, **kw):
        env = kw['env']; trained.append(env['HISTORY_RGB_MODE'])
        save_train(Path(env['OUTPUT_DIR']), env['HISTORY_RGB_MODE'])
        return SimpleNamespace(wait=lambda timeout: 0)
    monkeypatch.setattr(tool.subprocess, 'Popen', popen)

    def execute(output, name, argv):
        from qwen3vl_local.sft_new_loop_phase3 import eval as evaluation
        evaluated.append(name)
        monkeypatch.setattr('sys.argv', ['eval', *argv[3:]])
        args = evaluation.parse_args()  # exercise the real CLI saved-contract resolution
        if name == failure:
            raise RuntimeError('synthetic eval failure ' + name)
        root = Path(args.output_dir); root.mkdir()
        tool.write_json(root / 'evaluation_plan.json', {'plan': {'selected_ids': ['fixture']}})
        tool.write_json(root / 'metrics.json', {})
        (root / 'cases.jsonl').write_bytes(cases[1].read_bytes())
    monkeypatch.setattr(server_phase3_prepare, 'execute', execute)
    output = tmp_path / 'new'; output.mkdir()
    code = tool.run(tmp_path, output, tmp_path, tmp_path / 'base', tmp_path / 'prior' if reuse else None)
    state = json.loads((output / 'epoch_result.json').read_text())
    assert code == (2 if failure else 0)
    assert state['status'] == ('blocked' if failure else 'first_epoch_verified')
    assert trained == (([] if reuse else ['2rgb_short']) + ([] if failure == '2rgb_short_reload_val' else ['4rgb']))
    if failure:
        assert failure in state['error']
    else:
        assert evaluated == ['2rgb_short_reload_val', '2rgb_short_test', '4rgb_reload_val', '4rgb_test']
        assert state['completed_modes'] == list(tool.MODES)
    assert tool.package(output, code) == code
