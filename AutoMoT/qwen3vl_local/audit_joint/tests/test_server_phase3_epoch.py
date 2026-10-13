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
