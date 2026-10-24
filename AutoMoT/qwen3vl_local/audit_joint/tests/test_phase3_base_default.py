"""Synthetic artifact regressions; no model inference or GPU quality claims."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest
from qwen3vl_local.audit_joint import phase3_base_default as default
from qwen3vl_local.audit_joint import phase3_compare


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def pack(run, config, config_path, monkeypatch):
    archive = run / config['archive']
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'w') as bundle:
        for path in sorted(run.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl') and 'final' not in path.parts:
                bundle.write(path, str(path.relative_to(run)))
        bundle.writestr('train.log', b'truncated log tail')
    config['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    write(config_path, config)
    monkeypatch.setattr(default, 'CONFIG', config_path)
    return archive


@pytest.fixture
def selected(tmp_path, monkeypatch):
    config = json.loads(default.CONFIG.read_text())
    run = tmp_path / config['run']
    for name in default.REQUIRED: write(run / name, {})
    write(run / 'experiment/train/train_run_manifest.json', {
        k: config[k] for k in ('prompt_variant', 'history_rgb_mode', 'action_output_mode', 'production_prompt_sha256')})
    write(run / config['adapter'] / 'sft_new_loop_phase3_adapter_config.json', {
        k: config[k] for k in ('prompt_variant', 'history_rgb_mode', 'action_output_mode')})
    pack(run, config, tmp_path / 'selection.json', monkeypatch)
    monkeypatch.setattr(phase3_compare, 'verify_completion', lambda *args: {})
    return tmp_path, run, config


def test_selection_is_previous_base_without_creating_artifacts(tmp_path):
    chosen = default.selection(tmp_path)
    assert chosen['metrics']['val_correct'] == 251 and chosen['metrics']['test_correct'] == 267
    assert chosen['refinement_status'] == 'paused_by_user'
    assert not chosen['adapter_present'] and not chosen['weights_verified']
    assert not list(tmp_path.iterdir())
    assert Path(chosen['run_path']) == phase3_compare.root_for(tmp_path, 'base') / 'run_base_bounded_v3'


def test_latest_is_ignored_and_matching_evidence_succeeds(selected, monkeypatch, capsys):
    app, run, config = selected
    (run.parent / 'latest').symlink_to('run_base_new_20261024')
    (run / 'train.log').write_text('full log, unlike exported truncated tail')
    seen = []
    def verify(app, method):
        seen.append((app, method)); print('native diagnostic')
    monkeypatch.setattr(phase3_compare, 'verify_completion', verify)
    assert default.verified_adapter(app) == run / config['adapter']
    assert seen == [(app, 'base')]
    captured = capsys.readouterr()
    assert not captured.out and 'native diagnostic' in captured.err


@pytest.mark.parametrize('name', ['experiment/job.json', 'experiment/train/train_run_manifest.json',
    'experiment/train/final_generation_val_cases.jsonl', 'experiment/test/cases_rank0.jsonl'])
def test_old_archive_with_replaced_run_is_rejected(selected, monkeypatch, name):
    app, run, _ = selected
    (run / name).write_text('{"replacement":true}')
    monkeypatch.setattr(phase3_compare, 'verify_completion', lambda *a: pytest.fail('must reject before native verifier'))
    with pytest.raises(ValueError, match='run_evidence_mismatch'): default.verified_adapter(app)


def test_wrong_package_rejected(selected):
    app, run, config = selected
    (run / config['archive']).write_bytes(b'newer package')
    with pytest.raises(ValueError, match='archive_mismatch'): default.verified_adapter(app)


@pytest.mark.parametrize('field,value', [('method', 'single'), ('prompt_variant', 'single_candidate'),
    ('history_rgb_mode', '1rgb'), ('automatic_promotion', True), ('refinement_status', 'active')])
def test_selection_cannot_silently_change_method_or_policy(tmp_path, monkeypatch, field, value):
    config = json.loads(default.CONFIG.read_text()); config[field] = value
    path = tmp_path / 'wrong.json'; write(path, config)
    monkeypatch.setattr(default, 'CONFIG', path)
    with pytest.raises(ValueError): default.selection(tmp_path)


def test_single_adapter_cannot_be_used_as_base(selected):
    app, run, config = selected
    write(run / config['adapter'] / 'sft_new_loop_phase3_adapter_config.json', dict(history_rgb_mode='1rgb'))
    with pytest.raises(ValueError, match='adapter_semantics_mismatch'): default.verified_adapter(app)


def test_native_verification_failure_is_not_swallowed(selected, monkeypatch):
    def reject(*args): raise ValueError('adapter files changed')
    monkeypatch.setattr(phase3_compare, 'verify_completion', reject)
    with pytest.raises(ValueError, match='adapter files changed'): default.verified_adapter(selected[0])


def test_changes_during_native_verification_are_rejected(selected, monkeypatch):
    app, run, _ = selected
    def mutate(*args): (run / 'experiment/job.json').write_text('{} ')
    monkeypatch.setattr(phase3_compare, 'verify_completion', mutate)
    with pytest.raises(ValueError, match='run_evidence_mismatch'): default.verified_adapter(app)


def test_missing_assets_never_fall_back_to_latest(tmp_path):
    with pytest.raises(FileNotFoundError): default.verified_adapter(tmp_path)


def test_status_cli_has_no_single_import_or_file_creation(tmp_path):
    code = ('import sys; from pathlib import Path; '
        'from qwen3vl_local.audit_joint import phase3_base_default as d; '
        'd.selection(Path(sys.argv[1])); '
        'assert not any("phase3_single" in m for m in sys.modules)')
    subprocess.run([sys.executable, '-B', '-c', code, str(tmp_path)],
        env=dict(os.environ, PYTHONPATH=str(default.APP), PYTHONDONTWRITEBYTECODE='1'), check=True)
    assert not list(tmp_path.iterdir())


def test_shell_status_from_unrelated_cwd(tmp_path):
    result = subprocess.run(['bash', str(default.CONFIG.with_suffix('.sh')), 'status'],
        cwd=tmp_path, env=dict(os.environ, PYTHON=sys.executable), capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['selection'] == 'previous_bounded_v3'


from qwen3vl_local.audit_joint.tests.test_phase3_compare_completion import completed, completed_template


def test_resolver_with_real_native_completion_check(completed, monkeypatch, tmp_path):
    # Full native CPU verification of synthetic 9600-step records and weight bytes.
    app, run, _ = completed
    config = json.loads(default.CONFIG.read_text())
    manifest = run / 'experiment/train/train_run_manifest.json'
    value = json.loads(manifest.read_text())
    value.update({k: config[k] for k in ('prompt_variant', 'history_rgb_mode', 'action_output_mode', 'production_prompt_sha256')})
    write(manifest, value)
    adapter = run / config['adapter']; cfg = adapter / 'sft_new_loop_phase3_adapter_config.json'
    value = json.loads(cfg.read_text())
    value.update({k: config[k] for k in ('prompt_variant', 'history_rgb_mode', 'action_output_mode')})
    write(cfg, value)
    from qwen3vl_local.audit_joint.phase3_base_coverage import adapter_files
    receipt = adapter / 'coverage_receipt.json'
    value = json.loads(receipt.read_text()); value['files'] = adapter_files(adapter); write(receipt, value)
    for name in default.REQUIRED:
        if not (run / name).exists(): write(run / name, {})
    pack(run, config, tmp_path / 'selection.json', monkeypatch)
    assert default.verified_adapter(app) == adapter
    (adapter / 'adapter_model.safetensors').write_bytes(b'corrupted weight')
    with pytest.raises(subprocess.CalledProcessError): default.verified_adapter(app)
