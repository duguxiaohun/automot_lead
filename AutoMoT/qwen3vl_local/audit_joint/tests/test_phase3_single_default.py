"""Default routing and artifact rejection tests; native GPU execution is not exercised."""
import json
from pathlib import Path
import pytest
from qwen3vl_local.audit_joint import phase3_single_default as default
from qwen3vl_local.audit_joint import phase3_compare as compare


def test_canonical_single_routes_all_stages_to_v3(tmp_path):
    selected = default.selection(tmp_path)
    assert Path(selected['run_path']) == compare.root_for(tmp_path, 'single') / 'runs/six_1280'
    assert compare.package('single') == 'phase3_single_v3'
    assert compare.commands(tmp_path, 'cpu', 'single')[0][3].endswith('phase3_single_v3.cpu')
    assert 'phase3_single_v3/remote.sh' in compare.commands(tmp_path, 'train', 'single')[0][1]
    assert not selected['weights_verified'] and not selected['state_present']
    assert not list(tmp_path.iterdir())


@pytest.fixture
def saved(tmp_path, monkeypatch):
    config = json.loads(default.CONFIG.read_text())
    run = tmp_path / config['run']
    for name in config['evidence_sha256']:
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    state = run / 'epoch_3/state.pt'
    state.write_bytes(b'synthetic old weight')
    config['state_sha256'] = default.sha(state)
    (run / 'epoch_3/receipt.json').write_text(json.dumps({'state_sha256': config['state_sha256']}))
    config['evidence_sha256'] = {name: default.sha(run / name) for name in config['evidence_sha256']}
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(default, 'CONFIG', path)
    return tmp_path, run


def test_pinned_checkpoint_ignores_latest_and_calls_native_verifier(saved, monkeypatch):
    app, run = saved
    (run.parent / 'latest').symlink_to('single_refine')
    calls = []
    monkeypatch.setattr(compare, 'verify_completion', lambda app, method: calls.append((app, method)))
    assert default.verified_checkpoint(app) == run / 'epoch_3'
    assert calls == [(app, 'single')]


@pytest.mark.parametrize('name', ['training_complete.json', 'epoch_3/receipt.json', 'test/cases.json', 'epoch_3/state.pt'])
def test_replaced_results_or_weights_rejected_before_native_verifier(saved, monkeypatch, name):
    app, run = saved
    (run / name).write_bytes(b'replacement')
    monkeypatch.setattr(compare, 'verify_completion', lambda *a: pytest.fail('must reject before native verification'))
    with pytest.raises(ValueError, match='mismatch'):
        default.verified_checkpoint(app)


def test_native_failure_propagates(saved, monkeypatch):
    def fail(*args): raise ValueError('native source mismatch')
    monkeypatch.setattr(compare, 'verify_completion', fail)
    with pytest.raises(ValueError, match='native source mismatch'):
        default.verified_checkpoint(saved[0])


def test_mutation_during_native_verification_rejected(saved, monkeypatch):
    app, run = saved
    monkeypatch.setattr(compare, 'verify_completion', lambda *a: (run / 'epoch_3/state.pt').write_bytes(b'changed'))
    with pytest.raises(ValueError, match='weights_changed'):
        default.verified_checkpoint(app)


@pytest.mark.parametrize('key,value', [('package', 'phase3_single_refine'), ('automatic_promotion', True), ('checkpoint', 'epoch_2')])
def test_wrong_default_rejected(saved, key, value):
    config = json.loads(default.CONFIG.read_text())
    config[key] = value
    default.CONFIG.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='selection_mismatch'):
        default.selection(saved[0])


def test_missing_old_run_does_not_fall_back(tmp_path):
    with pytest.raises(FileNotFoundError):
        default.verified_checkpoint(tmp_path)
