"""Launcher orchestration only; model generation is an explicit test double."""
import json
from pathlib import Path
import sys
import pytest
from qwen3vl_local.audit_joint.phase1_visual_refine_v2 import remote as r, experiment as e


def setup(tmp_path, monkeypatch, candidate, layout):
    app = tmp_path/layout/'AutoMoT'
    app.mkdir(parents=True)
    assets = tmp_path/'assets'
    assets.mkdir()
    if layout == 'cr02':
        (app/'checkpoints').symlink_to(assets, target_is_directory=True)
    else:
        (app/'checkpoints').mkdir()
    model = app/'checkpoints/model'
    model.mkdir()
    (model/'model.safetensors.index.json').write_text('{}')
    for mode in r.MODES:
        adapter = app/f'checkpoints/run_{mode}/best_generation'
        adapter.mkdir(parents=True)
        (adapter/'phase12_adapter_identity.json').write_text('{}')
        folder = app/f'checkpoints/pipeline/{mode}/eval_review/lora_production'
        folder.mkdir(parents=True)
        (folder/'evaluation_inputs.json').write_text(json.dumps(dict(
            phase=1, history_rgb_mode=mode,
            arguments={'adapter_dir':f'/other/server/AutoMoT/checkpoints/run_{mode}/best_generation'})))
    discover = r.discover
    monkeypatch.setattr(r, 'discover', lambda pipeline, mode: discover(pipeline, mode, app=app))
    monkeypatch.setattr(r, 'APP', app)
    monkeypatch.setattr(r, 'admit', lambda *a, **kw: (app/'lead_data', {'test_double': True}))
    monkeypatch.setattr(r, 'choose_gpu', lambda: '0')
    monkeypatch.setenv('GPU_IDS', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setattr(sys, 'argv', ['remote', 'all', '--candidate', candidate,
        '--pipeline-root', str(app/'checkpoints/pipeline'), '--model-dir', str(model),
        '--output-root', str(app/'checkpoints/outputs')])
    return app


@pytest.mark.parametrize('layout', ['cr02', 'dsw'])
@pytest.mark.parametrize('candidate', ['rs2_same_direction', 'rs4_signal_head'])
def test_both_modes_prepared_before_replay_and_quality_failure_preserved(tmp_path, monkeypatch, layout, candidate):
    app = setup(tmp_path, monkeypatch, candidate, layout)
    prepared, replayed, packaged = [], [], []
    def prepare(inputs, data_root, reference, selected):
        assert selected == candidate and reference == inputs.parent
        prepared.append(inputs.parents[2].name)
    def replay(inputs, data_root, model, adapter, out, reference, selected):
        assert prepared == list(r.MODES)
        assert selected == candidate and reference == inputs.parent
        assert adapter.is_relative_to(app)
        replayed.append(out.name)
        out.mkdir()
        return {'status': 'review_required' if len(replayed) == 1 else 'no_observed_regression'}
    def package(out, inputs):
        packaged.append(out.name)
        return out.with_suffix('.tar.gz')
    monkeypatch.setattr(e, 'prepare', prepare)
    monkeypatch.setattr(e, 'replay', replay)
    monkeypatch.setattr(r, 'package', package)
    with pytest.raises(SystemExit) as exc:
        r.main()
    assert exc.value.code == 2
    assert replayed == packaged == list(r.MODES)
    status = json.loads(next((app/'checkpoints/outputs').glob('*/status.json')).read_text())
    assert status['status'] == 'review_required'
    assert not status['training_started'] and not status['production_default_changed']


@pytest.mark.parametrize('stage', ['rgb', 'reference'])
def test_second_mode_admission_failure_prevents_any_replay(tmp_path, monkeypatch, stage):
    app = setup(tmp_path, monkeypatch, 'rs2_same_direction', 'cr02')
    def check(inputs, *args, **kwargs):
        if inputs.parents[2].name == '4rgb':
            raise ValueError('synthetic second-mode admission failure')
        return app/'lead_data', {'test_double': True}
    monkeypatch.setattr(e, 'replay', lambda *a, **kw: pytest.fail('No model replay before all admissions'))
    if stage == 'rgb':
        monkeypatch.setattr(r, 'admit', check)
        monkeypatch.setattr(r, 'choose_gpu', lambda: pytest.fail('No GPU before RGB admission'))
    else:
        monkeypatch.setattr(e, 'prepare', check)
    with pytest.raises(ValueError, match='second-mode'):
        r.main()
    assert not (app/'checkpoints/outputs').exists()
