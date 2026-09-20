"""中途审计ZIP的原子替换、恢复过滤和体积边界测试，不加载模型。"""
from pathlib import Path
import json
from types import SimpleNamespace
import zipfile
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_prior import training_audit


def publish(root, step=3, **kwargs):
    args = dict(step=step, cursor=dict(epoch=1, micro=0), best=1.0, plan={},
                args=SimpleNamespace(optimizer='muon_adamw'), train_metrics=dict(samples=5, loss=2),
                validation=dict(samples=2, route_ade_m=1), performance={}, reason='validation_complete')
    args.update(kwargs)
    return training_audit.publish(root, **args)


def test_pack_failure_preserves_previous_valid_zip(tmp_path, monkeypatch):
    archive = publish(tmp_path)
    previous = archive.read_bytes()
    original = zipfile.ZipFile.writestr
    def interrupted(self, name, data, *args, **kwargs):
        original(self, name, data, *args, **kwargs)
        raise RuntimeError('simulated interrupted zip write')
    monkeypatch.setattr(zipfile.ZipFile, 'writestr', interrupted)
    with pytest.raises(RuntimeError, match='interrupted zip'):
        publish(tmp_path, step=6, cursor=dict(epoch=2, micro=0))
    assert archive.read_bytes() == previous
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert json.loads(z.read('metrics.json'))['latest']['optimizer_step'] == 3


def test_rollback_excludes_future_records_and_never_reads_weights(tmp_path):
    publish(tmp_path, step=6, cursor=dict(epoch=2, micro=0))
    training_audit.record_window(tmp_path, 2, {'loss': 2})
    training_audit.record_window(tmp_path, 6, {'loss': 1})
    validation = tmp_path / 'validation'
    validation.mkdir()
    (validation / 'epoch_002_step00000006.json').write_text('{"future": true}')
    (tmp_path / 'latest.pt').write_bytes(b'private weights')
    (tmp_path / 'train.log').write_text('not included raw log')
    archive = publish(tmp_path, step=3)
    with zipfile.ZipFile(archive) as z:
        state = json.loads(z.read('metrics.json'))
        assert [e['epoch'] for e in state['epoch_history']] == [1]
        assert 'audit/step_00000002.json' in z.namelist()
        assert 'audit/step_00000006.json' not in z.namelist()
        assert not any(name.endswith(('.pt', '.log')) for name in z.namelist())
        assert not any('step00000006' in name for name in z.namelist())


def test_termination_after_completed_validation_keeps_same_step_result(tmp_path):
    publish(tmp_path)
    archive = publish(tmp_path, validation=None, train_metrics={}, reason='terminated')
    with zipfile.ZipFile(archive) as z:
        latest = json.loads(z.read('metrics.json'))['latest']
        assert latest['validation']['samples'] == 2 and latest['train_metrics']['samples'] == 5
        assert not latest['validation_pending']
