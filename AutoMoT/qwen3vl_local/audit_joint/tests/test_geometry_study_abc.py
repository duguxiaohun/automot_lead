import json
from types import SimpleNamespace
import pytest
from qwen3vl_local.audit_joint import geometry_study_abc as abc


def test_gpu_auto_select_four_and_explicit_order():
    query = lambda *a, **k: '0, 100\n1, 900\n2, 500\n3, 800\n4, 700\n'
    assert abc.selected_gpus(query=query) == '1,3,4,2'
    assert abc.selected_gpus('4,2,0,1', query=query) == '4,2,0,1'


@pytest.mark.parametrize('explicit,rows', [
    (None, '0, 100\n1, 100\n2, 100'),
    ('0,0,1,2', '0, 100\n1, 100\n2, 100\n3, 100'),
    ('0,1,2,7', '0, 100\n1, 100\n2, 100\n3, 100'),
    ('0,1,2', '0, 100\n1, 100\n2, 100\n3, 100'),
])
def test_wrong_gpu_count_duplicate_or_absent_rejected(explicit, rows):
    with pytest.raises(ValueError, match='four distinct'):
        abc.selected_gpus(explicit, query=lambda *a, **k: rows)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_existing_later_arm_blocks_all_before_expensive_checks(tmp_path, monkeypatch):
    (tmp_path/'checkpoints/geometry_study_v1/runs/c_1280').mkdir(parents=True)
    monkeypatch.setattr(abc, 'ready', lambda *a: pytest.fail('check collisions first'))
    with pytest.raises(FileExistsError):
        abc.preflight(tmp_path, 1280, ['a', 'b', 'c'])


def test_admission_failure_never_launches(tmp_path, monkeypatch):
    def rejected(*a): raise ValueError('study_cpu_admission_rejected')
    monkeypatch.setattr(abc, 'ready', rejected)
    with pytest.raises(ValueError, match='admission_rejected'):
        abc.run_training(tmp_path, 1280, ['a'], launch=lambda *a, **k: pytest.fail('no GPU launch'))


def fixture_ready(monkeypatch):
    value = dict(dataset='/verified/dataset.json', dataset_sha256='verified', optimizer_steps=[3200]*3)
    monkeypatch.setattr(abc, 'preflight', lambda *a: (value, '3,2,1,0'))
    return value


def test_sequential_three_arms_completion_and_status(tmp_path, monkeypatch):
    fixture_ready(monkeypatch)
    monkeypatch.setenv('STUDY_RUN_NAME', 'should_not_leak')
    calls = []
    def launch(command, **kwargs):
        variant = command[-1]; calls.append(variant)
        assert kwargs['env']['GPU_IDS'] == '3,2,1,0'
        assert 'STUDY_RUN_NAME' not in kwargs['env']
        root = tmp_path/f'checkpoints/geometry_study_v1/runs/{variant}_1280'
        write(root/'training_complete.json', dict(total_steps=9600, selected_checkpoint='epoch_2',
            binding=dict(contract=dict(variant=variant), dataset_sha256='verified')))
        write(root/'test/metrics.json', {'accuracy': .5})
        write(root/'epoch_1.json', dict(result=dict(steps=3200, loss=.3, elapsed_seconds=60, status='epoch_complete')))
        return SimpleNamespace(returncode=0)
    assert abc.run_training(tmp_path, 1280, ['a', 'b', 'c'], launch=launch) == 0
    assert calls == ['a', 'b', 'c']
    status = abc.status_report(tmp_path, 1280)
    assert status['runs']['b']['operation']['status'] == 'completed'
    assert status['runs']['b']['selected_checkpoint'] == 'epoch_2'
    assert status['runs']['b']['epochs'][0]['loss'] == .3
    assert status['runs']['b']['test_complete']


def test_failure_stops_before_next_arm(tmp_path, monkeypatch):
    fixture_ready(monkeypatch); calls = []
    def launch(command, **kwargs):
        calls.append(command[-1]); return SimpleNamespace(returncode=7)
    assert abc.run_training(tmp_path, 1280, ['a', 'b', 'c'], launch=launch) == 7
    assert calls == ['a']
    assert abc.status_report(tmp_path, 1280)['runs']['a']['operation']['status'] == 'failed'


def test_zero_exit_without_test_receipt_is_not_success(tmp_path, monkeypatch):
    fixture_ready(monkeypatch)
    with pytest.raises(FileNotFoundError):
        abc.run_training(tmp_path, 1280, ['a'], launch=lambda *a, **k: SimpleNamespace(returncode=0))
    assert abc.status_report(tmp_path, 1280)['runs']['a']['operation']['status'] == 'failed'


def test_status_without_dataset_never_builds(tmp_path):
    assert abc.status_report(tmp_path, 1280)['ready'] == {'status': 'not_prepared'}
    assert not list(tmp_path.iterdir())
