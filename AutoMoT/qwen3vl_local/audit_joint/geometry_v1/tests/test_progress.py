import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from qwen3vl_local.audit_joint.geometry_progress import Progress, install
from qwen3vl_local.audit_joint.geometry_v1 import train
from .test_pipeline import Tiny, packed
from .test_geometry import record


def test_observer_preserves_real_updates_scheduler_rng_and_result(tmp_path, capsys):
    results = []
    for observed in (False, True):
        torch.manual_seed(17)
        model = Tiny()
        bundle = SimpleNamespace(model=model, device=torch.device('cpu'))
        opt = torch.optim.AdamW(model.parameters(), lr=.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1/(i+1))
        before_keys = set(sched.__dict__)
        work = [record(8+i % 9) for i in range(12)]
        if observed:
            p = Progress(tmp_path, 3)
            result = p.epoch_call(train.train_epoch, bundle, work, opt, sched, packed)
        else:
            result = train.train_epoch(bundle, work, opt, sched, packed)
        assert set(sched.__dict__) == before_keys
        result.pop('elapsed_seconds')
        results.append((copy.deepcopy(model.state_dict()), copy.deepcopy(sched.state_dict()), result, torch.get_rng_state()))
    a, b = results
    assert all(torch.equal(a[0][k], b[0][k]) for k in a[0])
    assert a[1] == b[1] and a[2] == b[2] and torch.equal(a[3], b[3])
    messages = [json.loads(line.split('] ', 1)[1]) for line in capsys.readouterr().out.splitlines()]
    assert [m['step'] for m in messages if m['stage'] == 'train'] == [0, 1, 10, 12]
    assert json.loads((tmp_path/'progress.json').read_text())['global_step'] == 12


def test_budget_stop_and_exception_restore_scheduler(tmp_path):
    model = Tiny()
    bundle = SimpleNamespace(model=model, device=torch.device('cpu'))
    opt = torch.optim.AdamW(model.parameters())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
    keys = set(sched.__dict__)
    progress = Progress(tmp_path, 3)
    r = progress.epoch_call(train.train_epoch, bundle, [record()], opt, sched, packed,
                            max_epoch_hours=1e-12, timing_steps=1)
    assert r['status'] == 'budget_stop'
    assert json.loads((tmp_path/'progress.json').read_text())['stage'] == 'budget_stop'
    def bad(*args, **kwargs): raise RuntimeError('failed')
    with pytest.raises(RuntimeError, match='failed'):
        progress.epoch_call(bad, bundle, [record()], opt, sched, packed)
    assert set(sched.__dict__) == keys


def test_stage_observers_preserve_original_calls(tmp_path):
    calls = []
    fake = SimpleNamespace(
        train_epoch=lambda *a, **k: None,
        training_members=lambda a: [1, 2],
        save_checkpoint=lambda *a, **k: calls.append('saved'),
        planning=SimpleNamespace(evaluation=lambda *a: [1, 2, 3]),
        objective=SimpleNamespace(predict=lambda x: x+1))
    def run(args):
        fake.training_members(None)
        fake.save_checkpoint()
        assert [fake.objective.predict(i) for i in (1, 2)] == [2, 3]
        calls.append('trained')
    def evaluate(args):
        for i in fake.planning.evaluation(None): fake.objective.predict(i)
    fake.run = run
    fake.evaluate = evaluate
    install(fake)
    args = SimpleNamespace(output=tmp_path, epochs=3)
    fake.run(args)
    assert calls == ['saved', 'trained']
    assert json.loads((tmp_path/'progress.json').read_text())['stage'] == 'training_complete'
    fake.evaluate(args)
    assert json.loads((tmp_path/'progress.json').read_text())['stage'] == 'evaluation_complete'


def _worker(rank, init, output):
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method=init, rank=rank, world_size=2)
    try:
        torch.manual_seed(11)
        model = torch.nn.parallel.DistributedDataParallel(Tiny())
        bundle = SimpleNamespace(model=model, device=torch.device('cpu'))
        opt = torch.optim.AdamW(model.parameters())
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
        fn = train.train_epoch if rank else lambda *a, **k: Progress(output, 3).epoch_call(train.train_epoch, *a, **k)
        r = fn(bundle, [record(i) for i in range(8, 12)], opt, sched, packed, rank=rank, world=2)
        assert r['steps'] == 2 and sum(r['actual_exposure'].values()) == 4
    finally:
        dist.destroy_process_group()


def test_rank0_observer_with_real_two_rank_loop(tmp_path):
    torch.multiprocessing.spawn(_worker, args=('file://'+str(tmp_path/'init'), str(tmp_path)), nprocs=2, join=True)
    result = json.loads((tmp_path/'progress.json').read_text())
    assert result['global_step'] == 2 and result['stage'] == 'epoch_complete'


@pytest.mark.parametrize('operation', ['plan', 'train', 'evaluate'])
def test_shell_routes_only_model_operations_through_observer(tmp_path, operation):
    fake = tmp_path/'python'
    fake.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    result = subprocess.run(['bash', str(Path(__file__).parents[1]/'run.sh'), operation, '--help'],
        env=dict(os.environ, PYTHON=str(fake), GPU_IDS='0,1,2,3'), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout.splitlines()[-1])
    module = 'qwen3vl_local.audit_joint.geometry_v1' if operation == 'plan' else 'qwen3vl_local.audit_joint.geometry_progress'
    assert args[-3:] == [module, operation, '--help']
