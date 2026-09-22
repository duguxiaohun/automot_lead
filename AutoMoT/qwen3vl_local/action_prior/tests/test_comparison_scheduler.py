"""真实CPU子进程测试GPU队列的调度/环境隔离/失败回收；不宣称GPU推理验收。"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_prior import comparison_scheduler as cs
from qwen3vl_local.action_prior.comparison_cases import read_json


@pytest.mark.parametrize("requested,models,expected", [(4, 6, ["2", "0", "3", "1"]), (4, 2, ["2", "0", "3", "1"]), (8, 6, ["2", "0", "3", "1"]), (1, 3, ["2"])])
def test_auto_selection_idle_order_and_count(monkeypatch, requested, models, expected):
    monkeypatch.delenv("GPU_IDS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "77")
    monkeypatch.setattr(cs.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="0, 100, 0\n1, 900, 0\n2, 0, 0\n3, 100, 50\n"))
    plan = cs.select_gpus(requested, models)
    assert plan["selected_ids"] == expected and plan["parallel_models"] == min(models, len(expected))
    assert plan['parallel_workers'] == len(expected)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "77"  # parent untouched


def test_gpu_ids_explicit_overrides_requested_and_skips_probe(monkeypatch):
    monkeypatch.setenv("GPU_IDS", "7, 3, 2")
    monkeypatch.setattr(cs.subprocess, "run", lambda *a, **k: pytest.fail("explicit pin must skip nvidia-smi"))
    plan = cs.select_gpus(1, 4)
    assert plan["selected_ids"] == ["7", "3", "2"] and plan["source"] == "GPU_IDS"


@pytest.mark.parametrize("ids", ["0,0", "0,00", "0,", "-1", "GPU-abc", "1 2"])
def test_invalid_gpu_pin_rejected(monkeypatch, ids):
    monkeypatch.setenv("GPU_IDS", ids)
    with pytest.raises(ValueError, match="GPU_IDS"):
        cs.select_gpus()


def test_no_gpus_fail(monkeypatch):
    monkeypatch.delenv("GPU_IDS", raising=False)
    monkeypatch.setattr(cs.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    with pytest.raises(ValueError, match="未检测到GPU"):
        cs.select_gpus()


def test_worker_environment_isolated(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("GPU_IDS", "0,1,2,3")
    env = cs.worker_environment("2")
    assert env["CUDA_VISIBLE_DEVICES"] == env["GPU_IDS"] == "2"
    assert env["ACTION_PRIOR_GPU_READY"] == "1"
    assert "WORLD_SIZE" not in env and "LOCAL_RANK" not in env
    assert os.environ["WORLD_SIZE"] == "4"


def mock_commands(tmp_path, durations):
    worker = tmp_path / "worker.py"
    worker.write_text('''import json,os,sys,time
from pathlib import Path
out=Path(sys.argv[1]); delay=float(sys.argv[2])
start=time.monotonic()
time.sleep(delay)
out.write_text(json.dumps(dict(start=start,end=time.monotonic(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],pin=os.environ['GPU_IDS'],ready=os.environ['ACTION_PRIOR_GPU_READY'],world=os.environ.get('WORLD_SIZE'))))
print('finished test worker',flush=True)
''')
    return [[sys.executable, str(worker), str(tmp_path / f"result_{i}.json"), str(delay)] for i, delay in enumerate(durations)]


@pytest.mark.parametrize("gpus,tasks", [(1, 3), (2, 5), (4, 2), (4, 4)])
def test_queue_real_subprocess_no_gpu_overlap(tmp_path, gpus, tasks):
    commands = mock_commands(tmp_path, [.04] * tasks)
    gpu_plan = dict(selected_ids=[str(i) for i in range(gpus)], parallel_models=min(gpus, tasks))
    old_handler = signal.getsignal(signal.SIGTERM)
    records = cs.run_queue(commands, gpu_plan, tmp_path, poll_interval=.005)
    values = [read_json(tmp_path / f"result_{i}.json") for i in range(tasks)]
    for value in values:
        assert value["gpu"] == value["pin"] and value["ready"] == "1" and value["world"] is None
    for i, left in enumerate(values):
        for right in values[i+1:]:
            if left["gpu"] == right["gpu"]:
                assert left["end"] <= right["start"] or right["end"] <= left["start"]
    assert len({v["gpu"] for v in values}) == min(gpus, tasks)
    assert all(record["status"] == "complete" for record in records)
    assert read_json(tmp_path / "status.json")["completed"] == tasks
    assert signal.getsignal(signal.SIGTERM) == old_handler
    assert len(list((tmp_path / "logs").glob("*.log"))) == tasks


def test_fast_gpu_takes_next_job_before_slow_gpu_finishes(tmp_path):
    commands = mock_commands(tmp_path, [.7, .04, .04])
    cs.run_queue(commands, dict(selected_ids=["0", "1"]), tmp_path, poll_interval=.005)
    slow, fast, next_job = [read_json(tmp_path / f"result_{i}.json") for i in range(3)]
    assert fast["gpu"] == next_job["gpu"] == "1"
    assert next_job["start"] < slow["end"]


def test_failure_kills_other_worker_and_cancels_pending(tmp_path):
    sleeping = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(20)"
    failing = "import time; time.sleep(.08); raise SystemExit(7)"
    commands = [[sys.executable, "-c", source] for source in (sleeping, failing, "raise SystemExit(0)")]
    with pytest.raises(RuntimeError, match="退出码7"):
        cs.run_queue(commands, dict(selected_ids=["0", "1"]), tmp_path, poll_interval=.005, stop_timeout=.05)
    records = read_json(tmp_path / "scheduler.json")["jobs"]
    assert [r["status"] for r in records] == ["cancelled", "failed", "cancelled"]
    assert "pid" not in records[2]
    for record in records[:2]:
        with pytest.raises(ProcessLookupError):
            os.kill(record["pid"], 0)


def test_failed_spawn_records_failure(tmp_path):
    with pytest.raises(FileNotFoundError):
        cs.run_queue([[str(tmp_path / "does_not_exist")]], dict(selected_ids=["0"]), tmp_path)
    assert read_json(tmp_path / "scheduler.json")["jobs"][0]["status"] == "failed"


def test_sigterm_parent_reaps_running_workers(tmp_path):
    # worker自行通知父进程SIGTERM，验证真实信号路径而非只调用清理函数。
    worker = "import os,signal,time; time.sleep(.1); os.kill(os.getppid(),signal.SIGTERM); time.sleep(20)"
    code = f"from qwen3vl_local.action_prior.comparison_scheduler import run_queue; run_queue({[[sys.executable, '-c', worker]]!r}, {{'selected_ids':['0']}}, {str(tmp_path)!r}, poll_interval=.005, stop_timeout=.05)"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3])}
    run = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=5)
    assert run.returncode == 143, run.stderr
    record = read_json(tmp_path / "scheduler.json")["jobs"][0]
    assert record["status"] == "cancelled"
    with pytest.raises(ProcessLookupError):
        os.kill(record["pid"], 0)


@pytest.mark.parametrize("plan_only", [False, True])
def test_main_output_sibling_and_queue_integration(tmp_path, monkeypatch, plan_only):
    from qwen3vl_local.action_prior import compare_checkpoints as main
    from qwen3vl_local.action_prior import comparison_render as render
    from qwen3vl_local.action_prior import comparison_shards as shards
    automot = tmp_path / "AutoMoT"
    automot.mkdir()
    (automot / "checkpoints").mkdir()
    for name in ("run_a", "run_b"):
        (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "AUTOMOT_ROOT", automot)
    monkeypatch.setattr(main, "prepare", lambda cli, out: ({"models": []}, [{}, {}]))
    def plan_jobs(jobs, manifest, plan, out):
        assert len(jobs) == 2
        return [dict(job=str(out / '_plan' / f'worker_{i}.json')) for i in range(4)]
    monkeypatch.setattr(shards, 'plan_shards', plan_jobs)
    calls = []
    def select(requested, count):
        assert not plan_only and requested == 4 and count == 2
        return dict(selected_ids=["2", "3", "0", "1"], parallel_models=2)
    def queue(commands, plan, out):
        assert len(commands) == 4 and all("--worker-job" in c for c in commands)
        assert plan["selected_ids"] == ["2", "3", "0", "1"]
        calls.append(out)
    monkeypatch.setattr(cs, "select_gpus", select)
    monkeypatch.setattr(cs, "run_queue", queue)
    monkeypatch.setattr(render, "publish", lambda out, manifest: calls.append(out))
    monkeypatch.setattr(sys, "argv", ["compare_checkpoints.py", "run_a", "run_b", *(["--plan-only"] if plan_only else [])])
    main.main()
    results = list((automot / "test").glob("run_*"))
    assert len(results) == 1 and not list((automot / "checkpoints").iterdir())
    assert read_json(results[0] / "status.json")["status"] == ("planned_only" if plan_only else "complete")
    assert len(calls) == (0 if plan_only else 2)
