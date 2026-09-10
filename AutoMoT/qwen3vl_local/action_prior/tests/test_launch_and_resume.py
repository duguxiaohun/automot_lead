"""启动参数、恢复和累积边界检查，不调用 GPU、torchrun 或实际训练。"""

import json
import signal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from types import SimpleNamespace
import pytest
from qwen3vl_local.action_prior import launch, resume
from qwen3vl_local.action_prior.train import accumulation_state


@pytest.fixture
def launch_env(monkeypatch):
    for name in (
        "GPU_IDS",
        "ACTION_PRIOR_GPU_READY",
        "ACTION_PRIOR_RUN_READY",
        "RESUME",
        "NO_RUN_SUBDIR",
        "DDP_GPU_COUNT",
        "NPROC_PER_NODE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GPU_IDS", "2,3")
    monkeypatch.setenv("RUN_TAG", "test")


def test_launch_does_not_overwrite(tmp_path, monkeypatch, launch_env):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(sys, "argv", ["launch", "train", "--learning-rate", ".0001"])
    commands = []
    monkeypatch.setattr(launch, "run_logged", lambda c, path: commands.append(c))
    launch.main()
    assert (tmp_path / "runs/latest").resolve() == tmp_path / "runs/run_test"
    assert "--nproc_per_node=2" in commands[0]
    with pytest.raises(FileExistsError):
        launch.main()
    assert len(commands) == 1


def test_auto_gpu_sort_and_mask_override(monkeypatch):
    monkeypatch.delenv("GPU_IDS", raising=False)
    monkeypatch.delenv("ACTION_PRIOR_GPU_READY", raising=False)
    monkeypatch.delenv("ACTION_PRIOR_RUN_READY", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "99")
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout="0, 5000, 40\n1, 30, 2\n2, 30, 0\n"),
    )
    assert launch.ensure_gpu(2) == 2
    assert launch.os.environ["CUDA_VISIBLE_DEVICES"] == "2,1"


def test_probe_does_not_overwrite_full_eval(tmp_path, monkeypatch, launch_env):
    monkeypatch.setattr(
        sys, "argv", ["launch", "probe", "--checkpoint", str(tmp_path / "best.pt")]
    )
    commands = []
    monkeypatch.setattr(launch, "run_logged", lambda c, path: commands.append(c))
    monkeypatch.setattr(launch.subprocess, "run", lambda c, **kw: commands.append(c))
    launch.main()
    assert str(tmp_path / "probe_test") in commands[0]
    assert commands[1][-1] == str(tmp_path / "probe_test/cases")


def test_resume_recovers_actual_config_and_selected_priors(
    tmp_path, monkeypatch, launch_env
):
    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                learning_rate=0.0003,
                grad_accum_steps=5,
                data_dir="index_original",
                phase1_adapter="",
                phase2_adapter="",
                selection_policy="available",
                selection_manifest="/unavailable/original/selection.json",
                selection_output="/unavailable/original/selection.json",
                checkpoint_roots=["/shared/server a", "/shared/server b"],
                use_bev=True,
            )
        )
    )
    (tmp_path / "training_plan.json").write_text(json.dumps({"world_size": 2}))
    (tmp_path / "selected_priors.json").write_text(
        json.dumps(
            {
                "phase1": {"path": "one/best_generation"},
                "phase2": {"path": "two/best_generation"},
            }
        )
    )
    commands = []
    monkeypatch.setattr(resume.subprocess, "run", lambda c, **kw: commands.append(c))
    monkeypatch.setattr(sys, "argv", ["resume", str(tmp_path / "latest.pt")])
    resume.main()
    command = commands[0]
    assert command[command.index("--learning-rate") + 1] == "0.0003"
    assert command[command.index("--grad-accum-steps") + 1] == "5"
    assert command[command.index("--phase1-adapter") + 1] == "one/best_generation"
    assert "--output-dir" not in command
    assert "--selection-manifest" not in command and "--selection-output" not in command
    assert command[command.index("--checkpoint-roots") + 1:command.index("--checkpoint-roots") + 3] == [
        "/shared/server a", "/shared/server b"]
    assert command[command.index("--selection-policy") + 1] == "available"


def test_tail_accumulation_keeps_mean_scale():
    assert [accumulation_state(i, 10, 4) for i in range(10)] == [
        (4, False),
        (4, False),
        (4, False),
        (4, True),
        (4, False),
        (4, False),
        (4, False),
        (4, True),
        (2, False),
        (2, True),
    ]
    assert sum(1 / accumulation_state(i, 10, 4)[0] for i in (8, 9)) == 1


def test_launcher_signal_targets_torchrun_workers(monkeypatch):
    """多卡时只先通知 worker，让 torchrun 留在原位等待它们完成安全保存。"""
    original_read_text = Path.read_text

    def fake_read_text(path, *args, **kwargs):
        if str(path) == "/proc/123/task/123/children":
            return "201 202"
        return original_read_text(path, *args, **kwargs)

    sent = []
    monkeypatch.setattr(Path, "read_text", fake_read_text)
    monkeypatch.setattr(launch.os, "kill", lambda pid, signum: sent.append((pid, signum)))
    targets = launch._signal_training_process(
        SimpleNamespace(pid=123), signal.SIGTERM, torchrun=True
    )
    assert targets == [201, 202]
    assert sent == [(201, signal.SIGTERM), (202, signal.SIGTERM)]
    sent.clear()
    targets = launch._signal_training_process(SimpleNamespace(pid=123), signal.SIGINT)
    assert targets == [123]
    assert sent == [(123, signal.SIGINT)]


def test_run_logged_appends_both_streams_and_propagates_failure(tmp_path, capsys):
    import subprocess
    path = tmp_path / "run with spaces" / "train.log"
    launch.run_logged([sys.executable, "-c", "print('first run')"], path)
    with pytest.raises(subprocess.CalledProcessError) as error:
        launch.run_logged([sys.executable, "-c",
                           "import sys; print('second run'); print('failure detail', file=sys.stderr); sys.exit(7)"], path)
    assert error.value.returncode == 7
    saved = path.read_text()
    assert "first run" in saved and "second run" in saved and "failure detail" in saved
    assert "[exit] code=7" in saved
    console = capsys.readouterr().out
    assert "second run" in console and "failure detail" in console


def test_run_logged_forwards_sigterm_and_returns_signal_code(tmp_path):
    """launcher 自身收到 SIGTERM 时通知单卡训练子进程，并阻止后续流水线。"""
    import subprocess

    code = """
import os
import signal
import sys
import time

def stop(_signum, _frame):
    print("child received SIGTERM", flush=True)
    sys.exit(0)

signal.signal(signal.SIGTERM, stop)
os.kill(os.getppid(), signal.SIGTERM)
while True:
    time.sleep(0.01)
"""
    log = tmp_path / "signal.log"
    with pytest.raises(subprocess.CalledProcessError) as caught:
        launch.run_logged([sys.executable, "-c", code], log)
    assert caught.value.returncode == 128 + signal.SIGTERM
    saved = log.read_text(encoding="utf-8")
    assert "child received SIGTERM" in saved
    assert "[launcher] forwarded SIGTERM" in saved
    assert "[exit] code=143" in saved
