"""启动参数、恢复和累积边界检查，不调用 GPU、torchrun 或实际训练。"""

import json
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
    monkeypatch.setattr(launch.subprocess, "run", lambda c, **kw: commands.append(c))
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
