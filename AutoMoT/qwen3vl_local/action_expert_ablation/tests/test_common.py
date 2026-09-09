"""Lightweight checks for action expert ablation wrappers."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from qwen3vl_local.action_expert_ablation import common
from qwen3vl_local.action_expert_ablation import launch


def test_parser_hides_prior_inputs():
    """Ablation train CLI should not accept RS/EVENT prior knobs."""

    parser = common.parser("qwen_simple")
    option_strings = {item for action in parser._actions for item in action.option_strings}
    assert "--phase1-adapter" not in option_strings
    assert "--phase2-adapter" not in option_strings
    assert "--dataset-priors" not in option_strings
    assert "--prior-labels" not in option_strings
    assert "--analysis-review" not in option_strings
    assert "--condition-mode" not in option_strings


def test_bev_only_parser_hides_qwen_model_inputs():
    """BEV-only should not ask for Qwen paths in its public CLI."""

    parser = common.parser("bev_only")
    option_strings = {item for action in parser._actions for item in action.option_strings}
    assert "--model-dir" not in option_strings
    assert "--qwen-dtype" not in option_strings
    assert "--qwen-load-stagger-s" not in option_strings


def test_scalar_metrics_use_sample_mean_and_keep_condition_counts():
    """TensorBoard train/* scalars keep action_prior-style sample denominators."""

    metrics = common.scalar_metrics_from_counts(
        Counter(
            {
                "samples": 4,
                "loss": 8.0,
                "route_fm_mse": 4.0,
                "waypoint_fm_mse": 12.0,
                "condition/qwen_prefills": 4,
            }
        )
    )
    assert metrics["loss"] == 2.0
    assert metrics["route_fm_mse"] == 1.0
    assert metrics["waypoint_fm_mse"] == 3.0
    assert metrics["condition/qwen_prefills"] == 1.0
    assert metrics["count/condition/qwen_prefills"] == 4


def test_budget_complete_waits_for_pending_validation():
    """Resume should not train past the budget, but must finish owed validation."""

    plan = {"actual_step_limit": 2}
    cursor = {"epoch": 1, "micro": 0}
    assert common.budget_complete(2, plan, cursor)
    pending = common.with_validation_pending(cursor, validation_epoch=0, full_epoch=True)
    assert not common.budget_complete(2, plan, pending)
    assert common.budget_complete(2, plan, common.clear_validation_pending(pending))


def test_launcher_consumes_train_output_dir_and_resume(monkeypatch):
    """CLI output/resume should enter the same run-directory logic as env vars."""

    monkeypatch.setenv("OUTPUT_DIR", "env_out")
    monkeypatch.setenv("RESUME", "env_resume.pt")
    base, resume, cleaned = launch.resolve_train_launch_args(
        "qwen_simple",
        ["--output-dir", "cli_out", "--resume=cli_resume.pt", "--learning-rate", "1e-4"],
    )
    assert base == Path("cli_out")
    assert resume == "cli_resume.pt"
    assert cleaned == ["--learning-rate", "1e-4"]


def test_resume_args_restore_saved_non_default_config(tmp_path):
    """A plain --resume should recover the original run args before loading data."""

    run_dir = tmp_path / "run_custom"
    run_dir.mkdir()
    ckpt = run_dir / "latest.pt"
    ckpt.write_bytes(b"placeholder")
    config = vars(common.parser("qwen_simple").parse_args([]))
    config.update(
        output_dir=str(run_dir),
        data_dir=str(tmp_path / "custom_index"),
        learning_rate=1e-4,
        grad_accum_steps=8,
        num_epochs=20,
    )
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    args = common.parse_train_args("qwen_simple", ["--resume", str(ckpt)])
    assert args.output_dir == str(run_dir)
    assert args.resume == str(ckpt.resolve())
    assert args.data_dir == str(tmp_path / "custom_index")
    assert args.learning_rate == 1e-4
    assert args.grad_accum_steps == 8
    assert args.num_epochs == 20


def test_resume_args_keep_explicit_overrides(tmp_path):
    """Relocated paths passed on the resume command line should still win."""

    run_dir = tmp_path / "run_custom"
    run_dir.mkdir()
    ckpt = run_dir / "latest.pt"
    ckpt.write_bytes(b"placeholder")
    config = vars(common.parser("qwen_simple").parse_args([]))
    config.update(data_dir="old_index", learning_rate=1e-4)
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    args = common.parse_train_args(
        "qwen_simple",
        ["--resume", str(ckpt), "--data-dir", "new_index", "--learning-rate", "0.0003"],
    )
    assert args.data_dir == "new_index"
    assert args.learning_rate == 3e-4
    assert args.output_dir == str(run_dir)


def test_launcher_restores_resume_world_size_default(tmp_path, monkeypatch):
    """Launcher should recover non-default DDP size before selecting GPUs."""

    run_dir = tmp_path / "run_custom"
    run_dir.mkdir()
    ckpt = run_dir / "latest.pt"
    ckpt.write_bytes(b"placeholder")
    (run_dir / "training_plan.json").write_text(json.dumps({"world_size": 2}), encoding="utf-8")
    for name in ("GPU_IDS", "DDP_GPU_COUNT", "NPROC_PER_NODE"):
        monkeypatch.delenv(name, raising=False)

    launch.restore_resume_world_size_default(str(ckpt))
    assert os.environ["DDP_GPU_COUNT"] == "2"


def _capture_train_sh_args(
    tmp_path: Path,
    variant: str,
    extra: list[str],
    env_extra: dict[str, str] | None = None,
) -> list[str]:
    """Run a variant train.sh through a fake python binary and capture argv."""

    fake_bin = tmp_path / f"bin_{variant}_{len(extra)}"
    fake_bin.mkdir()
    capture = tmp_path / f"{variant}_argv.json"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
Path(os.environ["CAPTURE_ARGV"]).write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    for name in (
        "DATA_ROOT",
        "DATA_DIR",
        "MODEL_DIR",
        "LEAD_BEV_CKPT",
        "NUM_EPOCHS",
        "LR",
        "GRAD_ACCUM",
        "VAL_STEPS",
        "SAVE_STEPS",
        "LOGGING_STEPS",
        "NUM_WORKERS",
        "FLOW_SAMPLE_STEPS",
        "FLOW_ROUTE_COORDINATE_SCALE_M",
        "FLOW_WAYPOINT_COORDINATE_SCALE_M",
        "FLOW_TIME_EMBED_DIM",
        "FLOW_TRAJECTORY_LAYERS",
        "FLOW_TRAJECTORY_HEADS",
        "TRAIN_SAMPLED_METRICS",
        "RESUME",
    ):
        env.pop(name, None)
    env.update(env_extra or {})
    env["CAPTURE_ARGV"] = str(capture)
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    subprocess.run(
        ["bash", f"qwen3vl_local/action_expert_ablation/{variant}/train.sh", *extra],
        cwd=common.AUTOMOT_ROOT,
        env=env,
        check=True,
    )
    return json.loads(capture.read_text(encoding="utf-8"))


def test_train_sh_resume_does_not_inject_default_overrides(tmp_path):
    """Shell defaults should not masquerade as user overrides during resume."""

    for variant in ("qwen_simple", "bev_only"):
        argv = _capture_train_sh_args(tmp_path, variant, ["--resume", "run/latest.pt"])
        assert "--resume" in argv
        for option in (
            "--data-root",
            "--data-dir",
            "--model-dir",
            "--lead-bev-ckpt",
            "--num-epochs",
            "--learning-rate",
            "--grad-accum-steps",
            "--val-steps",
            "--save-steps",
            "--logging-steps",
            "--num-workers",
            "--flow-sample-steps",
        ):
            assert option not in argv


def test_train_sh_resume_keeps_environment_overrides(tmp_path):
    """Environment variables remain explicit resume overrides."""

    argv = _capture_train_sh_args(
        tmp_path,
        "qwen_simple",
        ["--resume", "run/latest.pt"],
        {"LR": "0.0001", "DATA_DIR": "custom_index", "TRAIN_SAMPLED_METRICS": "0"},
    )
    assert argv[argv.index("--learning-rate") + 1] == "0.0001"
    assert argv[argv.index("--data-dir") + 1] == "custom_index"
    assert "--no-train-sampled-metrics" in argv
    assert "--num-epochs" not in argv


def test_pipeline_stale_lock_file_does_not_block_index_build(tmp_path):
    """A leftover .build.lock file should not block a later flock-based build."""

    index_dir = tmp_path / "index"
    script = f'''
set -euo pipefail
source qwen3vl_local/action_expert_ablation/pipeline_common.sh
mkdir -p "{index_dir}"
: > "{index_dir}/.build.lock"
python() {{
  local out=""
  local prev=""
  for arg in "$@"; do
    if [[ "$prev" == "--output-dir" ]]; then
      out="$arg"
    fi
    prev="$arg"
  done
  printf 'train\\n' > "$out/train.jsonl"
  printf 'val\\n' > "$out/val.jsonl"
  printf 'test\\n' > "$out/test.jsonl"
}}
action_ablation_build_index_if_needed lead_data "{index_dir}"
test -s "{index_dir}/train.jsonl"
test -s "{index_dir}/val.jsonl"
test -s "{index_dir}/test.jsonl"
'''
    subprocess.run(["bash", "-c", script], cwd=common.AUTOMOT_ROOT, check=True, timeout=5)


def test_pipeline_resume_restores_saved_data_dir_before_build(tmp_path):
    """Full pipeline should build the checkpoint's saved index path on resume."""

    run_dir = tmp_path / "run_custom"
    run_dir.mkdir()
    ckpt = run_dir / "latest.pt"
    best = run_dir / "best.pt"
    ckpt.write_bytes(b"placeholder")
    best.write_bytes(b"placeholder")
    saved_index = tmp_path / "saved_index"
    config = vars(common.parser("bev_only").parse_args([]))
    config.update(data_root="saved_root", data_dir=str(saved_index), lead_bev_ckpt="saved_bev.pth")
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    build_out = tmp_path / "build_out.txt"
    bash_log = tmp_path / "bash_log.txt"
    script = f'''
set -euo pipefail
source qwen3vl_local/action_expert_ablation/pipeline_common.sh
python() {{
  local out=""
  local prev=""
  for arg in "$@"; do
    if [[ "$prev" == "--output-dir" ]]; then
      out="$arg"
    fi
    prev="$arg"
  done
  echo "$out" > "{build_out}"
  mkdir -p "$out"
  printf 'train\\n' > "$out/train.jsonl"
  printf 'val\\n' > "$out/val.jsonl"
  printf 'test\\n' > "$out/test.jsonl"
}}
bash() {{
  printf '%s\\n' "$*" >> "{bash_log}"
}}
action_ablation_run_full_pipeline bev_only checkpoints/action_expert_ablation/bev_only --resume "{ckpt}"
'''
    subprocess.run(["bash", "-c", script], cwd=common.AUTOMOT_ROOT, check=True, timeout=5)
    assert build_out.read_text(encoding="utf-8").strip() == str(saved_index)
    calls = bash_log.read_text(encoding="utf-8")
    assert f"--data-dir {saved_index}" in calls
    assert "--data-dir checkpoints/action_prior_data" not in calls


def test_execution_fingerprint_covers_shared_runtime_dependencies():
    """Condition identity should move when shared train/decoder helpers move."""

    fingerprint = common._execution_fingerprint(
        common.AUTOMOT_ROOT,
        common.contract_source_paths("bev_only"),
    )
    code = fingerprint["code"]
    assert "qwen3vl_local/action_prior/training_core.py" in code
    assert "qwen3vl_local/action_prior/progress.py" in code
    assert "qwen3vl_local/leadmot/query_bank.py" in code
    assert "qwen3vl_local/leadmot/config.py" in code
    assert "qwen3vl_local/sft_new_loop_phase1/prompts.py" not in code
    assert "qwen3vl_local/sft_new_loop_phase2/prompts.py" not in code
    assert "torch" in fingerprint["packages"]
