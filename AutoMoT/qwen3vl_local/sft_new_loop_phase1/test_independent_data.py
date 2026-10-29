"""Phase isolation, bounded CPU stages and separate real CPU data admission."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from . import data_stage, run_integrity as integrity
from .test_integrity import fixture_assets


@pytest.mark.parametrize("phase", [1, 2])
def test_sources_exclude_other_phase_training_and_dataset(tmp_path, monkeypatch, phase):
    helper = tmp_path / 'lead_video_tools'
    helper.mkdir()
    for name in ('__init__.py', 'abnormal_duration_filter.py'):
        (helper / name).write_text('# shared native audit dependency fixture\n')
    for n in (1,2):
        root = tmp_path / f"qwen3vl_local/sft_new_loop_phase{n}"
        root.mkdir(parents=True)
        for name in ("train.py", "eval.py", "prompts.py", "build_dataset.py", "training_policy.py", "run_integrity.py"):
            (root / name).write_text(f"# phase {n} {name}")
    monkeypatch.setattr(integrity, "ROOT", tmp_path)
    before = integrity.sources(phase)
    assert f"qwen3vl_local/sft_new_loop_phase{phase}/train.py" in before
    other = tmp_path / f"qwen3vl_local/sft_new_loop_phase{3-phase}"
    for name in ("train.py", "eval.py", "prompts.py", "build_dataset.py"):
        (other / name).write_text("# changed unrelated phase")
    assert integrity.sources(phase) == before
    own = tmp_path / f"qwen3vl_local/sft_new_loop_phase{phase}/train.py"
    own.write_text("# changed own phase")
    assert integrity.sources(phase) != before


def test_stage_success_failure_collision_and_logs(tmp_path):
    good = tmp_path / "good.json"
    result = data_stage.run([sys.executable, "-c", "print('real child output')"], good, timeout=10)
    assert result["status"] == "completed"
    assert good.with_suffix(".log").read_text() == "real child output\n"
    with pytest.raises(FileExistsError):
        data_stage.run([sys.executable, "-c", "raise SystemExit(2)"], good)
    bad = tmp_path / "bad.json"
    with pytest.raises(subprocess.CalledProcessError):
        data_stage.run([sys.executable, "-c", "raise SystemExit(17)"], bad)
    assert integrity.read(bad)["returncode"] == 17
    assert integrity.read(bad)["status"] == "failed"


def test_timeout_stops_only_owned_process_group(tmp_path):
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        status = tmp_path / "timeout.json"
        with pytest.raises(TimeoutError):
            data_stage.run([sys.executable, "-c", "import time; time.sleep(30)"], status, timeout=.2, heartbeat=.05)
        record = integrity.read(status)
        assert record["status"] == "timed_out" and record["returncode"] is not None
        assert other.poll() is None
    finally:
        other.terminate(); other.wait(timeout=5)


@pytest.mark.parametrize("phase", [1,2])
def test_real_cpu_prepare_reuses_data_without_model_or_other_phase(tmp_path, phase):
    index, _ = fixture_assets(tmp_path, phase)
    before = integrity.sha(index)
    out = tmp_path / "cpu"
    command = [sys.executable, "-B", "-m", "qwen3vl_local.sft_new_loop_phase1.data_stage", "prepare",
               "--phase", str(phase), "--audit-dir", str(out), "--run-build", "0", "--timeout", "60",
               "--modes", "2rgb_endpoints", "--cases-per-bin", "4", "--focus-balance-count", "6", "--action-index", "off",
               "--", "--output-dir", str(index.parent), "--data-root", str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    report = integrity.read(out / "status.json")
    assert report["status"] == "ready" and not report["model_loaded"]
    assert integrity.read(out / "2rgb_endpoints_data.json")["model_checked"] is False
    assert (out / "2rgb_endpoints_plan/training_budget.json").is_file()
    assert not list(out.rglob("training_complete.json"))
    assert not list(out.rglob("*.safetensors"))
    assert integrity.sha(index) == before
    # An invalid RGB must reject admission; it cannot become ready after planning.
    (tmp_path / "3.jpg").write_bytes(b"broken")
    command[command.index(str(out))] = str(tmp_path / "bad_cpu")
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    assert integrity.read(tmp_path / "bad_cpu/status.json")["status"] == "rejected"


@pytest.mark.parametrize("phase", [1,2])
def test_training_launcher_never_implicitly_builds_missing_data(tmp_path, phase):
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "GPU_IDS": "0,1,2,3", "PIPELINE_ROOT": str(tmp_path / "pipeline"),
           "DATA_OUTPUT_DIR": str(tmp_path / "missing"), "INDEX": str(tmp_path / "missing/frame_index.jsonl")}
    env.pop("RUN_BUILD", None)
    env.pop("TRAIN_OUTPUT_DIR", None)
    result = subprocess.run(["bash", str(root / f"qwen3vl_local/sft_new_loop_phase{phase}/run_full_pipeline.sh")],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "RUN_BUILD=0 but dataset missing" in result.stdout + result.stderr
    assert not (tmp_path / "missing").exists()
    assert not (tmp_path / "pipeline").exists()
    # Even an explicit build in a training launch must reject missing Qwen assets first.
    env.update(RUN_BUILD="1", MODEL_DIR=str(tmp_path / "missing_model"))
    result = subprocess.run(["bash", str(root / f"qwen3vl_local/sft_new_loop_phase{phase}/run_full_pipeline.sh")],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert integrity.read(tmp_path / "pipeline/model_before_build.json")["status"] == "failed"
    assert not (tmp_path / "missing").exists()
