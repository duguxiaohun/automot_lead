"""推荐结果与生产选择器一致，并保留每项指标和不兼容原因。"""

import json
from pathlib import Path
import subprocess
import sys

import pytest
from qwen3vl_local.action_prior.rank_loras import scan, show
from qwen3vl_local.action_prior.contracts import select_adapter
from test_contracts import fixture_adapter


@pytest.mark.parametrize("phase", [1, 2])
def test_recommendation_matches_training_and_excludes_other_slots(tmp_path, phase):
    fixture_adapter(tmp_path / "a", phase, score=0.7)
    best = fixture_adapter(tmp_path / "b", phase, score=0.9)
    fixture_adapter(tmp_path / "c", phase, slot="final", score=1)
    fixture_adapter(tmp_path / "d", phase, slot="best_generation_balanced", score=1)
    result = scan(tmp_path, phase, tmp_path / "base")
    assert (
        result["recommended"]
        == select_adapter(tmp_path, phase, tmp_path / "base")["path"]
        == str(best)
    )
    assert len(result["candidates"]) == 2


def test_phase2_all_slice_and_guard_details_are_printed(tmp_path, capsys):
    path = fixture_adapter(tmp_path / "p2", 2)
    file = path.parent / "best_generation.json"
    record = json.loads(file.read_text())
    record.update(
        generation={
            "samples": 384,
            "exact_accuracy": 0.8,
            "slice/ue3_target_recall": 0.75,
            "slice/highway_ue3_samples": 12,
            "invalid_subgroup/source/a_exact": 0.9,
        },
        generation_guards={
            "values": {"ue3_target_recall": 0.75},
            "passed": {"ue3_target_recall": True},
        },
    )
    file.write_text(json.dumps(record))
    result = scan(tmp_path, 2, tmp_path / "base")
    show(result)
    text = capsys.readouterr().out
    assert "slice/ue3_target_recall: 0.75" in text
    assert "slice/highway_ue3_samples: 12" in text
    assert "invalid_subgroup/source/a_exact: 0.9" in text
    assert "passed/ue3_target_recall: true" in text


def test_wrong_prompt_and_guard_failure_remain_visible(tmp_path):
    path = fixture_adapter(tmp_path / "a", 2, score=0.99)
    file = path / "sft_new_loop_phase2_adapter_config.json"
    cfg = json.loads(file.read_text())
    cfg["prompt_name"] = "old"
    file.write_text(json.dumps(cfg))
    b = fixture_adapter(tmp_path / "b", 2, score=0.98)
    file = b.parent / "best_generation.json"
    cfg = json.loads(file.read_text())
    cfg["generation_guards_ok"] = False
    file.write_text(json.dumps(cfg))
    result = scan(tmp_path, 2, tmp_path / "base")
    assert result["recommended"] is None
    assert "prompt" in result["candidates"][0]["rejection"]
    assert "guards" in result["candidates"][1]["rejection"]


def test_phase1_uses_saved_step_never_later_test_or_loss(tmp_path):
    path = fixture_adapter(tmp_path / "p1")
    log = path.parent / "train_eval_metrics.jsonl"
    with log.open("a") as f:
        for row in [
            dict(step=200, type="generation", exact_accuracy=1, format_valid_rate=1),
            dict(step=100, type="teacher_forced", loss=0.23, value_token_acc=0.98),
        ]:
            f.write(json.dumps(row) + "\n")
    result = scan(tmp_path, 1, tmp_path / "base")["candidates"][0]
    assert result["generation"]["step"] == 100
    assert result["generation_exact"] == 0.8
    assert result["teacher_forced"]["loss"] == 0.23


def test_cli_empty_reports_both_phases_no_gpu(tmp_path):
    script = Path(__file__).resolve().parents[1] / "rank_loras.py"
    out = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint-root",
            str(tmp_path / "missing"),
            "--output-dir",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    data = json.loads((out / "report.json").read_text())
    assert len(data["phases"]) == 2 and all(
        r["recommended"] is None for r in data["phases"]
    )
    assert "无可推荐权重" in result.stdout and (out / "log.txt").is_file()
