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


def test_phase2_only_final_is_found_but_never_recommended(tmp_path, capsys):
    path = fixture_adapter(tmp_path / "p2", 2, slot="final")
    # best 的 generation 明细绝不能冒充 final 保存 step 的结果。
    file = path.parent / "best_generation.json"
    record = json.loads(file.read_text())
    record.update(step=200, generation={"exact_accuracy": 1.0})
    file.write_text(json.dumps(record))
    result = scan(tmp_path, 2, tmp_path / "base")
    assert result["discovery_status"] == "only_non_best_slots"
    assert not result["candidates"] and result["recommended"] is None
    assert result["other_checkpoints"][0]["generation"]["step"] == 100
    show(result, summary_only=True)
    assert "非 best_generation 保存点" in capsys.readouterr().out


def test_all_contract_failures_are_reported_together(tmp_path, capsys):
    path = fixture_adapter(tmp_path / "p2", 2)
    file = path / "sft_new_loop_phase2_adapter_config.json"
    cfg = json.loads(file.read_text())
    cfg.update(prompt_name="old_v3", production_prompt_sha256="old_hash", git={"commit": None},
               history_rgb_count=2, base_model_dir="/wrong/base")
    file.write_text(json.dumps(cfg))
    (path / "adapter_model.safetensors").unlink()
    file = path.parent / "best_generation.json"
    record = json.loads(file.read_text())
    record.update(step=101, generation_guards_ok=False)
    file.write_text(json.dumps(record))
    result = scan(tmp_path, 2, tmp_path / "base")
    row = result["candidates"][0]
    checks = {c["name"]: c for c in row["checks"]}
    for name in ("prompt_name", "prompt_hash", "training_git_commit", "rgb_count",
                 "base_model_dir", "weight_files", "generation_guards", "selection_step"):
        assert checks[name]["status"] == "fail"
    assert checks["prompt_name"]["actual"] == "old_v3"
    assert checks["prompt_name"]["expected"] != "old_v3"
    assert result["discovery_status"] == "best_generation_rejected"
    assert "generation" not in row  # 错 step 的高分也不展示成该 adapter 的分数。
    show(result, summary_only=True)
    output = capsys.readouterr().out
    assert "[fail] prompt_name" in output and "[fail] training_git_commit" in output


def test_different_training_commit_is_not_a_rejection(tmp_path):
    path = fixture_adapter(tmp_path / "p2", 2)
    result = scan(tmp_path, 2, tmp_path / "base")
    assert result["recommended"] == str(path)
    check = next(c for c in result["candidates"][0]["checks"] if c["name"] == "training_git_commit")
    assert check["actual"] == "abcdef" and check["status"] == "pass"


def test_legacy_metadata_is_separate_from_new_phase_candidates(tmp_path, capsys):
    path = fixture_adapter(tmp_path / "sft_loop_phase2_old", 2)
    file = path / "sft_new_loop_phase2_adapter_config.json"
    file.rename(path / "sft_loop_phase2_adapter_config.json")
    result = scan(tmp_path, 2, tmp_path / "base")
    assert not result["candidates"] and not result["other_checkpoints"]
    row = result["discovery"]["excluded_other_packages"][0]
    assert not row["expected_metadata_present"]
    assert "sft_loop_phase2_adapter_config.json" in row["alternative_metadata"]
    show(result, summary_only=True)
    assert "其它训练包保存点已排除 1 个" in capsys.readouterr().out


def test_missing_metadata_in_new_package_still_reports_failure(tmp_path):
    path = fixture_adapter(tmp_path / "sft_new_loop_phase2_runs", 2)
    (path / "sft_new_loop_phase2_adapter_config.json").unlink()
    result = scan(tmp_path, 2, tmp_path / "base")
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["checks"][0]["status"] == "fail"


def test_metadata_identifies_new_package_in_custom_or_legacy_named_root(tmp_path):
    path = fixture_adapter(tmp_path / "sft_loop_phase2_augment_runs", 2)
    result = scan(tmp_path, 2, tmp_path / "base")
    assert result["recommended"] == str(path)
    assert not result["discovery"]["excluded_other_packages"]


def test_combined_phase1_and_old_rs_are_never_new_event_candidates(tmp_path):
    fixture_adapter(tmp_path / "sft_new_loop_phase1_runs" / "combined_phase1_phase2", 1)
    path = fixture_adapter(tmp_path / "sft_loop_phase2_augment_runs", 2)
    (path / "sft_new_loop_phase2_adapter_config.json").rename(
        path / "sft_loop_phase2_augment_adapter_config.json")
    result = scan(tmp_path, 2, tmp_path / "base")
    assert not result["candidates"] and not result["other_checkpoints"]
    assert len(result["discovery"]["excluded_other_packages"]) == 2
    # 即使显式扫描旧包的 best 目录，也不能冒充新 Phase2。
    assert not scan(path, 2, tmp_path / "base")["candidates"]


def test_external_directory_links_followed_deduplicated_and_cycles_stop(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    path = fixture_adapter(tmp_path / "external" / "run", 2)
    (root / "phase2_latest").symlink_to(path.parent, target_is_directory=True)
    (root / "second_alias").symlink_to(path.parent, target_is_directory=True)
    (path.parent / "cycle").symlink_to(root, target_is_directory=True)
    (root / "broken_phase2").symlink_to(tmp_path / "absent", target_is_directory=True)
    result = scan(root, 2, tmp_path / "external" / "base")
    assert result["recommended"] == str(path)
    assert len(result["candidates"]) == 1
    assert result["discovery"]["visited_directories"] < 10
    assert any(r["status"] == "unavailable" for r in result["discovery"]["links"])


def test_link_alias_cannot_make_final_into_best_generation(tmp_path):
    path = fixture_adapter(tmp_path / "external", 2, slot="final")
    alias = tmp_path / "best_generation"
    alias.symlink_to(path, target_is_directory=True)
    result = scan(alias, 2, tmp_path / "base")
    assert result["recommended"] is None and not result["candidates"]
    assert result["other_checkpoints"][0]["path"] == str(path)


def test_unknown_checkpoint_and_empty_phase2_directory_are_visible(tmp_path):
    path = tmp_path / "sft_new_loop_phase2_runs" / "empty"
    path.mkdir(parents=True)
    unknown = tmp_path / "unlabelled" / "best_generation"
    unknown.mkdir(parents=True)
    (unknown / "adapter_config.json").write_text("{}")
    result = scan(tmp_path, 2, tmp_path / "base")
    assert str(path.parent) in result["discovery"]["phase_directories"]
    assert result["discovery"]["unclassified_slots"][0]["path"] == str(unknown)
    assert result["discovery_status"] == "no_phase_checkpoint_found"


def test_bad_metadata_does_not_abort_remaining_candidates(tmp_path):
    path = fixture_adapter(tmp_path / "a", 2)
    (path / "sft_new_loop_phase2_adapter_config.json").write_text("[]")
    best = fixture_adapter(tmp_path / "b", 2)
    result = scan(tmp_path, 2, tmp_path / "base")
    assert result["recommended"] == str(best)
    assert len(result["candidates"]) == 2


def test_scandir_permission_failure_is_reported(tmp_path, monkeypatch):
    from qwen3vl_local.action_prior import lora_audit

    def denied(path):
        raise PermissionError("test access denied")

    monkeypatch.setattr(lora_audit.os, "scandir", denied)
    result = scan(tmp_path, 2, tmp_path / "base")
    assert "test access denied" in result["discovery"]["errors"][0]["error"]
    assert result["recommended"] is None


def test_audit_metadata_copy_and_archive_are_not_loadable_weights(tmp_path):
    path = fixture_adapter(tmp_path / "sft_new_loop_phase2_audit_bundle" / "adapter_metadata", 2,
                           slot="adapter")
    (path / "adapter_model.safetensors").unlink()
    archive = tmp_path / "sft_new_loop_phase2_archive.zip"
    archive.write_bytes(b"not opened or extracted")
    result = scan(tmp_path, 2, tmp_path / "base")
    assert result["other_checkpoints"][0]["artifact_kind"] == "audit_metadata_only"
    assert result["discovery"]["archives"][0]["path"] == str(archive)
    assert result["recommended"] is None
