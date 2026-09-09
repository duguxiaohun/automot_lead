"""入口脚本的先验开关组装；用桩 launcher 断言参数，不加载模型也不启动训练。"""

import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parents[1]
DEFAULT_LABELS = "checkpoints/action_prior_labels/prior_labels.jsonl"
STUB = """#!/usr/bin/env bash
case "$1" in
  */launch.py|*/build_prior_labels.py|*/build_dataset.py|*/resume.py|*/audit_bundle.py) echo "STUB $*"; exit 0 ;;
esac
exec {python} "$@"
"""


@pytest.fixture
def stub(tmp_path):
    """拦截真正的入口，只回显组装好的命令行。"""
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "python").write_text(STUB.format(python=sys.executable))
    (binary / "python").chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}{os.pathsep}{os.environ['PATH']}")
    for name in ("DATASET_PRIORS", "PRIOR_LABELS", "PRIOR_NOISE", "ANALYSIS_REVIEW",
                 "RESUME", "RUN_TAG", "OUTPUT_DIR", "DATA_DIR"):
        env.pop(name, None)
    return env


def run(script, args, env, **overrides):
    result = subprocess.run(
        ["bash", str(SCRIPTS / script), *args],
        cwd=ROOT, env=dict(env, **overrides), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def flags(output, prefix="STUB"):
    line = next(l for l in output.splitlines() if l.startswith(prefix))
    return line.split()


def all_flags(output, prefix="STUB"):
    return [line.split() for line in output.splitlines() if line.startswith(prefix)]


def value_of(tokens, flag):
    return tokens[tokens.index(flag) + 1]


def test_lora_default_keeps_the_independent_review(stub):
    tokens = flags(run("train.sh", [], stub))
    assert "--analysis-review" in tokens
    assert "--dataset-priors" not in tokens and "--prior-labels" not in tokens


@pytest.mark.parametrize(
    "args,overrides",
    [
        ([], {"DATASET_PRIORS": "1"}),
        (["--dataset-priors"], {}),
    ],
)
def test_dataset_default_drops_the_review_from_either_entrypoint(stub, args, overrides):
    """环境变量与命令行开关必须给出同一套默认值，否则会多出一次 base 生成。"""
    tokens = flags(run("train.sh", args, stub, **overrides))
    assert "--no-analysis-review" in tokens and "--analysis-review" not in tokens
    assert tokens.count("--dataset-priors") == 1
    assert value_of(tokens, "--prior-labels") == DEFAULT_LABELS
    assert value_of(tokens, "--prior-noise") == "0"


def test_explicit_options_are_never_duplicated_or_overridden(stub):
    tokens = flags(run(
        "train.sh",
        ["--dataset-priors", "--prior-labels", "/custom/x.jsonl",
         "--prior-noise", "0.1", "--analysis-review"],
        stub,
        DATASET_PRIORS="1", PRIOR_LABELS=DEFAULT_LABELS, PRIOR_NOISE="0.5",
    ))
    assert tokens.count("--prior-labels") == 1 and tokens.count("--prior-noise") == 1
    assert value_of(tokens, "--prior-labels") == "/custom/x.jsonl"
    assert value_of(tokens, "--prior-noise") == "0.1"
    assert "--analysis-review" in tokens and "--no-analysis-review" not in tokens


def test_analysis_review_env_overrides_the_dataset_default(stub):
    tokens = flags(run("train.sh", [], stub, DATASET_PRIORS="1", ANALYSIS_REVIEW="1"))
    assert "--analysis-review" in tokens and "--no-analysis-review" not in tokens


def test_eval_only_fills_missing_label_paths(stub):
    plain = flags(run("eval.sh", ["--checkpoint", "/tmp/best.pt", "--split", "test"], stub))
    assert "--prior-labels" not in plain
    filled = flags(run("eval.sh", ["--checkpoint", "/tmp/best.pt", "--dataset-priors"], stub))
    assert value_of(filled, "--prior-labels") == DEFAULT_LABELS
    for form in ("--prior-labels=/custom/y.jsonl", "--prior-labels /custom/y.jsonl"):
        tokens = flags(run(
            "eval.sh", ["--checkpoint", "/tmp/best.pt", "--dataset-priors", *form.split()], stub
        ))
        assert tokens.count("--prior-labels") + tokens.count("--prior-labels=/custom/y.jsonl") == 1
        assert "/custom/y.jsonl" in " ".join(tokens) and DEFAULT_LABELS not in tokens


def test_pipeline_never_builds_a_custom_label_index_silently(stub, tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_full_pipeline.sh"), "--dataset-priors",
         "--prior-labels", str(tmp_path / "missing.jsonl")],
        cwd=ROOT, env=dict(stub, OUTPUT_DIR=str(tmp_path / "out")),
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    combined = result.stdout + result.stderr
    assert "build it explicitly" in combined
    assert "build_prior_labels.py --output-dir checkpoints" not in combined


def test_resume_keeps_a_relocated_label_index(stub, tmp_path):
    """搬迁标签后旧 config.json 的路径已失效，续训必须收到新路径。"""
    checkpoint = tmp_path / "run" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")
    moved = tmp_path / "moved" / "prior_labels.jsonl"
    # 续训之后的 test/probe 需要真权重，这里只看续训入口拿到什么参数。
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_full_pipeline.sh"), "--dataset-priors",
         "--prior-labels", str(moved)],
        cwd=ROOT,
        env=dict(stub, RESUME=str(checkpoint), OUTPUT_DIR=str(tmp_path / "out")),
        capture_output=True, text=True,
    )
    tokens = flags(result.stdout)
    assert tokens[1].endswith("resume.py")
    assert value_of(tokens, "--prior-labels") == str(moved)


def test_resume_keeps_a_relocated_label_index_from_env(stub, tmp_path):
    """PRIOR_LABELS 环境变量也是显式路径，不能在续训时退回旧 config.json。"""
    checkpoint = tmp_path / "run" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")
    moved = tmp_path / "moved" / "prior_labels.jsonl"
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_full_pipeline.sh"), "--dataset-priors"],
        cwd=ROOT,
        env=dict(
            stub,
            RESUME=str(checkpoint),
            OUTPUT_DIR=str(tmp_path / "out"),
            PRIOR_LABELS=str(moved),
        ),
        capture_output=True,
        text=True,
    )
    tokens = flags(result.stdout)
    assert tokens[1].endswith("resume.py")
    assert value_of(tokens, "--prior-labels") == str(moved)


def test_resume_infers_dataset_mode_and_remaps_final_eval_probe(stub, tmp_path):
    """旧 dataset run 只传新标签路径时，续训和最终 test/probe 都要用新路径。"""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "latest.pt"
    checkpoint.write_bytes(b"")
    (run_dir / "best.pt").write_bytes(b"")
    (run_dir / "config.json").write_text(
        json.dumps({"dataset_priors": True}), encoding="utf-8"
    )
    moved = tmp_path / "moved" / "prior_labels.jsonl"
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_full_pipeline.sh"), "--prior-labels", str(moved)],
        cwd=ROOT,
        env=dict(stub, RESUME=str(checkpoint), OUTPUT_DIR=str(tmp_path / "out")),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = all_flags(result.stdout)
    resume, eval_call, probe_call = calls[:3]
    assert resume[1].endswith("resume.py")
    assert eval_call[1].endswith("launch.py") and eval_call[2] == "eval"
    assert probe_call[1].endswith("launch.py") and probe_call[2] == "probe"
    assert value_of(resume, "--prior-labels") == str(moved)
    assert value_of(eval_call, "--prior-labels") == str(moved)
    assert value_of(probe_call, "--prior-labels") == str(moved)


def test_lora_resume_does_not_leak_prior_labels_env(stub, tmp_path):
    """LoRA 先验续训不应收到 dataset-only 的 prior-labels 参数。"""
    checkpoint = tmp_path / "run" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_full_pipeline.sh")],
        cwd=ROOT,
        env=dict(
            stub,
            RESUME=str(checkpoint),
            OUTPUT_DIR=str(tmp_path / "out"),
            PRIOR_LABELS=str(tmp_path / "labels.jsonl"),
        ),
        capture_output=True,
        text=True,
    )
    tokens = flags(result.stdout)
    assert tokens[1].endswith("resume.py")
    assert "--prior-labels" not in tokens


def test_bench2drive_resume_reuses_the_pinned_adapters(tmp_path, monkeypatch):
    """上游出现新 best 时，续跑必须先用旧 manifest 固定的那一组权重。"""
    import torch
    from qwen3vl_local.action_prior import bench2drive, config, contracts

    torch.save(
        dict(
            schema="action_prior_checkpoint_v4",
            trajectory_decoder="conditional_joint_trajectory_flow_matching_v2",
            args=dict(dataset_priors=True, prior_noise=0.0, prior_labels="labels.jsonl",
                      phase1_adapter="", phase2_adapter=""),
            qwen_backbone={"schema": contracts.SCHEMA, "identity": "trained"},
        ),
        tmp_path / "best.pt",
    )
    seen = {}

    def fake_build_contract(args):
        seen.update(phase1=args.phase1_adapter, phase2=args.phase2_adapter,
                    dataset_priors=args.dataset_priors, prior_noise=args.prior_noise)
        return dict(
            identity="resolved",
            phase1={"path": args.phase1_adapter or "auto/new_best"},
            phase2={"path": args.phase2_adapter or "auto/new_best2"},
        )

    monkeypatch.setattr(config, "build_contract", fake_build_contract)
    monkeypatch.setattr(config, "validate_args", lambda args: None)
    monkeypatch.setattr(contracts, "require_contract", lambda *a, **k: "prior_source_override")
    monkeypatch.setenv("ACTION_DATASET_PRIORS", "0")
    for name in ("ACTION_PHASE1_ADAPTER", "ACTION_PHASE2_ADAPTER"):
        monkeypatch.delenv(name, raising=False)
    cli = type("Cli", (), {key: "" for key in (
        "model_dir", "lead_bev_ckpt", "phase1_adapter", "phase2_adapter",
        "phase1_training_index", "phase2_training_index")})()
    cli.checkpoint = str(tmp_path / "best.pt")

    identity, pinned = bench2drive.validate_checkpoint(
        cli, {"phase1": "pinned/best_generation", "phase2": "pinned/best_generation2"}
    )
    assert seen["phase1"] == "pinned/best_generation"
    assert seen["dataset_priors"] is False and seen["prior_noise"] == 0.0
    assert pinned == {"phase1": "pinned/best_generation",
                      "phase2": "pinned/best_generation2"}
    assert identity == "resolved"
    # 首次运行没有旧 manifest，仍然自动选并把结果固定下来。
    assert bench2drive.validate_checkpoint(cli)[1] == {
        "phase1": "auto/new_best", "phase2": "auto/new_best2"}
