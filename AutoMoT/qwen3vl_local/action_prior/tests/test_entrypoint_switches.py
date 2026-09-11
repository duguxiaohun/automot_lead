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
  */prepare_event_balance.py) echo "PREPARE $*" >&2; echo "/auto/full_event_mapping.jsonl"; exit 0 ;;
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
                 "EVENT_BALANCED", "EVENT_BALANCE_INDEX", "EVENT_BALANCED_SCENE_PRIORS",
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


def test_event_balanced_env_forwards_the_full_event_mapping_source(stub):
    source = "/tmp/action_prior/full_event_mapping.jsonl"
    tokens = flags(run("train.sh", [], stub, EVENT_BALANCED="1", EVENT_BALANCE_INDEX=source))
    assert value_of(tokens, "--sampling-mode") == "event_balanced"
    assert value_of(tokens, "--event-balance-index") == source


def test_uniform_scene_priors_env_also_forwards_the_full_event_mapping_source(stub):
    source = "/tmp/action_prior/full_event_mapping.jsonl"
    tokens = flags(run(
        "train.sh", [], stub,
        DATASET_PRIORS="1", EVENT_BALANCED_SCENE_PRIORS="1", EVENT_BALANCE_INDEX=source,
    ))
    assert "--sampling-mode" not in tokens  # uniform 是 parser 默认值
    assert "--event-balanced-scene-priors" in tokens
    assert value_of(tokens, "--event-balance-index") == source


def test_resume_and_final_eval_remap_event_balance_index(stub, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "latest.pt"
    checkpoint.write_bytes(b"")
    (run_dir / "best.pt").write_bytes(b"")
    moved = tmp_path / "moved" / "full_event_mapping.jsonl"
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_full_pipeline.sh"),
         "--event-balance-index", str(moved)],
        cwd=ROOT, env=dict(stub, RESUME=str(checkpoint), OUTPUT_DIR=str(tmp_path / "out")),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    resume, eval_call, probe_call = all_flags(result.stdout)[:3]
    assert value_of(resume, "--event-balance-index") == str(moved)
    assert value_of(eval_call, "--event-balance-index") == str(moved)
    assert value_of(probe_call, "--event-balance-index") == str(moved)


def test_explicit_uniform_mode_overrides_event_balanced_env(stub):
    tokens = flags(run("train.sh", ["--sampling-mode", "uniform"], stub, EVENT_BALANCED="1"))
    assert value_of(tokens, "--sampling-mode") == "uniform"
    assert "--event-balance-index" not in tokens


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


def test_explicit_scene_prior_disable_overrides_env_without_requiring_unused_index(stub):
    tokens = flags(run("train.sh", ["--no-event-balanced-scene-priors"], stub,
                       EVENT_BALANCED_SCENE_PRIORS="1"))
    assert "--event-balanced-scene-priors" not in tokens
    assert "--event-balance-index" not in tokens
    assert "--no-event-balanced-scene-priors" in tokens


def test_event_preflight_demo_forwards_mode_dataset_and_index(stub):
    tokens = flags(run("train.sh", ["--dataset-priors"], stub,
                       ACTION_MODE="preflight", DATA_DIR="checkpoints/action_prior_data_event_v1",
                       EVENT_BALANCED="1", EVENT_BALANCE_INDEX="checkpoints/action_prior_event_balance_v2/full_event_mapping.jsonl"))
    assert tokens[2] == "preflight"
    assert value_of(tokens, "--sampling-mode") == "event_balanced"
    assert value_of(tokens, "--data-dir") == "checkpoints/action_prior_data_event_v1"
    assert value_of(tokens, "--event-balance-index").endswith("full_event_mapping.jsonl")
    assert "--dataset-priors" in tokens


@pytest.mark.parametrize("args,env_overrides", [
    (["--event-balanced"], {}),
    (["--sampling-mode=event_balanced"], {}),
    ([], {"EVENT_BALANCED": "1"}),
    (["--event-balanced-scene-priors"], {}),
])
def test_pipeline_auto_prepares_before_preflight_and_passes_map_to_eval(stub, tmp_path, args, env_overrides):
    """验证真正 shell 顺序，所有昂贵入口使用桩；无需传 DATA_DIR 或索引路径。"""
    out = tmp_path / "out"
    run_dir = out / "run_demo"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"stub")
    labels = tmp_path / "labels.jsonl"
    labels.write_text("{}\n")
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_full_pipeline.sh"), "--dataset-priors", *args],
        cwd=ROOT, env=dict(stub, OUTPUT_DIR=str(out), RUN_TAG="demo", PRIOR_LABELS=str(labels), **env_overrides),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PREPARE" in result.stdout + result.stderr
    calls = all_flags(result.stdout)
    assert calls[0][1].endswith("build_dataset.py")
    preflight = next(c for c in calls if len(c) > 2 and c[2] == "preflight")
    training = next(c for c in calls if len(c) > 2 and c[2] == "train")
    for tokens in (preflight, training):
        assert value_of(tokens, "--event-balance-index") == "/auto/full_event_mapping.jsonl"
        assert value_of(tokens, "--data-dir").endswith("action_prior_data/run_demo")
    for mode in ("eval", "probe"):
        tokens = next(c for c in calls if len(c) > 2 and c[2] == mode)
        assert value_of(tokens, "--event-balance-index") == "/auto/full_event_mapping.jsonl"


def test_pipeline_disable_skips_auto_preparation_and_preserves_custom_paths(stub, tmp_path):
    out = tmp_path / "out"
    (out / "run_demo").mkdir(parents=True)
    (out / "run_demo" / "best.pt").write_bytes(b"stub")
    labels = tmp_path / "labels.jsonl"
    labels.write_text("{}\n")
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_full_pipeline.sh"), "--dataset-priors", "--no-event-balanced",
         "--data-dir", str(tmp_path / "custom index"), "--data-root", str(tmp_path / "custom data")],
        cwd=ROOT, env=dict(stub, OUTPUT_DIR=str(out), RUN_TAG="demo", PRIOR_LABELS=str(labels), EVENT_BALANCED="1"),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PREPARE" not in result.stdout + result.stderr
    assert str(tmp_path / "custom index") in result.stdout
    assert str(tmp_path / "custom data") in result.stdout
    training = next(c for c in all_flags(result.stdout) if len(c) > 2 and c[2] == "train")
    assert value_of(training, "--sampling-mode") == "uniform"
    assert "--event-balance-index" not in training


def test_pipeline_balanced_resume_does_not_rebuild_inputs(stub, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "latest.pt").write_bytes(b"")
    (run_dir / "best.pt").write_bytes(b"")
    (run_dir / "config.json").write_text(json.dumps(dict(dataset_priors=True, sampling_mode="event_balanced")))
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_full_pipeline.sh")],
        cwd=ROOT, env=dict(stub, RESUME=str(run_dir / "latest.pt"), OUTPUT_DIR=str(tmp_path / "out")),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PREPARE" not in result.stdout + result.stderr
    assert "build_dataset.py" not in result.stdout
    assert "--sampling-mode" not in all_flags(result.stdout)[0]


@pytest.mark.parametrize("options,env_extra,prepared", [
    (["--dataset-priors", "--event-balanced"], {}, True),
    (["--dataset-priors", "--sampling-mode=event_balanced"], {}, True),
    (["--dataset-priors"], {"EVENT_BALANCED": "1"}, True),
    (["--dataset-priors", "--no-event-balanced"], {"EVENT_BALANCED": "1"}, False),
    (["--dataset-priors", "--event-balanced-scene-priors"], {}, True),
    (["--dataset-priors", "--no-event-balanced-scene-priors"], {"EVENT_BALANCED_SCENE_PRIORS": "1"}, False),
])
def test_pipeline_simple_switch_prepares_inputs_before_preflight_and_propagates_index(stub, tmp_path, options, env_extra, prepared):
    """执行真实 shell，用桩边界验证一条命令的准备/训练/评测顺序。"""
    out, data = tmp_path / "out", tmp_path / "my data"
    run_dir = out / "run_simple"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"stub")
    labels = tmp_path / "labels.jsonl"
    labels.write_text("{}")
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_full_pipeline.sh"), *options,
         "--data-dir", str(data), "--data-root", str(tmp_path / "raw data"),
         "--prior-labels", str(labels)],
        cwd=ROOT, env=dict(stub, OUTPUT_DIR=str(out), RUN_TAG="simple", **env_extra),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = all_flags(result.stdout)
    preflight = next(c for c in calls if len(c) > 2 and c[2] == "preflight")
    training = next(c for c in calls if len(c) > 2 and c[2] == "train")
    evaluations = [c for c in calls if len(c) > 2 and c[2] in ("eval", "probe")]
    assert "--event-balanced" not in training  # alias 已翻译为 canonical sampling-mode
    if prepared:
        assert "PREPARE" in result.stdout + result.stderr
        assert result.stdout.index("build_dataset.py") < result.stdout.index("launch.py preflight")
        for call in (preflight, training, *evaluations):
            assert value_of(call, "--event-balance-index") == "/auto/full_event_mapping.jsonl"
    else:
        assert "PREPARE" not in result.stdout + result.stderr
        assert "--event-balance-index" not in training


def test_pipeline_explicit_event_index_does_not_build_replacement(stub, tmp_path):
    out = tmp_path / "out"
    (out / "run_explicit").mkdir(parents=True)
    (out / "run_explicit/best.pt").write_bytes(b"stub")
    labels = tmp_path / "labels.jsonl"
    labels.write_text("{}")
    supplied = tmp_path / "supplied.jsonl"
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_full_pipeline.sh"), "--dataset-priors", "--event-balanced",
         "--event-balance-index", str(supplied), "--prior-labels", str(labels)],
        cwd=ROOT, env=dict(stub, OUTPUT_DIR=str(out), RUN_TAG="explicit", DATA_DIR=str(tmp_path / "data")),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PREPARE" not in result.stdout + result.stderr
    for call in all_flags(result.stdout):
        if len(call) > 2 and call[2] in ("train", "eval", "probe", "preflight"):
            assert value_of(call, "--event-balance-index") == str(supplied)
