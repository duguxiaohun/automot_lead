"""执行真实 pipeline/resume 配置恢复，仅将模型 launcher 换成命令记录器。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen3vl_local.action_prior.config import parser


@pytest.fixture
def pipeline(tmp_path):
    """真实 resume.py 读取旧配置；缺少 train/builder 桩使误走新训练立即失败。"""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("run_full_pipeline.sh", "train.sh", "event_balance_common.sh", "resume.sh",
                 "resume.py", "eval.sh", "probe.sh", "scene_policy.py"):
        shutil.copy(SCRIPTS / name, scripts / name)
    (scripts / "launch.py").write_text('''import json, os, sys
from pathlib import Path
with open(os.environ["TRACE_FILE"], "a") as f:
    f.write(json.dumps(dict(argv=sys.argv[1:], resume=os.environ.get("RESUME"),
                           world=os.environ.get("DDP_GPU_COUNT"))) + "\\n")
if sys.argv[1] == "train":
    link = Path(os.environ["LATEST_LINK"])
    link.unlink()
    link.symlink_to(os.environ["OTHER_RUN"], target_is_directory=True)
''')
    (scripts / "audit_bundle.py").write_text("# 本测试不生成审计包。\n")
    run = tmp_path / "original run"
    run.mkdir()
    (run / "latest.pt").write_bytes(b"fixture")
    (run / "best.pt").write_bytes(b"fixture")
    config = vars(parser().parse_args([]))
    config.update(action_token_separation_weight=0.025, action_token_separation_margin=0.65, dataset_priors=True, prior_labels="old labels.jsonl",
                  sampling_mode="event_balanced", event_balance_index="old map.jsonl",
                  event_balanced_epoch_samples=48, event_balance_max_frame_repeats=4,
                  data_dir="old index", data_root="old data", learning_rate=0.0001,
                  num_epochs=7, grad_accum_steps=8, analysis_review=False)
    (run / "config.json").write_text(json.dumps(config))
    (run / "training_plan.json").write_text(json.dumps(dict(world_size=2)))
    (run / "selected_priors.json").write_text("{}")
    other = tmp_path / "other run"
    other.mkdir()
    (other / "latest.pt").write_bytes(b"other")
    link = tmp_path / "latest"
    link.symlink_to(run, target_is_directory=True)
    env = os.environ.copy()
    for key in ("RESUME", "DATASET_PRIORS", "PRIOR_LABELS", "EVENT_BALANCED", "EVENT_BALANCE_INDEX",
                "EVENT_BALANCED_SCENE_PRIORS", "EVENT_BALANCED_EPOCH_SAMPLES", "EVENT_BALANCE_MAX_FRAME_REPEATS",
                "EVENT_BALANCE_ROUTE_DIVERSE", "BEST_SELECTION_METRIC", "DATA_ROOT", "DATA_DIR", "MODEL_DIR",
                "LEAD_BEV_CKPT", "DDP_GPU_COUNT", "NPROC_PER_NODE", "BENCH2DRIVE", "PIPELINE_LOG", "NO_RUN_SUBDIR",
                "GENERATE_ANALYSIS", "HIGH_LEVEL_PLANNING", "HIGH_LEVEL_ACTION_PRIOR", "HIGH_LEVEL_ACTION_INDEX", "ANALYSIS_REVIEW", "NUM_EPOCHS", "LR", "GRAD_ACCUM", "VAL_STEPS",
                "SAVE_STEPS", "NUM_WORKERS", "LOGGING_STEPS", "PRIOR_NOISE", "PRIOR_NOISE_INVALID_SHARE",
                "CHECKPOINT_ROOT", "SELECTION_POLICY", "PHASE1_ADAPTER", "PHASE2_ADAPTER", "FLOW_SAMPLE_STEPS",
                "FLOW_ROUTE_COORDINATE_SCALE_M", "FLOW_WAYPOINT_COORDINATE_SCALE_M", "FLOW_TIME_EMBED_DIM",
                "FLOW_TRAJECTORY_LAYERS", "FLOW_TRAJECTORY_HEADS", "TRAIN_SAMPLED_METRICS"):
        env.pop(key, None)
    env.update(PYTHONPATH=str(ROOT), TRACE_FILE=str(tmp_path / "trace.jsonl"),
               LATEST_LINK=str(link), OTHER_RUN=str(other), OUTPUT_DIR=str(tmp_path / "output"), RUN_TAG="test")
    return scripts, run, link, env


@pytest.mark.parametrize("entry", ["train.sh", "run_full_pipeline.sh"])
@pytest.mark.parametrize("override", [False, True])
def test_optimization_resume_preserves_saved_values_and_cli_priority(pipeline, entry, override):
    """两个 shell 恢复入口均传递原优化配置；显式更改由 checkpoint 合同进一步拒绝。"""
    scripts, run, link, env = pipeline
    from qwen3vl_local.action_prior.optimization_config import OPTIMIZATION_ENV
    for key in OPTIMIZATION_ENV:
        env.pop(key, None)
    config_path = run / "config.json"
    saved = json.loads(config_path.read_text())
    saved.update(optimizer="adamw", lr_scheduler="cosine")
    config_path.write_text(json.dumps(saved))
    extra = []
    if override:
        env.update(OPTIMIZER="muon_adamw", LR_SCHEDULER="cosine_restarts")
        extra = ["--optimizer", "adamw"]
    result = subprocess.run(["bash", str(scripts / entry), "--resume", str(link / "latest.pt"), *extra],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    training = json.loads(Path(env["TRACE_FILE"]).read_text().splitlines()[0])
    args = parser().parse_args(training["argv"][1:])
    assert args.optimizer == "adamw"
    assert args.lr_scheduler == ("cosine_restarts" if override else "cosine")


@pytest.mark.parametrize('budget,cap', [(0, 8), (95136, 8), (116256, 11), (None, None)])
def test_resume_retains_sampling_budget_including_legacy_missing_fields(pipeline, budget, cap):
    scripts, run, link, env = pipeline
    saved = json.loads((run / 'config.json').read_text())
    for key, value in [('event_balanced_epoch_samples', budget), ('event_balance_max_frame_repeats', cap)]:
        if value is None:
            saved.pop(key)
        else:
            saved[key] = value
    (run / 'config.json').write_text(json.dumps(saved))
    result = subprocess.run(['bash', str(scripts / 'run_full_pipeline.sh'), '--resume', str(link / 'latest.pt')],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    call = json.loads(Path(env['TRACE_FILE']).read_text().splitlines()[0])
    restored = parser().parse_args(call['argv'][1:])
    assert restored.event_balanced_epoch_samples == (0 if budget is None else budget)
    assert restored.event_balance_max_frame_repeats == (8 if cap is None else cap)


@pytest.mark.parametrize("style", ["separate", "equals", "environment"])
def test_resume_restores_config_and_pins_real_run(pipeline, style):
    """三种入口一致，且训练期间 latest 改指不会把最终评测切换到另一 run。"""
    scripts, run, link, env = pipeline
    checkpoint = str(link / "latest.pt")
    if style == "separate":
        args = ["--resume", checkpoint]
        env["RESUME"] = "ignored-environment-path.pt"
    elif style == "equals":
        args = [f"--resume={checkpoint}"]
    else:
        args = []
        env["RESUME"] = checkpoint
    result = subprocess.run(["bash", str(scripts / "run_full_pipeline.sh"), *args],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    train, evaluate, probe = [json.loads(line) for line in Path(env["TRACE_FILE"]).read_text().splitlines()]
    assert [c["argv"][0] for c in (train, evaluate, probe)] == ["train", "eval", "probe"]
    restored = parser().parse_args(train["argv"][1:])
    assert (restored.optimizer, restored.lr_scheduler) == ("muon_adamw", "cosine_restarts")
    assert restored.action_token_separation_weight == 0.025
    assert restored.action_token_separation_margin == 0.65
    assert restored.dataset_priors is True
    assert restored.generate_analysis is False
    assert restored.data_dir == "old index"
    assert restored.prior_labels == "old labels.jsonl"
    assert restored.sampling_mode == "event_balanced"
    assert restored.event_balance_index == "old map.jsonl"
    assert restored.event_balanced_epoch_samples == 48
    assert restored.event_balance_max_frame_repeats == 4
    assert (restored.learning_rate, restored.num_epochs, restored.grad_accum_steps) == (0.0001, 7, 8)
    assert train["world"] == "2"
    assert train["resume"] == str(run / "latest.pt")
    for call in (evaluate, probe):
        argv = call["argv"]
        assert argv[argv.index("--checkpoint") + 1] == str(run / "best.pt")
        assert "--data-dir" not in argv
        assert "--prior-labels" not in argv
    assert not (Path(env["OUTPUT_DIR"]) / "run_test").exists()


@pytest.mark.parametrize("option", ["generate_analysis", "high_level_action_prior"])
@pytest.mark.parametrize("entrypoint", ["run_full_pipeline.sh", "train.sh"])
@pytest.mark.parametrize("saved,environment,cli,expected", [
    (True, None, [], True), (False, None, [], False), (None, None, [], True),
    (True, "0", [], False), (False, "1", [], True),
    (True, "1", ["--no-generate-analysis"], False),
    (False, "0", ["--generate-analysis"], True),
])
def test_resume_preserves_generation_mode_and_explicit_precedence(pipeline, saved, environment, cli, expected, entrypoint, option):
    """真实 resume 恢复摘要配置；显式覆盖仍由后续正式 checkpoint 合同守卫拒绝错配。"""
    scripts, run, link, env = pipeline
    cli = [value.replace("generate-analysis", option.replace("_", "-")) for value in cli]
    if saved is None and option != "generate_analysis":
        expected = False
    cfg = json.loads((run / "config.json").read_text())
    if saved is None:
        cfg.pop(option)
    else:
        cfg[option] = saved
    (run / "config.json").write_text(json.dumps(cfg))
    if environment is not None:
        env[option.upper()] = environment
    result = subprocess.run(
        ["bash", str(scripts / entrypoint), "--resume", str(link / "latest.pt"), *cli],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line)["argv"] for line in Path(env["TRACE_FILE"]).read_text().splitlines()]
    assert getattr(parser().parse_args(calls[0][1:]), option) is expected
    # 最终 test/probe 读取同一 checkpoint，不能被 launcher 环境临时改变输入分布。
    for call in calls[1:]:
        assert "--" + option.replace("_", "-") not in call
        assert "--no-" + option.replace("_", "-") not in call


@pytest.mark.parametrize("style", ["separate", "equals", "environment"])
def test_direct_train_resume_recovers_full_config_before_defaults(pipeline, style):
    """底层 train.sh 三种续训写法恢复原 LR、索引、摘要及卡数，CLI 路径优先。"""
    scripts, run, link, env = pipeline
    cfg = json.loads((run / "config.json").read_text())
    cfg["generate_analysis"] = True
    (run / "config.json").write_text(json.dumps(cfg))
    checkpoint = str(link / "latest.pt")
    env["RESUME"] = checkpoint if style == "environment" else "/ignored/environment.pt"
    args = (["--resume", checkpoint] if style == "separate"
            else ["--resume=" + checkpoint] if style == "equals" else [])
    result = subprocess.run(["bash", str(scripts / "train.sh"), *args], cwd=ROOT,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in Path(env["TRACE_FILE"]).read_text().splitlines()]
    assert len(calls) == 1 and calls[0]["argv"][0] == "train"
    restored = parser().parse_args(calls[0]["argv"][1:])
    for key in ("generate_analysis", "analysis_review", "learning_rate", "num_epochs", "grad_accum_steps",
                "dataset_priors", "prior_labels", "data_dir", "data_root", "sampling_mode", "event_balance_index"):
        assert getattr(restored, key) == cfg[key]
    assert calls[0]["resume"] == str(run / "latest.pt") and calls[0]["world"] == "2"
    assert not Path(env["OUTPUT_DIR"]).exists()


@pytest.mark.parametrize("cli_override", [False, True])
def test_direct_train_resume_only_forwards_explicit_env_overrides(pipeline, cli_override):
    """默认不覆盖，显式环境仍有效，CLI 对标量、路径和布尔参数均优先。"""
    scripts, run, link, env = pipeline
    env.update(LR="0.0003", DATA_ROOT="environment data", ANALYSIS_REVIEW="1",
               FLOW_SAMPLE_STEPS="12", NUM_WORKERS="3")
    cli = (["--learning-rate", "0.0004", "--data-root", "CLI data", "--no-analysis-review"]
           if cli_override else [])
    result = subprocess.run(["bash", str(scripts / "train.sh"), "--resume", str(link / "latest.pt"), *cli],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    call = json.loads(Path(env["TRACE_FILE"]).read_text())
    restored = parser().parse_args(call["argv"][1:])
    assert restored.learning_rate == (0.0004 if cli_override else 0.0003)
    assert restored.data_root == ("CLI data" if cli_override else "environment data")
    assert restored.analysis_review is (not cli_override)
    assert restored.flow_sample_steps == 12 and restored.num_workers == 3
    assert restored.num_epochs == 7 and restored.grad_accum_steps == 8


@pytest.mark.parametrize("style", ["cli", "environment"])
def test_resume_path_overrides_reach_training_test_and_probe(pipeline, style):
    """显式路径覆盖贯穿旧 best.pt 的评测，CLI 同时覆盖环境变量。"""
    scripts, run, link, env = pipeline
    args = ["--resume", str(link / "latest.pt")]
    paths = dict(data_root="new data", data_dir="new index", model_dir="new qwen",
                 lead_bev_ckpt="new bev.pth", prior_labels="new labels.jsonl", event_balance_index="new map.jsonl", high_level_action_index="new actions.jsonl")
    for key, value in paths.items():
        env[key.upper()] = value if style == "environment" else "ignored path"
        if style == "cli":
            args.append("--" + key.replace("_", "-") + "=" + value)
    result = subprocess.run(["bash", str(scripts / "run_full_pipeline.sh"), *args],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line)["argv"] for line in Path(env["TRACE_FILE"]).read_text().splitlines()]
    restored = parser().parse_args(calls[0][1:])
    for key, value in paths.items():
        assert getattr(restored, key) == value
        option = "--" + key.replace("_", "-")
        for argv in calls[1:]:
            assert argv[argv.index(option) + 1] == value


@pytest.mark.parametrize("entrypoint", ["run_full_pipeline.sh", "train.sh"])
@pytest.mark.parametrize("args", [["--resume"], ["--resume="], ["--resume", "--event-balanced"],
                                  ["--resume", "/missing/action/checkpoint.pt"]])
def test_bad_resume_fails_before_creating_run(pipeline, args, entrypoint):
    """缺值或不存在的 checkpoint 不得误走新训练或数据构建。"""
    scripts, run, link, env = pipeline
    result = subprocess.run(["bash", str(scripts / entrypoint), *args],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not Path(env["TRACE_FILE"]).exists()
    assert not Path(env["OUTPUT_DIR"]).exists()


def test_legacy_resume_does_not_inherit_new_separation_default(pipeline):
    scripts, run, link, env = pipeline
    path = run / "config.json"
    cfg = json.loads(path.read_text())
    cfg.pop("action_token_separation_weight")
    cfg.pop("action_token_separation_margin")
    path.write_text(json.dumps(cfg))
    result = subprocess.run(["bash", str(scripts / "train.sh"), "--resume", str(link / "latest.pt")],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    invocation = json.loads(Path(env["TRACE_FILE"]).read_text().splitlines()[0])
    restored = parser().parse_args(invocation["argv"][1:])
    assert restored.action_token_separation_weight == 0
