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
    for name in ("run_full_pipeline.sh", "event_balance_common.sh", "resume.sh",
                 "resume.py", "eval.sh", "probe.sh"):
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
    config.update(dataset_priors=True, prior_labels="old labels.jsonl",
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
                "LEAD_BEV_CKPT", "DDP_GPU_COUNT", "NPROC_PER_NODE", "BENCH2DRIVE", "PIPELINE_LOG", "NO_RUN_SUBDIR"):
        env.pop(key, None)
    env.update(PYTHONPATH=str(ROOT), TRACE_FILE=str(tmp_path / "trace.jsonl"),
               LATEST_LINK=str(link), OTHER_RUN=str(other), OUTPUT_DIR=str(tmp_path / "output"), RUN_TAG="test")
    return scripts, run, link, env


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
    assert restored.dataset_priors is True
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


@pytest.mark.parametrize("style", ["cli", "environment"])
def test_resume_path_overrides_reach_training_test_and_probe(pipeline, style):
    """显式路径覆盖贯穿旧 best.pt 的评测，CLI 同时覆盖环境变量。"""
    scripts, run, link, env = pipeline
    args = ["--resume", str(link / "latest.pt")]
    paths = dict(data_root="new data", data_dir="new index", model_dir="new qwen",
                 lead_bev_ckpt="new bev.pth", prior_labels="new labels.jsonl", event_balance_index="new map.jsonl")
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


@pytest.mark.parametrize("args", [["--resume"], ["--resume="], ["--resume", "--event-balanced"],
                                  ["--resume", "/missing/action/checkpoint.pt"]])
def test_bad_resume_fails_before_creating_run(pipeline, args):
    """缺值或不存在的 checkpoint 不得误走新训练或数据构建。"""
    scripts, run, link, env = pipeline
    result = subprocess.run(["bash", str(scripts / "run_full_pipeline.sh"), *args],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not Path(env["TRACE_FILE"]).exists()
    assert not Path(env["OUTPUT_DIR"]).exists()
