"""消融的均衡入口、输入边界与内容合同回归，不加载真实模型。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_expert_ablation import common
from qwen3vl_local.action_prior import config, event_balance, metrics, train
from qwen3vl_local.action_prior.tests.test_event_balance import _write_source
from qwen3vl_local.action_expert_ablation.tests.test_common import _capture_train_sh_args


@pytest.mark.parametrize("variant", ["qwen_simple", "bev_only"])
def test_aliases_validation_and_prior_boundary(variant):
    parser = common.parser(variant)
    for options, mode in [([], "uniform"), (["--event-balanced"], "event_balanced"),
                          (["--event-balanced", "--no-event-balanced"], "uniform"),
                          (["--no-event-balanced", "--sampling-mode=event_balanced"], "event_balanced")]:
        assert parser.parse_args(options).sampling_mode == mode
        assert config.parser().parse_args(options).sampling_mode == mode
    with pytest.raises(ValueError, match="require --event-balance-index"):
        common.validate_args(parser.parse_args(["--event-balanced"]), variant)
    with pytest.raises(ValueError, match="do not accept"):
        common.validate_args(parser.parse_args(["--event-balanced-scene-priors"]), variant)
    with pytest.raises(ValueError, match="repeat cap positive"):
        common.validate_args(parser.parse_args(["--event-balance-max-frame-repeats", "0"]), variant)


@pytest.mark.parametrize("variant", ["qwen_simple", "bev_only"])
def test_contract_binds_sampling_and_allows_source_relocation(tmp_path, variant):
    source = _write_source(tmp_path)
    bev = tmp_path / "bev.pth"
    bev.write_bytes(b"fake-bev")
    base = tmp_path / "qwen"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"fake-base")
    args = common.parser(variant).parse_args(["--event-balanced", "--event-balance-index", str(source)])
    args.lead_bev_ckpt = str(bev)
    args.model_dir = str(base)
    original = common.build_contract(args, variant)
    assert original["identity_payload"]["event_balanced_sampling"] == event_balance.sampling_contract(args)
    moved = tmp_path / "moved"
    shutil.copytree(source.parent, moved)
    args.event_balance_index = str(moved / source.name)
    common.require_contract(original, common.build_contract(args, variant))
    for key, value in [("event_balance_max_frame_repeats", 2), ("event_balanced_epoch_samples", 24),
                       ("event_balance_route_diverse", False), ("best_selection_metric", "event_balanced_ade"),
                       ("sampling_mode", "uniform")]:
        previous = getattr(args, key)
        setattr(args, key, value)
        with pytest.raises(ValueError, match="contract mismatch"):
            common.require_contract(original, common.build_contract(args, variant))
        setattr(args, key, previous)


def test_event_metrics_match_mainline_without_prior_groups():
    sample = dict(event_balance_status="special_filtered", event_balance_all_special_buckets=["UE2", "RE2"])
    planning = dict(route_ade_m=3.0, waypoint_ade_m=2.0)
    runtime = SimpleNamespace(last_audit={"samples": 1})
    counts = common.metric_hooks().sample_counts(runtime, sample, planning)
    expected = metrics.grouped_counts({}, sample, planning)
    assert {k: v for k, v in expected.items() if "event_balance" in k} == {
        k: v for k, v in counts.items() if "event_balance" in k}
    assert not any("UNKNOWN" in k or "accepted" in k for k in counts)
    assert common.scalar_metrics_from_counts(counts) == train.metrics_from_counts(counts)


@pytest.mark.parametrize("variant", ["qwen_simple", "bev_only"])
def test_train_environment_and_cli_precedence(tmp_path, variant):
    argv = _capture_train_sh_args(
        tmp_path, variant,
        ["--event-balanced-epoch-samples=24", "--event-balance-max-frame-repeats", "3",
         "--best-selection-metric=natural_ade", "--no-event-balance-route-diverse"],
        {"EVENT_BALANCED": "1", "EVENT_BALANCE_INDEX": "full map.jsonl",
         "EVENT_BALANCED_EPOCH_SAMPLES": "48", "EVENT_BALANCE_MAX_FRAME_REPEATS": "4",
         "BEST_SELECTION_METRIC": "event_balanced_ade", "EVENT_BALANCE_ROUTE_DIVERSE": "1"},
    )
    # argv 前四项是 launcher/mode/variant；检查真正训练 parser 的最终取值。
    args = common.parser(variant).parse_args(argv[4:])
    assert args.sampling_mode == "event_balanced"
    assert args.event_balance_index == "full map.jsonl"
    assert args.event_balanced_epoch_samples == 24
    assert args.event_balance_max_frame_repeats == 3
    assert args.best_selection_metric == "natural_ade"
    assert args.event_balance_route_diverse is False


@pytest.mark.parametrize("variant", ["qwen_simple", "bev_only"])
@pytest.mark.parametrize("mode", ["auto", "explicit", "disabled", "resume"])
def test_pipeline_shares_preparation_and_forwards_eval_source(tmp_path, variant, mode):
    """执行真实 shell，仅替换 Python 的重型入口，检查自动准备与 train/eval 实参。"""
    index = tmp_path / "action index"
    index.mkdir()
    (index / "manifest.json").write_text("{}\n")
    for split in ("train", "val", "test"):
        (index / f"{split}.jsonl").write_text("{}\n")
    run = tmp_path / "run"
    run.mkdir()
    (run / "best.pt").write_bytes(b"fake")
    (run / "latest.pt").write_bytes(b"fake")
    source = str(tmp_path / "full map.jsonl")
    args = ["--event-balanced"]
    if mode == "explicit":
        args += ["--event-balance-index", source]
    elif mode == "disabled":
        args = ["--no-event-balanced"]
    elif mode == "resume":
        saved = vars(common.parser(variant).parse_args([]))
        saved.update(data_dir=str(index), sampling_mode="event_balanced", event_balance_index="old/map.jsonl")
        (run / "config.json").write_text(json.dumps(saved))
        args = ["--resume", str(run / "latest.pt"), "--event-balance-index", source]
    binary = tmp_path / "bin"
    binary.mkdir()
    stub = binary / "python"
    stub.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
if sys.argv[1] == "-":
    os.execv(os.environ["REAL_PYTHON"], [os.environ["REAL_PYTHON"], *sys.argv[1:]])
with open(os.environ["CALL_LOG"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1].endswith("prepare_event_balance.py"):
    print(os.environ["AUTO_SOURCE"])
''')
    stub.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("EVENT_", "DATA_", "RESUME", "OUTPUT_DIR"))}
    env.update(PATH=str(binary) + os.pathsep + env["PATH"], CALL_LOG=str(log), REAL_PYTHON=sys.executable,
               AUTO_SOURCE=source, DATA_DIR=str(index), OUTPUT_DIR=str(run), NO_RUN_SUBDIR="1")
    if mode == "disabled":
        env["EVENT_BALANCED"] = "1"
    result = subprocess.run(["bash", f"qwen3vl_local/action_expert_ablation/{variant}/run_full_pipeline.sh", *args],
                            cwd=common.AUTOMOT_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    prep = [call for call in calls if call[0].endswith("prepare_event_balance.py")]
    assert len(prep) == int(mode == "auto")
    training, evaluation = [call for call in calls if call[0].endswith("launch.py")]
    if mode != "disabled":
        assert training[training.index("--event-balance-index") + 1] == source
        assert evaluation[evaluation.index("--event-balance-index") + 1] == source
    else:
        assert training[training.index("--sampling-mode") + 1] == "uniform"
        assert "--event-balance-index" not in evaluation
    if mode == "resume":
        assert "--sampling-mode" not in training
        assert "--event-balanced-epoch-samples" not in training


def test_mainline_and_ablation_share_first_build_lock(tmp_path):
    """三条入口同时首次建索引只执行一次 builder，不提前消费未发布 manifest 的 split。"""
    script = r'''
set -euo pipefail
source qwen3vl_local/action_expert_ablation/pipeline_common.sh
python() {
  local out="${@: -1}"
  echo build >> "$out/build_calls"
  printf '{}\n' > "$out/train.jsonl"
  printf '{}\n' > "$out/val.jsonl"
  printf '{}\n' > "$out/test.jsonl"
  # 在 builder 内验证三 split 已写好仍不能当作完整索引。
  if action_shared_index_ready "$out"; then return 3; fi
  sleep 0.15
  printf '{}\n' > "$out/manifest.json"
}
action_build_shared_index_if_needed lead_data "$1" &
first=$!
action_ablation_build_index_if_needed lead_data "$1" &
second=$!
action_ablation_build_index_if_needed lead_data "$1" &
third=$!
wait "$first"
wait "$second"
wait "$third"
action_shared_index_ready "$1"
'''
    index = tmp_path / "shared index"
    result = subprocess.run(["bash", "-c", script, "test", str(index)],
                            cwd=common.AUTOMOT_ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (index / "build_calls").read_text().splitlines() == ["build"]
