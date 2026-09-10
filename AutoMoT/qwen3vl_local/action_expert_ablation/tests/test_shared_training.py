"""CPU 小模型回归：真实入口共用同预算循环，不加载 Qwen/BEV 或数据集。"""
from pathlib import Path
import json
import os
import signal
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_expert_ablation import common
from qwen3vl_local.action_prior import train, runtime, launch, flow_matching, lora_bundle
from qwen3vl_local.action_prior.training_core import GracefulTerminationExit
from qwen3vl_local.action_prior.tests.test_training_loop import (
    TinyConfig, TinyFlowDecoder, lightweight_old_helpers,
)
import qwen3vl_local.leadmot as leadmot


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """只替换冻结模型和日志 IO，执行真实 FM loss、优化器、EMA 与保存/恢复。"""
    old = lightweight_old_helpers()
    monkeypatch.setitem(sys.modules, "qwen3vl_local.leadmot.train", old)
    monkeypatch.setattr(leadmot, "train", old, raising=False)
    monkeypatch.setattr(common, "_OLD_TRAIN", old)
    monkeypatch.setattr(leadmot, "LeadMoTPlanningDecoderConfig", TinyConfig)
    monkeypatch.setattr(flow_matching, "ConditionalFlowMatchingDecoder", TinyFlowDecoder)
    monkeypatch.setattr(common, "training_device", lambda rank: torch.device("cpu"))
    monkeypatch.setattr(train, "training_device", lambda rank: torch.device("cpu"))
    monkeypatch.setattr(launch, "ensure_gpu", lambda: 1)
    for key in ("ACTION_PRIOR_RUN_READY", "ACTION_ABLATION_RUN_READY", "WORLD_SIZE"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(lora_bundle, "preserve_for_training", lambda contract, out: contract)
    contract = {"identity": "cpu_test", "phase1": {}, "phase2": {}}
    monkeypatch.setattr(common, "build_contract", lambda *a: contract.copy())
    monkeypatch.setattr(train, "build_contract", lambda *a: contract.copy())
    rows = {}
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "val", "test"):
        rows[split] = [dict(scenario=split, run_id="route", route_group=f"{split}/route",
                            anchor=i, split=split) for i in range(5 if split == "train" else 2)]
        (data / f"{split}.jsonl").write_text("\n".join(map(json.dumps, rows[split])))
    monkeypatch.setattr(common, "read_rows", lambda args, split: rows[split])
    monkeypatch.setattr(train, "read_rows", lambda args, split: rows[split])
    state = SimpleNamespace(failure=None, train_cases=[], eval_cases=[], logs={})
    tb = ModuleType("torch.utils.tensorboard")
    def writer(path):
        logs = state.logs.setdefault(str(Path(path).parent), [])
        return SimpleNamespace(add_scalar=lambda *item: logs.append(item), flush=lambda: None,
                               close=lambda: None)
    tb.SummaryWriter = writer
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tb)

    class Runtime:
        device = torch.device("cpu")
        def __init__(self):
            self.prior = SimpleNamespace(last_audit={"invalid": {}, "analysis_truncated": False})
            self.last_audit = {"samples": 1}
        def forward_sample(self, sample, decoder, config, dtype, clip=None, **kwargs):
            if sample["split"] == "train":
                assert kwargs["sample_trajectory"] is False
                if state.failure == "mid_epoch" and len(state.train_cases) == 2:
                    raise RuntimeError("injected interruption")
                if state.failure == "sigterm" and len(state.train_cases) == 2:
                    os.kill(os.getpid(), signal.SIGTERM)
                state.train_cases.append(sample["anchor"])
            else:
                assert kwargs["sample_trajectory"] is True
                if state.failure == "validation":
                    raise RuntimeError("injected interruption")
                if state.failure == "sigterm_validation":
                    state.failure = "sigterm_validation_sent"
                    os.kill(os.getpid(), signal.SIGTERM)
                state.eval_cases.append(sample["anchor"])
            return decoder(**kwargs)
    monkeypatch.setattr(common, "make_runtime", lambda *a: Runtime())
    monkeypatch.setattr(runtime, "make_runtime", lambda *a: Runtime())

    def run(variant, out, *, resume=False, limit=0, val_steps=2):
        argv = ["train", "--data-dir", str(data), "--output-dir", str(out),
                "--num-epochs", "2", "--grad-accum-steps", "2", "--num-workers", "0",
                "--val-steps", str(val_steps), "--save-steps", "1", "--logging-steps", "1",
                "--max-train-steps", str(limit), "--decoder-dtype", "float32"]
        if resume:
            argv += ["--resume", str(out / "latest.pt")]
        monkeypatch.setattr(sys, "argv", argv)
        if variant == "prior":
            train.main()
        else:
            common.train_main(variant)
        return torch.load(out / "latest.pt", weights_only=False)
    return run, state


def test_three_entries_have_identical_budget_updates_and_core_metrics(harness, tmp_path):
    """相同模拟条件下，三条入口的样本、尾窗口、验证、参数与核心 loss 完全一致。"""
    run, state = harness
    results = []
    for variant in ("prior", "qwen_simple", "bev_only"):
        state.train_cases.clear(); state.eval_cases.clear()
        out = tmp_path / variant
        checkpoint = run(variant, out)
        core_tags = {"train/loss", "train/route_fm_mse", "train/waypoint_fm_mse", "train/lr",
                     "train/samples", "train/samples_seen", "train/step_samples", "val/loss", "val_epoch/route_ade_m", "val_epoch/waypoint_ade_m"}
        logs = state.logs[str(out)]
        results.append((checkpoint, list(state.train_cases), list(state.eval_cases),
                        [item for item in logs if item[0] in core_tags]))
        assert checkpoint["step"] == 6
        assert [value for tag, value, step in logs if tag == "train/samples"] == [2, 2, 1, 2, 2, 1]
        assert [value for tag, value, step in logs if tag == "train/samples_seen"] == [2, 4, 5, 7, 9, 10]
        assert [value for tag, value, step in logs if tag == "train/step_samples"] == [2, 2, 1, 2, 2, 1]
        if variant != "prior":
            assert not (out / "audit").exists()
            assert not (out / "epoch_audit").exists()
            assert not any("prior/" in tag or "group/" in tag for tag, *_ in logs)
    for result in results[1:]:
        assert result[1:] == results[0][1:]
        assert torch.equal(result[0]["decoder"]["weight"], results[0][0]["decoder"]["weight"])
        assert torch.equal(result[0]["ema_state_dict"]["shadow"]["weight"],
                           results[0][0]["ema_state_dict"]["shadow"]["weight"])


@pytest.mark.parametrize("variant", ["qwen_simple", "bev_only"])
@pytest.mark.parametrize("failure,limit", [("mid_epoch", 0), ("validation", 0), ("validation", 2)])
def test_ablation_resume_and_completed_budget(harness, tmp_path, variant, failure, limit):
    """覆盖中途/epoch 验证/截断预算验证中断，以及完成后续训不再增加 step。"""
    run, state = harness
    baseline = tmp_path / "baseline"
    reference = run(variant, baseline, limit=limit, val_steps=100)
    state.train_cases.clear(); state.eval_cases.clear()
    state.failure = failure
    out = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="injected interruption"):
        run(variant, out, limit=limit, val_steps=100)
    partial = torch.load(out / "latest.pt", weights_only=False)
    assert bool(partial["cursor"].get("validation_pending")) == (failure == "validation")
    state.failure = None
    restored = run(variant, out, resume=True, limit=limit, val_steps=100)
    assert restored["step"] == reference["step"]
    assert restored["cursor"] == reference["cursor"]
    assert (out / "best.pt").is_file()
    for key in ("decoder",):
        assert torch.equal(restored[key]["weight"], reference[key]["weight"])
    assert torch.equal(restored["ema_state_dict"]["shadow"]["weight"],
                       reference["ema_state_dict"]["shadow"]["weight"])
    calls = (len(state.train_cases), len(state.eval_cases))
    finished = run(variant, out, resume=True, limit=limit, val_steps=100)
    assert finished["step"] == reference["step"]
    assert (len(state.train_cases), len(state.eval_cases)) == calls


def test_real_tensorboard_resume_trimming(tmp_path):
    """有 TensorBoard 的环境执行真实事件裁剪；缺依赖时明确跳过。"""
    pytest.importorskip("tensorboard")
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from qwen3vl_local.action_prior.training_core import trim_tensorboard_for_resume

    tb = tmp_path / "tb"
    writer = EventFileWriter(str(tb))
    for step in (1, 2, 3):
        writer.add_event(Event(step=step, summary=Summary(value=[
            Summary.Value(tag="train/loss", simple_value=1.0 / step),
        ])))
    writer.close()
    original = set(tb.glob("events.out.tfevents.*"))
    trim_tensorboard_for_resume(tb, 2)
    def steps():
        return sorted(event.step for path in tb.glob("events.out.tfevents.*")
                      for event in EventFileLoader(str(path)).Load()
                      if any(value.tag == "train/loss" for value in event.summary.value))
    assert steps() == [1, 2]
    archived = list((tmp_path / "tb_resume_archive").rglob("events.out.tfevents.*"))
    assert {p.name for p in archived} == {p.name for p in original}
    trim_tensorboard_for_resume(tb, 2)
    assert steps() == [1, 2]


@pytest.mark.parametrize("variant", ["qwen_simple", "bev_only"])
def test_sigterm_saves_safe_cursor_and_resumes_exactly(harness, tmp_path, variant):
    """SIGTERM 完成当前累积窗后保存，以 143 退出，随后可从安全 cursor 精确恢复。"""
    run, state = harness
    baseline = tmp_path / "baseline"
    reference = run(variant, baseline, val_steps=100)
    state.train_cases.clear()
    state.eval_cases.clear()
    state.failure = "sigterm"
    out = tmp_path / "terminated"
    with pytest.raises(GracefulTerminationExit) as caught:
        run(variant, out, val_steps=100)
    assert caught.value.code == 128 + signal.SIGTERM

    partial = torch.load(out / "latest.pt", weights_only=False)
    marker = json.loads((out / "termination.json").read_text(encoding="utf-8"))
    assert partial["step"] == marker["optimizer_step"] == 2
    assert partial["cursor"] == marker["cursor"] == {"epoch": 0, "micro": 4}
    assert marker["signal_name"] == "SIGTERM"

    state.failure = None
    restored = run(variant, out, resume=True, val_steps=100)
    assert not (out / "termination.json").exists()
    assert list((out / "termination_history").glob("termination_*.json"))
    assert restored["cursor"] == reference["cursor"]
    assert restored["step"] == reference["step"]
    assert torch.equal(restored["decoder"]["weight"], reference["decoder"]["weight"])
    assert torch.equal(
        restored["ema_state_dict"]["shadow"]["weight"],
        reference["ema_state_dict"]["shadow"]["weight"],
    )


def test_sigterm_during_validation_preserves_pre_validation_cursor(harness, tmp_path):
    """验证中止不发布残缺指标，并从验证前已经完成的 optimizer cursor 恢复。"""
    run, state = harness
    baseline = tmp_path / "baseline_validation"
    reference = run("bev_only", baseline, val_steps=1)
    state.train_cases.clear()
    state.eval_cases.clear()
    state.failure = "sigterm_validation"
    out = tmp_path / "terminated_validation"
    with pytest.raises(GracefulTerminationExit):
        run("bev_only", out, val_steps=1)
    partial = torch.load(out / "latest.pt", weights_only=False)
    assert partial["step"] == 1
    assert partial["cursor"] == {"epoch": 0, "micro": 2}
    assert not (out / "validation/step_00000001.json").exists()

    state.failure = None
    restored = run("bev_only", out, resume=True, val_steps=1)
    assert restored["step"] == reference["step"]
    assert restored["cursor"] == reference["cursor"]
    assert torch.equal(restored["decoder"]["weight"], reference["decoder"]["weight"])
