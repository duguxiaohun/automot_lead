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
    TinyConfig, TinyFlowDecoder, lightweight_old_helpers, stub_toy_sampling,
)
import qwen3vl_local.leadmot as leadmot


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """替换模型/数据/日志 IO，小样本采样用夹具；执行真实 loss、优化器、EMA 与保存/恢复。"""
    # 测试使用 CPU 小模型；不能由其它入口选卡后的宿主 CUDA 状态触发 GPU 内存统计。
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    stub_toy_sampling(monkeypatch)
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
    from qwen3vl_local.action_prior.contracts import SCHEMA
    contract = {"schema": SCHEMA, "identity": "cpu_test", "phase1": {}, "phase2": {}}
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
    state = SimpleNamespace(failure=None, train_cases=[], eval_cases=[], logs={}, rows=rows)
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
                if state.failure == "after_epoch" and len(state.train_cases) == 5:
                    raise RuntimeError("injected after epoch interruption")
                if state.failure == "sigterm" and len(state.train_cases) == 2:
                    os.kill(os.getpid(), signal.SIGTERM)
                if state.failure == "sigterm_epoch" and len(state.train_cases) == 4:
                    os.kill(os.getpid(), signal.SIGTERM)
                state.train_cases.append(sample["anchor"])
            else:
                assert kwargs["sample_trajectory"] is True
                if state.failure == "cycle_validation" and len(state.train_cases) == 5:
                    raise RuntimeError("injected cycle interruption")
                if state.failure == "sigterm_cycle" and len(state.train_cases) == 5:
                    state.failure = "sigterm_cycle_sent"
                    os.kill(os.getpid(), signal.SIGTERM)
                if state.failure == "validation":
                    raise RuntimeError("injected interruption")
                if state.failure == "sigterm_validation":
                    state.failure = "sigterm_validation_sent"
                    os.kill(os.getpid(), signal.SIGTERM)
                state.eval_cases.append(sample["anchor"])
            return decoder(**kwargs)
    monkeypatch.setattr(common, "make_runtime", lambda *a: Runtime())
    monkeypatch.setattr(runtime, "make_runtime", lambda *a: Runtime())

    def run(variant, out, *, resume=False, limit=0, val_steps=2, extra=()):
        argv = ["train", "--data-dir", str(data), "--output-dir", str(out),
                "--num-epochs", "2", "--grad-accum-steps", "2", "--num-workers", "0",
                "--val-steps", str(val_steps), "--save-steps", "1", "--logging-steps", "1",
                "--max-train-steps", str(limit), "--decoder-dtype", "float32"]
        argv += list(extra)
        if resume:
            argv += ["--resume", str(out / "latest.pt")]
        monkeypatch.setattr(sys, "argv", argv)
        if variant == "prior":
            train.main()
        else:
            common.train_main(variant)
        return torch.load(out / "latest.pt", weights_only=False)
    return run, state


@pytest.mark.parametrize("variant", ["prior", "qwen_simple", "bev_only"])
@pytest.mark.parametrize("failure", ["cycle_validation", "sigterm_cycle"])
def test_cycle_full_validation_deduplicates_and_resumes(harness, tmp_path, variant, failure):
    """周期3/9与epoch重合，最终12与epoch/小验证重合；验证中断后无更新丢失。"""
    run, state = harness
    cli = ("--num-epochs", "4", "--val-max-samples", "1")
    baseline = tmp_path / "cycle_baseline"
    reference = run(variant, baseline, extra=cli, val_steps=4)
    # 完整验证3/6/9/12，各2样本；step4/8各小验证1样本。
    assert len(state.eval_cases) == 10
    cycle_file = baseline / "validation/epoch_001_step00000003.json"
    assert json.loads(cycle_file.read_text())["samples"] == 2
    final = json.loads((baseline / "validation/epoch_004_step00000012.json").read_text())
    assert final["validation_cycle"] == 3 and final["validation_final"]
    assert (baseline / "validation/step_00000004.json").exists()
    assert not (baseline / "validation/step_00000012.json").exists()
    tags = state.logs[str(baseline)]
    assert [step for tag, _, step in tags if tag == 'val_cycle/route_ade_m'] == [3, 9, 12]
    assert any(tag.startswith('train/update/') for tag, _, _ in tags)

    state.train_cases.clear(); state.eval_cases.clear()
    state.failure = failure
    out = tmp_path / "cycle_interrupted"
    expected = RuntimeError if failure == "cycle_validation" else GracefulTerminationExit
    with pytest.raises(expected):
        run(variant, out, extra=cli, val_steps=4)
    partial = torch.load(out / "latest.pt", weights_only=False)
    assert partial['step'] == 3 and partial['cursor']['validation_cycle'] == 1
    assert partial['cursor']['epoch'] == 1 and partial['cursor']['micro'] == 0
    assert not (out / "validation/epoch_001_step00000003.json").exists()
    state.failure = None
    resumed = run(variant, out, resume=True, extra=cli, val_steps=4)
    assert len(state.train_cases) == 20
    assert resumed['cursor'] == reference['cursor']
    for name in reference['decoder']:
        assert torch.equal(reference['decoder'][name], resumed['decoder'][name])
    assert resumed['best_sampled_trajectory_score'] == reference['best_sampled_trajectory_score']
    if variant == 'prior':
        for path in (baseline / 'epoch_audit').glob('*.json'):
            assert json.loads(path.read_text()) == json.loads((out / 'epoch_audit' / path.name).read_text())


@pytest.mark.parametrize("variant", ["prior", "qwen_simple", "bev_only"])
def test_cycle_validation_can_select_best(harness, tmp_path, monkeypatch, variant):
    """周期完整验证默认参与best，最终成绩独立于历史best。"""
    run, state = harness
    module = train if variant == 'prior' else common
    original = module.evaluate
    def evaluate(*args, **kwargs):
        metrics = original(*args, **kwargs)
        score = 0.0 if len(state.train_cases) == 5 else 10.0
        return dict(metrics, route_ade_m=score, waypoint_ade_m=score)
    monkeypatch.setattr(module, 'evaluate', evaluate)
    cli = ("--num-epochs", "4")
    out = tmp_path / 'cycle_best'
    run(variant, out, extra=cli, val_steps=100)
    assert torch.load(out / 'best.pt', weights_only=False)['step'] == 3
    final = json.loads((out / 'validation/final.json').read_text())
    assert final['optimizer_step'] == 12 and final['selection_score'] > final['best_selection_score']


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
    assert partial["cursor"] == marker["cursor"]
    assert partial["cursor"]["epoch"] == 0 and partial["cursor"]["micro"] == 4
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
    assert partial["cursor"]["epoch"] == 0 and partial["cursor"]["micro"] == 2
    assert not (out / "validation/step_00000001.json").exists()

    state.failure = None
    restored = run("bev_only", out, resume=True, val_steps=1)
    assert restored["step"] == reference["step"]
    assert restored["cursor"] == reference["cursor"]
    assert torch.equal(restored["decoder"]["weight"], reference["decoder"]["weight"])


@pytest.mark.parametrize("sampling", ["event_balanced", "action_balanced"])
def test_three_entries_event_balanced_updates_metrics_and_resume(harness, tmp_path, monkeypatch, sampling):
    """真实共享循环：同源均衡课程、best 指标、样本顺序和参数更新三组一致。"""
    from qwen3vl_local.action_prior.tests.test_event_balance import _write_source, _action_row
    from qwen3vl_local.action_prior.event_balance import EventBalanceIndex

    run, state = harness
    source = _write_source(tmp_path)
    from qwen3vl_local.action_prior.contracts import file_hash
    source_rows = [json.loads(line) for line in source.read_text().splitlines()]
    source.write_text("".join(json.dumps(dict(row, scenario=split, source_split=split)) + "\n"
                              for split in state.rows for row in source_rows))
    manifest_path = source.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["index_sha256"] = file_hash(source)
    manifest_path.write_text(json.dumps(manifest))
    index = EventBalanceIndex(source)
    for split in state.rows:
        state.rows[split] = [dict(_action_row(i), scenario=split, split=split,
                                 route_group=f"{split}/route_{i % 3}") for i in range(40)]
        index.annotate(state.rows[split])
    if sampling == "action_balanced":
        from qwen3vl_local.action_prior import action_token
        # Source IO is covered separately; this fixture exercises the actual shared loop.
        for rr in state.rows.values():
            for row in rr:
                name = "STOP" if row["event_balance_status"] == "special_eligible" else "UNCOND"
                row.update(action_token_id=action_token.ACTION_TOKEN_NAMES.index(name),
                           action_token=dict(name=name, reason="fixture", version=action_token.TOKEN_VERSION))
            from qwen3vl_local.action_prior.tests.test_action_balance import row as action_row
            # 全局动作均衡需六种语义动作都有独立容量；不再用全STOP伪造完整动作域。
            split = rr[0]["split"]
            for action in action_token.ACTION_TOKEN_NAMES[1:]:
                for i in range(2):
                    rr.append(action_row(100 + len(rr), ["UE2"], action, split=split))
    extra = ["--sampling-mode", sampling, "--event-balance-index", str(source),
             "--event-balanced-epoch-samples", "24", "--event-balance-max-frame-repeats", "2",
             "--best-selection-metric", "event_balanced_ade"]
    results = []
    for variant in ("prior", "qwen_simple", "bev_only"):
        state.train_cases.clear(); state.eval_cases.clear()
        out = tmp_path / variant
        ckpt = run(variant, out, extra=extra, val_steps=100)
        audit = json.loads((out / "sampling/epoch_001.json").read_text())
        assert audit["total"] == 24
        if sampling == "action_balanced":
            assert audit["sampled_cells"] == {k: v for k, v in audit["cell_quotas"].items() if v}
            assert "conditioner.action_embedding.weight" not in ckpt["decoder"]
        assert audit["max_frame_repeats"] <= 2
        metrics = json.loads(next((out / "validation").glob("*.json")).read_text())
        assert metrics["event_balance_bucket_coverage_complete"] == 1
        assert "event_balanced_route_ade_m" in metrics
        event_metrics = {k: v for k, v in metrics.items() if "event_balance" in k}
        results.append((ckpt, list(state.train_cases), list(state.eval_cases), audit, event_metrics))
        if sampling == "action_balanced":
            expected_order = list(state.train_cases)
            state.train_cases.clear(); state.eval_cases.clear()
            state.failure = "mid_epoch"
            interrupted = tmp_path / (variant + "_action_interrupted")
            with pytest.raises(RuntimeError, match="injected interruption"):
                run(variant, interrupted, extra=extra, val_steps=100)
            state.failure = None
            restored_action = run(variant, interrupted, resume=True, extra=extra, val_steps=100)
            assert state.train_cases == expected_order
            assert torch.equal(restored_action["decoder"]["weight"], ckpt["decoder"]["weight"])
            assert restored_action["cursor"] == ckpt["cursor"]
        if variant != "prior":
            assert not any("prior/" in tag or "group/condition/" in tag
                           for tag, *_ in state.logs[str(out)])
            calls = len(state.train_cases)
            restored = run(variant, out, resume=True, val_steps=100)
            assert restored["args"]["sampling_mode"] == sampling
            assert restored["args"]["event_balanced_epoch_samples"] == 24
            assert len(state.train_cases) == calls
    for result in results[1:]:
        assert result[1:] == results[0][1:]
        assert torch.equal(result[0]["decoder"]["weight"], results[0][0]["decoder"]["weight"])
        assert result[0]["best_sampled_trajectory_score"] == results[0][0]["best_sampled_trajectory_score"]


@pytest.mark.parametrize("optimizer", ["adamw", "muon_adamw"])
@pytest.mark.parametrize("schedule", ["cosine", "cosine_restarts"])
def test_optimizer_matrix_three_entries_and_resume(harness, tmp_path, monkeypatch, optimizer, schedule):
    """四种组合均执行真实矩阵更新，三入口结果相同，混合动量可精确恢复。"""
    class MatrixFlow(TinyFlowDecoder):
        def __init__(self, config, flow_config):
            super().__init__(config, flow_config)
            self.trajectory_blocks = torch.nn.Linear(2, 2)

        def forward(self, **kwargs):
            base = self.trajectory_blocks(super().forward(**kwargs)["flow_velocity"])
            return dict(flow_velocity=base, pred_route=base[:, :10], pred_future_waypoints=base[:, 10:])

    monkeypatch.setattr(flow_matching, "ConditionalFlowMatchingDecoder", MatrixFlow)
    run, state = harness
    cli = ("--optimizer", optimizer, "--lr-scheduler", schedule)
    reference = None
    for variant in ("prior", "qwen_simple", "bev_only"):
        state.train_cases.clear(); state.eval_cases.clear()
        result = run(variant, tmp_path / variant, extra=cli)
        assert result["optimization_contract"]["schedule"]["optimizer"] == optimizer
        if optimizer == "muon_adamw":
            assert any("momentum_buffer" in s for s in result["optimizer"]["state"].values())
        if reference is not None:
            assert reference["optimization_contract"] == result["optimization_contract"]
            for name, value in reference["decoder"].items():
                assert torch.equal(value, result["decoder"][name])
        reference = result
        state.train_cases.clear(); state.eval_cases.clear()
        state.failure = "mid_epoch"
        interrupted = tmp_path / (variant + "_resume")
        with pytest.raises(RuntimeError, match="injected interruption"):
            run(variant, interrupted, extra=cli)
        state.failure = None
        resumed = run(variant, interrupted, resume=True, extra=cli)
        for name, value in reference["decoder"].items():
            assert torch.equal(value, resumed["decoder"][name])
        with pytest.raises(ValueError, match="resume schedule mismatch"):
            run(variant, interrupted, resume=True, extra=(*cli, "--learning-rate", "0.01"))



@pytest.mark.parametrize('variant', ['prior', 'qwen_simple', 'bev_only'])
@pytest.mark.parametrize('failure', ['after_epoch', 'validation'])
def test_intermediate_audit_available_before_training_finishes(harness, tmp_path, variant, failure):
    """第一轮后异常退出，ZIP已经可审计；验证中断明确标pending，恢复后累积更新历史。"""
    import hashlib
    import zipfile
    run, state = harness
    state.failure = failure
    out = tmp_path / 'intermediate'
    with pytest.raises(RuntimeError, match='interruption'):
        run(variant, out, val_steps=100)
    archive = out / 'training_audit.zip'
    assert archive.is_file() and archive.stat().st_size < 30_000_000
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        content = json.loads(z.read('metrics.json'))
        latest = content['latest']
        assert latest['optimizer_step'] == 3 and latest['epoch'] == 1
        assert latest['epoch_training_complete'] and latest['train_metrics']['samples'] == 5
        assert latest['validation_pending'] == (failure == 'validation')
        assert (latest['validation'] is None) == (failure == 'validation')
        assert content['epoch_history'][0]['training_complete']
        manifest = json.loads(z.read('AUDIT_MANIFEST.json'))
        for item in manifest['included']:
            assert hashlib.sha256(z.read(item['path'])).hexdigest() == item['sha256']
        assert not any(name.endswith(('.pt', '.sqlite', '.jpg')) for name in z.namelist())
        assert any(name.startswith('audit/step_') for name in z.namelist())
    state.failure = None
    resumed = run(variant, out, resume=True, val_steps=100)
    with zipfile.ZipFile(archive) as z:
        content = json.loads(z.read('metrics.json'))
        assert content['latest']['optimizer_step'] == resumed['step'] == 6
        assert not content['latest']['validation_pending']
        assert [e['train']['samples'] for e in content['epoch_history']] == [5, 5]
        assert all(e['validation']['samples'] == 2 for e in content['epoch_history'])


@pytest.mark.parametrize('optimizer', ['adamw', 'muon_adamw'])
@pytest.mark.parametrize('scheduler', ['cosine', 'cosine_restarts'])
def test_default_shared_validation_and_audit_for_all_combinations(harness, tmp_path, optimizer, scheduler):
    """无验证开关时四组合仍有相同完整验证机会，每个epoch都有可审计训练计数。"""
    import zipfile
    run, state = harness
    cli = ('--num-epochs', '4', '--optimizer', optimizer, '--lr-scheduler', scheduler)
    out = tmp_path / 'shared_validation'
    ckpt = run('bev_only', out, extra=cli, val_steps=100)
    paths = sorted(p for p in (out / 'validation').glob('*_step*.json'))
    assert sorted(json.loads(p.read_text())['optimizer_step'] for p in paths) == [3, 6, 9, 12]
    assert ckpt['optimization_contract']['schedule']['shared_validation_updates'] == [3, 9, 12]
    final = json.loads((out / 'validation/final.json').read_text())
    assert final['optimizer_step'] == 12 and final['weight_view'] == 'ema'
    with zipfile.ZipFile(out / 'training_audit.zip') as z:
        history = json.loads(z.read('metrics.json'))['epoch_history']
        assert [item['epoch'] for item in history] == [1, 2, 3, 4]
        assert all(item['training_complete'] and item['train']['samples'] == 5 for item in history)


@pytest.mark.parametrize("variant", ["prior", "qwen_simple", "bev_only"])
def test_seven_epochs_cycle_boundaries_and_audit(harness, tmp_path, variant):
    """三条真实训练入口共用首轮warmup与1/2/4周期，最终审计覆盖完整七轮。"""
    import zipfile
    run, state = harness
    out = tmp_path / variant
    checkpoint = run(variant, out, extra=("--num-epochs", "7"), val_steps=100)
    schedule = checkpoint["optimization_contract"]["schedule"]
    assert checkpoint["step"] == 21 and checkpoint["args"]["num_epochs"] == 7
    assert schedule["optimizer_steps_per_epoch"] == 3
    assert schedule["warmup_steps"] == 1
    assert schedule["cycle_steps"] == [2, 6, 12]
    assert schedule["shared_validation_updates"] == [3, 9, 21]
    assert [step for tag, _, step in state.logs[str(out)] if tag == "val_cycle/route_ade_m"] == [3, 9, 21]
    with zipfile.ZipFile(out / "training_audit.zip") as archive:
        history = json.loads(archive.read("metrics.json"))["epoch_history"]
        assert [item["epoch"] for item in history] == list(range(1, 8))
        assert all(item["training_complete"] and item["train"]["samples"] == 5 for item in history)
    # 即使保存参数相同，旧版计划也不能被新代码静默续训。
    checkpoint["optimization_contract"]["schedule"]["version"] = "action_optimization_v4"
    torch.save(checkpoint, out / "latest.pt")
    with pytest.raises(ValueError, match="optimization contract mismatch"):
        run(variant, out, resume=True, extra=("--num-epochs", "7"), val_steps=100)


@pytest.mark.parametrize("variant", ["prior", "qwen_simple", "bev_only"])
def test_token_separation_reaches_shared_loop_and_logs(harness, tmp_path, monkeypatch, variant):
    """Real loss/accumulation/optimizer/EMA/checkpoint; only costly model/data IO is stubbed."""
    from qwen3vl_local.action_prior import action_token
    run, state = harness
    monkeypatch.setattr(action_token, "token_contract", lambda args: {"fixture": True})
    for rows in state.rows.values():
        for row in rows:
            row.update(action_token_id=2, action_token=dict(
                name="STOP", reason="fixture", version=action_token.TOKEN_VERSION))

    class TokenDecoder(TinyFlowDecoder):
        def __init__(self, config, flow_config):
            super().__init__(config, flow_config)
            self.conditioner = torch.nn.Module()
            self.conditioner.action_embedding = torch.nn.Embedding(7, 16)
            with torch.no_grad():
                self.conditioner.action_embedding.weight.copy_(
                    torch.ones(7, 16) * 0.02 + torch.randn(7, 16) * 0.002)
        def forward(self, **kwargs):
            result = super().forward(**kwargs)
            # Zero FM contribution isolates regularizer gradients and its weighting.
            result["flow_velocity"] = result["flow_velocity"] + self.conditioner.action_embedding.weight.sum() * 0
            return result

    monkeypatch.setattr(flow_matching, "ConditionalFlowMatchingDecoder", TokenDecoder)
    out = tmp_path / (variant + "_separation")
    cli = ("--high-level-action-token", "--action-token-separation-weight", "0.01")
    checkpoint = run(variant, out, limit=1, extra=cli)
    logs = {tag: value for tag, value, _ in state.logs[str(out)]}
    assert logs["train/action_token_separation_loss"] > 0
    assert logs["train/action_token_separation_weighted"] == pytest.approx(
        logs["train/action_token_separation_loss"] * 0.01)
    assert logs["train/loss"] == pytest.approx(logs["train/fm_loss"] + logs["train/action_token_separation_weighted"])
    initial = json.loads((out / "action_token_initial.json").read_text())["metrics"]
    assert logs["train/action_token/cosine_mean"] < initial["action_token/cosine_mean"]
    window = json.loads((out / "training_audit/windows/step_00000001.json").read_text())
    assert window["metrics"]["action_token/cosine_max"] == logs["train/action_token/cosine_max"]
    assert "conditioner.action_embedding.weight" in checkpoint["decoder"]
    plan = json.loads((out / "training_plan.json").read_text())
    assert plan["action_token_separation"]["enabled"]

    state.train_cases.clear(); state.eval_cases.clear()
    disabled = tmp_path / (variant + "_disabled_separation")
    run(variant, disabled, limit=1, extra=(*cli, "--action-token-separation-weight", "0"))
    logs = {tag: value for tag, value, _ in state.logs[str(disabled)]}
    assert logs["train/action_token_separation_weighted"] == 0
    assert logs["train/loss"] == logs["train/fm_loss"]
    initial = json.loads((disabled / "action_token_initial.json").read_text())["metrics"]
    assert logs["train/action_token/cosine_mean"] == initial["action_token/cosine_mean"]
