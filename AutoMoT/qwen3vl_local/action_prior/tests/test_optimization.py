"""优化器数值、参数路由、周期边界与状态恢复回归；不加载冻结大模型。"""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_prior.config import parser
from qwen3vl_local.action_prior.optimization import build_optimization, parameter_groups, orthogonalized_update
from qwen3vl_local.action_prior.optimization_config import optimization_plan, lr_factor, validate_optimization


class SmallDecoder(torch.nn.Module):
    """覆盖真实 decoder 的隐藏矩阵、输入/输出层和二维 embedding 角色。"""

    def __init__(self):
        super().__init__()
        self.conditioner = torch.nn.Module()
        self.conditioner.blocks = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 4))
        self.conditioner.route_query_bank = torch.nn.Embedding(2, 4)
        self.trajectory_blocks = torch.nn.TransformerEncoderLayer(4, 2, 8, dropout=0, batch_first=True)
        self.point_type_embedding = torch.nn.Embedding(2, 4)
        self.coordinate_encoder = torch.nn.Linear(2, 4)
        self.velocity_head = torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 4), torch.nn.SiLU(), torch.nn.Linear(4, 2))
        self.frozen = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)


def make(model, *cli, total=61, steps_per_epoch=10):
    args = parser().parse_args(list(cli))
    optimizer, scheduler = build_optimization(args, model, {"actual_step_limit": total, "optimizer_steps_per_epoch": steps_per_epoch})
    return optimizer, scheduler


def advance(model, optimizer, scheduler, index):
    """给每个参数稳定但随 step 变化的梯度，不让恢复检查依赖测试 RNG。"""
    optimizer.zero_grad(set_to_none=True)
    for i, p in enumerate(model.parameters()):
        if p.requires_grad:
            p.grad = torch.sin(torch.arange(p.numel()).reshape(p.shape) + index + i).to(p)
    optimizer.step()
    scheduler.step()


def test_parameter_roles_and_exact_coverage():
    model = SmallDecoder()
    groups = parameter_groups(model, parser().parse_args([]))
    routed = {n: g for g in groups for n in g["param_names"]}
    params = [p for g in groups for p in g["params"]]
    assert len(params) == len(set(map(id, params)))
    assert set(routed) == {n for n, p in model.named_parameters() if p.requires_grad}
    for name in ("conditioner.blocks.0.weight", "conditioner.blocks.2.weight", "trajectory_blocks.self_attn.in_proj_weight", "velocity_head.1.weight"):
        assert routed[name]["algorithm"] == "muon"
    for name in ("conditioner.blocks.1.weight", "conditioner.blocks.0.bias", "coordinate_encoder.weight", "velocity_head.3.weight"):
        assert routed[name]["algorithm"] == "adamw"
    for name in ("point_type_embedding.weight", "conditioner.route_query_bank.weight"):
        assert routed[name]["algorithm"] == "adamw" and routed[name]["weight_decay"] == 0


def test_real_decoder_parameter_routing_without_allocating_weights():
    """在 meta device 构造真实结构，验证真实层名都进入预期组，无需 Qwen/BEV 权重。"""
    from qwen3vl_local.leadmot.config import LeadMoTPlanningDecoderConfig
    from qwen3vl_local.action_prior.flow_matching import ConditionalFlowMatchingDecoder
    with torch.device("meta"):
        model = ConditionalFlowMatchingDecoder(LeadMoTPlanningDecoderConfig())
    groups = parameter_groups(model, parser().parse_args([]))
    routed = {n: g for g in groups for n in g["param_names"]}
    assert set(routed) == {n for n, p in model.named_parameters() if p.requires_grad}
    assert routed["trajectory_blocks.layers.0.self_attn.in_proj_weight"]["algorithm"] == "muon"
    for name in routed:
        if name.startswith("conditioner.blocks.") and name.endswith("_proj.weight"):
            assert routed[name]["algorithm"] == "muon"
    assert routed["velocity_head.3.weight"]["algorithm"] == "adamw"
    for name in ("point_type_embedding.weight", "point_index_embedding.weight", "conditioner.bev_projector.pos_embed"):
        assert routed[name]["algorithm"] == "adamw" and routed[name]["weight_decay"] == 0


def test_auxiliary_adamw_matches_torch_across_scheduler_steps():
    model = SmallDecoder()
    optimizer, scheduler = make(model)
    aux = [g for g in optimizer.param_groups if g["algorithm"] == "adamw"]
    reference_groups = [dict(params=[torch.nn.Parameter(p.detach().clone()) for p in g["params"]], lr=g["lr"], weight_decay=g["weight_decay"]) for g in aux]
    reference = torch.optim.AdamW(reference_groups, betas=(0.9, 0.95), foreach=False)
    for step in range(20):
        for source, dest in zip(aux, reference.param_groups):
            dest["lr"] = source["lr"]
            for p, q in zip(source["params"], dest["params"]):
                p.grad = torch.full_like(p, 0.03 * (step + 1))
                q.grad = p.grad.clone()
        optimizer.step(); reference.step(); scheduler.step()
        for source, dest in zip(aux, reference.param_groups):
            for p, q in zip(source["params"], dest["params"]):
                assert torch.equal(p, q)


@pytest.mark.parametrize("mode", ["adamw", "muon_adamw"])
@pytest.mark.parametrize("schedule", ["cosine", "cosine_restarts"])
def test_serialized_resume_matches_uninterrupted_updates(mode, schedule):
    torch.manual_seed(7)
    original = SmallDecoder()
    optimizer, scheduler = make(original, "--optimizer", mode, "--lr-scheduler", schedule)
    for step in range(12):
        advance(original, optimizer, scheduler, step)
    data = BytesIO()
    torch.save(dict(model=original.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), data)
    data.seek(0)
    state = torch.load(data, weights_only=True)
    resumed = SmallDecoder()
    restored_optimizer, restored_scheduler = make(resumed, "--optimizer", mode, "--lr-scheduler", schedule)
    resumed.load_state_dict(state["model"])
    restored_optimizer.load_state_dict(state["optimizer"])
    restored_scheduler.load_state_dict(state["scheduler"])
    for step in range(12, 61):
        advance(original, optimizer, scheduler, step)
        advance(resumed, restored_optimizer, restored_scheduler, step)
        assert scheduler.get_last_lr() == restored_scheduler.get_last_lr()
        for a, b in zip(original.parameters(), resumed.parameters()):
            assert torch.equal(a, b)


def test_restart_budget_peaks_troughs_and_terminal_zero():
    args = parser().parse_args(["--warmup-ratio", "0.05"])
    plan = optimization_plan(args, 7000, steps_per_epoch=1000)
    assert plan["warmup_steps"] == 50
    assert plan["cycle_steps"] == [950, 2000, 4000]
    assert lr_factor(0, plan) == 1 / 50
    assert lr_factor(49, plan) == 1
    start = 50
    for length in plan["cycle_steps"]:
        values = [lr_factor(i, plan) for i in range(start, start + length)]
        assert values[0] == 1 and values[-1] == 0
        assert values == sorted(values, reverse=True)
        start += length
    assert start == 7000 and lr_factor(7000, plan) == lr_factor(9000, plan) == 0


@pytest.mark.parametrize("total", [1, 2, 3, 5, 10, 20, 61])
def test_short_runs_keep_nonempty_cycles_and_at_least_one_update(total):
    plan = optimization_plan(parser().parse_args([]), total, steps_per_epoch=10)
    assert plan["warmup_steps"] + sum(plan["cycle_steps"]) == total
    assert all(n > 0 for n in plan["cycle_steps"])
    assert any(lr_factor(i, plan) > 0 for i in range(total))


def test_cosine_baseline_uses_same_first_epoch_warmup():
    from qwen3vl_local.action_prior.tests.test_training_loop import lightweight_old_helpers
    model = SmallDecoder()
    opt, scheduler = make(model, "--optimizer", "adamw", "--lr-scheduler", "cosine", total=100, steps_per_epoch=40)
    args = parser().parse_args([])
    old = lightweight_old_helpers()
    reference_model = deepcopy(model)
    reference_groups = parameter_groups(reference_model, parser().parse_args(["--optimizer", "adamw"]))
    reference = torch.optim.AdamW(reference_groups, lr=args.learning_rate, betas=(0.9, 0.95))
    reference_scheduler = old._make_scheduler(reference, 100, 0.02)  # 40 updates × 5% = 2 warmup updates
    for step in range(100):
        assert scheduler.get_last_lr() == reference_scheduler.get_last_lr()
        advance(model, opt, scheduler, step)
        advance(reference_model, reference, reference_scheduler, step)
        assert all(torch.equal(p, q) for p, q in zip(model.parameters(), reference_model.parameters()))


@pytest.mark.parametrize("shape", [(2, 8), (8, 2), (4, 4)])
def test_muon_direction_scale_invariance_and_zero(shape):
    gradient = torch.arange(1, 1 + shape[0] * shape[1], dtype=torch.float32).reshape(shape)
    update = orthogonalized_update(gradient, 5)
    assert update.shape == gradient.shape and torch.isfinite(update).all()
    torch.testing.assert_close(update, orthogonalized_update(gradient * 10, 5), atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(orthogonalized_update(torch.zeros_like(gradient), 5)) == 0


def test_muon_updates_matrix_and_skips_absent_grad():
    model = SmallDecoder()
    optimizer, scheduler = make(model, "--warmup-ratio", "0")
    p = model.conditioner.blocks[0].weight
    untouched = model.conditioner.blocks[2].weight.detach().clone()
    initial = p.detach().clone()
    p.grad = torch.ones_like(p)
    optimizer.step()
    assert not torch.equal(p, initial)
    assert torch.equal(model.conditioner.blocks[2].weight, untouched)
    assert optimizer.state[p]["momentum_buffer"].dtype == torch.float32
    with pytest.raises(ValueError, match="routing"):
        state = optimizer.state_dict()
        state["param_groups"][0]["param_names"] = ["wrong"]
        optimizer.load_state_dict(state)


@pytest.mark.parametrize("cli", [["--optimizer", "unknown"], ["--lr-scheduler", "unknown"], ["--learning-rate", "nan"], ["--weight-decay", "inf"]])
def test_invalid_optimization_rejected(cli):
    with pytest.raises(ValueError):
        validate_optimization(parser().parse_args(cli))


class DistributedMatrix(torch.nn.Module):
    """DDP 小模型同时包含 Muon 和辅助 AdamW 参数。"""
    def __init__(self):
        super().__init__()
        self.trajectory_blocks = torch.nn.Linear(2, 2)
        self.output_head = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.output_head(torch.tanh(self.trajectory_blocks(x)))


def _distributed_worker(rank, rendezvous, result):
    """每 rank 累积两个 micro-batch，只在窗口末同步梯度。"""
    from contextlib import nullcontext
    from datetime import timedelta
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(123)
        model = DistributedMatrix()
        wrapped = torch.nn.parallel.DistributedDataParallel(model)
        optimizer, scheduler = make(model, total=20)
        inputs = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 8
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                with wrapped.no_sync() if micro == 0 else nullcontext():
                    (wrapped(inputs[rank + micro * 2:rank + micro * 2 + 1]).square().mean() / 2).backward()
            optimizer.step(); scheduler.step()
        collected = [None, None]
        dist.all_gather_object(collected, model.state_dict())
        assert all(torch.equal(collected[0][k], collected[1][k]) for k in collected[0])
        if rank == 0:
            torch.save(model.state_dict(), result)
    finally:
        dist.destroy_process_group()


def test_two_rank_accumulation_matches_global_batch(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip("requires CPU Gloo")
    torch.multiprocessing.spawn(_distributed_worker, args=((tmp_path / "rendezvous").as_uri(), str(tmp_path / "ddp.pt")), nprocs=2)
    torch.manual_seed(123)
    model = DistributedMatrix()
    optimizer, scheduler = make(model, total=20)
    inputs = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 8
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
        optimizer.step(); scheduler.step()
    actual = torch.load(tmp_path / "ddp.pt", weights_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(actual[key], value, atol=1e-7, rtol=1e-6)


def test_decay_policy_independent_of_optimizer():
    model = SmallDecoder()
    routes = []
    for mode in ('adamw', 'muon_adamw'):
        args = parser().parse_args(['--optimizer', mode])
        routes.append({name: group['weight_decay'] for group in parameter_groups(model, args)
                       for name in group['param_names']})
    assert routes[0] == routes[1]
    assert routes[0]['point_type_embedding.weight'] == 0


@pytest.mark.parametrize('mode', ['adamw', 'muon_adamw'])
def test_monitor_measures_actual_updates_without_changing_them(mode):
    from qwen3vl_local.action_prior.optimization import optimizer_step_with_metrics
    model = SmallDecoder()
    reference = deepcopy(model)
    opt, _ = make(model, '--optimizer', mode)
    ref, _ = make(reference, '--optimizer', mode)
    before = {n: p.clone() for n, p in model.named_parameters()}
    for p, q in zip(model.parameters(), reference.parameters()):
        if p.requires_grad:
            p.grad = torch.ones_like(p)
            q.grad = torch.ones_like(q)
    metrics = optimizer_step_with_metrics(opt, monitor=True)
    assert optimizer_step_with_metrics(ref, monitor=False) == {}
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), reference.parameters()))
    for group in opt.param_groups:
        name, p = group['param_names'][0], group['params'][0]
        prefix = f"update/{group['group_name']}/{name}"
        delta = p - before[name]
        assert metrics[prefix + '/rms'] == delta.square().mean().sqrt().item()
        assert metrics[prefix + '/relative_norm'] == (delta.norm() / (before[name].norm() + 1e-12)).item()
    assert metrics['optimizer_time_ms/adamw'] >= 0
    if mode == 'muon_adamw':
        assert metrics['optimizer_time_ms/muon'] >= 0
    assert opt._action_timings is None


def test_cycle_status_uses_completed_updates():
    from qwen3vl_local.action_prior.optimization_config import cycle_status
    plan = optimization_plan(parser().parse_args([]), 7000, steps_per_epoch=1000)
    assert cycle_status(50, plan)['cycle'] == 0
    assert cycle_status(51, plan) == dict(cycle=1, progress=0, cycle_end=False)
    assert cycle_status(1000, plan) == dict(cycle=1, progress=1, cycle_end=True)
    assert cycle_status(1001, plan) == dict(cycle=2, progress=0, cycle_end=False)
    assert cycle_status(7000, plan) == dict(cycle=3, progress=1, cycle_end=True)
    assert cycle_status(7001, plan)['cycle_end'] is False


def test_advanced_switches_removed_and_defaults_recorded():
    from qwen3vl_local.action_prior.optimization_config import OPTIMIZATION_SETTINGS, full_validation_due
    options = parser()._option_string_actions
    for name in ('decay-policy', 'cycle-validation', 'optimizer-monitor-steps', 'full-validation-policy',
                 'full-validation-steps', 'muon-momentum', 'muon-ns-steps', 'muon-lr-scale',
                 'cosine-restart-cycles', 'cosine-restart-mult'):
        assert '--' + name not in options
    plans = []
    for scheduler in ('cosine', 'cosine_restarts'):
        plan = optimization_plan(parser().parse_args(['--lr-scheduler', scheduler]), 12, steps_per_epoch=3)
        assert all(plan[key] == value for key, value in OPTIMIZATION_SETTINGS.items())
        plans.append([step for step in range(1, 13) if full_validation_due(
            step, plan, full_epoch=step % 3 == 0)])
    assert plans[0] == plans[1] == [3, 6, 9, 12]


def test_timing_excludes_validation_checkpoint_and_preserves_final_overhead():
    from qwen3vl_local.action_prior.training_core import TrainingTiming
    now = [0.0]
    timer = TrainingTiming(clock=lambda: now[0])
    now[0] += 2
    timer.add_samples(4)
    with timer.measure('validation'):
        now[0] += 10
    with timer.measure('checkpoint'):
        now[0] += 5
    now[0] += 2
    assert timer.window_metrics() == dict(samples_per_second=1, overall_samples_per_second=4/19,
                                         training_seconds=4, overall_seconds=19)
    timer.add_samples(2)
    now[0] += 1
    assert timer.window_metrics()['samples_per_second'] == 2
    # 最后一条训练日志后的验证、保存也进入累计口径；异常退出也计时。
    with pytest.raises(RuntimeError):
        with timer.measure('validation'):
            now[0] += 3
            raise RuntimeError('interrupted')
    with timer.measure('checkpoint'):
        now[0] += 2
    summary = timer.summary()
    assert summary['samples'] == 6 and summary['training_seconds'] == 5
    assert summary['validation_seconds'] == 13 and summary['checkpoint_seconds'] == 7
    assert summary['overall_samples_per_second'] == 6/25


@pytest.mark.parametrize("scheduler", ["cosine", "cosine_restarts"])
def test_warmup_counts_first_epoch_including_partial_accumulation(scheduler):
    """真实 plan 使用每轮 ceil(micro/accum)，缩短总轮数不应缩短 warmup。"""
    from qwen3vl_local.action_prior.config import training_plan
    from qwen3vl_local.action_prior.tests.test_action_balance import row
    from qwen3vl_local.action_prior.event_balance import SPECIAL_BUCKETS
    # 十二份均衡预算204，每份17；累积5仍有4帧尾窗口，共41次更新。
    rows = {split: [row(i, [bucket] if bucket else [], action="STOP" if bucket else "UNCOND", split=split)
                    for i, bucket in enumerate((*SPECIAL_BUCKETS, "", ""))]
            for split in ("train", "val", "test")}
    for epochs in (7, 15):
        args = parser().parse_args(["--num-epochs", str(epochs), "--grad-accum-steps", "5",
                                  "--lr-scheduler", scheduler, "--event-balanced-epoch-samples", "204",
                                  "--event-balance-max-frame-repeats", "17"])
        plan = training_plan(args, rows, 1)
        schedule = plan["optimization"]["schedule"]
        assert plan["optimizer_steps_per_epoch"] == 41
        assert schedule["warmup_steps"] == 2
        assert schedule["shared_validation_updates"][:3] == [41, 123, 287]
        assert plan["actual_step_limit"] == 41 * epochs


def test_short_budget_truncates_without_squeezing_epoch_cycles():
    """max-train-steps只截断，不能把原本后面数轮的峰谷压进smoke。"""
    args = parser().parse_args([])
    full = optimization_plan(args, 7000, 1000)
    short = optimization_plan(args, 1500, 1000)
    assert short["warmup_steps"] == full["warmup_steps"] == 50
    assert short["cycle_steps"] == [950, 500]
    assert short["nominal_cycle_steps"] == [950, 2000]
    assert [lr_factor(i, short) for i in range(1500)] == [lr_factor(i, full) for i in range(1500)]
    assert lr_factor(1500, short) == 0


def test_zero_warmup_restarts_at_epoch_boundaries():
    args = parser().parse_args(["--warmup-ratio", "0"])
    plan = optimization_plan(args, 700, 100)
    assert plan["warmup_steps"] == 0 and plan["cycle_steps"] == [100, 200, 400]
    for step in (0, 100, 300):
        assert lr_factor(step, plan) == 1
    for step in (99, 299, 699, 700):
        assert lr_factor(step, plan) == 0


@pytest.mark.parametrize("cut", [3, 4, 5, 11, 12, 13])
def test_serialized_resume_at_trough_and_restart(cut):
    """跨首/次周期谷底前后恢复，LR及Muon's/AdamW状态产生完全相同更新。"""
    original = SmallDecoder()
    opt, scheduler = make(original, total=28, steps_per_epoch=4)
    for step in range(cut):
        advance(original, opt, scheduler, step)
    restored = deepcopy(original)
    opt2, scheduler2 = make(restored, total=28, steps_per_epoch=4)
    opt2.load_state_dict(deepcopy(opt.state_dict()))
    scheduler2.load_state_dict(deepcopy(scheduler.state_dict()))
    for step in range(cut, 28):
        assert scheduler.get_last_lr() == scheduler2.get_last_lr()
        advance(original, opt, scheduler, step)
        advance(restored, opt2, scheduler2, step)
        assert all(torch.equal(p, q) for p, q in zip(original.parameters(), restored.parameters()))
