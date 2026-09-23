"""动作 token 的证据门控、真实反传、来源合同与恢复配置。"""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from qwen3vl_local.action_prior import action_token as token
from qwen3vl_local.action_prior.config import parser
from qwen3vl_local.action_prior.contracts import file_hash
from qwen3vl_local.action_prior.tests.test_prepare_action_priors import sources
from qwen3vl_local.action_prior.tests.test_entrypoint_switches import stub
from qwen3vl_local.action_expert_ablation import common
from qwen3vl_local.leadmot import LeadMoTPlanningDecoderConfig
from qwen3vl_local.action_prior.flow_matching import ConditionalFlowMatchingDecoder, FlowMatchingConfig


def eligible(*buckets):
    return dict(status="special_eligible", eligible_buckets=buckets)


def test_keep_is_one_category_and_distinct_from_unconditioned():
    assert token.ACTION_TOKEN_NAMES.count("KEEP") == 1
    assert len(token.ACTION_TOKEN_NAMES) == 7
    longitudinal = dict(DECELERATE=False, STOP=False, RESUME=False)
    for evidence in (longitudinal, dict(longitudinal, LANE_CHANGE_LEFT=False, LANE_CHANGE_RIGHT=False)):
        assert token.project_token(eligible("UE2"), {"UE2": evidence})["name"] == "KEEP"
    for status in ("confirmed_regular", "special_filtered", "unconfirmed"):
        assert token.project_token(dict(status=status), {}) == dict(name="UNCOND", reason=status)
    assert token.project_token(None, {})["name"] == "UNCOND"
    for evidence in ({}, {"UE2": dict(longitudinal, RESUME=None)}):
        with pytest.raises(ValueError):
            token.project_token(eligible("UE2"), evidence)
    with pytest.raises(ValueError, match="conflicting"):
        token.project_token(eligible("UE1", "UE2"), {"UE1": longitudinal, "UE2": dict(longitudinal, STOP=True)})


def test_primary_stop_then_crossing_then_speed():
    evidence = dict(DECELERATE=True, STOP=False, RESUME=False, LANE_CHANGE_LEFT=True, LANE_CHANGE_RIGHT=False)
    assert token.project_token(eligible("UE2"), {"UE2": evidence})["name"] == "LANE_CHANGE_LEFT"
    evidence.update(DECELERATE=False, STOP=True)
    assert token.project_token(eligible("UE2"), {"UE2": evidence})["name"] == "STOP"


def complete_source(sources):
    """补真实 choice_annotation 所需完整证据，不替换 Phase3 校验器。"""
    _, data, full, _ = sources
    manifest_path = full.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    candidate = Path(manifest["candidate_index"])
    rows = [json.loads(line) for line in candidate.read_text().splitlines()]
    for row in rows:
        row["action_evidence"].update(
            longitudinal_decision=dict(eligible=True, action="STOP" if row["frame_id"] == 1 else "NONE"),
            lateral_observation_complete=True,
            lane_change_direction="LEFT" if row["frame_id"] == 1 else "RIGHT",
        )
    candidate.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest["candidate_sha256"] = file_hash(candidate)
    manifest_path.write_text(json.dumps(manifest))
    return SimpleNamespace(high_level_action_token=True, event_balance_index=str(full), data_dir=str(data)), candidate


def test_real_source_contract_projection_and_corruption(sources):
    args, candidate = complete_source(sources)
    source = token.token_source(args)
    rows = [dict(scenario="S", run_id="R", anchor=i) for i in range(1, 6)]
    source.annotate(rows)
    assert [row["action_token"]["name"] for row in rows] == ["STOP", "LANE_CHANGE_RIGHT", "UNCOND", "UNCOND", "UNCOND"]
    assert source.identity["privileged_action_conditioning"] is True
    assert source.identity["keep_scope"] == "merged"
    assert token.token_source(args) is source
    candidate.write_text(candidate.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        token.token_source(args)


def test_source_rejects_incomplete_evidence_before_keep(sources):
    args, candidate = complete_source(sources)
    rows = [json.loads(line) for line in candidate.read_text().splitlines()]
    rows[1]["action_evidence"]["lateral_observation_complete"] = False
    candidate.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = Path(args.event_balance_index).with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["candidate_sha256"] = file_hash(candidate)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete lateral"):
        token.token_source(args)


@pytest.mark.parametrize("prefix_length", [0, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_action_token_affects_flow_and_receives_gradients(prefix_length, dtype):
    cfg = LeadMoTPlanningDecoderConfig(hidden_size=16, num_kv_heads=2, head_dim=8, num_heads=2,
        num_layers=2, rope_type="none", bev_channels=4, bev_grid=(2, 2), use_high_level_action_token=True)
    flow_cfg = FlowMatchingConfig(trajectory_heads=2, trajectory_layers=1)
    torch.manual_seed(15)
    baseline = ConditionalFlowMatchingDecoder(replace(cfg, use_high_level_action_token=False), flow_cfg)
    torch.manual_seed(15)
    model = ConditionalFlowMatchingDecoder(cfg, flow_cfg)
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
    assert cfg.total_gen_tokens() == baseline.config.total_gen_tokens() + 1
    assert cfg.slice_layout()["action"] == (4, 5)
    kwargs = dict(pooled_kv=[(torch.randn(1, 2, prefix_length, 8), torch.randn(1, 2, prefix_length, 8)) for _ in range(2)],
        bev=torch.randn(1, 4, 2, 2), speed=torch.ones(1), target_point=torch.ones(1, 2),
        target_point_next=torch.ones(1, 2), final_goal=torch.ones(1, 2),
        flow_state=torch.randn(1, 18, 2), flow_time=torch.tensor([0.4]), sample_trajectory=False)
    with torch.autocast("cpu", enabled=dtype == torch.bfloat16, dtype=torch.bfloat16):
        first = model(**kwargs, action_token_id=torch.tensor([0]))["flow_velocity"]
        second = model(**kwargs, action_token_id=torch.tensor([2]))["flow_velocity"]
    assert not torch.equal(first, second)
    (first.float().square().mean() + second.float().square().mean()).backward()
    grad = model.conditioner.action_embedding.weight.grad
    assert grad[0].abs().sum() > 0 and grad[2].abs().sum() > 0
    assert torch.isfinite(grad).all()
    from qwen3vl_local.action_prior.optimization import parameter_groups
    groups = parameter_groups(model, parser().parse_args([]))
    group = next(g for g in groups if "conditioner.action_embedding.weight" in g["param_names"])
    assert group["algorithm"] == "adamw" and group["weight_decay"] == 0
    with pytest.raises(ValueError, match="explicit"):
        model(**kwargs)
    with pytest.raises(ValueError, match="disabled"):
        baseline(**kwargs, action_token_id=torch.tensor([0]))


@pytest.mark.parametrize("variant", ["qwen_simple", "bev_only"])
def test_resume_restores_switches_and_cli_wins(tmp_path, monkeypatch, variant):
    saved = vars(common.parser(variant).parse_args(["--high-level-action-token", "--rgb-frame-count", "1"]))
    saved.update(action_token_separation_weight=0.03, action_token_separation_margin=0.6)
    (tmp_path / "config.json").write_text(json.dumps(saved))
    checkpoint = tmp_path / "latest.pt"
    args = common.parse_train_args(variant, ["--resume", str(checkpoint)])
    assert args.high_level_action_token and args.rgb_frame_count == 1
    assert args.action_token_separation_weight == 0.03 and args.action_token_separation_margin == 0.6
    monkeypatch.setenv("HIGH_LEVEL_ACTION_TOKEN", "1")
    monkeypatch.setenv("RGB_FRAME_COUNT", "1")
    args = common.parse_train_args(variant, ["--resume", str(checkpoint), "--no-high-level-action-token", "--rgb-frame-count", "4"])
    assert not args.high_level_action_token and args.rgb_frame_count == 4
    saved.pop("action_token_separation_weight")
    saved.pop("action_token_separation_margin")
    (tmp_path / "config.json").write_text(json.dumps(saved))
    legacy = common.parse_train_args(variant, ["--resume", str(checkpoint)])
    assert legacy.action_token_separation_weight == 0


def test_resume_never_rebuilds_missing_token_source():
    with pytest.raises(ValueError, match="saved"):
        token.ensure_token_inputs(SimpleNamespace(high_level_action_token=True, event_balance_index="", resume="old.pt"))
    assert token.token_contract(SimpleNamespace(high_level_action_token=False)) is None


@pytest.mark.parametrize("variant", ["prior", "qwen_simple", "bev_only"])
def test_contract_rejects_changed_token_or_image_condition(sources, tmp_path, monkeypatch, variant):
    from qwen3vl_local.action_prior import config, provenance, contracts
    from qwen3vl_local.action_prior.tests.test_dataset_priors import dataset_args
    source_args, _ = complete_source(sources)
    # 本机缺少只读 runner 源码；此测试只替换执行源码扫描，真实标签/模型文件合同保留。
    monkeypatch.setattr(provenance, "execution_fingerprint", lambda: {"test": "same_source"})
    monkeypatch.setattr(common, "_execution_fingerprint", lambda *a: {"test": "same_source"})
    args = dataset_args(tmp_path) if variant == "prior" else common.parser(variant).parse_args([])
    if variant != "prior":
        bev = tmp_path / "bev.pth"
        bev.write_bytes(b"bev")
        base = tmp_path / "base"
        base.mkdir()
        (base / "model.safetensors").write_bytes(b"base")
        args.lead_bev_ckpt, args.model_dir = str(bev), str(base)
    args.event_balance_index, args.data_dir = source_args.event_balance_index, source_args.data_dir
    args.high_level_action_token = True
    build = (lambda: config.build_contract(args)) if variant == "prior" else (lambda: common.build_contract(args, variant))
    require = contracts.require_contract if variant == "prior" else common.require_contract
    original = build()
    assert original["identity_payload"]["high_level_action_token"]["vocabulary"] == list(token.ACTION_TOKEN_NAMES)
    for field in ("action_token_separation_weight", "action_token_separation_margin"):
        previous = getattr(args, field)
        setattr(args, field, previous + 0.1)
        with pytest.raises(ValueError, match="contract mismatch"):
            require(original, build())
        setattr(args, field, previous)
    args.rgb_frame_count = 1
    with pytest.raises(ValueError, match="contract mismatch"):
        require(original, build())
    args.rgb_frame_count = 4
    args.high_level_action_token = False
    with pytest.raises(ValueError, match="contract mismatch"):
        require(original, build())
    args.high_level_action_token = True
    require(original, build())
    args.sampling_mode = "action_balanced"
    with pytest.raises(ValueError, match="contract mismatch"):
        require(original, build())
    action_contract = build()
    assert action_contract["identity_payload"]["event_balanced_sampling"]["action_labels"] == token.token_source(args).identity


@pytest.mark.parametrize("variant", ["qwen_simple", "bev_only"])
def test_shell_switch_environment_and_cli_precedence(tmp_path, variant):
    from qwen3vl_local.action_expert_ablation.tests.test_common import _capture_train_sh_args
    argv = _capture_train_sh_args(tmp_path, variant,
        ["--no-high-level-action-token", "--rgb-frame-count", "4"],
        {"HIGH_LEVEL_ACTION_TOKEN": "1", "RGB_FRAME_COUNT": "1"})
    args = common.parser(variant).parse_args(argv[4:])
    assert args.high_level_action_token is False and args.rgb_frame_count == 4


def test_main_shell_switches(stub):
    from qwen3vl_local.action_prior.tests.test_entrypoint_switches import run, flags
    argv = flags(run("train.sh", ["--no-high-level-action-token", "--rgb-frame-count", "4"],
                     stub, HIGH_LEVEL_ACTION_TOKEN="1", RGB_FRAME_COUNT="1"))
    offset = argv.index("train") + 1
    args = parser().parse_args(argv[offset:])
    assert args.high_level_action_token is False and args.rgb_frame_count == 4


@pytest.mark.parametrize("bf16", [False, True])
def test_separation_penalty_reduces_near_collapsed_cosines(bf16):
    torch.manual_seed(91)
    weight = torch.nn.Parameter(torch.ones(7, 32) * 0.02 + torch.randn(7, 32) * 0.002)
    initial = token.embedding_diagnostics(weight)["action_token/cosine_mean"]
    optimizer = torch.optim.AdamW([weight], lr=0.003, weight_decay=0)
    for _ in range(80):
        optimizer.zero_grad()
        with torch.autocast("cpu", enabled=bf16, dtype=torch.bfloat16):
            loss = token.separation_loss(weight, 0.5)
        assert loss.dtype == torch.float32
        loss.backward()
        assert torch.isfinite(weight.grad).all()
        if _ == 0:
            # Every row gets separation gradients even if a batch has only one label.
            assert (weight.grad.norm(dim=1) > 0).all()
        optimizer.step()
    assert initial > 0.9
    assert token.embedding_diagnostics(weight)["action_token/cosine_max"] < 0.6


def test_separation_is_scale_invariant_and_inactive_for_distinct_rows():
    weight = torch.eye(7, 16, requires_grad=True)
    loss = token.separation_loss(weight, 0.5)
    loss.backward()
    assert loss.item() == 0 and weight.grad.count_nonzero() == 0
    torch.manual_seed(21)
    value = torch.ones(7, 16) + torch.randn(7, 16) * 0.1
    assert torch.allclose(token.separation_loss(value, 0.5),
                          token.separation_loss(value * torch.arange(1, 8)[:, None], 0.5))
    values = token.embedding_diagnostics(value)
    assert len([k for k in values if k.startswith("action_token/cosine/")]) == 21
    assert len([k for k in values if k.startswith("action_token/norm/")]) == 7
    assert torch.isfinite(token.separation_loss(torch.zeros(7, 16), 0.5))


@pytest.mark.parametrize("field,value", [("weight", -1), ("weight", float("nan")),
    ("weight", float("inf")), ("margin", -0.1), ("margin", 1.0), ("margin", float("nan"))])
def test_bad_separation_configuration_rejected(field, value):
    args = parser().parse_args([])
    setattr(args, "action_token_separation_" + field, value)
    with pytest.raises(ValueError, match="action-token-separation"):
        token.separation_contract(args)


def test_separation_default_switch_and_environment(monkeypatch):
    args = parser().parse_args([])
    assert not token.separation_contract(args)["enabled"]
    args.high_level_action_token = True
    assert token.separation_contract(args)["enabled"]
    args.action_token_separation_weight = 0
    assert not token.separation_contract(args)["enabled"]
    monkeypatch.setenv("ACTION_TOKEN_SEPARATION_WEIGHT", "0.02")
    monkeypatch.setenv("ACTION_TOKEN_SEPARATION_MARGIN", "0.6")
    args = parser().parse_args([*token.conditioning_env_args(), "--action-token-separation-weight", "0"])
    assert args.action_token_separation_weight == 0
    assert args.action_token_separation_margin == 0.6
    assert token.separation_contract(SimpleNamespace(high_level_action_token=True))["weight"] == 0


def _separation_ddp_worker(rank, rendezvous, result):
    from contextlib import nullcontext
    from datetime import timedelta
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(7)
        model = torch.nn.Embedding(7, 16)
        with torch.no_grad():
            model.weight.copy_(torch.ones(7, 16) * 0.02 + torch.randn(7, 16) * 0.002)
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        for micro in range(3):
            with ddp.no_sync() if micro < 2 else nullcontext():
                fm = ddp(torch.tensor([rank * 3 + micro])).square().mean()
                ((fm + 0.01 * token.separation_loss(model.weight, 0.5)) / 3).backward()
        if rank == 0:
            torch.save(dict(weight=model.weight.detach(), grad=model.weight.grad), result)
    finally:
        dist.destroy_process_group()


def test_separation_ddp_and_accumulation_match_global_objective(tmp_path):
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires CPU Gloo")
    result = tmp_path / "gradient.pt"
    torch.multiprocessing.spawn(_separation_ddp_worker,
        args=((tmp_path / "rendezvous").as_uri(), str(result)), nprocs=2, join=True)
    saved = torch.load(result, weights_only=True)
    weight = saved["weight"].requires_grad_()
    # Six distinct samples, one copy of the full-table penalty, no extra world factor.
    (weight[:6].square().mean() + 0.01 * token.separation_loss(weight, 0.5)).backward()
    assert torch.allclose(saved["grad"], weight.grad, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("mode,policy", [("action_balanced", "global_action"), ("event_balanced", "smooth_cap")])
def test_action_sampling_loads_same_labels_without_enabling_token(sources, mode, policy):
    args, _ = complete_source(sources)
    args.high_level_action_token = False
    args.sampling_mode = mode
    args.sampling_policy = policy
    args.seed = 2026
    rows = [dict(scenario="S", run_id="R", anchor=i) for i in range(1, 6)]
    token.annotate_tokens(args, rows)
    assert rows[0]["action_token"]["name"] == "STOP"
    assert token.token_contract(args) is None
    cfg = SimpleNamespace(use_high_level_action_token=False)
    assert token.token_tensor(rows[0], cfg, "cpu") is None
    from qwen3vl_local.action_prior.event_balance import sampling_contract
    args.event_balance_route_diverse = True
    args.event_balanced_scene_priors = False
    args.event_balanced_epoch_samples = 0
    args.event_balance_max_frame_repeats = 8
    args.best_selection_metric = "natural_ade"
    contract = sampling_contract(args)
    assert contract["action_labels"] == token.token_source(args).identity
    assert contract["mode"] == mode
    assert contract["sampling_policy"] == policy
