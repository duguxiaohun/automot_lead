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
    (tmp_path / "config.json").write_text(json.dumps(saved))
    checkpoint = tmp_path / "latest.pt"
    args = common.parse_train_args(variant, ["--resume", str(checkpoint)])
    assert args.high_level_action_token and args.rgb_frame_count == 1
    monkeypatch.setenv("HIGH_LEVEL_ACTION_TOKEN", "1")
    monkeypatch.setenv("RGB_FRAME_COUNT", "1")
    args = common.parse_train_args(variant, ["--resume", str(checkpoint), "--no-high-level-action-token", "--rgb-frame-count", "4"])
    assert not args.high_level_action_token and args.rgb_frame_count == 4


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
    args.rgb_frame_count = 1
    with pytest.raises(ValueError, match="contract mismatch"):
        require(original, build())
    args.rgb_frame_count = 4
    args.high_level_action_token = False
    with pytest.raises(ValueError, match="contract mismatch"):
        require(original, build())
    args.high_level_action_token = True
    require(original, build())


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
