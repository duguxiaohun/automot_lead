"""标签覆盖、重复支持与训练前拒绝使用同一个共享合同。"""
from copy import deepcopy

import pytest

from qwen3vl_local.action_prior import action_token, config
from qwen3vl_local.action_prior.audit_data_capacity import audit
from qwen3vl_local.action_expert_ablation import common


def row(name="UNCOND", split="train", frame=0):
    return dict(scenario="S", run_id=f"Town01_Rep0_{split}_route0_01_09_12_00_00",
                anchor=frame, route_group=f"S/{split}", event_balance_original_split=split,
                action_token=dict(name=name, reason="outside_phase3_mapping",
                                  version=action_token.TOKEN_VERSION),
                action_token_id=action_token.ACTION_TOKEN_NAMES.index(name))


def test_support_counts_frames_and_physical_routes_not_presentations():
    first = row("RESUME")
    repeated = deepcopy(first)
    other_rep = dict(first, run_id=first["run_id"].replace("Rep0", "Rep1"))
    rows = dict(train=[first, repeated, other_rep], val=[row("STOP", "val")])
    report = action_token.token_support(rows)
    assert report["train"]["presentations"]["RESUME"] == 3
    assert report["train"]["unique_frames"]["RESUME"] == 2
    assert report["train"]["physical_routes"]["RESUME"] == 1
    assert report["val"]["actions_without_train_support"] == ["STOP"]
    assert report["train"]["unique_frames"]["STOP"] == 0
    assert action_token.token_coverage(rows)["train"]["STOP"] == 0
    action_token.require_conditioned_training(report, stage="test")


@pytest.mark.parametrize("variant", ["prior", "qwen_simple", "bev_only"])
def test_all_uncond_rejected_by_all_training_plans_before_model_load(variant):
    args = config.parser().parse_args(["--high-level-action-token"])
    rows = {s: [row(split=s)] for s in ("train", "val", "test")}
    with pytest.raises(ValueError, match="no conditioned training frames.*outside_phase3_mapping"):
        if variant == "prior":
            config.training_plan(args, rows, 1)
        else:
            common.training_plan(args, rows, 1, variant)


def test_sparse_valid_actions_reported_without_manufacturing_coverage():
    args = config.parser().parse_args(["--high-level-action-token"])
    rows = {s: [row("STOP", s)] for s in ("train", "val", "test")}
    plan = config.training_plan(args, rows, 1)
    assert "RESUME" in plan["action_token_support"]["train"]["missing_actions"]
    assert plan["action_token_support"]["train"]["conditioned_presentations"] == 1


def test_capacity_audit_replays_ddp_tail_instead_of_reporting_pool_support(tmp_path, monkeypatch):
    from qwen3vl_local.action_prior import audit_data_capacity as module
    args = config.parser().parse_args(["--high-level-action-token", "--event-balance-index", "unused"])
    args.world_sizes, args.num_epochs, args.data_dir = [4], 1, str(tmp_path)
    # 找到确定 seed 下被 DDP 尾部截掉的唯一动作帧：源池有支持，实际 epoch 无支持。
    rows = {s: [row(split=s)] for s in ("val", "test")}
    rows["train"] = [row(frame=i) for i in range(5)]
    import random
    order = list(range(5)); random.Random(args.seed).shuffle(order)
    rows["train"][order[-1]] = row("RESUME", frame=order[-1])
    for split in rows:
        (tmp_path / f"{split}.jsonl").write_text("fixture\n")
    monkeypatch.setattr(module, "read_rows", lambda args, split: rows[split])
    with pytest.raises(ValueError, match="epoch=1.*no conditioned training frames"):
        audit(args)
