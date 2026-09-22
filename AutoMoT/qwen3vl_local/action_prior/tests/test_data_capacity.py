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


def balanced_rows(split):
    from qwen3vl_local.action_prior import event_balance as eb
    special = [dict(row("STOP", split, i), event_balance_status=eb.SPECIAL_ELIGIBLE,
                    event_balance_buckets=[bucket], event_balance_all_special_buckets=[bucket])
               for i, bucket in enumerate(eb.SPECIAL_BUCKETS)]
    background = [dict(row("UNCOND", split, 20 + i), event_balance_status=eb.CONFIRMED_REGULAR,
                       event_balance_buckets=[], event_balance_all_special_buckets=[]) for i in range(2)]
    return special + background


def test_sparse_valid_actions_reported_without_manufacturing_coverage():
    args = config.parser().parse_args(["--high-level-action-token"])
    rows = {s: balanced_rows(s) for s in ("train", "val", "test")}
    plan = config.training_plan(args, rows, 1)
    assert "RESUME" in plan["action_token_support"]["train"]["missing_actions"]
    assert plan["action_token_support"]["train"]["conditioned_presentations"] == 10
    assert plan["ddp_tail_per_epoch"] == 0


def test_capacity_audit_checks_actual_epoch_instead_of_only_pool_support(tmp_path, monkeypatch):
    from qwen3vl_local.action_prior import audit_data_capacity as module
    args = config.parser().parse_args(["--high-level-action-token"])
    args.world_sizes, args.num_epochs, args.data_dir = [4], 1, str(tmp_path)
    rows = {s: balanced_rows(s) for s in ("train", "val", "test")}
    for split in rows:
        (tmp_path / f"{split}.jsonl").write_text("fixture\n")
    # 数据/来源 IO 用夹具；计划保留真实事件预算，故障注入模拟实际 epoch 错误丢失动作条件。
    monkeypatch.setattr(module, "read_rows", lambda args, split: rows[split])
    from qwen3vl_local.action_prior import event_balance as eb
    monkeypatch.setattr(eb, "source_contract", lambda args: {"fixture": True})
    monkeypatch.setattr(eb, "source_audit", lambda args: {})
    args.event_balance_index = "fixture"
    monkeypatch.setattr(module, "build_balanced_epoch", lambda rows, **kw: (
        [row(frame=i) for i in range(kw["total"])], {"total": kw["total"]}))
    with pytest.raises(ValueError, match="epoch=1.*no conditioned training frames"):
        audit(args)
