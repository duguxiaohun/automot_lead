"""两层均衡、共享帧容量、DDP预算和CLI合同的CPU回归。"""
from collections import Counter
import json
from types import SimpleNamespace

import pytest

from qwen3vl_local.action_prior import action_balance as ab, event_balance as eb
from qwen3vl_local.action_prior.action_token import ACTION_TOKEN_NAMES, TOKEN_VERSION
from qwen3vl_local.action_prior.config import parser
from qwen3vl_local.action_expert_ablation import common


def row(frame, events=(), action="UNCOND", status=None, split="train"):
    return dict(scenario=split, run_id=f"r{frame % 5}", route_group=f"{split}/r{frame % 5}",
        anchor=frame, split=split, event_balance_buckets=list(events),
        event_balance_all_special_buckets=list(events),
        event_balance_status=status or (eb.SPECIAL_ELIGIBLE if events else eb.CONFIRMED_REGULAR),
        action_token_id=ACTION_TOKEN_NAMES.index(action),
        action_token=dict(name=action, version=TOKEN_VERSION, reason="fixture"))


def pool():
    rows = []
    frame = 0
    for i, event in enumerate(eb.SPECIAL_BUCKETS):
        # Unequal label frequencies, and different action domains across events.
        for action in (("STOP", "KEEP", "RESUME") if i == 0 else ("STOP", "KEEP")):
            for _ in range(6 if action == "STOP" else 3):
                rows.append(row(frame, [event], action)); frame += 1
    for _ in range(30):
        rows.append(row(frame)); frame += 1
    rows.extend([row(frame, ["UE1"], status=eb.SPECIAL_FILTERED),
                 row(frame + 1, status=eb.UNCONFIRMED)])
    return rows


@pytest.mark.parametrize("world", [1, 4, 7])
def test_capacity_return_event_quotas_repeat_caps_and_seven_epoch_replay(world):
    rows = pool()
    total = ab.action_balanced_total(rows, requested=0, repeat_cap=8, world=world)
    assert total % world == 0
    plan = ab.action_balance_plan(rows, total, repeat_cap=8)
    assert plan["supported_actions"]["UE1"] == ["STOP", "RESUME", "KEEP"]
    assert "LANE_CHANGE_LEFT" not in plan["missing_actions"]["UE1"]
    assert "DECELERATE" in plan["missing_actions"]["UE1"]
    for epoch in range(7):
        selected, audit = ab.build_action_balanced_epoch(rows, total=total, seed=epoch, repeat_cap=8)
        assert selected == ab.build_action_balanced_epoch(rows, total=total, seed=epoch, repeat_cap=8)[0]
        assert len(selected) == total
        counts = Counter(r["event_balance_bucket"] for r in selected)
        assert all(counts[e] == total // 12 for e in eb.SPECIAL_BUCKETS)
        assert counts[eb.REGULAR_BACKGROUND] == total // 6
        for event, actions in plan["supported_actions"].items():
            actual = Counter(r["action_token"]["name"] for r in selected if r["event_balance_bucket"] == event)
            assert set(actual) == set(actions)
            assert dict(actual) == {a: plan["cell_quotas"][f"{event}/{a}"] for a in actions}
        assert audit["max_frame_repeats"] <= 8
        assert audit["sampled_cells"] == plan["cell_quotas"]
        assert audit["unique_frames"] == audit["optimal_unique_frames"]
        assert all(r["event_balance_status"] in (eb.SPECIAL_ELIGIBLE, eb.CONFIRMED_REGULAR) for r in selected)
        assert sum(len(selected[rank::world]) for rank in range(world)) == total
    assert ab.build_action_balanced_epoch(rows, total=total, seed=0, repeat_cap=8)[0] != selected


def test_shared_frame_capacity_is_global_across_events():
    # Ten event pools all reference the same STOP frame; independent capacities overcount it.
    rows = [row(0, eb.SPECIAL_BUCKETS, "STOP"), row(1), row(2)]
    with pytest.raises(ValueError, match="no jointly feasible"):
        ab.action_balanced_total(rows, requested=0, repeat_cap=8, world=1)
    with pytest.raises(ValueError, match="infeasible"):
        ab.action_balanced_total(rows, requested=24, repeat_cap=10, world=1)
    assert ab.action_balanced_total(rows, requested=0, repeat_cap=10, world=1) == 12
    selected, audit = ab.build_action_balanced_epoch(rows, total=12, seed=4, repeat_cap=10)
    assert Counter(r["anchor"] for r in selected)[0] == 10
    assert audit["unique_frames"] == 3


def test_budget_validation_and_missing_data_are_not_synthesized():
    rows = pool()
    with pytest.raises(ValueError, match="divisible"):
        ab.action_balanced_total(rows, requested=13, repeat_cap=8, world=1)
    with pytest.raises(ValueError, match="infeasible"):
        ab.action_balanced_total(rows, requested=72000, repeat_cap=1, world=1)
    with pytest.raises(ValueError, match="duplicate"):
        ab.action_groups([*rows, rows[0]])
    with pytest.raises(ValueError, match="missing event"):
        ab.action_groups([r for r in rows if "UE7" not in r["event_balance_buckets"]])
    rows[0].pop("action_token")
    with pytest.raises(ValueError, match="validated Phase3"):
        ab.action_groups(rows)


@pytest.mark.parametrize("variant", ["prior", "qwen_simple", "bev_only"])
def test_sampling_aliases_are_independent_of_model_conditioning(variant):
    p = parser() if variant == "prior" else common.parser(variant)
    assert p.parse_args(["--action-balanced"]).sampling_mode == "action_balanced"
    assert not p.parse_args(["--action-balanced"]).high_level_action_token
    assert p.parse_args(["--event-balanced", "--action-balanced"]).sampling_mode == "action_balanced"
    assert p.parse_args(["--action-balanced", "--event-balanced"]).sampling_mode == "event_balanced"
    assert p.parse_args(["--action-balanced", "--no-action-balanced"]).sampling_mode == "uniform"


def test_sampling_contract_restores_without_online_label_files(monkeypatch):
    from qwen3vl_local.action_prior import action_token
    args = parser().parse_args(["--action-balanced"])
    args.event_balance_source_identity = {"hash": "source"}
    args.event_balance_index = "fixture"
    monkeypatch.setattr(eb, "source_contract", lambda args: args.event_balance_source_identity)
    monkeypatch.setattr(action_token, "token_source", lambda args: SimpleNamespace(identity={"hash": "actions"}))
    expected = eb.sampling_contract(args)
    args.event_balance_index = ""
    monkeypatch.setattr(action_token, "token_source", lambda args: pytest.fail("online label access"))
    assert eb.sampling_contract(args) == expected
    assert expected["action_balance"]["event_weights"][eb.REGULAR_BACKGROUND] == 2
    assert expected["action_labels"] == {"hash": "actions"}
    args.action_balance_label_identity = {"hash": "changed"}
    assert eb.sampling_contract(args) != expected


def test_same_primary_token_as_model_even_for_concurrent_events():
    # The single token is global for this frame, never relabeled to a local context action.
    rows = [row(i, [event], "STOP") for i, event in enumerate(eb.SPECIAL_BUCKETS)]
    rows += [row(99, ["UE1", "UE2"], "LANE_CHANGE_LEFT"), row(100), row(101)]
    groups, available = ab.action_groups(rows)
    assert "UE1/LANE_CHANGE_LEFT" not in groups
    assert groups["UE2/LANE_CHANGE_LEFT"][0]["action_token"]["name"] == "LANE_CHANGE_LEFT"
    assert available["UE1"]["LANE_CHANGE_LEFT"] == 0
    assert rows[-3]["event_balance_buckets"] == ["UE1", "UE2"]


def test_re5_lateral_membership_excluded_without_relabeling_or_losing_frame():
    rows = pool() + [row(999, ["RE2", "RE5"], "LANE_CHANGE_RIGHT")]
    plan = ab.action_balance_plan(rows, 120)
    assert plan["excluded_out_of_domain"] == {"RE5/LANE_CHANGE_RIGHT": 1}
    assert plan["available"]["RE2"]["LANE_CHANGE_RIGHT"] == 1
    assert plan["available"]["RE5"]["LANE_CHANGE_RIGHT"] == 0
    assert rows[-1]["action_token"]["name"] == "LANE_CHANGE_RIGHT"
    assert rows[-1]["event_balance_buckets"] == ["RE2", "RE5"]


def test_rare_action_does_not_limit_epoch_and_is_not_independently_repeated():
    rows = []
    for i, event in enumerate(eb.SPECIAL_BUCKETS):
        rows += [row(i * 1000 + j, [event], "STOP") for j in range(100)]
    rows += [row(20000 + j) for j in range(200)]
    rows += [row(30000, ["UE1"], "KEEP")]
    assert ab.action_balanced_total(rows, requested=0, repeat_cap=8, world=4) == 9600
    selected, audit = ab.build_action_balanced_epoch(rows, total=1200, seed=1)
    assert audit["sampled_cells"]["UE1/KEEP"] == 1
    assert audit["sampled_cells"]["UE1/STOP"] == 99
    assert next(r for r in selected if r["anchor"] == 30000)["action_token"]["name"] == "KEEP"


def test_joint_conflict_returns_quota_instead_of_reducing_event_budget():
    # UE1 的 KEEP 只有共享帧；UE2 优先分给 KEEP 的目标必须回流一份到 STOP。
    rows = [row(0, ["UE1", "UE2"], "KEEP"), row(1, ["UE2"], "STOP")]
    rows += [row(i + 10, [e], "STOP") for i, e in enumerate(eb.SPECIAL_BUCKETS[2:])]
    rows += [row(100), row(101)]
    selected, audit = ab.build_action_balanced_epoch(rows, total=12, seed=1, repeat_cap=1)
    assert len(selected) == 12
    assert audit["max_frame_repeats"] == 1
    assert audit["quotas"] == eb.weighted_quotas(12)
    assert audit["sampled_cells"]["UE2/STOP"] == 1
    assert audit["action_quota_overflow"] == 1


def test_orphan_out_of_domain_action_is_error_not_background():
    with pytest.raises(ValueError, match="no supporting event"):
        ab.action_groups(pool() + [row(999, ["RE5"], "LANE_CHANGE_RIGHT")])


def test_joint_return_matches_exhaustive_small_allocation():
    import itertools
    import random
    rng = random.Random(19)
    for _ in range(20):
        rows = [row(i, rng.choice([["UE1"], ["UE2"], ["UE1", "UE2"]]),
                    rng.choice(["KEEP", "STOP"])) for i in range(4)]
        groups = {}
        for r in rows:
            for event in r["event_balance_buckets"]:
                groups.setdefault(f'{event}/{r["action_token"]["name"]}', []).append(r)
        if {k.split('/')[0] for k in groups} != {"UE1", "UE2"}:
            continue
        available = {e: {k.split('/')[1]: len(v) for k, v in groups.items() if k.startswith(e + '/')}
                     for e in ("UE1", "UE2")}
        from qwen3vl_local.sft_new_loop_phase3.sampling import support_aware_quota
        desired = {f'{e}/{a}': n for e in available
                   for a, n in support_aware_quota(available[e], 2).items()}
        choices = []
        for r in rows:
            cells = [f'{e}/{r["action_token"]["name"]}' for e in r["event_balance_buckets"]]
            choices.append([dict(zip(cells, counts)) for counts in itertools.product(range(3), repeat=len(cells))
                            if sum(counts) <= 2])
        optimum = None
        for solution in itertools.product(*choices):
            counts = Counter()
            for assignment in solution:
                counts.update(assignment)
            if any(sum(n for k, n in counts.items() if k.startswith(e + '/')) != 2 for e in available):
                continue
            objective = (sum(max(0, counts[k] - desired[k]) for k in groups),
                         -sum(sum(v.values()) > 0 for v in solution))
            optimum = objective if optimum is None else min(optimum, objective)
        allocation = eb._joint_allocation(rows, desired, repeat_cap=2, groups=groups,
                                           event_quotas={"UE1": 2, "UE2": 2})
        if optimum is None:
            assert allocation is None
        else:
            audit = allocation[3]
            assert (audit["action_quota_overflow"], -audit["optimal_unique_frames"]) == optimum


@pytest.mark.parametrize('variant', ['prior', 'qwen_simple', 'bev_only'])
def test_repeat_default_follows_final_mode_and_explicit_cap_wins(variant):
    p = parser() if variant == 'prior' else common.parser(variant)
    for flags, expected in [([], 8), (['--event-balanced'], 8), (['--action-balanced'], 2),
                            (['--sampling-mode', 'action_balanced'], 2),
                            (['--action-balanced', '--event-balanced'], 8),
                            (['--event-balanced', '--action-balanced'], 2),
                            (['--event-balance-max-frame-repeats', '8', '--action-balanced'], 8),
                            (['--action-balanced', '--event-balance-max-frame-repeats', '1'], 1)]:
        assert p.parse_args(flags).event_balance_max_frame_repeats == expected


@pytest.mark.parametrize('variant', ['qwen_simple', 'bev_only'])
@pytest.mark.parametrize('saved_cap', [2, 8])
def test_resume_preserves_saved_repeat_cap(tmp_path, variant, saved_cap):
    cfg = vars(common.parser(variant).parse_args(['--action-balanced', '--event-balance-max-frame-repeats', str(saved_cap)]))
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    restored = common.parse_train_args(variant, ['--resume', str(tmp_path / 'latest.pt')])
    assert restored.sampling_mode == 'action_balanced'
    assert restored.event_balance_max_frame_repeats == saved_cap


def test_support_audit_flags_do_not_change_labels_or_quotas():
    rows = pool()
    before = json.dumps(rows, sort_keys=True)
    total = ab.action_balanced_total(rows, requested=0, repeat_cap=2, world=1)
    plan = ab.action_balance_plan(rows, total)
    selected, audit = ab.build_action_balanced_epoch(rows, total=total, seed=1)
    support = audit['support']['cells']['UE1/RESUME']
    assert support['review_flags'] == ['fewer_than_100_frames', 'fewer_than_10_physical_routes']
    assert support['diagnostic_only']
    assert support['presentations'] == audit['cell_quotas']['UE1/RESUME']
    assert audit['max_frame_repeats'] <= 2
    assert plan['support'] == audit['support']
    assert json.dumps(rows, sort_keys=True) == before
