"""全局动作比例、温和事件权重、共同预算与CLI合同的CPU回归。"""
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
    for event in eb.SPECIAL_BUCKETS:
        for action in ("STOP", "KEEP", "RESUME", "DECELERATE"):
            for _ in range(6 if action == "STOP" else 3):
                rows.append(row(frame, [event], action)); frame += 1
    for action in ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"):
        for _ in range(50):
            rows.append(row(frame, ["RE2"], action)); frame += 1
    for _ in range(60):
        rows.append(row(frame)); frame += 1
    rows.extend([row(frame, ["UE1"], status=eb.SPECIAL_FILTERED),
                 row(frame + 1, status=eb.UNCONFIRMED)])
    return rows


@pytest.mark.parametrize("world", [1, 4, 7])
def test_global_action_ratios_event_reference_budget_and_seven_epochs(world):
    rows = pool()
    total = ab.action_balanced_total(rows, requested=0, repeat_cap=8, world=world)
    assert total == eb.event_balanced_total(rows, requested=0, repeat_cap=8, world=world)
    assert total % world == 0
    accumulated = Counter()
    for epoch in range(7):
        selected, audit = ab.build_action_balanced_epoch(rows, total=total, seed=epoch)
        assert selected == ab.build_action_balanced_epoch(rows, total=total, seed=epoch)[0]
        plan = ab.action_balance_plan(rows, total, seed=epoch)
        counts = Counter(r["action_token"]["name"] for r in selected)
        assert counts == ab.global_action_quotas(total, seed=epoch)
        assert counts["UNCOND"] == total // 6
        semantic = [counts[a] for a in ab.SEMANTIC_ACTIONS]
        assert max(semantic) - min(semantic) <= 1
        assert audit["max_frame_repeats"] <= 8
        assert audit["cell_quotas"] == plan["cell_quotas"]
        assert audit["support"] == plan["support"]
        assert len(selected) == total
        assert all(r["event_balance_status"] in (eb.SPECIAL_ELIGIBLE, eb.CONFIRMED_REGULAR) for r in selected)
        assert sum(len(selected[rank::world]) for rank in range(world)) == total
        if epoch < 6:
            accumulated.update(counts)
    assert len({accumulated[a] for a in ab.SEMANTIC_ACTIONS}) == 1
    # 横向动作只来自RE2；全局动作均衡不强行维护事件1:1。
    assert audit["quotas"]["RE2"] > audit["quotas"]["UE1"]


def test_soft_event_boost_is_bounded_and_rare_frames_not_forced_to_equal_event_counts():
    rows = []
    for action in ab.SEMANTIC_ACTIONS:
        # UE2/RE2均为机动域；每个动作中大事件1000帧、小事件10帧。
        for event, n in (("UE2", 1000), ("RE2", 10)):
            for _ in range(n):
                rows.append(row(len(rows), [event], action))
    for _ in range(2400):
        rows.append(row(len(rows)))
    _, audit = ab.build_action_balanced_epoch(rows, total=14400, seed=8)
    assert audit["event_boosts"]["UE2"] == 1
    assert audit["event_boosts"]["RE2"] == 2
    for action in ab.SEMANTIC_ACTIONS:
        large = audit["cell_quotas"][f"UE2/{action}"]
        small = audit["cell_quotas"][f"RE2/{action}"]
        assert large + small == 2000
        assert 1.8 < (small / 10) / (large / 1000) <= 2.1
        assert small < large / 20
    assert audit["max_frame_repeats"] <= 4


def test_concurrent_events_do_not_duplicate_capacity_or_multiply_boost():
    rows = pool() + [row(999, ["UE1", "UE2"], "STOP")]
    data = ab._sampling_pools(rows)
    assert data["counts"]["STOP"] == 61
    selected, audit = ab.build_action_balanced_epoch(rows, total=120, seed=4, repeat_cap=1)
    assert max(Counter(eb._identity(r) for r in selected).values()) == 1
    assert audit["unique_frames"] == 120
    assert rows[-1]["event_balance_buckets"] == ["UE1", "UE2"]


def test_shared_frames_are_counted_once_even_with_several_attributions():
    rows = pool()
    for r in rows:
        if r["action_token"]["name"] == "STOP":
            r["event_balance_buckets"] = list(eb.SPECIAL_BUCKETS)
    selected, audit = ab.build_action_balanced_epoch(rows, total=720, seed=19, repeat_cap=4)
    assert audit["action_unique_frames"]["STOP"] == 60
    assert max(Counter(eb._identity(r) for r in selected).values()) <= 4
    assert sum(audit["quotas"].values()) == 720


def test_capacity_shortage_reports_action_and_does_not_shrink_requested_or_auto_budget():
    rows = [r for r in pool() if r["action_token"]["name"] != "RESUME"]
    rows.append(row(999, ["UE1"], "RESUME"))
    for requested in (0, 120):
        with pytest.raises(ValueError, match="RESUME.*Budget was not shortened"):
            ab.action_balanced_total(rows, requested=requested, repeat_cap=8, world=1)
    # 首轮余数可能没有落到某动作；仍按所有epoch的最大需要量预检。
    rows = pool()
    rows = [r for r in rows if r["action_token"]["name"] != "RESUME"] + [row(999, ["UE1"], "RESUME")]
    with pytest.raises(ValueError, match="RESUME"):
        ab.build_action_balanced_epoch(rows, total=12, seed=5, repeat_cap=1)


def test_missing_global_action_is_error_but_missing_event_is_allowed_with_explicit_budget():
    rows = [r for r in pool() if "UE7" not in r["event_balance_buckets"]]
    assert ab.action_balanced_total(rows, requested=120, repeat_cap=8, world=1) == 120
    with pytest.raises(ValueError, match="missing global action.*KEEP"):
        ab.action_balance_plan([r for r in rows if r["action_token"]["name"] != "KEEP"], 120)


def test_budget_validation_and_bad_labels_are_not_synthesized():
    rows = pool()
    with pytest.raises(ValueError, match="divisible"):
        ab.action_balanced_total(rows, requested=13, repeat_cap=8, world=1)
    with pytest.raises(ValueError, match="infeasible"):
        ab.action_balanced_total(rows, requested=72000, repeat_cap=1, world=1)
    with pytest.raises(ValueError, match="duplicate"):
        ab.action_groups([*rows, rows[0]])
    rows[0].pop("action_token")
    with pytest.raises(ValueError, match="validated Phase3"):
        ab.action_groups(rows)


def test_out_of_domain_memberships_never_change_labels_or_facts():
    rows = pool() + [row(999, ["RE2", "RE5"], "LANE_CHANGE_RIGHT")]
    before = json.dumps(rows, sort_keys=True)
    plan = ab.action_balance_plan(rows, 120)
    assert plan["excluded_out_of_domain"] == {"RE5/LANE_CHANGE_RIGHT": 1}
    assert plan["available"]["RE5"]["LANE_CHANGE_RIGHT"] == 0
    ab.build_action_balanced_epoch(rows, total=120, seed=1)
    assert json.dumps(rows, sort_keys=True) == before
    with pytest.raises(ValueError, match="no supporting event"):
        ab.action_groups(pool() + [row(999, ["RE5"], "LANE_CHANGE_RIGHT")])


def test_weighted_capacity_return_and_seeded_integer_rounding():
    sizes, weights = {"large": 100, "small": 1}, {"large": 1, "small": 2}
    import random
    quotas = ab._weighted_capacity_quota(sizes, weights, 202, 2, random.Random(0))
    assert quotas == {"large": 200, "small": 2}
    with pytest.raises(ValueError, match="capacity infeasible"):
        ab._weighted_capacity_quota(sizes, weights, 203, 2, random.Random(0))


@pytest.mark.parametrize("variant", ["prior", "qwen_simple", "bev_only"])
def test_sampling_aliases_are_independent_of_model_conditioning(variant):
    p = parser() if variant == "prior" else common.parser(variant)
    assert p.parse_args(["--action-balanced"]).sampling_mode == "action_balanced"
    assert not p.parse_args(["--action-balanced"]).high_level_action_token
    assert p.parse_args(["--event-balanced", "--action-balanced"]).sampling_mode == "action_balanced"
    assert p.parse_args(["--action-balanced", "--event-balanced"]).sampling_mode == "event_balanced"
    assert p.parse_args([]).sampling_mode == "event_balanced"
    assert p.parse_args([]).event_balance_max_frame_repeats == 8
    assert p.parse_args(["--action-balanced"]).event_balance_max_frame_repeats == 8
    for removed in (["--no-action-balanced"], ["--no-event-balanced"], ["--sampling-mode", "uniform"]):
        with pytest.raises(SystemExit):
            p.parse_args(removed)


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
    assert expected["action_balance"]["background_fraction"] == "1/6"
    assert expected["action_labels"] == {"hash": "actions"}
    args.action_balance_label_identity = {"hash": "changed"}
    assert eb.sampling_contract(args) != expected

@pytest.mark.parametrize('variant', ['prior', 'qwen_simple', 'bev_only'])
def test_repeat_default_follows_final_mode_and_explicit_cap_wins(variant):
    p = parser() if variant == 'prior' else common.parser(variant)
    for flags, expected in [([], 8), (['--event-balanced'], 8), (['--action-balanced'], 8),
                            (['--sampling-mode', 'action_balanced'], 8),
                            (['--action-balanced', '--event-balanced'], 8),
                            (['--event-balanced', '--action-balanced'], 8),
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

@pytest.mark.parametrize("variant", ["prior", "qwen_simple", "bev_only"])
@pytest.mark.parametrize("options", [[], ["--action-balanced"]])
def test_two_modes_auto_prepare_mapping_without_enabling_tokens(monkeypatch, variant, options):
    from qwen3vl_local.action_prior import action_token, prepare_action_priors
    p = parser() if variant == "prior" else common.parser(variant)
    args = p.parse_args(options)
    prepared = []
    def prepare(args):
        prepared.append(args.sampling_mode)
        args.event_balance_index = "prepared/full_event_mapping.jsonl"
    monkeypatch.setattr(prepare_action_priors, "ensure_full_mapping", prepare)
    action_token.ensure_token_inputs(args)
    action_token.ensure_token_inputs(args)
    assert prepared == [args.sampling_mode]
    assert not args.high_level_action_token
    assert not getattr(args, "high_level_action_prior", False)
    args.event_balance_index, args.resume = "", "saved/latest.pt"
    with pytest.raises(ValueError, match="resume requires the saved"):
        action_token.ensure_token_inputs(args)
    assert len(prepared) == 1


@pytest.mark.parametrize("variant", ["prior", "qwen_simple", "bev_only"])
@pytest.mark.parametrize("saved", [{}, {"sampling_mode": "uniform"}])
def test_legacy_uniform_resume_is_not_silently_changed(tmp_path, monkeypatch, variant, saved):
    import sys
    from qwen3vl_local.action_prior import resume
    (tmp_path / "config.json").write_text(json.dumps(saved))
    checkpoint = str(tmp_path / "latest.pt")
    with pytest.raises(ValueError, match="uniform runs require their original source"):
        if variant == "prior":
            monkeypatch.setattr(sys, "argv", ["resume", checkpoint])
            resume.main()
        else:
            common.parse_train_args(variant, ["--resume", checkpoint])


def test_concurrent_membership_has_no_extra_sampling_mass():
    rows = []
    for action in ab.SEMANTIC_ACTIONS:
        for events in (["UE2"], ["RE2"], ["UE2", "RE2"]):
            for _ in range(10):
                rows.append(row(len(rows), events, action))
    for _ in range(36):
        rows.append(row(len(rows)))
    selected, audit = ab.build_action_balanced_epoch(rows, total=216, seed=0, repeat_cap=1)
    assert audit["event_boosts"]["UE2"] == audit["event_boosts"]["RE2"] == 1
    membership_counts = Counter(tuple(r["event_balance_buckets"]) for r in selected if r["action_token"]["name"] != "UNCOND")
    assert set(membership_counts.values()) == {60}
    assert audit["unique_frames"] == 216


@pytest.mark.parametrize("variant", ["prior", "qwen_simple", "bev_only"])
def test_plan_reports_actual_event_distribution_and_global_actions(variant):
    from qwen3vl_local.action_prior import config
    args = (parser() if variant == "prior" else common.parser(variant)).parse_args(
        ["--action-balanced", "--event-balanced-epoch-samples", "120"])
    rows = {s: [dict(r, route_group=s + "/" + r["route_group"], split=s) for r in pool()]
            for s in ("train", "val", "test")}
    plan = config.training_plan(args, rows, 4) if variant == "prior" else common.training_plan(args, rows, 4, variant)
    selected, audit = ab.build_action_balanced_epoch(rows["train"], total=120, seed=args.seed)
    assert plan["samples_per_epoch"] == 120
    expected = dict.fromkeys(plan["sampling"]["epoch_quotas"], 0)
    expected.update(audit["quotas"])
    assert plan["sampling"]["epoch_quotas"] == expected
    assert plan["sampling"]["action_balance"]["action_quotas"] == audit["sampled_actions"]
    assert plan["sampling"]["epoch_quotas"] != eb.weighted_quotas(120)
    assert plan["sampling"]["budget_reference"] == "explicit"
