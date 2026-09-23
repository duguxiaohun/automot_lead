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
    assert p.parse_args([]).event_balance_max_frame_repeats == 11
    assert p.parse_args(["--action-balanced"]).event_balance_max_frame_repeats == 11
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
    for flags, expected in [([], 11), (['--event-balanced'], 11), (['--action-balanced'], 11),
                            (['--sampling-mode', 'action_balanced'], 11),
                            (['--action-balanced', '--event-balanced'], 11),
                            (['--event-balanced', '--action-balanced'], 11),
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


@pytest.mark.parametrize('variant', ['prior', 'qwen_simple', 'bev_only'])
def test_new_training_defaults_and_explicit_comparisons(variant):
    p = parser() if variant == 'prior' else common.parser(variant)
    args = p.parse_args([])
    assert args.sampling_mode == 'event_balanced' and args.sampling_policy == 'smooth_cap'
    assert args.sampling_smooth_power == .5 and args.event_balance_max_frame_repeats == 11
    assert args.event_balanced_epoch_samples == 116256
    assert p.parse_args(['--action-balanced']).event_balanced_epoch_samples == 116256
    assert p.parse_args(['--event-balanced-epoch-samples', '0']).event_balanced_epoch_samples == 0
    assert p.parse_args(['--action-balanced']).sampling_policy == 'global_action'
    assert p.parse_args(['--sampling-policy', 'cycle_even']).sampling_policy == 'cycle_even'
    with pytest.raises(SystemExit):
        p.parse_args(['--action-balanced', '--sampling-policy', 'smooth_cap'])


@pytest.mark.parametrize('variant', ['prior', 'qwen_simple', 'bev_only'])
def test_fixed_default_budget_preserves_event_quotas_and_old_update_count(variant):
    from qwen3vl_local.action_prior import config
    args = (parser() if variant == 'prior' else common.parser(variant)).parse_args([])
    # 当前审计瓶颈958帧；每事件9688次需要cap11，cap8必须拒绝而非缩轮。
    train = [row(i * 958 + j, [event], 'STOP')
             for i, event in enumerate(eb.SPECIAL_BUCKETS) for j in range(958)]
    train.extend(row(10000 + i) for i in range(1916))
    splits = dict(train=train, val=[], test=[])
    plan = (config.training_plan(args, splits, 4) if variant == 'prior'
            else common.training_plan(args, splits, 4, variant))
    assert plan['samples_per_epoch'] == 116256
    assert plan['optimizer_steps_per_epoch'] == 1817
    assert plan['actual_step_limit'] == 12719
    audit = plan['sampling']['hierarchical']
    assert audit['sampled'] == {**dict.fromkeys(eb.SPECIAL_BUCKETS, 9688), eb.REGULAR_BACKGROUND: 19376}
    assert audit['max_frame_repeats'] == 11
    args.event_balance_max_frame_repeats = 8
    with pytest.raises(ValueError, match='infeasible.*shared-frame'):
        config.training_plan(args, splits, 4)
    args.event_balanced_epoch_samples = 0
    assert config.training_plan(args, splits, 4)['samples_per_epoch'] == 91968


@pytest.mark.parametrize('variant', ['qwen_simple', 'bev_only'])
@pytest.mark.parametrize('budget,cap', [(0, 8), (95136, 8), (116256, 11), (None, None)])
def test_resume_budget_does_not_inherit_new_training_defaults(tmp_path, variant, budget, cap):
    saved = vars(common.parser(variant).parse_args(['--action-balanced']))
    for key, value in [('event_balanced_epoch_samples', budget), ('event_balance_max_frame_repeats', cap)]:
        if value is None:
            saved.pop(key)
        else:
            saved[key] = value
    (tmp_path / 'config.json').write_text(json.dumps(saved))
    restored = common.parse_train_args(variant, ['--resume', str(tmp_path / 'latest.pt')])
    assert restored.event_balanced_epoch_samples == (0 if budget is None else budget)
    assert restored.event_balance_max_frame_repeats == (8 if cap is None else cap)


@pytest.mark.parametrize('policy', ['smooth_cap', 'cycle_even', 'global_action'])
def test_real_action_sampler_interface_and_cross_epoch_cursor(policy):
    rows = []
    for event in eb.SPECIAL_BUCKETS:
        for i in range(20):
            rows.append(row(len(rows), [event], 'STOP'))
    for i in range(40):
        rows.append(row(len(rows)))
    if policy == 'global_action':
        rows = pool()
    mode = 'action_balanced' if policy == 'global_action' else 'event_balanced'
    cursor, seen = {}, set()
    for epoch in range(2):
        chosen, audit = eb.build_balanced_epoch(
            rows if epoch == 0 else rows[::-1], mode=mode, sampling_policy=policy,
            total=120, seed=epoch, master_seed=13, repeat_cap=1, cursor_offsets=cursor,
        )
        assert audit['cursor_start'] == cursor
        cursor = audit['next_cursor_offsets']
        assert cursor and audit['max_frame_repeats'] == 1
        if policy != 'global_action':
            assert audit['quotas'] == eb.weighted_quotas(120)
        seen.update(eb._identity(r) for r in chosen)
    if policy != 'global_action':
        assert len(seen) == 240


def test_hierarchical_rare_action_cap_and_same_event_return():
    rows = []
    for event in eb.SPECIAL_BUCKETS:
        for _ in range(100):
            rows.append(row(len(rows), [event], 'STOP'))
        for _ in range(4):
            rows.append(row(len(rows), [event], 'KEEP'))
    for _ in range(200):
        rows.append(row(len(rows)))
    selected, audit = ab.build_hierarchical_epoch(rows, total=1200, seed=1, repeat_cap=2)
    assert len(selected) == 1200 and audit['max_frame_repeats'] <= 2
    assert audit['sampled_actions'] == {'STOP': 920, 'KEEP': 80, 'UNCOND': 200}
    assert audit['quotas'] == eb.weighted_quotas(1200)


def test_formal_action_sampler_fairly_exposes_all_fourteen_frames():
    """原固定最小费用同解只覆盖13/14；事件/动作/cap都不允许因公平性改变。"""
    rows = [row(i, [event], 'STOP') for i, event in enumerate(eb.SPECIAL_BUCKETS)]
    rows += [row(10), row(11), row(12, ['UE1'], 'STOP'), row(13, ['UE1', 'UE2'], 'STOP')]
    cursor, history, seen = {}, {}, set()
    for epoch in range(7):
        kwargs = dict(mode='event_balanced', sampling_policy='smooth_cap', total=12,
                      seed=epoch, master_seed=1, repeat_cap=1, cursor_offsets=cursor, pool_history=history)
        selected, audit = eb.build_balanced_epoch(rows, **kwargs)
        # JSON保存/恢复与输入倒序不能影响计划或公平历史。
        restored = json.loads(json.dumps(dict(cursor_offsets=cursor, pool_history=history)))
        replay, report = eb.build_balanced_epoch(rows[::-1], **{**kwargs, **restored})
        assert replay == selected and report == audit
        assert audit['sampled'] == eb.weighted_quotas(12)
        assert audit['sampled_actions'] == {'STOP': 10, 'UNCOND': 2}
        assert audit['max_frame_repeats'] == 1 and audit['unique_frames'] == 12
        assert audit['action_quota_overflow'] == 0
        for key, pool in audit['pools'].items():
            assert pool['cumulative_presentations'] == cursor.get(key, 0) + pool['presentations']
            assert pool['skipped_epochs'] == (0 if pool['presentations'] else history.get(key, {}).get('skipped_epochs', 0) + 1)
        cursor, history = audit['next_cursor_offsets'], audit['next_pool_history']
        seen.update(r['anchor'] for r in selected)
        if epoch == 1:
            assert seen == set(range(14))
    assert sum(p['presentations'] for p in history.values()) == 84
    assert max(p['skipped_epochs'] for p in history.values()) <= 1
