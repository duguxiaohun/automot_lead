"""INVALID来源内细分容量回流：复现远端瓶颈拓扑，保留1024/cap8和覆盖。"""
from collections import Counter
from dataclasses import replace
import itertools
import random
from types import SimpleNamespace

import pytest

from qwen3vl_local.sft_new_loop_phase3 import train
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS, CONTEXT_BY_ID
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity, signature_for_row
from qwen3vl_local.sft_new_loop_phase3.invalid_capacity import reallocate_invalid_targets
from qwen3vl_local.sft_new_loop_phase3.sampling import hierarchical_event_action_epoch_sample, _item_identity
from qwen3vl_local.sft_new_loop_phase3.test_validation_balance import candidate_rows


def item(source, asked, frame, *, rs='R4', route='rare', reason='wrong_road_structure'):
    row = train.FrameRow(scenario='S', route_id=route, town='Town01', frame_id=frame, true_rs=rs,
        prompt_road_structure='R3', context_id=asked, question_domain=CONTEXT_BY_ID[asked].question_domain,
        action_signature='INVALID', event='', split='train', goal_ego_xy=(10, 0),
        history_rgb_paths=['unused.jpg'] * 4, latest_rgb_path='unused.jpg', current_speed_mps=3,
        answers={**dict.fromkeys(train.ANSWER_KEYS, False), train.INVALID_KEY: True},
        invalid_source=f'source={source}|true_rs={rs}|asked_context={asked}', invalid_reason=reason)
    return SimpleNamespace(row=row)


def key(item):
    row = item.row
    return 'INVALID|' + '|'.join((row.invalid_source, row.prompt_road_structure, row.invalid_reason))


def rare_groups(frame_count=3):
    groups, targets = {}, {}
    asked = ['DYNAMIC_CUTIN', 'LEAD_BRAKE', 'ONCOMING_INVASION', 'POST_BYPASS_RETURN',
             'RAMP_MERGE_EXIT', 'STATIC_BLOCKAGE', 'VULNERABLE_CROSSING']
    for context, target in zip(asked, [8, 7, 8, 8, 7, 8, 8]):
        rows = [item('DYNAMIC_CUTIN', context, i) for i in range(frame_count)]
        groups[key(rows[0])] = rows
        targets[key(rows[0])] = target
    rows = [item('DYNAMIC_CUTIN', 'SIGNAL_FAILURE', i, rs='R1', route='donor') for i in range(40)]
    groups[key(rows[0])] = rows
    targets[key(rows[0])] = 151
    return groups, targets


@pytest.mark.parametrize('reserved_count', [0, 1])
def test_three_frame_seven_group_bottleneck_keeps_source_205_cap8(reserved_count):
    groups, targets = rare_groups()
    reserved = {('S', 'rare', '0'): reserved_count} if reserved_count else {}
    adjusted, report = reallocate_invalid_targets(groups, targets, repeat_cap=8, reserved=reserved)
    assert sum(adjusted.values()) == 205
    assert report['source_targets'] == report['source_actual'] == {'DYNAMIC_CUTIN': 205}
    assert report['shifted_presentations'] == 30 + reserved_count
    assert all(adjusted[e] >= 1 for e in targets)
    assert sum(n for e, n in adjusted.items() if 'true_rs=R4' in e) == 24 - reserved_count
    selected, audit = hierarchical_event_action_epoch_sample(
        groups, adjusted, repeat_cap=8, action_fn=lambda _: 'INVALID', initial_frame_usage=reserved)
    assert len(selected) == 205 and audit['max_frame_repeat'] <= 8


def test_cannot_borrow_other_source_budget_or_drop_coverage():
    groups, targets = rare_groups()
    donor = next(e for e in groups if 'true_rs=R1' in e)
    rows = [item('LEAD_BRAKE', 'SIGNAL_FAILURE', i, rs='R1', route='other-source') for i in range(40)]
    del groups[donor]
    target = targets.pop(donor)
    groups[key(rows[0])], targets[key(rows[0])] = rows, target
    assert reallocate_invalid_targets(groups, targets, repeat_cap=8) is None
    groups, targets = rare_groups()
    # 7个必须保留的细分组共用3帧，cap1时连覆盖下限都不可行，不能丢组。
    assert reallocate_invalid_targets(groups, targets, repeat_cap=1) is None


def test_original_feasible_target_vector_is_unchanged():
    groups, targets = rare_groups()
    for e in targets:
        if 'true_rs=R4' in e:
            targets[e] = 2
    adjusted, report = reallocate_invalid_targets(groups, targets, repeat_cap=8)
    assert adjusted == targets and report['shifted_presentations'] == 0


def test_positive_and_reviewed_cases_share_automatic_frame_capacity():
    groups, targets = rare_groups()
    positive = next(row for row in candidate_rows(0) if not row.invalid_source)
    positive = replace(positive, scenario='S', route_id='rare', frame_id=0)
    groups[positive.context_id], targets[positive.context_id] = [SimpleNamespace(row=positive)], 2
    reserved = {('S', 'rare', '1'): 1}
    adjusted, report = reallocate_invalid_targets(groups, targets, repeat_cap=8, reserved=reserved)
    assert adjusted[positive.context_id] == 2
    assert report['shifted_presentations'] == 33
    selected, audit = hierarchical_event_action_epoch_sample(
        groups, adjusted, repeat_cap=8, initial_frame_usage=reserved,
        action_fn=lambda it: 'INVALID' if it.row.invalid_source else 'KEEP')
    assert len(selected) == 207 and audit['max_frame_repeat'] <= 8


def test_reallocation_is_order_independent_and_binds_training_contract():
    groups, targets = rare_groups()
    original = reallocate_invalid_targets(groups, targets, repeat_cap=8)
    reversed_groups = {e: list(reversed(rows)) for e, rows in reversed(list(groups.items()))}
    assert reallocate_invalid_targets(reversed_groups, dict(reversed(list(targets.items()))),
                                      repeat_cap=8) == original
    args = SimpleNamespace(seed=20260904, action_output_mode='binary', sampling_policy='smooth_cap')
    config = train.sampling_config(args)['invalid_capacity_reallocation']
    assert config['version'] == original[1]['version'] and len(config['sha256']) == 64
    args.action_output_mode = 'choice'
    assert train.sampling_config(args)['invalid_capacity_reallocation'] is None
    args.action_output_mode, args.sampling_policy = 'binary', 'cycle_even'
    assert train.sampling_config(args)['invalid_capacity_reallocation'] is None


def test_reallocation_matches_exhaustive_minimum_deviation():
    rng = random.Random(97)
    for _ in range(45):
        groups, targets = {}, {}
        for context in ('LEAD_BRAKE', 'DYNAMIC_CUTIN', 'STATIC_BLOCKAGE'):
            rows = [item('DYNAMIC_CUTIN', context, i) for i in range(4) if rng.random() < .7]
            if not rows:
                rows = [item('DYNAMIC_CUTIN', context, 0)]
            groups[key(rows[0])] = rows
            targets[key(rows[0])] = rng.randrange(1, 3)
        cap, budget = rng.randrange(1, 3), sum(targets.values())
        choices = [(e, _item_identity(row)) for e, rows in groups.items() for row in rows]
        solutions = []
        # 枚举每个细分组每帧的次数，比逐呈现排列小得多。
        for counts in itertools.product(range(cap + 1), repeat=len(choices)):
            if sum(counts) != budget:
                continue
            events, frames = Counter(), Counter()
            for (e, frame), n in zip(choices, counts):
                events[e] += n
                frames[frame] += n
            if min(events.values()) >= 1 and max(frames.values()) <= cap:
                solutions.append(sum(max(0, events[e] - targets[e]) for e in targets))
        repaired = reallocate_invalid_targets(groups, targets, repeat_cap=cap)
        if not solutions:
            assert repaired is None
        else:
            assert repaired[1]['shifted_presentations'] == min(solutions)


@pytest.mark.parametrize('frame_count,manual_count', [(2, 52), (3, 52), (4, 53), (8, 0)])
def test_real_training_joint_path_preserves_12288_and_manual_cases_across_epochs(
        monkeypatch, frame_count, manual_count):
    """只固定原INVALID选样计划；实际分配、扩展、游标/历史推进走生产代码。"""
    groups, targets = rare_groups(frame_count)
    positives = [row for row in candidate_rows(0) if not row.invalid_source]
    positive_rows = [replace(row, route_id=f'positive-{row.context_id}', frame_id=i, split='train')
                     for row in positives for i in range(128)]
    sources = {source: 205 if i < 8 else 204 for i, source in enumerate(CONTEXT_IDS)}
    manual_source = CONTEXT_IDS[0]
    manual = [item(manual_source, 'LEAD_BRAKE', i, rs='R1', route='manual', reason='same_rs_wrong_event')
              for i in range(manual_count)]
    for source, budget in sources.items():
        if source == 'DYNAMIC_CUTIN':
            assert budget == 205
            continue
        rows = [item(source, 'SIGNAL_FAILURE', i, rs='R1', route=f'other-{source}') for i in range(40)]
        groups[key(rows[0])], targets[key(rows[0])] = rows, budget - (manual_count if source == manual_source else 0)
    planned = manual + [rows[i % len(rows)] for e, rows in groups.items() for i in range(targets[e])]
    assert len(planned) == 2048
    def fixed_invalid_plan(items, **kwargs):
        assert kwargs['target'] == 2048
        return [train._make_item(it.row, seed=kwargs['rng'].randrange(100), action_output_mode='binary') for it in planned]
    monkeypatch.setattr(train, 'balanced_invalid_items', fixed_invalid_plan)
    rows = positive_rows + [it.row for bucket in groups.values() for it in bucket] + [it.row for it in manual]
    cursors, history = {}, {}
    for epoch in range(7):
        audit = {}
        work, cursors = train._balanced_work(rows, target_per_bin=1024, invalid_multiplier=2,
            seed=20260904 + epoch, mode='smooth_cap', repeat_cap=8, sampling_audit=audit,
            cursor_state=cursors, pool_history=history, return_cursors=True)
        history = audit['next_pool_history']
        assert len(work) == 12288
        assert max(Counter(_item_identity(it) for it in work).values()) <= 8
        assert Counter(it.row.context_id for it in work if not it.row.invalid_source) == dict.fromkeys(CONTEXT_IDS, 1024)
        assert Counter(signature_for_row(it.row).source_class for it in work if it.row.invalid_source) == sources
        assert Counter(case_identity(it) for it in work if it.row.invalid_reason == 'same_rs_wrong_event') == Counter(map(case_identity, manual))
        expected_shift = max(0, 54 - frame_count * 8)
        if expected_shift:
            assert audit['invalid_capacity_reallocation']['shifted_presentations'] == expected_shift
            assert all(audit['invalid_capacity_reallocation']['coverage_floors'][e] <=
                       sum(1 for it in work if it.row.invalid_source and key(it) == e) for e in targets)
        else:
            assert 'invalid_capacity_reallocation' not in audit
