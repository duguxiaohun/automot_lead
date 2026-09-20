"""RGB 边界案例导出的时间窗回归；不通过调整阈值迎合模型输出。"""
from dataclasses import replace

import pytest

from qwen3vl_local.sft_new_loop_phase3.trajectory_action import (
    ACTION_RULE_VERSION, longitudinal_decision, label_actions, action_evidence,
)
from qwen3vl_local.sft_new_loop_phase3.prompts import (
    PROMPT_NAME, make_prompt_spec, build_action_prompt, SPEED_ACTION_RULES, LANE_ACTION_RULES,
)
from qwen3vl_local.sft_new_loop_phase3.test_action_contract import _signals


@pytest.mark.parametrize('tail', [[0, 0], [20, 20], [float('nan')]])
def test_longer_evidence_cannot_change_window_label(tail):
    # #236：+2s以后的增速不能进入纵向标签；也不能用未来缺值取消完整窗。
    signals = _signals(future_speeds=[5]*9+tail, future_speed_count=9+len(tail))
    assert label_actions(signals) == label_actions(_signals(future_speeds=[5]*9))
    decision = longitudinal_decision(signals['future_speeds'])
    assert decision['action'] == 'NONE' and decision['ignored_tail_samples'] == len(tail)


def test_reversal_after_two_seconds_does_not_quarantine_resume():
    speeds = [5, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0]
    assert longitudinal_decision(speeds)['action'] == 'RESUME'
    assert label_actions(_signals(future_speeds=speeds, future_speed_count=len(speeds)))['RESUME']


def test_diagnostic_slices_ignore_extra_future_samples():
    from qwen3vl_local.sft_new_loop_phase3.audit_temporal_slices import slices
    row = dict(history_rgb_paths=['a', 'b', 'c', 'd'], invalid_action_context=False,
               answers={'DECELERATE': False, 'RESUME': False},
               action_evidence={'future_speeds_exact_mps': [5]*9})
    base = slices(row)
    row['action_evidence']['future_speeds_exact_mps'] += [.5, 3.8]
    assert slices(row) == base


def test_gain_must_be_confirmed_by_deadline_and_use_anchor_baseline():
    # 最后一点首次到达增速阈值，+2.25s的第二点不能补确认。
    late = longitudinal_decision([5]*8+[7, 7])
    assert late['action'] == 'NONE' and late['gain_unconfirmed_at_2s_boundary']
    inside = longitudinal_decision([5]*7+[7, 7])
    assert inside['action'] == 'RESUME' and inside['gain_confirmed_s'] == 2
    # #300：从未来峰值算降幅会误标DECEL；应相对最新速度16.25，阈值3.25。
    baseline = longitudinal_decision([16.25, 16.7, 17.74, 18.46, 17.9, 16.07, 14.22, 13.29, 13.75])
    assert baseline['action'] == 'NONE' and baseline['delta_threshold_mps'] == 3.25


def test_wait_priority_and_mixed_windows_keep_distinct_reasons():
    wait = longitudinal_decision([0, .1, 2, 4, 4, 4, 4, 4, 4])
    mixed = longitudinal_decision([5, 7, 7, 5, 5, 5, 5, 5, 5])
    assert wait['action'] == 'STOP' and wait['reason'] == 'current_confirmed_wait'
    assert not mixed['eligible'] and mixed['action'] is None
    assert mixed['reason'] == 'mixed_longitudinal_phase' and mixed['reversal_s'] == .75
    invalid = longitudinal_decision([5]*8+[float('nan')])
    assert not invalid['eligible'] and invalid['reason'] == 'invalid_speed_sample'


def test_evidence_explains_label_without_changing_input_prompt():
    signals = _signals(future_speeds=[5, 4, 3, 3, 3, 3, 3, 3, 3])
    evidence = action_evidence(signals)
    assert evidence['longitudinal_decision']['action'] == 'DECELERATE'
    assert evidence['longitudinal_decision']['first_drop_s'] == .5
    assert label_actions(signals)['DECELERATE']


@pytest.mark.parametrize('mode', ['binary', 'choice'])
@pytest.mark.parametrize('rgb', ['4rgb', '2rgb_endpoints'])
def test_v12_compact_prompt_is_answer_independent_while_v10_calibration_stays_bounded(mode, rgb):
    """v12 补充场景目的，不能泄漏答案或倒退 v10 的离线判定器。"""
    spec = make_prompt_spec(variant='all_random_order', answers={'STOP': True}, seed_key='same-input',
        context_id='STATIC_BLOCKAGE', road_structure='R2', goal_xy=(30, 1),
        current_speed_mps=5, action_output_mode=mode)
    prompt = build_action_prompt(spec=spec, history_rgb_mode=rgb)
    changed = replace(spec, invalid_context=True,
        questions=tuple(replace(q, answer=not q.answer) for q in spec.questions))
    assert prompt == build_action_prompt(spec=changed, history_rgb_mode=rgb)
    assert PROMPT_NAME.endswith("v14_observed_progress")
    assert ACTION_RULE_VERSION == "current_wait_first_crossing_v8_bounded_window"
    if mode == "binary":
        assert SPEED_ACTION_RULES in prompt
        assert LANE_ACTION_RULES in prompt
        assert 'two consecutive samples' in prompt and 'max(1.2 m/s, 20% of current speed)' in prompt
    else:
        assert "Choose one primary action or NONE. Speed: next 2 seconds." in prompt
        assert 'two consecutive samples' in prompt and 'max(1.2 m/s, 20% of current speed)' in prompt
    assert 'crossings already in the input' in prompt
    assert 'earlier-started maneuver' not in prompt
    assert 'longitudinal_decision' not in prompt and 'future_speeds' not in prompt
