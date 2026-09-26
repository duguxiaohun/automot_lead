"""固定预算/上限的容量证书；诊断不把上界当可执行采样方案。"""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import itertools
import random

import pytest

from qwen3vl_local.sft_new_loop_phase3.capacity_diagnostic import (
    SharedFrameCapacityError, diagnose_joint_capacity,
)


def frame(number):
    return dict(scenario='S', route_id='R', frame_id=number)


def test_matches_logged_total_shortfall_without_claiming_real_data_reproduction():
    """同数值合成拓扑：并非训练机真实候选重放。"""
    shared = [frame(i) for i in range(252)]
    other = [frame(i + 1000) for i in range(1274)]
    targets = {'A': 1024, 'B': 1022, 'C': 10190}
    report = diagnose_joint_capacity({'A': shared, 'B': shared, 'C': other}, targets, repeat_cap=8)
    assert (report['target'], report['feasible'], report['shortfall']) == (12236, 12206, 30)
    assert report['bottleneck_events'] == ['A', 'B']
    assert report['bottleneck_unique_frames'] == 252
    assert report['bottleneck_target'] == 2046
    assert report['bottleneck_available_presentations'] == 2016
    assert report['minimum_repeat_cap'] == 9
    assert report['automatic_changes'] is False
    assert targets == {'A': 1024, 'B': 1022, 'C': 10190}


def test_relaxed_invalid_quota_result_is_only_a_coverage_unchecked_upper_bound():
    prefix = 'INVALID|source=LEAD_BRAKE|true_rs=R1|asked_context='
    groups = {'LEAD_BRAKE': [frame(0)],
              prefix + 'SIGNAL_FAILURE|R4|wrong_road_structure': [frame(0)],
              prefix + 'RAMP_MERGE_EXIT|R3|wrong_road_structure': [frame(1), frame(2)]}
    report = diagnose_joint_capacity(groups, dict.fromkeys(groups, 1), repeat_cap=1)
    assert report['feasible'] == 2 and report['target'] == 3
    upper = report['invalid_source_only_upper_bound']
    assert upper['feasible'] == 3 and upper['source_quotas_preserved'] is True
    assert upper['executable_plan'] is False
    assert upper['signature_prompt_rs_coverage_preserved'] is False


def test_reservations_and_missing_support_are_not_hidden():
    report = diagnose_joint_capacity({'A': [frame(0)], 'B': [frame(0)]}, {'A': 1, 'B': 1},
                                     repeat_cap=2, initial_frame_usage={('S', 'R', '0'): 1})
    assert report['feasible'] == 1 and report['minimum_repeat_cap'] == 3
    assert report['reserved_presentations'] == 1
    report = diagnose_joint_capacity({'A': []}, {'A': 1}, repeat_cap=8)
    assert report['feasible'] == 0 and report['minimum_repeat_cap'] is None


def test_capacity_and_minimum_cap_match_exhaustive_assignments():
    rng = random.Random(194)
    for _ in range(45):
        groups = {e: [frame(i) for i in range(3) if rng.random() < .6] for e in ('A', 'B')}
        targets = {e: rng.randrange(1, 3) for e in groups}
        cap = rng.randrange(1, 3)
        candidates = [[None, *[row['frame_id'] for row in groups[e]]]
                      for e, n in targets.items() for _ in range(n)]
        assignments = [Counter(i for i in assignment if i is not None)
                       for assignment in itertools.product(*candidates)]
        feasible = max(sum(counts.values()) for counts in assignments if max(counts.values(), default=0) <= cap)
        minimum = min((max(cap, max(counts.values())) for counts in assignments
                       if sum(counts.values()) == sum(targets.values())), default=None)
        report = diagnose_joint_capacity(groups, targets, repeat_cap=cap)
        assert report['feasible'] == feasible
        assert report['minimum_repeat_cap'] == minimum


def test_phase3_failure_reports_and_preserves_parameters_state_and_reviewed_cases(capsys):
    from qwen3vl_local.sft_new_loop_phase3 import train
    from qwen3vl_local.sft_new_loop_phase3.test_validation_balance import candidate_rows
    rows = candidate_rows(0)
    positives = [r for r in rows if not r.invalid_source]
    shared = positives[0]
    negatives = [replace(r, scenario=shared.scenario, route_id=shared.route_id, frame_id=shared.frame_id)
                 for r in rows if r.invalid_source]
    cursors, history, audit = {}, {}, {}
    original_rows = deepcopy(positives + negatives)
    with pytest.raises(SharedFrameCapacityError) as error:
        train._balanced_work(positives + negatives, target_per_bin=1, invalid_multiplier=1,
                             require_invalid_coverage=False, seed=4, mode='smooth_cap', repeat_cap=1,
                             cursor_state=cursors, pool_history=history, sampling_audit=audit)
    assert isinstance(error.value, ValueError)
    assert error.value.report['target_per_bin'] == 1
    assert error.value.report['repeat_cap'] == 1
    assert error.value.report['shortfall'] == 1
    assert audit['capacity_failure'] == error.value.report
    assert cursors == {} and history == {}
    assert positives + negatives == original_rows
    assert '[phase3-capacity]' in capsys.readouterr().out


def test_successful_sampling_never_calls_failure_diagnostics(monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import capacity_diagnostic, invalid_capacity, train
    from qwen3vl_local.sft_new_loop_phase3.test_validation_balance import candidate_rows

    def forbidden(*args, **kwargs):
        raise AssertionError('successful plan must not use diagnostics')

    monkeypatch.setattr(capacity_diagnostic, 'diagnose_joint_capacity', forbidden)
    monkeypatch.setattr(invalid_capacity, 'reallocate_invalid_targets', forbidden)
    work = train._balanced_work(candidate_rows(0), target_per_bin=1, invalid_multiplier=1,
                                require_invalid_coverage=False, seed=4, mode='smooth_cap', repeat_cap=8)
    assert len(work) == 11
