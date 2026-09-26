"""连续 RGB 复核所见的短暂停顿/确认截止边界必须分开，标签和输入保持隔离。"""
import json
from pathlib import Path

import pytest

from qwen3vl_local.sft_new_loop_phase3.action_review import near_stop_review
from qwen3vl_local.sft_new_loop_phase3.audit_label_boundaries import speed_boundaries
from qwen3vl_local.sft_new_loop_phase3.build_dataset import development_route_groups, _split
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import longitudinal_decision


def test_confirmation_boundary_is_not_a_released_single_sample():
    # RGB review 34: actual stop begins at +1.5 s, persists after the confirmation deadline.
    speeds = [7.866, 6.906, 6.310, 4.103, 4.314, 2.917, .021, 0, 0]
    before = longitudinal_decision(speeds)
    report = speed_boundaries(speeds)
    assert before['action'] == report['speed_action'] == 'DECELERATE'
    assert 'near_stop_pair_confirmation_crosses_immediate_boundary' in report['flags']
    assert 'single_near_stop_sample_released_in_window' not in report['flags']
    segment, = report['near_stop']['segments']
    assert segment['start_s'] == 1.5 and segment['confirmed_s'] == 1.75
    assert segment['open_at_window_end'] and segment['release_s'] is None
    assert longitudinal_decision(speeds) == before


def test_transient_pause_and_later_stop_are_independent_segments():
    speeds = [4, .1, 2, 3, 2, 1, 1, .2, .1]
    report = near_stop_review(speeds)
    assert set(report['flags']) == {'single_near_stop_sample_released_in_window',
                                  'near_stop_pair_starts_after_immediate_window'}
    assert report['segments'][0]['release_s'] == .5
    assert report['segments'][1]['confirmed_s'] == 2
    assert longitudinal_decision(speeds)['action'] == 'DECELERATE'


def test_last_sample_does_not_invent_a_release_or_confirmation_from_tail():
    speeds = [4]*8 + [.1]
    result = near_stop_review(speeds)
    assert result['flags'] == ['near_stop_at_window_end_unconfirmed']
    assert result == near_stop_review(speeds + [4, 4]) == near_stop_review(speeds + [0, 0])
    assert result['segments'][0]['confirmed_s'] is None


@pytest.mark.parametrize('speeds', [[1]*8, [1]*8+[float('nan')], [1]*8+[-1]])
def test_incomplete_or_invalid_evidence_does_not_become_a_clean_motion_window(speeds):
    assert near_stop_review(speeds)['eligible'] is False


def test_near_stop_samples_do_not_override_confirmed_pullaway():
    speeds = [0, .2, 1, 2, 3, 4, 5, 6, 7]
    review = near_stop_review(speeds, brake=False, throttle=1)
    assert review['segments'][0]['sample_count'] == 2
    assert review['flags'] == []
    assert longitudinal_decision(speeds, brake=False, throttle=1)['action'] == 'RESUME'


def test_exposed_physical_groups_are_train_only_for_new_splits():
    groups = json.loads(Path(__file__).with_name('development_route_groups_20260926.json').read_text())['groups']
    assert len(groups) > 118 and set(groups) <= development_route_groups()
    for group in groups:
        scenario, route = group.split('/', 1)
        # Repetition and recording time cannot recover a holdout identity.
        town, remainder = route.split('_', 1)
        repeated = f'{town}_Rep9_{remainder}_route0_09_26_01_02_03'
        for seed in (20260904, 20260920):
            assert _split(scenario, repeated, seed, .2, .2) == 'train'
