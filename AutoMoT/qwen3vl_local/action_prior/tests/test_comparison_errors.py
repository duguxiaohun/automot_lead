"""误差筛选判据：终点距离、任意模型对、严格阈值和非法输入。"""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_prior.comparison_errors import error_selection


def predictions():
    return {f"model_{i:02d}": {key: [[[0., 0.], [0., 0.]]] for key in
            ("pred_route", "gt_route", "pred_waypoints", "gt_waypoints")} for i in range(1, 4)}


@pytest.mark.parametrize("key,kind", [("pred_route", "route"), ("pred_waypoints", "waypoint")])
def test_any_model_gt_including_third(key, kind):
    data = predictions()
    data['model_03'][key][0][-1] = [0., 1.1]
    result = error_selection(data, True)
    assert result['kept']
    assert result['max_distances_m'][f'{kind}_vs_gt_m'] == pytest.approx(1.1)


@pytest.mark.parametrize("key,kind", [("pred_route", "route"), ("pred_waypoints", "waypoint")])
def test_pairwise_distance_not_difference_of_fde(key, kind):
    data = predictions()
    data['model_02'][key][0][-1] = [.75, 0.]
    data['model_03'][key][0][-1] = [-.75, 0.]
    result = error_selection(data, True)
    assert result['kept']
    assert result['max_distances_m'][f'{kind}_vs_gt_m'] == .75
    assert result['triggers'] == [dict(trajectory=kind, pair=['model_02', 'model_03'], distance_m=1.5)]


def test_strict_threshold_endpoint_only_and_switch_off():
    data = predictions()
    data['model_01']['pred_route'][0] = [[100., 100.], [1., 0.]]
    assert not error_selection(data, True, 1.)['kept']
    assert error_selection(data, False, 1.)['kept']
    assert error_selection(data, True, .9)['kept']


@pytest.mark.parametrize('threshold', [0., -1., float('nan'), float('inf')])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match='threshold'):
        error_selection(predictions(), True, threshold)


@pytest.mark.parametrize('trajectory', [[], [[]], [[[float('nan'), 0.]]], [[[0.]]]])
def test_invalid_endpoints_fail_instead_of_silently_hiding(trajectory):
    data = predictions()
    data['model_02']['pred_waypoints'] = trajectory
    with pytest.raises(ValueError, match='endpoint'):
        error_selection(data, True)
