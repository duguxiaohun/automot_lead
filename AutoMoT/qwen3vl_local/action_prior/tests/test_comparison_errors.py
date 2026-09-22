"""Waypoint ADE/FDE OR判据；route不参与，模型间按对应时刻距离计算。"""
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_prior.comparison_errors import error_selection


def predictions():
    return {f'model_{i:02d}': {key: [[[0., 0.] for _ in range(4)]] for key in
        ('pred_route','gt_route','pred_waypoints','gt_waypoints')} for i in range(1,4)}


@pytest.mark.parametrize('metric', ['ade','fde'])
def test_any_model_gt_independent_or(metric):
    data=predictions()
    pts=[[1.1,0.]]*4 if metric=='ade' else [[0.,0.]]*3+[[3.2,0.]]
    data['model_03']['pred_waypoints']=[pts]
    decision=error_selection(data,True)
    assert decision['kept']
    assert {t['metric'] for t in decision['triggers']} == {metric}


@pytest.mark.parametrize('metric', ['ade','fde'])
def test_nonfirst_model_pair_can_trigger_while_both_near_gt(metric):
    data=predictions()
    for mid,sign in [('model_02',1),('model_03',-1)]:
        data[mid]['pred_waypoints']=[[[sign*.75,0.]]*4 if metric=='ade' else [[0.,0.]]*3+[[sign*1.75,0.]]]
    result=error_selection(data,True)
    assert result['kept']
    assert len(result['triggers'])==1
    assert result['triggers'][0]['pair']==['model_02','model_03']
    assert result['triggers'][0]['metric']==metric


def test_route_ignored_strict_equal_and_switch_off():
    data=predictions()
    data['model_01']['pred_route']=[[[100.,100.]]*4]
    assert not error_selection(data,True)['kept']
    data['model_01']['pred_waypoints']=[[[1.,0.],[0.,0.],[0.,0.],[3.,0.]]]
    assert not error_selection(data,True)['kept']  # ADE=1, FDE=3 exactly
    assert error_selection(data,False)['kept']


@pytest.mark.parametrize('threshold',[0.,-1.,float('nan'),float('inf')])
@pytest.mark.parametrize('key',['ade_threshold_m','fde_threshold_m'])
def test_invalid_threshold(threshold,key):
    with pytest.raises(ValueError,match='threshold'):
        error_selection(predictions(),True,**{key:threshold})


@pytest.mark.parametrize('trajectory',[[],[[]],[[[float('nan'),0.]]],[[[0.]]],[[[0.,0.]]]])
def test_invalid_or_misaligned_trajectory(trajectory):
    data=predictions();data['model_02']['pred_waypoints']=trajectory
    with pytest.raises(ValueError):
        error_selection(data,True)
