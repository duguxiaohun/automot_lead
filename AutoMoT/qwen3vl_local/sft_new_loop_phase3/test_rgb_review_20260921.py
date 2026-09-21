"""真实 RGB 审计导出的重编号隔离、起步边界及曝光路线回归。"""
import json
from pathlib import Path

import pytest

from qwen3vl_local.sft_new_loop_phase3.build_dataset import _split
from qwen3vl_local.sft_new_loop_phase3.lateral_rgb_audit import lateral_uncertainty
from qwen3vl_local.sft_new_loop_phase3.prepare_error_review import optional_bool
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import (
    RouteTrajectory, longitudinal_decision, lane_change_direction_from_ids,
)

ROOT = Path(__file__).parent
TOPOLOGY = json.loads((ROOT/'lane_topology_evidence_20260921.json').read_text())


@pytest.mark.parametrize('row', TOPOLOGY)
def test_reviewed_section_renumbering_is_unknown_not_a_lane_change(row):
    transition = row['transition_frame']
    # 已审地图中先处于较大s的section，沿合法predecessor进入lane2。
    before, after = row['frames']
    assert before['projected_s_m'] > row['section_boundary_s_m'] > after['projected_s_m']
    assert (before['lane_id'], after['lane_id']) == (1, 2)
    assert before['section_id'] is None and after['section_id'] is None
    for anchor in range(transition-12, transition):
        assert lateral_uncertainty(row['scenario'], row['route_id'], anchor)
    for anchor in (transition-13, transition, transition+1):
        assert lateral_uncertainty(row['scenario'], row['route_id'], anchor) is None
    assert lateral_uncertainty(row['scenario'], row['route_id']+'_different', transition-1) is None
    # 用完整且原本会触发RIGHT的序列检查signals：不能用False冒充可靠KEEP。
    start = transition-1
    metas = {f: dict(speed=8.,road_id=int(row['road_id']),lane_id=1 if f<transition else 2,
                     lane_type_str='Driving',pos_global=[f,0],theta=0,next_target_points=[[100,0]])
             for f in range(start, start+14)}
    t=RouteTrajectory(Path(row['scenario'])/row['route_id'],tuple(metas),metas,{int(row['road_id']):1})
    assert t.lane_change(start) is None
    assert t.signals(start)['lateral_observation_complete'] is False
    assert t.signals(start)['lateral_rgb_uncertainty'] is not None
    # 普通跨线的方向规则仍在；只隔离审过的精确区间。
    assert lane_change_direction_from_ids(1, 2, 1) == 'RIGHT'
    assert lane_change_direction_from_ids(1, -1, 1) == 'LEFT'


@pytest.mark.parametrize('speeds, expected', [
    ([0., .54, 1.04, 2.02, 3.15, 4., 4.5, 5., 5.2], 'RESUME'),
    ([0., .1, 1.04, 2.02, 3.15, 4., 4.5, 5., 5.2], 'STOP'),
    ([1.53, .005, .527, 1.53, 2.53, 4.16, 5., 6., 7.], 'DECELERATE'),
    ([7.93, 6.78, 5.59, 6.01, 8.22, 7.8, 8., 7.9, 8.], 'DECELERATE'),
])
def test_immediate_pullaway_wait_and_brief_deceleration_remain_distinct(speeds, expected):
    assert longitudinal_decision(speeds)['action'] == expected


def test_exposed_routes_stay_train_only_across_repetitions():
    groups=json.loads((ROOT/'development_route_groups_20260921.json').read_text())['groups']
    assert len(groups)==223
    for group in groups:
        scenario,stem=group.split('/',1)
        town,rest=stem.split('_',1)
        route=f'{town}_Rep91_{rest}_route0_02_02_02_02_02'
        for seed in (20260920,20260921):
            assert _split(scenario,route,seed,.99,.01)=='train'


def test_missing_review_fields_are_not_evidence_of_no_braking_or_hazard():
    assert optional_bool({},'brake') is None
    assert optional_bool({'brake':None},'brake') is None
    assert optional_bool({'brake':False},'brake') is False
    assert optional_bool({'brake':True},'brake') is True
