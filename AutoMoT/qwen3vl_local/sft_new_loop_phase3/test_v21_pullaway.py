"""起步与等待、缺证据、输入隔离及生产分布分母的回归。"""
import json
from pathlib import Path

import pytest

from qwen3vl_local.sft_new_loop_phase3.trajectory_action import (
    longitudinal_decision, longitudinal_from_signals, label_actions, action_evidence,
    recorded_controls, validate_action_rule,
)
from qwen3vl_local.sft_new_loop_phase3.test_action_contract import _signals
from qwen3vl_local.sft_new_loop_phase3.test_action_review import route
from qwen3vl_local.sft_new_loop_phase3.action_review import build_action_review
from qwen3vl_local.sft_new_loop_phase3.audit_label_boundaries import diagnose
from qwen3vl_local.sft_new_loop_phase3.audit_temporal_slices import timing_diagnostics
from qwen3vl_local.sft_new_loop_phase3.choice_semantics import choice_annotation
from qwen3vl_local.sft_new_loop_phase3.sampling import primary_action_distribution
from qwen3vl_local.sft_new_loop_phase3.build_dataset import _split

ROOT = Path(__file__).parent
CASES = json.loads((ROOT/'pullaway_regressions_20260921.json').read_text())['cases']


@pytest.mark.parametrize('case', CASES, ids=lambda c:f"{c['route']}_f{c['frame']}")
def test_raw_audited_pullaway_regressions(case):
    # RouteTrajectory沿用原始极小负速度归零（>=-0.05），没有四舍五入。
    speeds = case['speeds']
    assert speeds == [max(0.0,v) for v in case['raw_meta_speeds']]
    controls = {k:case[k] for k in ('brake','throttle')}
    assert longitudinal_decision(speeds)['action'] == 'STOP'  # 速度对自身不能证明释放。
    decision = longitudinal_decision(speeds, **controls)
    assert decision['action'] == 'RESUME'
    assert decision['reason'] == 'current_confirmed_pullaway'
    assert decision['current_near_stop_pair'] and not decision['stop_qualifies']
    assert timing_diagnostics(speeds, **controls) == decision


@pytest.mark.parametrize('controls', [{}, {'brake':False}, {'throttle':1},
    {'brake':True,'throttle':1}, {'brake':False,'throttle':.1},
    {'brake':'False','throttle':1}, {'brake':0,'throttle':1},
    {'brake':False,'throttle':True}, {'brake':False,'throttle':'1'},
    {'brake':False,'throttle':float('nan')}, {'brake':False,'throttle':2}])
def test_unknown_or_unreleased_control_cannot_establish_pullaway(controls):
    decision = longitudinal_decision(CASES[0]['speeds'], **controls)
    assert decision['action'] == 'STOP' and not decision['confirmed_pullaway']


@pytest.mark.parametrize('speeds', [[0]*9, [0,0,0,0,0,1.3,1.5,2,3],
    [0,.1,.3,.4,.5,.6,.7,.8,.9], [0,.1,1.5,2,0,0,0,0,0],
    [0,.1,.099,1,2,3,4,5,6], [0,0,0,0,0,0,.1,.2,3]])
def test_later_release_creep_reversal_and_unconfirmed_gain_remain_stop(speeds):
    assert longitudinal_decision(speeds,brake=False,throttle=1)['action'] == 'STOP'


def test_missing_window_and_tail_do_not_create_launch():
    speeds = CASES[0]['speeds']
    assert not longitudinal_decision(speeds[:8],brake=False,throttle=1)['eligible']
    a = longitudinal_decision(speeds,brake=False,throttle=1)
    b = longitudinal_decision(speeds+[0,float('nan')],brake=False,throttle=1)
    assert b['action'] == a['action'] == 'RESUME' and b['ignored_tail_samples'] == 2


def test_label_evidence_review_primary_and_boundaries_share_controls(tmp_path):
    speeds = CASES[0]['speeds']
    t = route(tmp_path, speeds+[speeds[-1]]*5)
    for m in t.metas.values():m.update(brake=False,throttle=1)
    s=t.signals(0);labels=label_actions(s);evidence=action_evidence(s)
    assert labels['RESUME'] and not labels['STOP']
    assert evidence['longitudinal_decision'] == longitudinal_from_signals(s)
    review=build_action_review(t,0,'ONCOMING_INVASION',labels,s)
    report, audited=diagnose(t,0,'ONCOMING_INVASION')
    assert report['speed_action'] == review['primary_action'] == 'RESUME' and audited==labels
    assert 'pullaway_with_padded_startup_history' in review['flags']
    assert not any(m['action']=='STOP' for m in review['motion_milestones'])
    # 主要动作仍优先首次跨线；改变速度位不等于强制choice=RESUME。
    s.update(lane_change_direction='LEFT',lateral_observation_complete=True)
    assert choice_annotation(label_actions(s),'STATIC_BLOCKAGE',action_evidence(s))['primary_action']=='LANE_CHANGE_LEFT'
    # 从磁盘证据重放同样需要锚点控制。
    rebuilt=dict(future_speeds=evidence['future_speeds_exact_mps'],future_speed_count=9,
                 lane_change_direction='',brake=evidence['brake'],throttle=evidence['throttle'])
    assert label_actions(rebuilt)==labels
    del t.metas[0]['brake']
    assert t.signals(0)['brake'] is None and action_evidence(t.signals(0))['brake'] is None
    assert label_actions(t.signals(0))['STOP']


def test_old_action_contract_is_rejected():
    with pytest.raises(ValueError,match='action rule mismatch'):
        validate_action_rule({'action_evidence':{'rule_version':'current_wait_first_crossing_v8_bounded_window'}})


def test_production_primary_projection_has_explicit_denominators():
    report=primary_action_distribution({'STOP':3270,'STOP+LANE_CHANGE_LEFT':262,
        'STOP+LANE_CHANGE_RIGHT':222,'NONE':6486})
    assert report['counts']['STOP']==3754 and report['total']==report['valid']==10240
    assert report['invalid']==0 and report['stop_fraction_all']==pytest.approx(.3666015625)
    report=primary_action_distribution({'STOP':4648,'INVALID':2374,'RESUME':7222})
    assert (report['total'],report['valid'],report['invalid'])==(14244,11870,2374)
    assert report['stop_fraction_valid']==4648/11870
    assert primary_action_distribution({})['stop_fraction_valid'] is None
    with pytest.raises(ValueError):primary_action_distribution({'invented':1})


def test_noise_audit_routes_are_no_longer_blind_validation():
    exposure=json.loads((ROOT/'development_route_groups_noise_20260921.json').read_text())
    assert len(exposure['routes'])==242
    for r in exposure['routes']:
        assert _split(r['scenario'],r['route_id'],20260920,.99,.01)=='train'


def test_confirmed_launch_can_adjust_speed_after_initial_gain():
    # ConstructionObstacle/Town12_1256 f84：起步后的调速不应回写成持续等待。
    speeds=[.0000027983,.43181672,1.45917021,2.78185993,4.60153830,
            5.73759326,7.36907251,6.89047344,5.81992416]
    decision=longitudinal_decision(speeds,brake=False,throttle=1)
    assert decision['action']=='RESUME' and not decision['whole_window_nondecreasing']
    # 明显放弃增速，或速度重新回到近停，不能复用这个例外。
    for tail in ([1.,2.5], [0.,0.]):
        assert longitudinal_decision(speeds[:7]+tail,brake=False,throttle=1)['action']=='STOP'
