"""Lane forecast continuity and observable prompt semantics, without new approvals."""
import math
import random
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_events as t, route_prompts as prompts
from qwen3vl_local.sft_new_loop_phase4.tests.test_ten_event_teacher import frames,episode


def parallel(speed=28.):
    fs=frames('U-E3')
    for f in fs:
        f['meta'].update(is_junction=False,route=[[float(x),0.] for x in range(0,41,2)])
        f['actors'][1].update(position=[5.,-4.,0.],yaw=0.,extent=[1.,.9,1.],lane_id=2,
                             ego_velocity=[speed,0.],speed=speed)
    return fs


def test_lane_corner_occupancy_between_samples_reaches_native_clearance():
    fs=parallel();a=fs[-1]['actors'][1]
    nav=dict(points=[[0.,0.],[14.,0.],[14.,10.],[24.,10.]],half_width=1.75)
    assert all(t.route_intersection(p,nav,margin=.25) is False for p in t.lane_predictions(fs,a,nav))
    assert t.lane_intersection(fs,a,nav) is True
    assert t.clearance(fs,nav) is not True


def test_supported_parallel_lane_remains_clear_without_blanket_prediction_block():
    fs=parallel(10.);a=fs[-1]['actors'][1]
    nav=dict(points=[[0.,0.],[10.,0.],[20.,5.]],half_width=1.75)
    assert t.lane_intersection(fs,a,nav) is False
    assert t.clearance(fs,nav) is True


def test_missing_history_falls_back_to_generic_and_finite_endpoint_is_not_extended():
    fs=parallel();a=fs[-1]['actors'][1]
    nav=dict(points=[[0.,0.],[10.,0.]],half_width=1.75)
    assert t.lane_intersection(fs,a,nav,horizon_s=4.) is False
    fs[-1]['meta']['is_junction']=True
    assert t.lane_intersection(fs,a,nav) is None
    assert t.clearance(fs,nav) is True


@pytest.mark.parametrize('horizon',[0.,-1.,float('nan'),float('inf')])
def test_lane_horizon_validation_including_missing_evidence(horizon):
    with pytest.raises(ValueError,match='horizon'):t.lane_intersection([],{}, {},horizon_s=horizon)


def test_piecewise_forecast_contains_dense_samples_without_time_axis_mixing():
    rng=random.Random(46)
    nav=dict(points=[[0.,0.],[10.,0.],[15.,6.],[25.,6.]],half_width=1.75)
    for _ in range(35):
        fs=parallel(rng.uniform(1.,30.));a=fs[-1]['actors'][1]
        for f in fs:
            f['actors'][1]['position'][1]=rng.choice((-1,1))*rng.uniform(3.,6.)
        # Use a persistent lateral offset in all history frames.
        for f in fs:f['actors'][1]['position'][1]=a['position'][1]
        horizon=rng.uniform(.1,4.)
        sampled=[]
        for i in range(1,201):
            poses=t.lane_predictions(fs,a,nav,horizon_s=horizon*i/200)
            # Only the exact requested endpoint belongs to this dense probe.
            if poses:sampled.extend(poses)
        if any(t.route_intersection(p,nav,margin=.25) for p in sampled):
            assert t.lane_intersection(fs,a,nav,horizon_s=horizon) is True


@pytest.mark.parametrize('event,edge',[('U-E1','proceed'),('U-E7','proceed'),('U-E2','depart'),('R-E2','enter'),('R-E5','proceed')])
def test_native_messages_share_conjunctive_evidence_rubric(event,edge):
    ep=episode(event);obs=dict(frame_id=10,history_frames=[4,6,8,10],speed_mps=0.)
    msgs=prompts.messages(ep,edge,obs,['rgb']*4)
    assert msgs[0]['content']==prompts.SYSTEM
    assert 'every stated requirement together' in msgs[0]['content']
    assert 'movement alone does not establish permission' in msgs[0]['content']
    assert 'occluded does not establish clearance' in msgs[0]['content']
    assert 'successor has just been achieved' in msgs[0]['content']
    assert 'Current state:' in msgs[1]['content'][-1]['text']
    assert 'Candidate next state:' in msgs[1]['content'][-1]['text']
    assert prompts.parse_answer('NO\n')=='NO'
    with pytest.raises(ValueError):prompts.parse_answer('UNKNOWN')
