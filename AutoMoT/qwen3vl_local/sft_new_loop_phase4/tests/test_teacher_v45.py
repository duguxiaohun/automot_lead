"""Continuous linear occupancy and exact configurable lane prediction horizon."""
import math
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_events as t
from qwen3vl_local.sft_new_loop_phase4.tests.test_ten_event_teacher import frames


def test_fast_crossing_between_half_second_samples_is_not_clear():
    fs=frames('U-E7')
    for f in fs:f['actors'][1].update(position=[10.,-7.,0.],extent=[1.,.4,1.],yaw=math.pi/2,ego_velocity=[0.,28.],speed=28.)
    a=fs[-1]['actors'][1];nav=dict(points=[[0.,0.],[22.,0.]],half_width=1.75)
    assert not any(t.route_intersection(dict(a,position=[10.,-7.+28.*dt,0.]),nav,margin=.25) for dt in [0.,.5,1.,1.5,2.])
    assert t.linear_intersection(a,nav,a['ego_velocity'],2.) is True
    assert t.clearance(fs,nav,horizon_s=2.) is not True


def test_both_axes_must_intersect_at_the_same_time():
    a=dict(position=[-20.,0.,0.],extent=[.5,.5,1.],yaw=0.)
    nav=dict(points=[[0.,0.],[10.,0.]],half_width=1.)
    # Crosses lateral strip only near t=0, reaches route x only near t=2.
    assert t.linear_intersection(a,nav,[10.,10.],3.) is False
    assert t.linear_intersection(a,nav,[10.,0.],3.) is True
    assert t.linear_intersection(a,nav,[-10.,0.],3.) is False


def test_stationary_and_endpoint_intersection_match_existing_geometry():
    nav=dict(points=[[0.,0.],[10.,0.]],half_width=1.)
    for x,y in [(-2.,0.),(0.,0.),(10.,0.),(12.,0.),(5.,4.)]:
        a=dict(position=[x,y,0.],extent=[.5,.5,1.],yaw=.3)
        assert t.linear_intersection(a,nav,[0.,0.],2.) is t.route_intersection(a,nav,margin=.25)


def test_lane_path_receives_long_horizon_and_exact_endpoint():
    fs=frames('U-E3',True)
    for f in fs:
        f['meta'].update(is_junction=False,route=[[float(x),0.] for x in range(0,41,2)])
        f['actors'][1].update(position=[5.,4.,0.],yaw=0.,extent=[1.,.9,1.],lane_id=2,ego_velocity=[3.,-.3],speed=3.)
    a=fs[-1]['actors'][1];nav=dict(points=fs[-1]['meta']['route'],half_width=1.75)
    short=t.lane_predictions(fs,a,nav,horizon_s=2.)
    long=t.lane_predictions(fs,a,nav,horizon_s=4.)
    assert len(short)==4 and len(long)==8
    assert not any(t.route_intersection(p,nav,margin=.25) for p in short)
    assert any(t.route_intersection(p,nav,margin=.25) for p in long)
    assert t.prediction_times(2.2)==[.5,1.,1.5,2.,2.2]
    assert t.clearance(fs,nav,horizon_s=2.) is True
    assert t.clearance(fs,nav,horizon_s=4.) is not True


@pytest.mark.parametrize('horizon',[0.,-1.,float('nan'),float('inf')])
def test_invalid_horizon_is_rejected(horizon):
    with pytest.raises(ValueError,match='horizon'):t.prediction_times(horizon)


def test_continuous_check_does_not_miss_dense_sampled_intersections():
    import random
    rng=random.Random(45)
    nav=dict(points=[[0.,0.],[7.,0.],[12.,6.]],half_width=1.5)
    for _ in range(80):
        a=dict(position=[rng.uniform(-10,20),rng.uniform(-15,15),0.],extent=[2.,.8,1.],yaw=rng.uniform(-math.pi,math.pi))
        v=[rng.uniform(-15,15),rng.uniform(-15,15)]
        sampled=any(t.route_intersection(dict(a,position=[a['position'][0]+v[0]*i*.01,a['position'][1]+v[1]*i*.01,0.]),nav,margin=.25) for i in range(201))
        if sampled:assert t.linear_intersection(a,nav,v,2.) is True
