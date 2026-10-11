"""Road-relative completion; synthetic tests are not independent label approval."""
from copy import deepcopy
import math
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_rules as t, teacher_events as events, route_prompts
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import frames
from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import actor


def bend():
    fs=frames();angle=.4
    points=[[0.,0.],[4.,0.]]+[[4.+d*math.cos(angle),d*math.sin(angle)] for d in range(2,30,2)]
    for f in fs:
        f['meta'].update(speed=5.,route=deepcopy(points))
        f['actors'][1].update(position=[4.+8*math.cos(angle),8*math.sin(angle),0.],
            yaw=angle,speed=5.,ego_velocity=[5*math.cos(angle),5*math.sin(angle)])
    return fs


def completion(fs):
    return t.facts(fs,Episode('U-E1','e',state='PROCEED',longitudinal='FOLLOW'),2)


def test_bend_uses_actor_tangent_not_ego_lateral_velocity():
    fs=bend();a=fs[-1]['actors'][1]
    assert a['ego_velocity'][1]>1.
    values,reasons=completion(fs)
    assert values['stable_progress_established'] is True
    assert values['event_resolved'] is True
    assert not any('following_' in r for r in reasons)
    m=t.lead_route_motion(fs[-1],a)
    assert abs(m['ego_velocity'][1])<1e-10
    assert m['gap']==pytest.approx(12-a['extent'][0]-fs[-1]['actors'][0]['extent'][0])


@pytest.mark.parametrize('failure',['closing','oscillation','lateral','small_gap','priority','conflict','missing_lane','crossing'])
def test_curved_following_does_not_remove_other_completion_gates(failure):
    fs=bend();h=.4
    if failure in ('closing','oscillation','lateral'):
        for i,f in enumerate(fs):
            v=3. if failure=='closing' else 7. if failure=='oscillation' and i==6 else 5.
            lat=.3 if failure=='lateral' else 0.
            f['actors'][1]['ego_velocity']=[v*math.cos(h)-lat*math.sin(h),v*math.sin(h)+lat*math.cos(h)]
    elif failure=='small_gap':
        for f in fs:f['actors'][1]['position']=[4.+2*math.cos(h),2*math.sin(h),0.]
    elif failure=='priority':fs[-1]['meta']['stop_sign_hazard']=True
    elif failure=='conflict':
        for f in fs:
            f['actors'].append(actor(3,**{'class':'static'},position=[6.,1.,0.]));f['image_evidence']['3']={'quality_pass':True}
    elif failure=='missing_lane':
        for f in fs:f['meta']['lane_id']=None;f['actors'][1]['lane_id']=None
    else:
        for f in fs:f['actors'][1]['yaw']=h+math.pi/2
    values,_=completion(fs)
    assert values.get('stable_progress_established') is not True


def test_curved_queued_lead_can_seed_but_normal_cruise_cannot():
    fs=bend()
    assert not any(s['event']=='U-E1' for s in t.seeds(fs))
    for f in fs:f['actors'][1].update(speed=0.,ego_velocity=[0.,0.])
    assert [s['actor_id'] for s in t.seeds(fs) if s['event']=='U-E1']==[2]


@pytest.mark.parametrize('points',[
    [[0.,0.],[5.,0.],[10.,0.],[5.,0.],[0.,0.]],
    [[0.,0.],[5.,0.],[10.,0.],[10.,5.],[5.,5.],[0.,5.],[0.,0.],[5.,0.],[10.,0.]],
])
def test_multiple_stations_or_directions_abstain(points):
    fs=frames()
    for f in fs:f['meta']['route']=points
    assert t.route_motion(fs[-1],fs[-1]['actors'][1]) is None
    assert completion(fs)==({},['lead_route_motion_unresolved'])


def test_endpoint_not_extended_and_changed_plan_not_used():
    fs=frames()
    fs[-1]['meta']['route']=[[0.,0.],[5.,0.]]
    assert t.route_motion(fs[-1],fs[-1]['actors'][1]) is None
    fs[-1]['meta'].update(route=[[0.,0.],[10.,0.]],changed_route=True)
    assert t.route_motion(fs[-1],fs[-1]['actors'][1]) is None


def test_straight_following_and_diagnostic_gate_agree():
    fs=frames()
    for f in fs:
        f['meta']['speed']=2.;f['actors'][1].update(position=[10.,0.,0.],ego_velocity=[2.,0.],speed=2.)
    hs=[f['actors'][1] for f in fs[-5:]];gaps=[t.route_motion(f,a)['gap'] for f,a in zip(fs[-5:],hs)]
    ev=t.following_stability_evidence(fs[-5:],hs,gaps)
    assert ev['value'] is True and all(ev['checks'].values())
    assert completion(fs)[0]['stable_progress_established'] is True
    assert t.following_stable(fs[-4:],hs,gaps) is None
    assert t.following_stable(fs[-5:],hs,[float('nan')]*5) is None


@pytest.mark.parametrize('speeds,expected', [([1.,1.2,1.5],True),([1.,2.,1.],False),([0.,.2,.4],False)])
def test_geometric_exit_does_not_replace_longitudinal_completion(speeds,expected):
    fs=frames()
    for f,v in zip(fs[-3:],speeds):f['meta']['speed']=v
    report=events.progress_stability_evidence(fs)
    assert report['value'] is expected
    assert 'not_geometric_completion' in report['scope']
    assert events.progress_stability_evidence(fs[-2:])['value'] is None


def test_active_rubric_explains_curve_and_separate_completion_conditions():
    assert 'along the road' in route_prompts.SYSTEM
    assert 'assess these separately' in route_prompts.SYSTEM
    # Frozen manual/reference rubric remains an independent historical asset.
    assert 'along the road' not in route_prompts.frozen.SYSTEM


def test_replacement_cannot_compare_unsupported_old_route_station():
    fs=frames()
    for f in fs:
        f['actors'][1]['position'][0]=25.  # navigation ends at 20 m
        f['actors'].append(actor(3,position=[15.,0.,0.],speed=5.,ego_velocity=[5.,0.]))
        f['image_evidence']['3']=dict(quality_pass=True);f['meta']['speed']=5.
    assert t.lead_retirement(fs,Episode('U-E1','e'),2) is None
