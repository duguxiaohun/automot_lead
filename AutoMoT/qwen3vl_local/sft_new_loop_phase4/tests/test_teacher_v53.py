"""Native departure semantics; no mocked teacher, geometry or answers."""
from copy import deepcopy

import pytest

from qwen3vl_local.sft_new_loop_phase4 import teacher_events as t, teacher_rules as rules
from qwen3vl_local.sft_new_loop_phase4.tests.test_ten_event_teacher import frames, seed, episode


def departure(visible_exit=False):
    fs=frames('U-E5',True)
    s=dict(seed('U-E5'),restriction_frame=4,identity_world=[[float(x),0.,0.] for x in range(0,31,2)],identity_half_width=1.75)
    for f,y in zip(fs,[0.,0.,0.,0.,1.,4. if visible_exit else 3.,4.]):
        f['actors'][1]['position'][1]=y
    fs[-1]['image_evidence']['2']['quality_pass']=False
    return fs,s


def answer(fs,s):
    ep=episode('U-E5')
    facts,_=t.facts(fs,ep,s)
    return rules.answer(next(e for e in ep.questions() if e.key=='proceed').criteria,facts)


@pytest.mark.parametrize('visible_exit,expected',[(False,'UNKNOWN'),(True,'YES')])
@pytest.mark.parametrize('planner',['unchanged','shifted','missing'])
def test_historical_planner_cannot_change_frozen_instance_departure(visible_exit,expected,planner):
    fs,s=departure(visible_exit)
    for f in fs[:-1]:
        if planner=='shifted':f['meta']['route']=[[float(x),8.] for x in range(0,31,2)]
        if planner=='missing':f['meta']['route']=None
    assert answer(fs,s)==expected


@pytest.mark.parametrize('visible_exit',[False,True])
def test_reproject_each_past_ego_pose_not_current_local_corridor(visible_exit):
    fs,s=departure(visible_exit)
    # Translate past ego poses and actors equally: world geometry is unchanged.
    # Comparing those actors against today's local corridor would falsely clear.
    for i,f in enumerate(fs[:-1]):
        shift=5.+i*.1
        f['meta']['ego_matrix'][1][3]=shift
        f['actors'][1]['position'][1]-=shift
    nav=t.instance_navigation(fs[-1],s)
    assert t.visible_departure(fs,s,nav) is visible_exit


@pytest.mark.parametrize('failure',['missing_seed','missing_past_pose','hidden_exit','missing_actor'])
def test_missing_or_occluded_departure_does_not_authorize_release(failure):
    fs,s=departure(True)
    if failure=='missing_seed':s.pop('identity_world')
    if failure=='missing_past_pose':
        for f in fs[:-1]:f['meta']['ego_matrix']=None
    if failure=='hidden_exit':fs[-2]['image_evidence']['2']['quality_pass']=False
    if failure=='missing_actor':fs[-1]['actors']=fs[-1]['actors'][:1]
    assert t.visible_departure(fs,s,t.navigation(fs[-1])) is False
    if failure!='missing_seed':assert answer(fs,s)!='YES'


def test_visible_departure_does_not_remove_other_traffic_veto():
    fs,s=departure(True)
    for f in fs:
        other=deepcopy(f['actors'][1]);other['id']=3;other['position']=[12.,0.,0.]
        f['actors'].append(other)
        f['image_evidence']['3']=deepcopy(f['image_evidence']['2'])
        f['image_evidence']['3']['quality_pass']=True
    assert t.visible_departure(fs,s,t.instance_navigation(fs[-1],s)) is True
    assert answer(fs,s)=='NO'


def test_visible_departure_does_not_remove_priority_veto():
    fs,s=departure(True)
    fs[-1]['meta']['stop_sign_hazard']=True
    assert answer(fs,s)=='NO'
