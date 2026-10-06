from copy import deepcopy
import math
import pytest
from qwen3vl_local.sft_new_loop_phase4 import privileged_geometry as g,privileged_context as c,privileged_producer as p,automatic_context as auto
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.replay_safety import assess
from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import snapshot,actor,episode,request,nav_context
from qwen3vl_local.sft_new_loop_phase4.tests.test_twenty_second_audit import scene


@pytest.mark.parametrize('cls',['static','static_prop_car','car'])
def test_zero_semantic_pixels_are_not_empty_maneuver_space(cls):
    frames=[snapshot(f,[actor(**{'class':cls},position=[15.,-3.,0.],visible_pixels=0)]) for f in (8,9,10)]
    ep=episode();r=g.condition_proposal(frames,ep,'enter',geometry_context=nav_context())
    assert r['target']!='YES'
    if cls!='car':assert r['target']=='NO' and g.visible(frames[-1]['actors'][1],frames[-1]) is True
    else:assert r['target']=='UNKNOWN'
    for f in frames:f['image_evidence']['2']['quality_pass']=False
    assert g.condition_proposal(frames,ep,'enter',geometry_context=nav_context())['target']=='UNKNOWN'


@pytest.mark.parametrize('event',['U-E1','U-E3'])
def test_visible_lead_can_remain_in_corridor_when_following_gap_allows_progress(event):
    ep=Episode(event,'scene',state='YIELD',longitudinal='HOLD')
    frames=[snapshot(f,[actor()]) for f in (8,9,10)]
    context=scene(ep,corridor_bounds=[0,30,-1.75,1.75],event_actor_ids=[2])
    r=g.condition_proposal(frames,ep,'proceed',scene_context=context)
    assert r['target']=='YES'
    for f in frames:f['actors'].append(actor(3,**{'class':'walker'},position=[10.,0.,0.]));f['image_evidence']['3']={'quality_pass':True}
    assert g.condition_proposal(frames,ep,'proceed',scene_context=context)['target']=='NO'


def test_ten_metre_pedestrian_requires_wait_without_using_expert_stop_time():
    ep=Episode('U-E4','crossing',state='YIELD',longitudinal='APPROACH')
    frames=[snapshot(f,[actor(**{'class':'walker'},position=[10.,0.,0.])]) for f in (8,9,10)]
    r=g.condition_proposal(frames,ep,'hold',scene_context=scene(ep,corridor_bounds=[0,20,-1.75,1.75],event_actor_ids=[2]))
    assert r['target']=='YES'


def test_rear_deletion_only_blocks_inventory_clearance_not_rgb_condition():
    frames=[snapshot(8,[actor(position=[-10.,0.,0.])]),snapshot(9,[actor(position=[-10.,0.,0.])]),snapshot(10)]
    risks=g.disappearances(frames[-2],frames[-1]);assert risks and all(not r['rgb_relevant'] for r in risks)
    ep=episode()
    assert g.condition_proposal(frames,ep,'enter',geometry_context=nav_context())['target']=='YES'
    receipt,audit=assess(ep,'enter',frames,request(ep,frames))
    assert receipt is None and 'near_actor_disappearance' in audit['reasons']


@pytest.mark.parametrize('pixels,quality',[(0,True),(100,False),(100,True)])
def test_front_missing_actor_does_not_become_clear_when_visibility_fails(pixels,quality):
    a=actor(position=[10.,0.,0.],visible_pixels=pixels)
    frames=[snapshot(9,[a]),snapshot(10)];frames[0]['image_evidence']['2']['quality_pass']=quality
    assert g.disappearances(*frames)[0]['rgb_relevant']
    assert g.condition_proposal(frames,episode(),'enter',geometry_context=nav_context())['target']=='UNKNOWN'


def test_camera_bounds_near_edge_are_not_classified_as_inventory_only():
    a=actor(position=[5.,4.,0.],extent=[3.,2.,1.]);f=snapshot(9,[a])
    assert g.camera_possible(a,f) is True
    f['meta']['sensor_information']={};assert g.camera_possible(a,f) is None


def test_navigation_uses_planned_current_route_and_excludes_shifted_expert_route():
    f=snapshot(10);f['meta'].update(route=[[3.,0.],[4.,0.],[5.,0.]],changed_route=False)
    assert auto.navigation(f)['points']
    f['meta']['changed_route']=True;assert not auto.navigation(f)['points']
    f['meta']['route_original']=[[3.,1.],[4.,1.],[5.,1.]]
    assert auto.navigation(f)['source']=='route_original'
    f['meta']['future_positions']=[[99.,99.]]
    assert auto.navigation(f)['points']==f['meta']['route_original']


def test_curved_navigation_rejects_other_lane_but_near_boundary_generates_candidate():
    f=snapshot(10,[actor(position=[15.,3.5,0.],yaw=math.pi,lane_id=2)])
    f['meta'].update(route=[[3.,0.],[10.,0.],[15.,1.],[20.,3.]],changed_route=False)
    nav=auto.navigation(f)
    assert auto.route_intersection(f['actors'][1],nav,margin=.5)
    assert any(a['event']=='U-E5' for a in p.candidates([f],{}))
    f['actors'][1]['position'][1]=8.
    assert not any(a['event']=='U-E5' for a in p.candidates([f],{}))


def test_red_light_is_not_signal_malfunction_or_full_priority_proof():
    f=snapshot(10,[actor(7,**{'class':'traffic_light'},affects_ego=True,state='Red')])
    f['meta'].update(is_junction=True,junction_id=4)
    events={a['event'] for a in p.candidates([f],{})}
    assert 'U-E7' not in events and 'U-E6' not in events  # proximity/red alone is not an event
    r=auto.infer([f],[])
    assert r['priority']['stop_obligation_candidate'] is True
    assert r['priority']['priority_satisfied'] is None and r['signal_failure_established'] is None
    f['actors'][1]['state']='Green'
    assert auto.infer([f],[])['priority']['priority_satisfied'] is None


def test_negative_static_extents_allow_diagnostics_but_never_permission():
    from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import raw
    m,b=raw([actor(**{'class':'static_prop_car'},extent=[-2.,1.,1.])]);f=g.project(m,b,10)
    assert f['source_geometry_issues'] and f['actors'][1]['extent']==[2.,1.,1.]
    f.update(scenario='Scenario',route_id='Town01_Rep0_1_0_route0',sources=[],causal_sha256='synthetic')
    frames=[snapshot(9),f];ep=episode()
    assert g.condition_proposal(frames,ep,'enter',geometry_context=nav_context())['target']=='UNKNOWN'
    receipt,audit=assess(ep,'enter',frames,request(ep,frames))
    assert receipt is None and 'invalid_static_source_geometry' in audit['reasons']


def test_automatic_conditions_are_consumable_but_do_not_grant_unseen_clearance():
    frames=[snapshot(f,[actor(**{'class':'static'},position=[10.,0.,0.],visible_pixels=0)]) for f in (8,9,10)]
    for f in frames:f['meta'].update(route=[[3.,0.],[8.,0.],[13.,0.]],changed_route=False)
    ep=Episode('U-E2','block',state='YIELD',branch='in_lane_pass',longitudinal='HOLD')
    assert auto.condition_facts(frames,ep,{},g.DEFAULTS)['corridor_clear'] is False
    for f in frames:f['actors']=f['actors'][:1]
    assert auto.condition_facts(frames,ep,{},g.DEFAULTS).get('corridor_clear') is None


def test_rear_actor_teleported_into_front_view_remains_rgb_relevant():
    previous=snapshot(9,[actor(position=[-10.,0.,0.])])
    current=snapshot(10,[actor(position=[10.,0.,0.])])
    assert g.disappearances(previous,current)[0]['rgb_relevant'] is True


def test_stop_waypoint_proxy_is_not_a_physical_gap_blocker():
    frames=[snapshot(f,[actor(**{'class':'traffic_light'},position=[10.,0.,0.])]) for f in (8,9,10)]
    assert g.condition_proposal(frames,episode(),'enter',geometry_context=nav_context())['target']=='YES'
    receipt,audit=assess(episode(),'enter',frames,request(episode(),frames))
    assert receipt['clear'] is True and not audit['blocking_actor_ids']
    receipt,audit=assess(episode(),'enter',frames,request(episode(),frames,priority_verified=False))
    assert receipt is None and 'priority_verified_unconfirmed' in audit['reasons']
