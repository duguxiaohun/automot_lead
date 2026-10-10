from copy import deepcopy
import math
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_events as t,teacher_rules as rules,teacher_replay as replay
from qwen3vl_local.sft_new_loop_phase4 import privileged_geometry as g,teacher_approval as approval
from qwen3vl_local.sft_new_loop_phase4.full_pipeline import default_registry
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import snapshot,actor
from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS


def frames(event,blocked=False):
    out=[]
    for n in range(4,11):
        y=0. if blocked else 4.
        a=actor(2,position=[10.,y,0.],speed=0.,ego_velocity=[0.,0.])
        if event=='U-E5':a.update(yaw=math.pi)
        if event=='U-E2':a.update(**{'class':'static'})
        f=snapshot(n,[a]);f['meta'].update(speed=0.,distance_to_next_junction=math.inf,changed_route=False,
            route=[[float(x),0.] for x in range(0,31,2)])
        f['scene_visibility']={'quality_pass':True};out.append(f)
    return out


def seed(event):
    s=dict(event=event,actor_id=2,actor_ids=[2],instance_sources=[])
    if event in ('U-E2','R-E2','R-E3'):
        s.update(target_world=[[float(x),-4.,0.] for x in range(0,31,2)],original_world=[[float(x),0.,0.] for x in range(0,31,2)],
                 direction='LEFT',half_width=1.75)
    if event in ('U-E6','U-E7','R-E5'):
        s.update(junction_world=[10.,0.,0.],junction_seen=False,failure_established=event=='U-E7',negotiated_stop_sources=[{'frame_id':4}])
    return s


def episode(event):
    return Episode(event,'test',**(dict(direction='LEFT',target_corridor='target',return_direction='RIGHT',return_corridor='original')
                                  if event in ('U-E2','R-E2','R-E3') else {}))


@pytest.mark.parametrize('event',t.EVENTS)
@pytest.mark.parametrize('blocked',[False,True])
def test_each_added_event_has_evidence_based_yes_and_no(event,blocked):
    fs=frames(event,blocked);s=seed(event);ep=episode(event)
    if event in ('U-E2','R-E2','R-E3'):
        for f in fs:
            f['actors'][1]['position']=[10.,-4. if blocked else 0.,0.]
    facts,_=t.facts(fs,ep,s)
    edge=next(e for e in ep.questions() if e.key in ('proceed','enter','depart'))
    assert rules.answer(edge.criteria,facts)==('NO' if blocked else 'YES')


@pytest.mark.parametrize('event',t.EVENTS)
def test_dark_scene_cannot_create_release_or_entry_yes(event):
    fs=frames(event);ep=episode(event);s=seed(event)
    for f in fs:f['scene_visibility']['quality_pass']=False
    facts,_=t.facts(fs,ep,s)
    edge=next(e for e in ep.questions() if e.key in ('proceed','enter','depart'))
    assert rules.answer(edge.criteria,facts)!='YES'


def test_red_signal_is_not_automatically_a_failure():
    fs=frames('U-E6',True)
    for f in fs:
        f['meta'].update(is_junction=True,distance_to_next_junction=0.,traffic_light_state='Red')
        f['actors'].append(actor(9,**{'class':'traffic_light'},affects_ego=True,state='Red'))
    assert 'U-E7' not in {s['event'] for s in t.seeds(fs)}
    fs[-1]['meta']['current_active_scenario_type']='CrossJunctionDefectTrafficLight'
    assert 'U-E7' in {s['event'] for s in t.seeds(fs)}


def test_full_registry_covers_ten_events_and_lateral_edges():
    registry=default_registry();checked=approval.validate_registry(registry)
    assert {c.split('/')[1] for c in registry['rule_classes']}==set(EVENTS)
    for event,edge in [('U-E2','depart'),('U-E2','return'),('R-E2','enter'),('R-E3','enter'),('U-E7','proceed')]:
        assert rules.rule_class(event,edge,2) in checked['weak_classes']
    assert not checked['approved'] and registry['train_only']


def test_navigation_target_does_not_bypass_runtime_safety():
    fs=frames('R-E2');ep=episode('R-E2');s=seed('R-E2');facts,_=t.facts(fs,ep,s)
    answers={e.key:rules.answer(e.criteria,facts) for e in ep.questions()}
    assert answers['enter']=='YES'
    ep.advance(10,answers)
    assert ep.state=='WAIT' and ep.uncertain
    assert 'maneuver_safety_unconfirmed' in ep.wait_reason
    assert t.synchronize_observed(fs,ep,s) is None


def test_observed_entry_sync_is_not_a_permission_or_supervised_answer():
    fs=frames('R-E2');ep=episode('R-E2');s=seed('R-E2')
    for f in fs:f['meta']['ego_matrix'][1][3]=-4.
    sync=t.synchronize_observed(fs,ep,s)
    assert ep.state=='CROSS' and sync['scope']=='offline_observed_state_only_not_permission_or_label'
    assert not ep.history


def test_decisive_disappearance_never_becomes_clearance():
    fs=frames('U-E5',True);fs[-1]['actors']=fs[-1]['actors'][:1]
    facts,reasons=t.facts(fs,episode('U-E5'),seed('U-E5'))
    assert facts=={} and 'instance_rgb_discontinuity' in reasons


def test_normal_road_bend_without_lane_command_does_not_seed_re2():
    fs=frames('R-E2')
    for f in fs:f['meta']['route']=[[float(x),float(x)/4] for x in range(0,31,2)]
    assert 'R-E2' not in {s['event'] for s in t.seeds(fs)}
    fs[-1]['meta']['next_commands']=[6,4]
    assert 'R-E2' in {s['event'] for s in t.seeds(fs)}

@pytest.mark.parametrize('event',t.EVENTS)
def test_automatic_instance_extraction_for_added_events(event):
    fs=frames(event,True)
    for f in fs:
        if event=='U-E2':
            f['meta'].update(scenario_obstacles_ids=[2],route_original=deepcopy(f['meta']['route']),
                             route=[[float(x),-4.] for x in range(0,31,2)])
        elif event=='U-E3':
            f['actors'][1]['yaw']=.3
            f['actors'][1]['speed']=2.
            f['actors'][1]['position'][1]=4. if f['frame_id']<8 else 0.
            # Exported lane follows the actor; an unchanged lane vetoes generic U-E3 (v44).
            f['actors'][1]['lane_id']=2 if f['frame_id']<8 else 1
        elif event in ('U-E6','U-E7','R-E5'):
            f['meta'].update(is_junction=True,distance_to_next_junction=0.)
            f['actors'][1]['yaw']=math.pi/2
            if event=='U-E6':f['meta']['current_active_scenario_type']='OppositeVehicleRunningRedLight'
            if event=='U-E7':f['meta']['current_active_scenario_type']='CrossJunctionDefectTrafficLight'
            if event=='R-E5':f['meta']['current_active_scenario_type']='NonSignalizedJunctionLeftTurn'
        elif event in ('R-E2','R-E3'):
            f['meta'].update(next_commands=[5,4],route=[[float(x),-4.] for x in range(0,31,2)])
            if event=='R-E3':f['meta']['current_active_scenario_type']='HighwayExit'
    assert event in {s['event'] for s in t.seeds(fs)}


def test_explicit_merge_does_not_double_as_generic_lane_change():
    fs=frames('R-E3')
    for f in fs:f['meta'].update(current_active_scenario_type='HighwayExit',next_commands=[5,4],
                                route=[[float(x),-4.] for x in range(0,31,2)])
    assert {s['event'] for s in t.seeds(fs)}=={'R-E3'}


def test_local_corridor_is_bounded_by_arc_length_on_bends():
    nav=t.bounded(dict(points=[[0.,0.],[10.,0.],[10.,20.],[10.,50.]],half_width=1.75))
    assert nav['points']==[[0.,0.],[10.,0.],[10.,12.]]
    assert not t.route_intersection(actor(3,position=[10.,40.,0.]),nav)


def test_obstacle_return_can_be_positive_after_continuous_passage():
    fs=frames('U-E2');s=seed('U-E2');s['obstacle_group_visible']=True
    ep=episode('U-E2');ep.state='PASS';ep.longitudinal='RECOVER'
    for f in fs:
        f['meta']['ego_matrix'][1][3]=-4.
        f['actors'][1]['position']=[-8.,4.,0.]
    facts,_=t.facts(fs,ep,s)
    edge=next(e for e in ep.questions() if e.key=='return')
    assert rules.answer(edge.criteria,facts)=='YES'
    fs[-1]['actors'].pop()
    facts,_=t.facts(fs,ep,s)
    assert rules.answer(edge.criteria,facts)!='YES'


def test_failed_signal_does_not_cancel_stop_sign_obligation():
    fs=frames('U-E7');s=seed('U-E7')
    for f in fs:
        f['meta']['stop_sign_hazard']=True
        f['actors'].append(actor(9,**{'class':'stop_sign'},affects_ego=True))
    facts,_=t.facts(fs,episode('U-E7'),s)
    assert facts['priority_satisfied'] is False


def test_stale_executed_junction_wait_is_censored_without_a_label():
    fs=frames('U-E7');s=seed('U-E7');s['junction_seen']=True
    fs[-1]['meta']['ego_matrix'][0][3]=30.
    ep=episode('U-E7')
    assert t.retirement(fs,ep,s)=='observed_junction_exit_censors_stale_wait'
    assert ep.state=='YIELD' and not ep.history


def test_return_target_exhaustion_does_not_invent_no_return_branch():
    fs=frames('U-E2');s=seed('U-E2');ep=episode('U-E2');ep.state='PASS'
    fs[-1]['meta']['ego_matrix'][0][3]=60.
    assert t.retirement(fs,ep,s)=='frozen_navigation_exhausted_not_completion'
    assert ep.return_required and ep.state=='PASS'


def test_replay_rearms_navigation_actor_after_clear_visible_interval():
    fs=frames('R-E2');key=('R-E2',fs[-1]['actors'][0]['id'])
    seen={key};completed={key:dict(clear_observations=0,last_frame=6)}
    for f in fs[3:6]:replay.rearm_completed(completed,seen,[],f)
    assert key not in seen and not completed


def test_junction_normal_following_gap_is_not_an_occupancy_blocker(monkeypatch):
    fs=frames('U-E6',True);s=seed('U-E6')
    monkeypatch.setattr(rules,'ordinary_lead',lambda history,ident:ident==2)
    facts,_=t.facts(fs,episode('U-E6'),s)
    assert facts['corridor_clear'] is True
    assert facts['restricted_progress_established'] is True


def test_meta_lane_ownership_does_not_require_absent_ego_bbox_duplicate():
    f=frames('U-E6')[-1];f['meta'].update(is_junction=True,distance_to_next_junction=0.)
    f['actors'][0].pop('road_id',None);f['actors'][0].pop('lane_id',None)
    nav=t.bounded(t.navigation(f))
    assert t.control_priority(f,nav) is True
    f['actors'][0]['lane_id']=f['meta']['lane_id']+1
    assert t.control_priority(f,nav) is None
    f['meta']['light_hazard']=True;f['actors'][0].pop('lane_id')
    assert t.control_priority(f,nav) is False


def test_static_scenario_car_and_planner_shift_are_not_cutin():
    fs=frames('U-E3',True)
    for f in fs:
        f['actors'][1]['speed']=2.;f['actors'][1]['yaw']=.3
        f['actors'][1]['position'][1]=4.
    fs[-1]['meta']['route']=[[float(x),4.] for x in range(0,31,2)]
    assert 'U-E3' not in {s['event'] for s in t.seeds(fs)}
    for f in fs:f['meta']['scenario_obstacles_ids']=[2]
    assert 'U-E3' not in {s['event'] for s in t.seeds(fs)}
