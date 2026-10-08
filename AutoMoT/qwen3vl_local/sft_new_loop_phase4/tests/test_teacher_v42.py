"""Motion confounding, event identity and curved-lane regression cases."""
from copy import deepcopy
from collections import Counter
import math
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_events as t,teacher_rules as rules,teacher_replay as replay
from qwen3vl_local.sft_new_loop_phase4 import phase3_sampling as sampling
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.tests.test_ten_event_teacher import frames
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher_v32 import motion_frames
from qwen3vl_local.sft_new_loop_phase4.tests.test_fourteenth_audit_fixes import row


@pytest.mark.parametrize('event',['U-E1','U-E3','U-E4','U-E5','U-E6','U-E7','R-E5'])
def test_stale_longitudinal_wait_censored_for_all_yield_events(event):
    ep=Episode(event,'test');status={}
    assert rules.observed_resumption(frames('U-E3'),ep,status) is None
    assert rules.observed_resumption(motion_frames(),ep,status)
    assert ep.state=='YIELD'  # censoring grants no permission or completion


@pytest.mark.parametrize('event',['U-E2','R-E2','R-E3'])
def test_forward_rolling_is_not_lateral_entry(event):
    ep=Episode(event,'test',direction='LEFT',target_corridor='lane');status={}
    rules.observed_resumption(frames(event),ep,status)
    assert rules.observed_resumption(motion_frames(),ep,status) is None
    assert ep.state=='WAIT'


def test_red_queue_and_ordinary_cross_flow_are_not_ue6():
    fs=frames('U-E6',True)
    for f in fs:
        f['meta'].update(is_junction=True,current_active_scenario_type='ControlLoss')
        f['actors'][1]['yaw']=math.pi/2
    assert 'U-E6' not in {x['event'] for x in t.seeds(fs)}
    fs[-1]['meta']['current_active_scenario_type']='OppositeVehicleRunningRedLight'
    assert 'U-E6' in {x['event'] for x in t.seeds(fs)}


def test_ego_detour_into_opposing_lane_is_not_actor_invasion():
    fs=frames('U-E5',True)
    for f in fs:
        f['meta'].update(route_original=[[float(x),0.] for x in range(0,31,2)],
            route=[[float(x),-4.] for x in range(0,31,2)],changed_route=True)
        f['actors'][1]['position'][1]=-4.
    assert 'U-E5' not in {x['event'] for x in t.seeds(fs)}


def test_moving_obstacle_id_does_not_make_vehicle_static():
    fs=frames('U-E5',True)
    for f in fs:
        f['meta'].update(scenario_obstacles_ids=[2],current_active_scenario_type='InvadingTurn',scenario_actors_ids=[2])
        f['actors'][1].update(speed=6.,ego_velocity=[-6.,0.])
    assert not t.static_obstacle(fs[-1],fs[-1]['actors'][1])
    assert 'U-E2' not in {x['event'] for x in t.seeds(fs)}
    assert 'U-E5' in {x['event'] for x in t.seeds(fs)}


def test_normal_merge_without_cut_in_identity_is_not_ue3():
    fs=frames('U-E3',True)
    for f in fs:
        f['meta'].update(current_active_scenario_type='EnterActorFlow')
        f['actors'][1].update(speed=3.,yaw=.3)
        f['actors'][1]['position'][1]=4. if f['frame_id']<8 else 0.
    assert 'U-E3' not in {x['event'] for x in t.seeds(fs)}
    fs[-1]['meta']['cut_in_actors_ids']=[2]
    assert 'U-E3' in {x['event'] for x in t.seeds(fs)}


@pytest.mark.parametrize('closing,expected',[(1.,True),(5.,False)])
def test_curved_normal_following_uses_road_tangent_and_closing_ttc(closing,expected):
    fs=frames('U-E3',True);h=.35
    for f in fs:
        f['meta'].update(speed=9.,route=[[x*math.cos(h),x*math.sin(h)] for x in range(0,41,2)])
        f['actors'][1].update(position=[20*math.cos(h),20*math.sin(h),0.],yaw=h,
            speed=9.-closing,ego_velocity=[(9.-closing)*math.cos(h),(9.-closing)*math.sin(h)])
    assert rules.ordinary_lead(fs,2) is expected


def test_parallel_adjacent_lane_prediction_stays_in_its_curved_lane():
    fs=frames('U-E3',True)
    for f in fs:
        f['meta']['route']=[[float(x),0. if x<=10 else ((x-10)/10)**2] for x in range(0,41,2)]
        f['actors'][1].update(position=[7.,-4.,0.],yaw=0.,ego_velocity=[8.,0.],speed=8.,lane_id=2)
    predictions=t.lane_predictions(fs,fs[-1]['actors'][1],dict(points=fs[-1]['meta']['route'],half_width=1.75))
    assert predictions and predictions[-1]['yaw']>0
    assert all(t.route_intersection(a,dict(points=fs[-1]['meta']['route'],half_width=1.75)) is False for a in predictions)
    fs[-1]['actors'][1]['ego_velocity'][1]=2.
    assert t.lane_predictions(fs,fs[-1]['actors'][1],dict(points=fs[-1]['meta']['route'],half_width=1.75)) is None


def test_motion_cells_balance_when_supported_and_report_missing_cells():
    rows=[]
    for answer in ('YES','NO'):
        for speed in (0.,5.):
            for j in range(100):
                r=row(len(rows),'U-E3');r.update(edge='proceed',target=answer,slice='readiness',observation=dict(r['observation'],speed_mps=speed));rows.append(r)
    ids,report=sampling.plan(rows,epoch=0,budget=80)
    assert set(Counter((rows[i]['target'],sampling.motion(rows[i])) for i in ids).values())=={20}
    assert not report['missing_motion_cells']
    selected=[r for r in rows if not (r['target']=='YES' and sampling.motion(r)=='stationary')]
    _,report=sampling.plan(selected,epoch=0,budget=80)
    assert 'U-E3/proceed/YES/stationary' in report['missing_motion_cells']


def test_main_edge_weight_does_not_reward_a_one_answer_edge():
    rows=[]
    for edge in ('proceed','settle','depart'):
        for j in range(200):
            r=row(len(rows),'U-E2');r.update(edge=edge,target='NO' if edge=='depart' or j%2 else 'YES');rows.append(r)
    ids,_=sampling.plan(rows,epoch=0,budget=100)
    counts=Counter(rows[i]['edge'] for i in ids)
    assert counts=={'proceed':80,'settle':10,'depart':10}


@pytest.mark.parametrize('event,edge',[('U-E3','proceed'),('U-E7','proceed'),('U-E2','depart'),('R-E2','enter')])
def test_post_stop_yes_is_unapproved_catchup_not_readiness(monkeypatch,event,edge):
    from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import frames as sourced_frames
    source={f['frame_id']:f for f in sourced_frames()}
    for n in (11,12):
        f=deepcopy(source[10]);f['frame_id']=n;f['meta']['speed']=2.;f['meta']['ego_matrix'][0][3]=(n-10)*.5;source[n]=f
    monkeypatch.setattr(replay.g,'load_frame',lambda root,s,r,n:source[n])
    monkeypatch.setattr(replay,'instance_anomalies',lambda *args:[])
    ctx=dict(direction='LEFT',target_corridor='target') if edge!='proceed' else {}
    monkeypatch.setattr(rules,'seeds',lambda fs:[dict(event=event,actor_id=2,episode_context=ctx)])
    monkeypatch.setattr(t,'update_instance',lambda *args:None)
    monkeypatch.setattr(t,'retirement',lambda *args:None)
    monkeypatch.setattr(t,'synchronize_observed',lambda *args:None)
    def facts(fs,ep,seed):
        positive=fs[-1]['frame_id']>10
        return dict(event_restriction_observed=True,release_ready=positive,corridor_clear=True,
            priority_satisfied=True,target_corridor_known=True,entry_gap_clear=positive),[]
    monkeypatch.setattr(replay,'instance_facts',facts)
    r=dict(scenario='Scenario',route_id='r',physical_group='g',split='train',rgb_frames=list(source))
    qs=[q for r in replay.route_records(None,r) for q in r['questions'] if q['edge']==edge]
    assert any(q['rule_target']=='NO' for q in qs)
    later=[q for q in qs if q['frame_id']==11]
    assert later and all(q['slice']=='catchup' and q['rule_target']=='UNKNOWN' and q['state_rule_target']=='YES' for q in later)
