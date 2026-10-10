"""v44 RGB-audited U-E3 identity and U-E3/U-E5 stale-wait regressions."""
from qwen3vl_local.sft_new_loop_phase4 import teacher_events as t
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.tests.test_ten_event_teacher import frames


def cut_in(lateral_of_frame):
    fs=frames('U-E3',True)
    for f in fs:
        f['actors'][1].update(speed=3.,yaw=0.)
        f['actors'][1]['position'][1]=lateral_of_frame(f['frame_id'])
    return fs


def test_explicit_cut_in_actor_already_in_lane_is_an_ordinary_lead():
    # Town12 1681_0 actor 3697: lane centre since f120, re-seeded at f246.
    fs=cut_in(lambda n:0.)
    fs[-1]['meta']['cut_in_actors_ids']=[2]
    assert 'U-E3' not in {x['event'] for x in t.seeds(fs)}
    fs=cut_in(lambda n:4. if n<8 else 0.)
    fs[-1]['meta']['cut_in_actors_ids']=[2]
    assert 'U-E3' in {x['event'] for x in t.seeds(fs)}


def test_generic_cut_in_requires_actor_motion_toward_fixed_own_corridor():
    # Accident/ParkedObstacle replays: parallel lane traffic overlapped only via
    # a rotated footprint or the ego's planned detour. Lane offset stayed ~3.4 m.
    fs=cut_in(lambda n:3.2)
    old,a=fs[0]['actors'][1],fs[-1]['actors'][1]
    a['yaw']=.45
    assert t.route_intersection(a,dict(points=fs[-1]['meta']['route'],half_width=1.75)) is True
    assert t.crossed_fixed_corridor(fs[0],fs[-1],old,a) is False
    fs=cut_in(lambda n:4. if n<8 else 1.)
    assert t.crossed_fixed_corridor(fs[0],fs[-1],fs[0]['actors'][1],fs[-1]['actors'][1]) is True


def test_reference_corridor_ignores_planner_detour_onto_parallel_traffic():
    fs=cut_in(lambda n:-3.5)
    for f in fs[3:]:
        f['meta']['route']=[[float(x),-3.5] for x in range(0,31,2)]
    reference=dict(points=t.world_path(fs[0],[[float(x),0.] for x in range(0,31,2)]),half_width=1.75)
    assert t.crossed_fixed_corridor(fs[0],fs[-1],fs[0]['actors'][1],fs[-1]['actors'][1],reference) is False


def test_decisive_actor_behind_ego_censors_wait_without_completion():
    fs=frames('U-E5',True);ep=Episode('U-E5','test');ep.state='YIELD'
    s=dict(event='U-E5',actor_id=2,actor_ids=[2])
    assert t.retirement(fs,ep,s) is None
    for f in fs[-2:]:
        f['actors'][1]['position'][0]=-12.
    assert t.retirement(fs,ep,s)=='decisive_actor_passed_behind_censors_stale_wait'
    assert ep.state=='YIELD' and not ep.finished
    fs[-2]['actors'][1]['position'][0]=1.
    assert t.retirement(fs,ep,s) is None


def test_junction_release_from_rest_uses_longer_crossing_prediction():
    # Town03 002133 f20-23: cross traffic approaching the junction from the side.
    from qwen3vl_local.sft_new_loop_phase4.tests.test_ten_event_teacher import seed,episode
    import math
    def junction(speed):
        fs=frames('U-E7')
        for f in fs:
            f['meta']['speed']=speed
            f['actors'][1].update(position=[18.,-12.,0.],yaw=math.pi/2,speed=3.,ego_velocity=[0.,3.])
        return fs
    still,_=t.facts(junction(0.),episode('U-E7'),seed('U-E7'))
    moving,_=t.facts(junction(8.),episode('U-E7'),seed('U-E7'))
    # Synthetic frames lack pixel visibility for this actor: blocked or unknown, never clear.
    assert still['corridor_clear'] is not True and still['release_ready'] is not True
    assert moving['corridor_clear'] is True


def test_generic_cut_in_vetoed_when_exported_actor_lane_never_changes():
    # Accident 1819 actors 88/89: lane_id 1 throughout while the pre-obstacle
    # reference already bends toward that lane.
    fs=cut_in(lambda n:4. if n<8 else 1.)
    for f in fs:f['actors'][1].update(road_id=7,lane_id=1)
    assert not t.cut_in_identity(fs,fs[-1]['actors'][1],fs[0]['actors'][1])
    fs[-1]['actors'][1]['lane_id']=2
    assert t.cut_in_identity(fs,fs[-1]['actors'][1],fs[0]['actors'][1])
