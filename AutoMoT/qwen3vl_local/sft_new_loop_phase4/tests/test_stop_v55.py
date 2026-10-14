"""STOP evidence lifecycle regressions, not visual or legal certifications."""
from copy import deepcopy
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_controls as c, teacher_rules as r
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher_v29 import controlled_frames


def observe(fs):
    tracker=c.StopDutyTracker()
    for f in fs:f['stop_duties']=tracker.observe(f)
    return tracker


def clean(fs):
    for f in fs:
        f['actors']=[a for a in f['actors'] if a['class']!='walker']
    return fs


@pytest.mark.parametrize('copies', ['missing','matching','conflicting','invalid_meta','bool_copy'])
def test_meta_lane_ownership_optional_bbox_copy(copies):
    fs=controlled_frames()
    for f in fs:
        ego=f['actors'][0]
        if copies=='missing':
            ego.pop('road_id',None);ego.pop('lane_id',None)
        elif copies=='conflicting':ego['road_id']=999
        elif copies=='invalid_meta':f['meta']['lane_id']=None
        elif copies=='bool_copy':ego['road_id']=True
    observe(fs)
    assert fs[-1]['stop_duties']['50']['satisfied'] is (copies in ('missing','matching'))


@pytest.mark.parametrize('loss', ['flag','actor','gap','geometry','lane','identity'])
def test_pending_duty_never_disappears_into_permission(loss):
    fs=clean(controlled_frames(position=0.)) # invalid negative front gap
    tracker=observe(fs[:2])
    f=fs[2];stop=f['actors'][-1];stop['affects_ego']=False
    if loss=='actor':f['actors'].pop()
    elif loss=='gap':f['frame_id']+=3
    elif loss=='geometry':f['source_geometry_issues']=['invalid']
    elif loss=='lane':f['meta']['lane_id']=999
    elif loss=='identity':stop['position'][0]+=100
    f['stop_duties']=tracker.observe(f)
    assert not f['stop_duties']['50']['satisfied']
    assert r.local_priority(f,r.local_navigation(f,18.)) is None


def test_cleared_flag_still_allows_real_three_frame_stop_evidence():
    fs=clean(controlled_frames())
    for f in fs[1:]:f['actors'][-1]['affects_ego']=False
    observe(fs)
    assert fs[2]['stop_duties']['50']['satisfied']
    assert r.local_priority(fs[2],r.local_navigation(fs[2],18.)) is True
    fs[2]['meta']['light_hazard']=True
    assert r.local_priority(fs[2],r.local_navigation(fs[2],18.)) is False


def test_gap_breaks_stationary_observation_count_but_keeps_debt():
    fs=clean(controlled_frames());fs[2]['frame_id']=20
    tracker=observe(fs[:3])
    assert len(fs[2]['stop_duties']['50']['observations'])==1
    assert not fs[2]['stop_duties']['50']['satisfied']
    assert r.local_priority(fs[2],r.local_navigation(fs[2],18.)) is None


@pytest.mark.parametrize('fulfilled',[False,True])
def test_passing_the_entire_trigger_only_retires_fulfilled_duty(fulfilled):
    fs=clean(controlled_frames())
    if not fulfilled:
        for f in fs:f['meta']['speed']=1.
    tracker=observe(fs[:3]);f=deepcopy(fs[3])
    # Ego translates 20m; the stop remains at the same world position.
    for i in range(3):f['meta']['ego_matrix'][i][3]+=20*f['meta']['ego_matrix'][i][0]
    f['actors'][-1]['position'][0]-=20
    f['actors'][-1]['affects_ego']=False
    f['stop_duties']=tracker.observe(f)
    assert f['stop_duties']['50']['retired'] is fulfilled
    assert r.local_priority(f,r.local_navigation(f,18.)) is (True if fulfilled else None)
    if fulfilled:
        assert not tracker.entries
        again=deepcopy(fs[4]);again['stop_duties']=tracker.observe(again)
        assert not again['stop_duties']['50']['satisfied']


def test_retired_active_stop_does_not_restart_debt_behind_ego():
    fs=clean(controlled_frames());tracker=observe(fs[:3])
    for f in fs[3:]:
        for i in range(3):f['meta']['ego_matrix'][i][3]+=20*f['meta']['ego_matrix'][i][0]
        f['actors'][-1]['position'][0]-=20
        f['stop_duties']=tracker.observe(f)
        assert f['stop_duties']['50']['retired']
        assert r.local_priority(f,r.local_navigation(f,18.)) is True


def test_distant_ahead_duty_does_not_forbid_local_progress():
    fs=clean(controlled_frames(position=100.));observe(fs)
    assert not fs[-1]['stop_duties']['50']['satisfied']
    assert r.local_priority(fs[-1],r.local_navigation(fs[-1],18.)) is True


def test_unusable_source_invalidates_completion_not_debt():
    fs=clean(controlled_frames());tracker=observe(fs[:3])
    tracker.invalidate()
    f=fs[3];f['actors'][-1]['affects_ego']=False
    f['stop_duties']=tracker.observe(f)
    assert not f['stop_duties']['50']['satisfied']
    assert r.local_priority(f,r.local_navigation(f,18.)) is None


def test_missing_actor_after_satisfaction_keeps_valid_unsatisfied_receipt():
    fs=clean(controlled_frames());tracker=observe(fs[:3])
    f=fs[3];f['actors'].pop();f['stop_duties']=tracker.observe(f)
    assert f['stop_duties']['50']['observations']==[]
    assert not f['stop_duties']['50']['satisfied']
    assert r.local_priority(f,r.local_navigation(f,18.)) is None


def test_native_replay_does_not_reset_tracker_after_unreadable_source(monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4 import teacher_replay as replay
    fs=clean(controlled_frames());by_frame={f['frame_id']:f for f in fs}
    seen=[]
    original=c.StopDutyTracker.observe
    def load(root,scenario,route,n):
        if n==6:raise ValueError('unreadable source')
        return by_frame[n]
    def observe_frame(self,f):
        result=original(self,f);seen.append((f['frame_id'],result));return result
    for f in fs[3:]:f['actors'][-1]['affects_ego']=False
    monkeypatch.setattr(replay.g,'load_frame',load)
    monkeypatch.setattr(c.StopDutyTracker,'observe',observe_frame)
    route=dict(scenario=fs[0]['scenario'],route_id=fs[0]['route_id'],physical_group='fixture',
               split='train',rgb_frames=list(by_frame))
    list(replay.route_records(None,route))
    assert '50' in dict(seen)[7]
    assert not dict(seen)[7]['50']['satisfied']
