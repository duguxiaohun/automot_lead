"""Stale replay state must not label a later stop; no real approvals in fixtures."""
from copy import deepcopy
import math
import lzma
import pickle
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_rules as t,teacher_replay as replay,route_quality as quality
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import frames
from qwen3vl_local.sft_new_loop_phase4.preflight import stationary_readiness_support
from qwen3vl_local.sft_new_loop_phase4 import route_prompts as prompts


def motion_frames():
    fs=frames()
    for i,f in enumerate(fs):
        f['frame_id']=20+i;f['meta']['speed']=4.;f['meta']['ego_matrix'][0][3]=i
    return fs


@pytest.mark.parametrize('case',['normal','curve','reverse','sideways','jump','gap','no_stop','initial_moving','already_proceeding'])
def test_observed_execution_censors_only_proven_resumption(case):
    ep=Episode('U-E1','e',longitudinal='HOLD');status={}
    if case not in ('no_stop','initial_moving'):assert t.observed_resumption(frames(),ep,status) is None
    fs=motion_frames()
    if case=='curve':
        for i,f in enumerate(fs):
            angle=i*.03;m=f['meta']['ego_matrix'];m[0][0]=m[1][1]=math.cos(angle);m[0][1]=-math.sin(angle);m[1][0]=math.sin(angle);m[1][3]=i*i*.015
    if case=='reverse':
        for f in fs:f['meta']['ego_matrix'][0][3]*=-1
    if case=='sideways':
        for i,f in enumerate(fs):f['meta']['ego_matrix'][0][3]=0;f['meta']['ego_matrix'][1][3]=i
    if case=='jump':fs[-1]['meta']['ego_matrix'][0][3]=100
    if case=='gap':fs[-2]['frame_id']=99
    if case=='already_proceeding':ep.state='PROCEED'
    proof=t.observed_resumption(fs,ep,status)
    assert bool(proof)==(case in ('normal','curve'))
    assert ep.longitudinal=='HOLD'
    if proof:assert proof['motion_observations'][-1]['frame_id']==26


def test_replay_censors_before_later_stop_without_inventing_yes(monkeypatch):
    source={f['frame_id']:f for f in frames()}
    for n in range(11,30):
        f=deepcopy(source[10]);f['frame_id']=n
        f['meta']['speed']=4. if n<23 else 0.
        f['meta']['ego_matrix'][0][3]=min(n-10,12)
        source[n]=f
    monkeypatch.setattr(replay.g,'load_frame',lambda root,s,r,n:source[n])
    monkeypatch.setattr(t,'scoped_anomalies',lambda *a:[])
    monkeypatch.setattr(t,'lead_retirement',lambda *a:None)
    monkeypatch.setattr(t,'seeds',lambda fs:[dict(event='U-E1',actor_id=2)])
    monkeypatch.setattr(t,'facts',lambda fs,ep,ident:(dict(event_restriction_observed=True,release_ready=fs[-1]['frame_id']>=23,corridor_clear=True,priority_satisfied=True,stationary_wait_required=True),[]))
    route=dict(scenario='Scenario',route_id='r',physical_group='g',split='train',rgb_frames=list(source))
    rows=list(replay.route_records(None,route))
    ends=[e for row in rows for e in row['instance_ends']]
    assert any(e['reason']=='observed_resumption_censors_stale_wait' and e['completed'] is False for e in ends)
    assert not [q for row in rows if row['frame_id']>=15 for q in row['questions']]


def test_far_visible_lead_ends_scope_not_complete():
    fs=frames()
    for f in fs:f['actors'][1]['position'][0]=40.
    assert t.lead_retirement(fs,Episode('U-E1','e'),2)=='lead_visibly_beyond_event_horizon'
    fs[-1]['image_evidence']['2']['quality_pass']=False
    assert t.lead_retirement(fs,Episode('U-E1','e'),2) is None


def write_meta(tmp_path,n,speed=0.,x=0.,pose=True):
    p=tmp_path/'metas'/f'{n:04d}.pkl';p.parent.mkdir(exist_ok=True)
    m=frames()[0]['meta'];m['speed']=speed;m['ego_matrix'][0][3]=x
    if not pose:m.pop('ego_matrix')
    with lzma.open(p,'wb') as f:pickle.dump(m,f)


@pytest.mark.parametrize('case',['still','moves','pose_moves','missing','no_pose','short','gap'])
def test_whole_route_stall_needs_complete_speed_and_pose_evidence(tmp_path,monkeypatch,case):
    monkeypatch.setattr(quality,'MIN_FRAMES',5)
    nums=list(range(6))
    for n in nums:write_meta(tmp_path,n,speed=1. if n==5 and case=='moves' else 0.,x=1. if n==5 and case=='pose_moves' else 0.,pose=case!='no_pose')
    metas=nums[:-1] if case=='missing' else nums
    if case=='short':nums=nums[:3]
    if case=='gap':nums=[0,1,2,4,5]
    record=quality.stationarity_check(tmp_path,nums,metas)
    quality.validate_stationarity(record,nums)
    assert (record['status']=='whole_route_stationary')==(case=='still')
    if case=='still':
        record['observations'][-1]['speed']=3.
        with pytest.raises(ValueError):quality.validate_stationarity(record,nums)


def test_stationary_warning_does_not_count_moving_catchup_or_unknown():
    def row(speed,phase='readiness'):
        return dict(episode=dict(event='U-E1'),edge='proceed',target='YES',physical_group='g',observation=dict(speed_mps=speed),slice=phase)
    result=stationary_readiness_support([row(4.),row(0.,'catchup'),row(float('nan'))])
    assert result['blocking'] is False and result['missing_cells']['U-E1/proceed']==1
    assert result['cells']['U-E1/proceed']['moving_yes']==1
    assert 'R-E5/proceed' in result['missing_cells']
    assert 'U-E1/proceed' not in stationary_readiness_support([row(0.)])['missing_cells']


def test_following_branch_actual_messages_use_following_not_crossing():
    ep=Episode('U-E4','e',branch='cyclist_follow',longitudinal='HOLD')
    obs=dict(frame_id=10,history_frames=[6,10],speed_mps=0.)
    msg=prompts.messages(ep,'proceed',obs,['a','b'])[-1]['content'][-1]['text']
    assert 'cyclist may remain ahead' in msg and 'does not authorize overtaking' in msg
    assert prompts.prompt_version(ep,'proceed')=='phase4_state_pair_cyclist_follow_v9'
    plain=Episode('U-E4','e',longitudinal='HOLD')
    assert prompts.state_pair(plain,'proceed')==prompts.frozen.state_pair(plain,'proceed')
