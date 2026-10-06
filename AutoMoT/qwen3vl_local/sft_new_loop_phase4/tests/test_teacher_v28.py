"""Regression tests for local event scope; synthetic cases are not approvals."""
from copy import deepcopy
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_rules as t,teacher_replay as r,teacher_pool as pool,teacher_approval as approval
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import frames
from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import actor
from qwen3vl_local.sft_new_loop_phase4.controller import Episode


@pytest.mark.parametrize('position,affected',[([10.,6.,0.],False),([10.,0.,0.],True),([-10.,0.,0.],False)])
def test_disappearance_is_local_to_instance(position,affected):
    fs=frames('vru',True)
    for f in fs[:-1]:
        f['actors'].append(actor(3,position=position));f['image_evidence']['3']=dict(quality_pass=True)
    issues=t.scoped_anomalies(fs,2)
    assert bool(issues)==affected
    values,reasons=t.facts(fs,Episode('U-E4','vru',longitudinal='HOLD'),2)
    assert ('instance_rgb_discontinuity' in reasons)==affected
    if not affected:assert values['release_ready'] is True


def test_same_id_jump_into_local_region_is_not_ignored():
    fs=frames()
    for f in fs:
        f['actors'].append(actor(3,position=[10.,20.,0.]));f['image_evidence']['3']=dict(quality_pass=True)
    fs[-1]['actors'][-1]['position']=[10.,0.,0.]
    assert t.scoped_anomalies(fs,2)


def test_decisive_disappearance_is_never_clearance():
    fs=frames('vru',True);fs[-1]['actors'].pop()
    assert t.scoped_anomalies(fs,2)
    assert t.facts(fs,Episode('U-E4','vru'),2)[0]=={}


def test_clean_history_reacquires_a_new_instance_after_jump(monkeypatch):
    base=frames()[-1];data={}
    for n in range(4,23):
        f=deepcopy(base);f['frame_id']=n
        if n==11:f['actors'][1]['position'][0]=40.
        data[n]=f
    monkeypatch.setattr(r.g,'load_frame',lambda root,scenario,route,n:data[n])
    route=dict(scenario=base['scenario'],route_id=base['route_id'],physical_group='synthetic',split='train',rgb_frames=list(data))
    rows=list(r.route_records(None,route))
    assert any(e['reason']=='instance_rgb_discontinuity' for row in rows for e in row['instance_ends'])
    ids={q['episode']['instance_id'] for row in rows for q in row['questions']}
    assert len(ids)==2
    assert not any(row['questions'] for row in rows if 11<=row['frame_id']<=17)
    assert any(row['questions'] for row in rows if row['frame_id']==18)


def test_vru_twenty_meters_can_seed_but_darkness_cannot():
    fs=frames('vru')
    for f in fs:
        f['actors'][1]['position']=[20.,0.,0.]
        f['meta']['route'] += [[25.,0.],[30.,0.]]
    assert t.seeds(fs)[0]['event']=='U-E4'
    fs[-1]['image_evidence']['2']['quality_pass']=False
    assert t.seeds(fs)==[]


def test_braking_moving_lead_can_seed_but_cruise_cannot():
    fs=frames()
    for i,f in enumerate(fs):
        f['meta']['speed']=10.;f['actors'][1]['speed']=10.-i*.5
    assert t.seeds(fs)[0]['event']=='U-E1'
    for f in fs:f['actors'][1]['speed']=7.
    assert t.seeds(fs)==[]


def test_follow_recovery_accepts_sustained_opening_short_headway():
    fs=frames()
    for i,f in enumerate(fs):
        f['meta']['speed']=10.;f['actors'][1].update(position=[9.+i*.1,0.,0.],speed=11.,ego_velocity=[11.,0.])
    values,_=t.facts(fs,Episode('U-E1','e',state='PROCEED',longitudinal='RECOVER'),2)
    assert values['restricted_progress_established'] is True
    assert values['stable_progress_established'] is False
    values,_=t.facts(fs,Episode('U-E1','e',state='PROCEED',longitudinal='FOLLOW'),2)
    assert values['event_resolved'] and values['stable_progress_established']


@pytest.mark.parametrize('mode',['closing','oscillating','short','missing_velocity'])
def test_follow_recovery_rejects_unsafe_or_unresolved_track(mode):
    fs=frames();hs=[f['actors'][1] for f in fs[-5:]]
    for f in fs:f['meta']['speed']=10.
    for i,a in enumerate(hs):a.update(speed=11.,ego_velocity=[11.,0.])
    gaps=[6.]*5
    if mode=='closing':
        for a in hs:a['ego_velocity']=[8.,0.]
    if mode=='oscillating':hs[-1]['ego_velocity']=[12.8,0.]
    if mode=='short':gaps=[3.]*5
    if mode=='missing_velocity':hs[-1].pop('ego_velocity')
    assert t.following_stable(fs[-5:],hs,gaps) is not True


@pytest.mark.parametrize('case,expected',[('local_lane',True),('red',False),('stop',False),('missing_lane',None),('unknown_hazard',None),('unmatched_light',None)])
def test_near_junction_local_controls(case,expected):
    f=frames()[-1];f['meta']['distance_to_next_junction']=5.
    if case=='red':f['meta']['light_hazard']=True
    if case=='stop':f['meta']['stop_sign_hazard']=True
    if case=='missing_lane':f['meta'].pop('lane_id')
    if case=='unknown_hazard':f['meta'].pop('light_hazard')
    if case=='unmatched_light':f['actors'].append(actor(3,**{'class':'traffic_light'},state='Green'))
    assert t.local_priority(f,{'review_horizon_m':18.}) is expected


def test_instance_pool_filters_physical_identity_and_reserved_ownership():
    routes=[dict(scenario='S',route_id=str(i),physical_group=g,split=split) for i,(g,split) in enumerate([('a','val'),('a','val'),('b','train'),('c','test'),('d','val')])]
    assert {r['physical_group'] for r in pool.eligible_routes({'routes':routes},{'d'})}=={'a','c'}


def test_discovery_never_calls_condition_rules(monkeypatch):
    fs=frames();by={f['frame_id']:f for f in fs}
    monkeypatch.setattr(pool.g,'load_frame',lambda root,scenario,route,n:by[n])
    monkeypatch.setattr(t,'facts',lambda *a:pytest.fail('answer leakage'))
    route=dict(scenario='S',route_id='R',physical_group='G',split='val',rgb_frames=list(by))
    result=pool.discover((None,route))
    assert set(result['instances'])=={'U-E1'}
    assert 'rule_target' not in str(result)


def test_wilson_budget_reports_best_case_per_answer():
    budget=approval.budget_requirements()
    assert budget['answers']['YES']['min_all_correct_judgments']==73
    assert budget['answers']['NO']['min_all_correct_judgments']==35
    for answer in ('YES','NO'):
        n=budget['answers'][answer]['min_all_correct_judgments'];threshold=approval.CRITERIA['min_'+answer.lower()+'_precision_lower']
        assert approval.wilson_lower(n-1,n-1)<threshold<=approval.wilson_lower(n,n)


def test_ego_translation_does_not_erase_observed_vru_outward_motion():
    fs=frames('vru',True)
    for i,f in enumerate(fs):
        f['meta']['ego_matrix'][0][3]=i*.4
        f['actors'][1]['position'][0]-=i*.4
        f['meta']['speed']=1.6
    values,_=t.facts(fs,Episode('U-E4','vru',longitudinal='HOLD'),2)
    assert values['release_ready'] is True


def test_replay_does_not_resend_a_receipt_after_unknown_frame(monkeypatch):
    base=frames()[-1];data={}
    for n in range(4,14):
        f=deepcopy(base);f['frame_id']=n;data[n]=f
    monkeypatch.setattr(r.g,'load_frame',lambda root,scenario,route,n:data[n])
    monkeypatch.setattr(t,'execution_motion',lambda frames:True)
    def facts(frames,ep,ident):
        if frames[-1]['frame_id']==10:return dict(event_restriction_observed=True,release_ready=False,corridor_clear=False,priority_satisfied=True,restriction_present=True,stationary_wait_required=True),[]
        if frames[-1]['frame_id']==11:return dict(release_ready=True,corridor_clear=True,priority_satisfied=True,restriction_present=False,stationary_wait_required=False),[]
        return {},['unresolved']
    monkeypatch.setattr(t,'facts',facts)
    route=dict(scenario=base['scenario'],route_id=base['route_id'],physical_group='synthetic',split='train',rgb_frames=list(data))
    rows=list(r.route_records(None,route))
    assert rows[-3]['traces'][0]['execution_receipt']
    assert rows[-2]['traces'][0]['execution_receipt'] is None
    assert rows[-2]['traces'][0]['accepted']==[]
    moving_yes=[q for q in rows[-3]['questions'] if q['edge']=='proceed']
    assert moving_yes and all(q['slice']=='readiness' and q['rule_target']=='YES' for q in moving_yes)


@pytest.mark.parametrize('malformed',[0,1,'False',None])
def test_local_control_ownership_is_strictly_boolean(malformed):
    f=frames()[-1]
    f['actors'].append(actor(3,**{'class':'traffic_light'},affects_ego=malformed,state='Green'))
    assert t.local_priority(f,{'review_horizon_m':18.}) is None


def test_red_light_still_requires_hold_after_participant_releases():
    fs=frames('vru',True);fs[-1]['meta']['light_hazard']=True
    values,_=t.facts(fs,Episode('U-E4','e',longitudinal='RECOVER'),2)
    assert values['corridor_clear'] is True
    assert values['stationary_wait_required'] is True
    assert values['priority_satisfied'] is False


def test_event_equal_repetition_is_reported_separately_from_unique_coverage():
    from qwen3vl_local.sft_new_loop_phase4.preflight import sampling_repetition
    rows=[dict(episode=dict(event=e)) for e in ('U-E1','U-E1','U-E4')]
    report=sampling_repetition(rows,[0,1,2,2])
    assert report['U-E1']['repeated_presentations']==0
    assert report['U-E4']['distinct_questions']==1
    assert report['U-E4']['max_question_repeat']==2
    assert report['U-E4']['presentations_per_pool_question']==2


def test_frozen_pool_extension_has_instance_evidence_without_answer_screening():
    import json
    from qwen3vl_local.sft_new_loop_phase4.identity import ROOT,digest
    from qwen3vl_local.sft_new_loop_phase4 import dataset
    extension=json.loads((ROOT/'teacher_independent_pool_v1_20261004.json').read_text())
    assert extension['sha256']==digest({k:v for k,v in extension.items() if k!='sha256'})
    assert extension['teacher']['version']=='ue1_ue4_causal_teacher_v3'  # Historical answer-blind reservation evidence.
    assert extension['event_routes']=={'U-E1':40,'U-E4':40}
    assert len({r['physical_group'] for r in extension['reservations']})==len(extension['reservations'])
    owners=dataset.holdout_reservations()  # Includes retired duration exclusions; ownership never changes.
    for row in extension['reservations']:
        assert owners[row['physical_group']]==row['split']
        assert set(row['reserved_events'])<=set(row['instances'])
        assert 'rule_target' not in json.dumps(row)
