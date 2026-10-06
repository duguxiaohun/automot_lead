"""Causal scope regressions; synthetic examples are not rule approvals."""
from copy import deepcopy
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_rules as t, teacher_replay as replay, teacher_approval as approval
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import frames,question,seal
from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import actor
from qwen3vl_local.sft_new_loop_phase4.tests.test_fourteenth_audit_fixes import row
from qwen3vl_local.sft_new_loop_phase4.sampling import plan


def test_observable_stationary_start_does_not_require_full_meter_growth():
    fs=frames(release=True)
    for i,f in enumerate(fs):f['actors'][1]['position'][0]=8.+i*.2
    assert t.stationary_lead_start(fs,2)
    vals,_=t.facts(fs,Episode('U-E1','e',longitudinal='HOLD'),2)
    assert vals['release_ready'] is True
    for f in fs:f['image_evidence']['2']['crops'][0]['bounds']=[100,100,140,150]
    assert not t.stationary_lead_start(fs,2)
    assert t.facts(fs,Episode('U-E1','e'),2)[0]['release_ready'] is None


def test_subpixel_or_half_meter_stationary_start_stays_unresolved():
    fs=frames(release=True)
    for i,f in enumerate(fs):f['actors'][1]['position'][0]=8.+i*.12
    assert not t.stationary_lead_start(fs,2)
    assert t.facts(fs,Episode('U-E1','e'),2)[0]['release_ready'] is None


def test_ego_reverse_cannot_establish_lead_start():
    fs=frames(release=True)
    for i,f in enumerate(fs):f['meta']['ego_matrix'][0][3]=-i*.3
    assert not t.stationary_lead_start(fs,2)


def test_old_lead_departure_censors_scope_without_fabricated_complete():
    fs=frames()
    for f in fs[-3:]:f['actors'][1]['position']=[15.,6.,0.]
    assert t.lead_retirement(fs,Episode('U-E1','e'),2)=='lead_visibly_left_event_corridor'
    fs[-1]['image_evidence']['2']['quality_pass']=False
    assert t.lead_retirement(fs,Episode('U-E1','e'),2) is None


def test_replacement_requires_visible_ordinary_following_not_cut_in():
    fs=frames()
    for f in fs:
        f['actors'][1]['position'][0]=25.
        f['actors'].append(actor(3,position=[15.,0.,0.],speed=5.,ego_velocity=[5.,0.]))
        f['image_evidence']['3']=dict(quality_pass=True)
        f['meta']['speed']=5.
    assert t.lead_retirement(fs,Episode('U-E1','e'),2)=='lead_replaced_by_observed_moving_lead'
    fs[-1]['actors'][-1]['ego_velocity'][1]=1.
    assert t.lead_retirement(fs,Episode('U-E1','e'),2) is None


def test_normal_moving_lead_does_not_block_vru_release_but_stopped_lead_does():
    fs=frames('vru',True)
    for f in fs:
        f['actors'].append(actor(3,position=[10.,0.,0.],speed=2.,ego_velocity=[2.,0.]))
        f['image_evidence']['3']=dict(quality_pass=True)
    ep=Episode('U-E4','vru',longitudinal='HOLD')
    assert t.facts(fs,ep,2)[0]['release_ready'] is True
    fs[-1]['actors'][-1]['ego_velocity'][0]=0.
    assert t.facts(fs,ep,2)[0]['release_ready'] is False


def test_same_direction_cyclist_uses_follow_branch_and_does_not_require_stop():
    fs=frames()
    for f in fs:
        f['actors'][1].update(base_type='bicycle',speed=2.,ego_velocity=[2.,0.])
        f['meta']['speed']=2.
    seed=t.seeds(fs)[0];assert seed['branch']=='cyclist_follow'
    vals,_=t.facts(fs,Episode('U-E4','e',branch='cyclist_follow'),2)
    assert vals['stationary_wait_required'] is False
    assert vals['restricted_progress_established'] is True
    fs[-1]['actors'][1]['ego_velocity']=[0.,2.]
    assert t.seeds(fs)[0]['branch']=='default'


def test_visible_vru_deletion_still_abstains_even_if_previously_clear():
    fs=frames('vru',True);fs[-1]['actors'].pop()
    assert t.facts(fs,Episode('U-E4','e'),2)==({},['instance_rgb_discontinuity'])


@pytest.mark.parametrize('motion,target,expected',[('moving','YES','UNKNOWN'),('stationary','YES','YES'),('moving','NO','NO')])
def test_moving_yes_never_becomes_readiness_or_unverified_catchup(motion,target,expected):
    q=question(target);q.update(ego_motion=motion,ego_speed=2. if motion=='moving' else 0.);seal(q)
    a=approval.validate_registry(approval.weak_registry([q['rule_class']],'regression'))
    assert approval.target(q,a)[0]==expected
    assert q['rule_target']==target and q['slice']=='readiness'  # Raw fact stays intact.


def test_seed_restrict_not_weak_supervision():
    q=question();q['edge']='restrict';q['rule_class']=t.rule_class('U-E1','restrict',2)
    a=approval.validate_registry(approval.weak_registry([q['rule_class']],'regression'))
    assert approval.target(q,a)==('UNKNOWN','restriction_seed_tautology')


def test_single_answer_edge_is_not_weight_eight():
    rows=[]
    for edge,answers in [('proceed',('YES','NO')),('enter',('YES',))]:
        for answer in answers:
            for n in range(40):
                r=row(len(rows),'U-E1');r.update(edge=edge,target=answer);rows.append(r)
    _,a=plan(rows,epoch=0,budget=40,policy='event_weighted')
    assert a['effective_edge_weights']=={'U-E1/enter':1,'U-E1/proceed':8}
    assert a['edge_counts']['U-E1/proceed']>4*a['edge_counts']['U-E1/enter']


def test_motion_strata_balance_inside_answers_and_resume():
    rows=[]
    for answer in ('YES','NO'):
        for speed,n in [(0.,40),(3.,8)]:
            for _ in range(n):
                r=row(len(rows),'U-E1');r.update(edge='proceed',target=answer);r['observation']['speed_mps']=speed;rows.append(r)
    ids,a=plan(rows,epoch=0,budget=24,policy='event_weighted')
    assert set(a['edge_answer_motion_counts'].values())=={6}
    assert len(set(ids))==24
    direct=plan(rows,epoch=1,budget=24,policy='event_weighted')[0]
    assert plan(rows,epoch=1,budget=24,policy='event_weighted',history=a['next_history'])[0]==direct


def test_reviewed_unknown_cannot_be_overwritten_by_weak_yes():
    from qwen3vl_local.sft_new_loop_phase4.teacher_data import reconcile_weak_inputs
    manual=dict(id='m',model_input_sha256='same',target='UNKNOWN',label_basis='reviewed_transition_band')
    weak=dict(id='w',model_input_sha256='same',target='YES',label_basis='weak_rule_teacher')
    result,counts=reconcile_weak_inputs([manual,weak]);assert result==[manual]
    assert counts=={'manual_input_overlap':1}


def test_motion_balancing_does_not_make_catchup_half_the_positive_examples():
    rows=[]
    for target in ('YES','NO'):
        for phase,speed in [('readiness',0.),('catchup',3.)]:
            for n in range(40):
                r=row(len(rows),'U-E1');r.update(edge='proceed',target=target,slice=phase)
                r['observation']['speed_mps']=speed;rows.append(r)
    _,a=plan(rows,epoch=0,budget=40,policy='event_weighted')
    for answer in ('YES','NO'):
        c=a['edge_answer_phase_counts'];assert c['U-E1/proceed/'+answer+'/readiness']>=3*c['U-E1/proceed/'+answer+'/catchup']


def test_finished_previous_observation_is_not_rewritten_by_later_deletion(monkeypatch):
    fs=frames();data={f['frame_id']:f for f in fs}
    last=deepcopy(fs[-1]);last['frame_id']=11;last['actors'].pop();data[11]=last
    monkeypatch.setattr(replay.g,'load_frame',lambda root,scenario,route,n:data[n])
    class CompletedEpisode(Episode):
        def advance(self,frame,answers):
            super().advance(frame,answers)
            # Model a milestone already established in this observation.
            self.state='DONE';self.longitudinal='STABLE'
    monkeypatch.setattr(replay,'Episode',CompletedEpisode)
    route=dict(scenario=fs[0]['scenario'],route_id=fs[0]['route_id'],physical_group='synthetic',split='train',rgb_frames=list(data))
    rows=list(replay.route_records(None,route))
    assert rows[-1]['anomalies']  # Never relabel visible deletion as natural.
    assert rows[-1]['instance_ends'][0]['reason']=='finished'
    assert rows[-1]['instance_ends'][0]['completed'] is True
    assert not rows[-1]['questions']  # No post-deletion completion label.


def test_removing_motion_metadata_does_not_bypass_supervision_scope():
    q=question();q.pop('ego_speed');q.pop('ego_motion');seal(q)
    with pytest.raises(ValueError,match='motion'):replay.validate_question(q)
