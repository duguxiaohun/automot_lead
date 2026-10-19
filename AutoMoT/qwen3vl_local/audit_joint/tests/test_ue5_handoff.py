from copy import deepcopy
import math
import pytest
from qwen3vl_local.audit_joint.ue5_handoff import assess
from qwen3vl_local.sft_new_loop_phase4.tests.test_ten_event_teacher import frames
from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import actor
from qwen3vl_local.audit_joint.tests.test_departure_probe import frozen_seed


def sample():
    fs=frames('U-E5',True)
    for number in (11,12):
        f=deepcopy(fs[-1]);f['frame_id']=number;fs.append(f)
    for i,f in enumerate(fs):
        f['actors'][1].update(position=[-7.-i,0.,0.],ego_velocity=[-4.,0.])
        f['stop_duties']={}
    return fs,dict(frozen_seed(),restriction_frame=4)


def test_passage_routes_task_but_does_not_produce_re_or_release_label():
    fs,s=sample();before=deepcopy(fs);r=assess(fs,s)
    assert r['retire_instance'] and r['context']=='background_reclassification_candidate'
    assert r['supervised_target'] is None and not r['permission_to_move'] and not r['training_authorized']
    assert fs==before


@pytest.mark.parametrize('defect',['missing_actor','jump','gap','different_route','geometry','future_restriction','no_control_history'])
def test_insufficient_evidence_does_not_claim_background(defect):
    fs,s=sample()
    if defect=='missing_actor':fs[-1]['actors'].pop()
    if defect=='jump':fs[-1]['actors'][1]['position'][0]=-60.
    if defect=='gap':fs.pop(3)
    if defect=='different_route':fs[-1]['route_id']='other'
    if defect=='geometry':fs[-1]['source_geometry_issues']=['bad']
    if defect=='future_restriction':s['restriction_frame']=12
    if defect=='no_control_history':del fs[-1]['stop_duties']
    r=assess(fs,s)
    assert r['context']!='background_reclassification_candidate' and not r['permission_to_move']
    assert not r['retire_instance'] or defect=='no_control_history'


def test_hidden_but_still_alongside_does_not_retire_and_resumed_motion_is_not_proof():
    fs,s=sample()
    for f in fs:
        f['actors'][1]['position']=[-3.,-2.,0.]
        f['image_evidence']['2']['quality_pass']=False
        f['meta'].update(speed=5.,current_active_scenario_type=None,previous_active_scenario_type=None)
    assert not assess(fs,s)['retire_instance']


def test_whole_rotated_footprint_not_center_or_unrotated_length():
    fs,s=sample()
    for f in fs:f['actors'][1].update(position=[-4.,0.,0.],extent=[1.,3.,1.],yaw=math.pi/2)
    assert not assess(fs,s)['retire_instance']


def test_returning_actor_or_reentry_cancels_current_proposal():
    fs,s=sample();assert assess(fs,s)['retire_instance']
    fs[-1]['actors'][1]['position'][0]=fs[-2]['actors'][1]['position'][0]+.2
    assert not assess(fs,s)['retire_instance']


@pytest.mark.parametrize('blocker',['following_oncoming','red','stop','uncertain'])
def test_passed_first_car_cannot_clear_remaining_vehicle_or_control(blocker):
    fs,s=sample()
    for f in fs:
        if blocker=='following_oncoming':
            f['actors'].append(actor(3,position=[10.,0.,0.],yaw=math.pi,ego_velocity=[-3.,0.]))
            f['image_evidence']['3']={'quality_pass':True}
        elif blocker=='red':f['meta']['light_hazard']=True
        elif blocker=='stop':f['meta']['stop_sign_hazard']=True
        else:f['meta']['light_hazard']=None
    r=assess(fs,s);assert r['retire_instance'] and not r['permission_to_move']
    assert r['context']==('handoff_requires_reobservation' if blocker=='uncertain' else 'handoff_with_remaining_restriction')


def test_phase2_re_is_not_a_normal_progress_target():
    from qwen3vl_local.sft_new_loop_phase2.build_dataset import _target_class
    assert _target_class('R4',['R-E4'])=='RE'
    assert _target_class('R2',['U-E2'])=='RE'
    fs,s=sample();fs[-1]['meta']['light_hazard']=True
    assert assess(fs,s)['context']=='handoff_with_remaining_restriction'


def test_one_clear_frame_and_missing_navigation_do_not_skip_reobservation():
    fs,s=sample()
    fs[-2]['meta']['light_hazard']=True
    assert assess(fs,s)['context']=='handoff_requires_reobservation'
    fs[-1]['meta'].update(changed_route=True,light_hazard=True)
    assert assess(fs,s)['context']=='handoff_with_remaining_restriction'


def test_unknown_scene_does_not_become_clear_just_because_actor_passed():
    fs,s=sample();fs[-1]['scene_visibility']['quality_pass']=False
    assert assess(fs,s)['context']=='handoff_requires_reobservation'


def test_empty_phase4_is_unconditional_prior_not_keep_or_resume():
    from qwen3vl_local.sft_new_loop_phase4.controller import combine_priors
    assert combine_priors([])['action']=='UNCOND'


def test_other_stop_duty_survives_finished_ue5():
    from qwen3vl_local.sft_new_loop_phase4.controller import Episode, combine_priors
    passed=Episode('U-E5','passed',state='DONE',longitudinal='STABLE')
    remaining=Episode('R-E5','stop-duty',state='YIELD',longitudinal='HOLD')
    assert combine_priors([passed,remaining])['action']=='STOP'
