import pytest
from qwen3vl_local.audit_joint.transition_rechecks import screen


def row(frame,edge,target='YES',instance='a',mode=4,admission='YES'):
    return dict(question=dict(question_id=f'{frame}/{edge}/{instance}/{mode}',scenario='s',route_id='r',rgb_mode=mode,
         episode=dict(instance_id=instance,event='U-E7',state='YIELD',longitudinal='HOLD'),frame_id=frame,edge=edge,rule_target=target,facts={}),admission_target=admission)


def test_later_other_instance_or_other_mode_is_not_recheck():
    r=screen([row(10,'proceed'),row(11,'re_yield',instance='b'),row(12,'re_yield',mode=2)],{('s','r'):30})
    assert r['counts']['U-E7']['no_recheck_full_window']==1


def test_right_censoring_is_not_a_success_and_admission_is_separate():
    r=screen([row(10,'proceed',admission='UNKNOWN')],{('s','r'):15})
    assert r['counts']['U-E7']['right_censored']==1
    assert r['counts']['U-E7']['admitted_proceed_yes']==0
    assert r['windows'][0]['early_release_verdict']=='not_adjudicated'


def test_first_same_instance_recheck_within_inclusive_horizon_is_screen_only():
    r=screen([row(10,'proceed'),row(18,'re_yield'),row(19,'re_yield')],{('s','r'):30})
    assert r['counts']['U-E7']['rechecked']==1
    assert r['windows'][0]['first_recheck']==18
    assert r['windows'][0]['early_release_verdict']=='not_adjudicated'


def test_duplicate_identity_is_not_counted_twice():
    with pytest.raises(ValueError,match='duplicate'):screen([row(10,'proceed')]*2,{('s','r'):30})
