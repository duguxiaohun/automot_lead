"""Question-scoped STOP traffic relevance, including actors outside old radius."""
from copy import deepcopy
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_controls as c
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher_v29 import controlled_frames
from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import actor


NAV=dict(points=[[0.,0.],[22.,0.]],half_width=1.75)


@pytest.mark.parametrize('position,velocity,visible,expected',[
    ([-10,0,0],[0,0],False,True),
    ([-10,0,0],None,False,None),
    ([-10,0,0],[12,0],False,None),
    ([17,5,0],[0,-2],True,False),
    ([17,5,0],[0,-2],False,None),
    ([17,0,0],[0,0],True,False),
    ([17,0,0],[0,0],False,None),
    ([17,12,0],[0,0],False,True),
    ([45,2,0],[-20,0],True,False),
    ([45,2,0],[-20,0],False,None),
    ([10,10,0],[0,-2],True,False), # reaches the corridor after 2s, before 4s
])
def test_current_and_continuous_future_relevance_precede_visibility(position,velocity,visible,expected):
    f=controlled_frames()[-1];stop=f['actors'][-1]
    f['actors']=[f['actors'][0],stop,actor(3,position=position,ego_velocity=velocity)]
    for camera in f['meta']['sensor_information']['camera_calibration'].values():camera['fov']=150
    f['image_evidence']['3']=dict(quality_pass=visible)
    before=deepcopy(f)
    assert c.crossing_clear(f,stop,NAV) is expected
    assert f==before


@pytest.mark.parametrize('nav',[None,{},dict(points=[],half_width=1.),dict(points=[[0,0],[0,0]],half_width=1.),
                               dict(points=[[0,0],[22,0]],half_width=float('nan'))])
def test_missing_scope_never_falls_back_to_stop_box(nav):
    f=controlled_frames()[-1]
    assert c.crossing_clear(f,f['actors'][-1],nav) is None


def test_turn_uses_actual_polyline_not_axis_aligned_stop_region():
    f=controlled_frames()[-1];stop=f['actors'][-1]
    f['actors']=[f['actors'][0],stop,actor(3,position=[10,12,0],ego_velocity=[0,0])]
    for camera in f['meta']['sensor_information']['camera_calibration'].values():camera['fov']=150
    f['image_evidence']['3']=dict(quality_pass=True)
    assert c.crossing_clear(f,stop,dict(points=[[0,0],[10,0],[10,18]],half_width=1.75)) is False
    assert c.crossing_clear(f,stop,NAV) is True


def test_known_visible_conflict_wins_over_other_unknown_actor():
    f=controlled_frames()[-1];stop=f['actors'][-1]
    f['actors']=[f['actors'][0],stop,actor(3,position=[-10,0,0],ego_velocity=None),
                 actor(4,position=[15,0,0],ego_velocity=[0,0])]
    f['image_evidence']['4']=dict(quality_pass=True)
    assert c.crossing_clear(f,stop,NAV) is False


@pytest.mark.parametrize('position,velocity,visible,expected',[
    ([-10,0,0],[0,0],False,True),
    ([-10,0,0],None,False,None),
    ([17,5,0],[0,-2],True,False),
    ([17,5,0],[0,-2],False,None),
])
def test_native_priority_consumes_question_scope_after_completed_stop(position,velocity,visible,expected):
    from qwen3vl_local.sft_new_loop_phase4 import teacher_rules as rules
    from qwen3vl_local.sft_new_loop_phase4.tests.test_stop_v55 import clean,observe
    fs=clean(controlled_frames());observe(fs);f=fs[-1]
    assert f['stop_duties']['50']['satisfied']
    f['actors'].append(actor(3,position=position,ego_velocity=velocity))
    for camera in f['meta']['sensor_information']['camera_calibration'].values():camera['fov']=150
    f['image_evidence']['3']=dict(quality_pass=visible)
    assert rules.local_priority(f,dict(NAV,review_horizon_m=22.)) is expected
