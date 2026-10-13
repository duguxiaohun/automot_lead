"""Partial projections are diagnostic; departures need post-restriction evidence."""
from copy import deepcopy
from itertools import product

import numpy as np
import pytest
from PIL import Image

from qwen3vl_local.sft_new_loop_phase4 import privileged_geometry as g, privileged_visibility as v
from qwen3vl_local.sft_new_loop_phase4 import teacher_events as t
from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import snapshot, actor
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher_v53 import departure, answer


def test_near_clipping_keeps_edge_intersections_not_just_front_corners():
    vertices=list(product((-1.,2.),(-3.,3.),(-4.,4.)))
    points=v.clip_box_near_plane(vertices)
    assert len(points)==8
    assert set(points)==set(product((.1,2.),(-3.,3.),(-4.,4.)))


@pytest.mark.parametrize('depths',[(-2.,-1.),(-1.,.1)])
def test_box_wholly_behind_or_touching_near_plane_has_no_projection(depths):
    assert v.clip_box_near_plane(list(product(depths,(-1.,1.),(-1.,1.))))==[]


def test_clipping_preserves_fully_front_box_and_rejects_nonfinite():
    vertices=list(product((1.,2.),(-1.,1.),(-1.,1.)))
    assert v.clip_box_near_plane(vertices)==vertices
    vertices[0]=(float('nan'),0.,0.)
    assert v.clip_box_near_plane(vertices)==[]


@pytest.mark.parametrize('level',[None,0,25,255])
def test_partial_box_records_quality_without_granting_visibility(tmp_path,level):
    f=snapshot(10,[actor(position=[.5,0.,0.],extent=[1.,1.,1.],visible_pixels=900)])
    f['meta']['sensor_information']['camera_calibration']={'1':dict(
        width=384,height=384,pos=[0.,0.,0.],rot=[0.,0.,0.],fov=90.)}
    rgb=np.repeat((((np.indices((384,384)).sum(axis=0)%2)*190+30).astype('uint8'))[:,:,None],3,axis=2)
    if level is not None:rgb[:]=level
    path=tmp_path/'rgb.png';Image.fromarray(rgb).save(path)
    f['image_evidence']=v.image_evidence(path,f);e=f['image_evidence']['2']
    assert e['crops']==[] and e['quality_pass'] is False
    assert e['partial_quality_pass'] is (level is None)
    assert len(e['partial_crops'])==1
    assert e['partial_crops'][0]['bounds']==[0,0,384,384]
    assert g.visible(f['actors'][1],f) is None


@pytest.mark.parametrize('position,has_crop', [([10.,0.,0.],True),([-5.,0.,0.],False),([10.,100.,0.],False)])
def test_full_front_and_outside_projections(tmp_path,position,has_crop):
    f=snapshot(10,[actor(position=position,extent=[1.,1.,1.])])
    f['meta']['sensor_information']['camera_calibration']={'1':dict(
        width=384,height=384,pos=[0.,0.,0.],rot=[0.,0.,0.],fov=90.)}
    path=tmp_path/'rgb.png';Image.new('RGB',(384,384),(100,100,100)).save(path)
    e=v.image_evidence(path,f)['2']
    assert bool(e['crops']) is has_crop and e['partial_crops']==[]


def test_pre_intrusion_separation_is_not_visible_departure_or_yes():
    fs,s=departure(False);s['restriction_frame']=8
    for f,y in zip(fs,[4.,4.,4.,3.,1.,3.,4.]):f['actors'][1]['position'][1]=y
    assert t.visible_departure(fs,s,t.instance_navigation(fs[-1],s)) is False
    assert answer(fs,s)=='UNKNOWN'


@pytest.mark.parametrize('restriction',[None,True,-1,9,10,20])
def test_missing_invalid_or_noncausal_restriction_cannot_certify_exit(restriction):
    fs,s=departure(True);s['restriction_frame']=restriction
    assert t.visible_departure(fs,s,t.instance_navigation(fs[-1],s)) is False
    assert answer(fs,s)=='UNKNOWN'


def test_post_intrusion_visible_departure_remains_supported_but_partial_does_not():
    fs,s=departure(True)
    assert answer(fs,s)=='YES'
    fs[-2]['image_evidence']['2'].update(quality_pass=False,
        partial_quality_pass=True,partial_crops=[dict(usable=True,bounds=[0,0,100,100])])
    assert t.visible_departure(fs,s,t.instance_navigation(fs[-1],s)) is False
    assert answer(fs,s)=='UNKNOWN'
