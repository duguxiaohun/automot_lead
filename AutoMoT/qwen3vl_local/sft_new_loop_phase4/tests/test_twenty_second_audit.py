from copy import deepcopy
import json
import math
import numpy as np
import pytest
from PIL import Image
from qwen3vl_local.sft_new_loop_phase4 import privileged_geometry as g,privileged_producer as p,privileged_context as c
from qwen3vl_local.sft_new_loop_phase4.privileged_visibility import image_evidence
from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import snapshot,actor,episode,request,nav_context,write_source
from qwen3vl_local.sft_new_loop_phase4.replay_safety import assess
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.identity import digest
from qwen3vl_local.sft_new_loop_phase4.candidate_pool import scan
from qwen3vl_local.sft_new_loop_phase4.visible_scope import POLICY as VISIBLE_POLICY


def scene(ep,**kw):
    return dict(dict(frame_id=10,episode_sha256=digest(ep.to_dict()),source='causal_scene_review',evidence_id='synthetic',
        corridor_bounds=[0,8,-1.75,1.75],event_actor_ids=[],event_identity_established=True,priority_satisfied=True,
        visible_space_resolved=True,signal_failure_established=True,event_end_reached=True,stop_obligation_active=False),**kw)


def record(ep,frames,**kw):
    f=frames[-1]
    return dict(dict(policy=c.POLICY,scenario=f['scenario'],route_id=f['route_id'],frame_id=f['frame_id'],
        sources_sha256=digest([x['sources'] for x in frames]),episode=ep.to_dict(),state_evidence_id='synthetic-current-state',
        geometry_context=None,scene_context=None),**kw)


def test_same_id_jump_and_static_disappearance_are_not_clearance():
    frames=[snapshot(9,[actor()]),snapshot(10,[actor(position=[70.,0.,0.])])]
    risk=g.disappearances(*frames)
    assert risk[0]['kind']=='same_id_position_jump_candidate' and risk[0]['world_displacement_m']==55
    ep=episode();receipt,audit=assess(ep,'enter',frames,request(ep,frames))
    assert receipt is None and 'near_actor_disappearance' in audit['reasons']
    for cls in ('static','static_prop_car'):
        f=snapshot(9,[actor(**{'class':cls})]);assert g.disappearances(f,snapshot(10))


def test_turn_and_translation_do_not_create_false_teleport():
    before=snapshot(9,[actor(position=[15.,0.,0.],ego_velocity=[0.,0.])])
    after=snapshot(10,[actor(position=[0.,-14.,0.],ego_velocity=[0.,0.])])
    after['meta']['ego_matrix']=[[0.,-1.,0.,1.],[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
    assert g.world_point(before,before['actors'][1])==g.world_point(after,after['actors'][1])
    assert g.disappearances(before,after)==[]
    del after['meta']['ego_matrix']
    assert g.disappearances(before,after)[0]['kind']=='motion_compensation_unavailable'


def test_missing_pose_blocks_even_empty_inventory_safety():
    ep=episode();frames=[snapshot(9),snapshot(10)];del frames[-1]['meta']['ego_matrix']
    receipt,audit=assess(ep,'enter',frames,request(ep,frames))
    assert receipt is None and 'ego_motion_compensation_unavailable' in audit['reasons']


@pytest.mark.parametrize('base',['bicycle','motorcycle'])
def test_vulnerable_vehicle_is_not_lead_or_cutin(base):
    frames=[snapshot(f,[actor(base_type=base)]) for f in (8,9,10)]
    assert [a['event'] for a in p.candidates(frames,{})]==['U-E4']
    assert g.following(frames)[0] is None


@pytest.mark.parametrize('cls',['static','static_prop_car'])
def test_static_obstacle_intersection_creates_ue2_candidate(cls):
    frames=[snapshot(10,[actor(**{'class':cls},type_id='static.prop.constructioncone')])]
    assert p.candidates(frames,{})[0]['event']=='U-E2'
    frames[-1]['actors'][1]['position'][1]=5
    assert not p.candidates(frames,{})


def test_normal_opposing_lane_not_oncoming_intrusion():
    frames=[snapshot(10,[actor(position=[15.,3.5,0.],yaw=math.pi)])]
    assert p.candidates(frames,{})==[]
    frames[-1]['actors'][1]['position'][1]=1.
    assert p.candidates(frames,{})[0]['event']=='U-E5'


@pytest.mark.parametrize('level',[0,25,255])
def test_dark_or_flat_or_saturated_rgb_cannot_prove_visibility(tmp_path,level):
    f=snapshot(10,[actor(visible_pixels=900)])
    path=tmp_path/'rgb.png';Image.new('RGB',(384,384),(level,level,level)).save(path)
    f['image_evidence']=image_evidence(path,f)
    assert g.visible(f['actors'][1],f) is None
    ep=episode();f['actors'][1]['position'][1]=-3
    assert g.condition_proposal([f],ep,'enter',geometry_context=nav_context())['target']=='UNKNOWN'


@pytest.mark.parametrize('event',['U-E1','U-E3','U-E4','U-E5','U-E6','U-E7','R-E5'])
def test_proceed_yes_requires_current_scene_geometry_identity_and_priority(event):
    frames=[snapshot(f) for f in (8,9,10)];ep=Episode(event,'scene',state='YIELD',longitudinal='HOLD')
    assert g.condition_proposal(frames,ep,'proceed')['target']=='UNKNOWN'
    assert g.condition_proposal(frames,ep,'proceed',scene_context=scene(ep))['target']=='YES'
    assert g.condition_proposal(frames,ep,'proceed',scene_context=scene(ep,priority_satisfied=None))['target']=='UNKNOWN'
    assert g.condition_proposal(frames,ep,'proceed',scene_context=scene(ep,event_identity_established=False))['target']=='UNKNOWN'
    if event=='U-E7':assert g.condition_proposal(frames,ep,'proceed',scene_context=scene(ep,signal_failure_established=None))['target']=='UNKNOWN'


def test_hold_restrict_complete_can_be_positive_but_completion_needs_boundary():
    frames=[snapshot(f,[actor(position=[3.,0.,0.])]) for f in (8,9,10)]
    ep=Episode('U-E1','scene',state='YIELD',longitudinal='STABLE')
    for edge in ('hold','restrict'):
        assert g.condition_proposal(frames,ep,edge,scene_context=scene(ep))['target']=='YES'
    frames=[snapshot(f) for f in (8,9,10)]
    ep=Episode('U-E1','scene',state='PROCEED',longitudinal='STABLE')
    assert g.condition_proposal(frames,ep,'complete',scene_context=scene(ep))['target']=='YES'
    assert g.condition_proposal(frames,ep,'complete',scene_context=scene(ep,event_end_reached=None))['target']=='UNKNOWN'


def test_source_bound_navigation_reaches_route_producer_and_wrong_hash_rejected(tmp_path):
    for f in (8,9,10):s,r=write_source(tmp_path,f,[])
    frames=[g.load_frame(tmp_path,s,r,f) for f in (8,9,10)];ep=episode()
    rec=record(ep,frames,geometry_context=nav_context());contexts={(s,r,10):[rec]}
    rows=list(p.route_proposals(tmp_path,dict(scenario=s,route_id=r),contexts=contexts))
    enter=[v for v in rows[-1]['proposals'] if v['edge']=='enter']
    assert len(enter)==1 and enter[0]['proposal']['target']=='YES' and enter[0]['training_target']=='UNKNOWN'
    assert enter[0]['state_basis']=='source_bound_current_state'
    rec['sources_sha256']='bad'
    with pytest.raises(ValueError,match='causal history'):list(p.route_proposals(tmp_path,dict(scenario=s,route_id=r),contexts=contexts))


def test_new_visible_lateral_references_are_calibrated_and_not_skipped(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4 import dataset
    monkeypatch.setattr(dataset,'split_for',lambda *args,**kwargs:'train')
    for f in (8,9,10):s,r=write_source(tmp_path,f,[])
    frames=[g.load_frame(tmp_path,s,r,f) for f in (8,9,10)];ep=episode();rec=record(ep,frames,geometry_context=nav_context())
    ref=dict(scenario=s,route_id=r,frame_id=10,episode=ep.to_dict(),edge='enter',target='YES',scope=VISIBLE_POLICY,
        rgb_history_sha256=[f['sources'][-1]['sha256'] for f in frames],reviewer='synthetic',observation='empty visible target',observed_until=10)
    result=c.calibrate_references([ref],[rec],tmp_path,g.DEFAULTS)
    assert result['summary']=={'reference':1,'correct':1} and not result['supervision_approved']
    assert c.calibrate_references([ref],[],tmp_path,g.DEFAULTS)['summary']['abstained']==1
    ref['target']='NO'
    assert c.calibrate_references([ref],[rec],tmp_path,g.DEFAULTS)['summary']['wrong']==1
    ref['scope']='old_full_gap'
    with pytest.raises(ValueError,match='scope'):c.calibrate_references([ref],[rec],tmp_path,g.DEFAULTS)


def test_new_exposure_and_manual_check_reservations_are_enforced():
    from qwen3vl_local.sft_new_loop_phase4 import dataset,risk_review
    from qwen3vl_local.sft_new_loop_phase4.identity import ROOT
    ledger=json.loads((ROOT/'twenty_second_audit_exposure_20261002.json').read_text())
    assert len(ledger['train_only_groups'])==9
    assert set(ledger['train_only_groups'])<=dataset.groups()
    extra=dataset.producer_check_reservations()
    original=json.loads((ROOT/'producer_manual_check_plan_v2_20261002.json').read_text())['reservations']
    assert len(original)==40
    assert all(dataset.holdout_reservations()[r['physical_group']]==r['split'] for r in original)
    assert not set(extra)&dataset.groups()
    for group,split in extra.items():
        assert dataset.split_for(group,set(),seed=1)==split
        with pytest.raises(ValueError,match='development exposure'):dataset.split_for(group,{group})
        with pytest.raises(ValueError,match='model selection'):dataset.check_evaluation_protocol({'physical_group':group},extra)
    risks=risk_review.registry()
    for r in ledger['calibration_risks']:
        assert any(v['record']==r for v in risks[(r['scenario'],r['route_id'])])


@pytest.mark.parametrize('edge',['depart','return'])
def test_navigation_can_propose_bypass_permissions_and_retains_missing_evidence(edge):
    ep=Episode('U-E2','block',state='WAIT' if edge=='depart' else 'PASS',longitudinal='HOLD',
               direction='LEFT',target_corridor='left lane',return_direction='LEFT',return_corridor='left lane')
    frames=[snapshot(f,[actor(99,**{'class':'static'},position=[-10.,0.,0.])]) for f in (8,9,10)]
    nav=nav_context(obstacle_actor_ids=[99],obstacle_membership_complete=True)
    assert g.condition_proposal(frames,ep,edge,geometry_context=nav)['target']=='YES'
    assert g.condition_proposal(frames,ep,edge)['target']=='UNKNOWN'
    assert g.condition_proposal(frames,ep,edge,geometry_context=nav_context(priority_satisfied=None))['target']=='UNKNOWN'


def test_curved_road_oncoming_in_other_lane_is_not_intrusion():
    # Actual failure mode: far actors project across a straight ego-axis strip
    # on a curved road, while road/lane metadata places them in the normal lane.
    f=snapshot(10,[actor(position=[22.,.3,0.],yaw=-2.86,lane_id=-1)])
    assert p.candidates([f],{})==[]
