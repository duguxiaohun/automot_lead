from copy import deepcopy
import json
import lzma
import pickle
import pytest
from PIL import Image
from qwen3vl_local.sft_new_loop_phase4 import privileged_geometry as g, privileged_producer as p, replay_safety as safety
from qwen3vl_local.sft_new_loop_phase4 import candidate_pool,condition_builder as cb,dataset
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.evaluate import replay
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT,digest,file_sha
from qwen3vl_local.sft_new_loop_phase4.maneuver_safety import binding
from qwen3vl_local.sft_new_loop_phase4.tests.safety_fixtures import clearances


def actor(ident=2,**kw):
    return dict(dict(id=ident,**{'class':'car'},position=[15.,0.,0.],extent=[2.,.8,.8],yaw=0.,speed=2.,
                     ego_velocity=[2.,0.],road_id=1,lane_id=1,visible_pixels=100,num_points=300),**kw)


def raw(actors=()):
    return (dict(ego_matrix=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],speed=2.,road_id=1,lane_id=1,ego_lane_width=3.5,is_junction=False,light_hazard=False,
                 stop_sign_hazard=False,walker_hazard=False,vehicle_hazard=False,rear_danger_8=False,rear_danger_16=False,
                 sensor_information={'camera_calibration':{'1':dict(pos=[0.,0.,2.],rot=[0.,0.,0.],fov=60.,width=384,height=384,cropped_height=384)}}),
            [actor(1,**{'class':'ego_car'},position=[0.,0.,0.],extent=[2.,1.,1.]),*actors])


def snapshot(f,actors=()):
    m,b=raw(actors);r=g.project(m,b,f);r.update(scenario='Scenario',route_id='Town01_Rep0_1_0_route0',sources=[])
    r['image_evidence']={str(a['id']):{'quality_pass':True} for a in r['actors']}
    r['causal_sha256']=digest({k:r[k] for k in ('frame_id','meta','actors')});return r


def episode(**kw):
    return Episode(**dict(dict(event='R-E2',instance_id='lane-change',direction='LEFT',target_corridor='left lane',longitudinal='HOLD'),**kw))


def request(ep,frames,**kw):
    current=frames[-1]
    return dict(dict(policy=safety.POLICY,binding=binding(ep,'enter',current['frame_id']),causal_sha256=current['causal_sha256'],
        navigation_evidence_id='synthetic-current-nav',coverage_evidence_id='synthetic-inventory-contract',
        swept_envelope=[-3.,8.,-5.,1.5],target_footprint=[-2.,2.,-4.5,-2.5],coverage_radius_m=96.,horizon_s=1.,max_untracked_speed_mps=25.,
        acceleration_bound_mps2=2.,uncertainty_margin_m=.5,all_actor_inventory_complete=True,
        static_drivable_space_verified=True,priority_verified=True),**kw)


def test_future_arrays_controls_and_source_event_names_do_not_enter_causal_projection():
    m,b=raw([actor()]);baseline=g.project(m,b,10)
    for obj in (m,*b):obj.update(future_positions=[[999]*3],future_speeds=[-99],future_yaws=[5.],brake=1,throttle=1,steer=1,
                                 target_speed=99,vehicle_cuts_in=True,scenario='fake source event')
    assert g.project(m,b,10)==baseline
    b[1]['position'][0]=8
    assert g.project(m,b,10)!=baseline


def test_lidar_points_do_not_prove_camera_visibility_and_unknown_pixels_abstain():
    frame=snapshot(10,[actor()])
    assert g.visible(actor(visible_pixels=0,num_points=99999),frame) is False
    assert g.visible(actor(visible_pixels=-1),frame) is None
    assert g.visible(actor(position=[-8.,0.,0.]),frame) is None
    assert g.visible(actor(),frame) is True


@pytest.mark.parametrize('change',[{'position':[float('nan'),0.,0.]},{'extent':[0.,1.,1.]},{'yaw':float('inf')}])
def test_bad_geometry_rejected(change):
    m,b=raw([actor(**change)])
    with pytest.raises(ValueError):g.project(m,b,10)


def test_duplicate_actors_and_missing_ego_rejected():
    m,b=raw([actor(),actor()])
    with pytest.raises(ValueError,match='duplicate'):g.project(m,b,10)
    with pytest.raises(ValueError,match='ego'):g.project(m,[actor()],10)
    m,b=raw();m['speed']=-1e-7
    assert g.project(m,b,10)['meta']['speed']==-1e-7  # LEAD's speed is signed.


def test_following_is_causal_actor_scoped_and_not_motion_onset():
    frames=[snapshot(f,[actor()]) for f in (8,9,10)]
    assert g.following(frames)[:2]==(True,2)
    ep=Episode('U-E3','cutin',state='PROCEED',longitudinal='RECOVER')
    assert g.condition_proposal(frames,ep,'recover_follow')['target']=='YES'
    frames[-1]['actors'][1]['ego_velocity'][1]=2.
    assert g.following(frames)[0] is False
    frames[-1]['actors'][1]['id']=3
    assert g.following(frames)[0] is None


def test_disappearing_participant_never_becomes_following_or_completion_yes():
    frames=[snapshot(8,[actor()]),snapshot(9,[actor()]),snapshot(10)]
    risks=g.disappearances(frames[-2],frames[-1])
    assert risks[0]['actor_id']==2 and 'not_confirmed' in risks[0]['status']
    ep=Episode('U-E3','cutin',state='PROCEED',longitudinal='RECOVER')
    assert g.condition_proposal(frames,ep,'recover_follow')['target']=='UNKNOWN'


@pytest.mark.parametrize('case',['junction','invisible','new_participant','missing_history','unknown_hazard'])
def test_following_proposal_abstains_without_full_causal_evidence(case):
    frames=[snapshot(f,[actor()]) for f in (8,9,10)]
    if case=='junction':frames[-1]['meta']['is_junction']=True
    elif case=='invisible':frames[-1]['actors'][1]['visible_pixels']=-1
    elif case=='new_participant':frames[-1]['actors'].append(actor(3,**{'class':'walker'},position=[4.,0.,0.]))
    elif case=='missing_history':frames.pop(1)
    else:del frames[-1]['meta']['walker_hazard']
    assert g.following(frames)[0] is None


def nav_context(frame=10,**kw):
    return dict(dict(frame_id=frame,target_corridor='left lane',corridor_bounds=[0.,30.,-5.,-2.],
        target_road_id=1,target_lane_id=2,priority_satisfied=True,visible_space_resolved=True,
        obstacle_actor_ids=[],obstacle_membership_complete=False,source='planner',evidence_id='test-current-geometry'),**kw)


def test_visible_gap_requires_navigation_priority_and_visible_space_evidence():
    ep=episode();frames=[snapshot(f) for f in (8,9,10)]
    assert g.condition_proposal(frames,ep,'enter')['target']=='UNKNOWN'
    assert g.condition_proposal(frames,ep,'enter',geometry_context=nav_context())['target']=='YES'
    assert g.condition_proposal(frames,ep,'enter',geometry_context=nav_context(priority_satisfied=None))['target']=='UNKNOWN'
    frames[-1]['actors'].append(actor(position=[15.,-3.,0.]))
    frames[-1]['image_evidence']['2']={'quality_pass':True}
    assert g.condition_proposal(frames,ep,'enter',geometry_context=nav_context())['target']=='NO'
    with pytest.raises(ValueError):g.condition_proposal(frames,ep,'enter',geometry_context=nav_context(frame=11))


def test_return_cannot_infer_passage_from_missing_obstacle_id():
    ep=Episode('U-E2','obstacle',state='PASS',return_direction='LEFT',return_corridor='left lane')
    frames=[snapshot(f) for f in (8,9,10)]
    result=g.condition_proposal(frames,ep,'return',geometry_context=nav_context(obstacle_actor_ids=[99],obstacle_membership_complete=True))
    assert result['facts']['obstacle_passed'] is None and result['target']=='UNKNOWN'


def test_clearance_computes_rear_conflict_even_with_false_rear_danger():
    ep=episode();frames=[snapshot(f,[actor(position=[-8.,-3.,0.],ego_velocity=[18.,0.],visible_pixels=0)]) for f in (9,10)]
    receipt,audit=safety.assess(ep,'enter',frames,request(ep,frames))
    assert receipt['clear'] is False and audit['blocking_actor_ids']==[2]
    assert frames[-1]['meta']['rear_danger_8'] is False
    ep.advance(10,{q.key:'YES' if q.key=='enter' else 'NO' for q in ep.questions()},maneuver_clearances=[receipt])
    assert ep.state=='WAIT' and ep.longitudinal=='HOLD' and not ep.uncertain


@pytest.mark.parametrize('field,value',[('all_actor_inventory_complete',False),('static_drivable_space_verified',False),
                                      ('priority_verified',False),('coverage_radius_m',10.)])
def test_missing_safety_prerequisite_produces_no_receipt(field,value):
    ep=episode();frames=[snapshot(f) for f in (9,10)]
    receipt,audit=safety.assess(ep,'enter',frames,request(ep,frames,**{field:value}))
    assert receipt is None and audit['reasons']


@pytest.mark.parametrize('change',[{'horizon_s':0.},{'horizon_s':float('nan')},{'uncertainty_margin_m':-1.},
                                   {'swept_envelope':[0.,5.,-4.,-2.]},{'target_footprint':[-2.,2.,-8.,-6.]},
                                   {'target_footprint':[-2.,2.,-.5,.5]},{'binding':{}},{'causal_sha256':'wrong'}])
def test_invalid_safety_request_rejected(change):
    ep=episode();frames=[snapshot(f) for f in (9,10)]
    with pytest.raises(ValueError):safety.assess(ep,'enter',frames,request(ep,frames,**change))


def test_current_full_geometry_can_authorize_entry_but_does_not_acknowledge_execution():
    ep=episode();frames=[snapshot(f) for f in (9,10)]
    receipt,audit=safety.assess(ep,'enter',frames,request(ep,frames))
    assert receipt['clear'] and not audit['reasons']
    ep.advance(10,{q.key:'YES' if q.key=='enter' else 'NO' for q in ep.questions()},maneuver_clearances=[receipt])
    assert ep.state=='CROSS' and not ep.history[-1]['committed']


def test_safety_unknown_after_near_disappearance_and_no_visible_pixel_shortcut():
    ep=episode();frames=[snapshot(9,[actor()]),snapshot(10)]
    receipt,audit=safety.assess(ep,'enter',frames,request(ep,frames))
    assert receipt is None and 'near_actor_disappearance' in audit['reasons']


def test_known_denial_waits_beyond_stall_limit_and_unknown_still_rechecks():
    ep=episode(stall_limit=2)
    for f in range(10,15):
        proof=clearances(ep,f,'enter');proof[0]['clear']=False
        ep.advance(f,{q.key:'YES' if q.key=='enter' else 'NO' for q in ep.questions()},maneuver_clearances=proof)
        assert ep.prior()['action']=='STOP' and not ep.needs_recheck and ep.stalled_observations==0
    for f in (15,16):ep.advance(f,{q.key:'YES' if q.key=='enter' else 'NO' for q in ep.questions()})
    assert ep.needs_recheck


def write_source(root,frame,actors=()):
    scenario='Scenario';route='Town01_Rep0_1_0_route0';run=root/scenario/route;m,b=raw(actors)
    for kind,value in [('metas',m),('bboxes',b)]:
        path=run/kind/f'{frame:04d}.pkl';path.parent.mkdir(parents=True,exist_ok=True)
        with lzma.open(path,'wb') as f:pickle.dump(value,f)
    rgb=run/'rgb'/f'{frame:04d}.jpg';rgb.parent.mkdir(exist_ok=True);__import__('numpy');Image.fromarray(((__import__('numpy').indices((384,384)).sum(axis=0)%2)*255).astype('uint8')).convert('RGB').save(rgb)
    return scenario,route


def test_real_file_adapter_is_wired_into_replay_and_binds_current_rgb(tmp_path):
    s,r=write_source(tmp_path,9);write_source(tmp_path,10)
    frames=[g.load_frame(tmp_path,s,r,f) for f in (9,10)];ep=episode()
    obs=dict(frame_id=10,truth={'enter':'YES'},visible_truth_scope='visible_maneuver_conditions_v1',
             image_sha256=[frames[-1]['sources'][-1]['sha256']],
             privileged_source=dict(scenario=s,route_id=r,source_sha256=digest([f['sources'] for f in frames])),
             maneuver_safety_requests=[request(ep,frames)],execution_committed=True)
    predictor=lambda ep,key,obs:'YES' if key=='enter' else 'NO'
    report=replay(ep.to_dict(),[obs],predictor,safety_adapter=safety.ReplaySafetyAdapter(tmp_path))
    assert report['steps'][0]['after']['state']=='CROSS' and report['execution_delay_frames']==[0]
    assert report['steps'][0]['safety_audit'][0]['blocking_actor_ids']==[]
    with pytest.raises(ValueError,match='explicit safety adapter'):replay(ep.to_dict(),[obs],predictor)
    obs['image_sha256']=['wrong']
    with pytest.raises(ValueError,match='same current RGB'):replay(ep.to_dict(),[obs],predictor,safety_adapter=safety.ReplaySafetyAdapter(tmp_path))


def test_stream_is_split_bound_and_cannot_be_used_as_reviewed_labels(tmp_path):
    root=tmp_path/'data'
    for f in (8,9,10):s,r=write_source(root,f,[actor()])
    pool=candidate_pool.scan(root,tmp_path/'pool.json');split=pool['routes'][0]['split'];out=tmp_path/'proposals.jsonl'
    report=p.produce(pool,root,out,splits=[split],route_keys={(s,r)})
    assert report['frames']==3 and report['approved_questions']==0 and not report['full_causal_production']
    rows=[json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]['supervision_approved'] is False
    assert any(pr['proposal']['geometric_target']=='YES' for row in rows[1:] for pr in row['proposals'])
    assert all(pr['training_target']=='UNKNOWN' for row in rows[1:] for pr in row['proposals'])
    with pytest.raises(ValueError,match='header'):cb.compile_stream(out,root)
    with pytest.raises(FileExistsError):p.produce(pool,root,out,splits=[split])
    with pytest.raises(ValueError,match='outside'):p.produce(pool,root,tmp_path/'bad.jsonl',splits=[split],route_keys={('other','Town01_Rep0_2_0_route0')})


def test_calibration_counts_disagreements_and_abstention_without_self_approval(tmp_path):
    for f in (8,9,10):s,r=write_source(tmp_path,f,[actor()])
    ep=dict(event='U-E1',instance_id='following',state='PROCEED',longitudinal='RECOVER')
    ann=dict(scenario=s,route_id=r,episode=ep,edge='recover_follow',start=10,end=10,
             facts={'restricted_progress_established':False},slice='readiness',label_basis='per_frame_conditions',context_valid=True,
             evidence_id='synthetic-calibration',frame_sha256={'10':file_sha(tmp_path/s/r/'rgb/0010.jpg')})
    report=p.calibrate([ann],tmp_path,scope='all')
    assert report['summary']==dict(reference=1,wrong=1) and report['conditional_error_rate']==1.
    assert not report['supervision_approved']
    ann['frame_sha256']['10']='wrong'
    with pytest.raises(ValueError,match='identity'):p.calibrate([ann],tmp_path,scope='all')


def test_compiled_lateral_conditions_roundtrip_as_quarantine_not_provenance_failure(tmp_path):
    for f in range(4,11):s,r=write_source(tmp_path,f)
    f=g.load_frame(tmp_path,s,r,10)
    producer=dict(name='synthetic reviewed conditions',version='1',source='causal_geometry_review',rubric_sha256=cb.rubric_identity())
    record=dict(scenario=s,route_id=r,episode={k:v for k,v in episode().to_dict().items() if k in dataset.EPISODE_FIELDS},
                frame_id=10,observed_until=10,context_valid=True,
                facts=dict(target_corridor_known=True,entry_gap_clear=True,priority_satisfied=True),
                sources=[f['sources'][-1]])
    annotations=cb.annotations_for_record(record,producer,tmp_path)
    out=tmp_path/'compiled';dataset.build(annotations,tmp_path,out,rgb_mode=2)
    dataset.load_dataset(out)
    row=next(r for r in dataset.read_rows(out/'review_queue.jsonl') if r['edge']=='enter')
    assert row['target']=='UNKNOWN' and row['review_reason']=='visible_scope_reaudit_required'
    cb.validate_row(row,tmp_path)
    row['target']='YES'
    with pytest.raises(ValueError,match='differs'):cb.validate_row(row,tmp_path)


def test_new_exposure_is_explicit_and_old_confirmed_risk_not_double_counted():
    from qwen3vl_local.sft_new_loop_phase4 import risk_review
    ledger=json.loads((ROOT/'twenty_first_audit_exposure_20261001.json').read_text())
    assert len(ledger['train_only_groups'])==6 and set(ledger['previous_splits'].values())=={'train'}
    for group in ledger['train_only_groups']:
        assert all(dataset.split_for(group,dataset.groups(),seed)=='train' for seed in (1,2,2026))
    assert len(ledger['calibration_risks'])==7
    assert not any('146_1' in r['route_id'] for r in ledger['calibration_risks'])
    for r in ledger['calibration_risks']:assert (r['scenario'],r['route_id']) in risk_review.registry()


def test_replay_separately_reports_conflicting_answers_and_missing_or_denied_safety():
    ep=Episode('U-E3','following',state='PROCEED',longitudinal='RECOVER')
    r=replay(ep.to_dict(),[dict(frame_id=10)],lambda ep,key,o:'YES' if key in ('stable','recover_follow') else 'NO')
    assert r['answer_conflict_frames']==[10] and r['safety_missing_frames']==[]
    ep=episode();proof=clearances(ep,10,'enter');proof[0]['clear']=False
    r=replay(ep.to_dict(),[dict(frame_id=10,maneuver_clearances=proof),dict(frame_id=11)],
             lambda ep,key,o:'YES' if key=='enter' else 'NO')
    assert r['safety_denied_frames']==[10] and r['safety_missing_frames']==[11]
    assert r['answer_conflict_frames']==[]


def test_actor_sweep_and_planner_envelope_use_same_fixed_current_frame():
    ep=episode()
    frames=[snapshot(f,[actor(position=[-10.,-3.,0.],ego_velocity=[15.,0.],visible_pixels=0)]) for f in (9,10)]
    for frame in frames:
        frame['meta']['speed']=20.
        frame['causal_sha256']=digest({k:frame[k] for k in ('frame_id','meta','actors')})
    # In the fixed current frame the car sweeps into the planner envelope.
    # Subtracting ego speed would incorrectly predict it moving backwards.
    receipt,audit=safety.assess(ep,'enter',frames,request(ep,frames))
    assert receipt['clear'] is False and audit['blocking_actor_ids']==[2]
