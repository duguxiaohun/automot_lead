"""Synthetic regressions, never independent certifications."""
from copy import deepcopy
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_rules as rules,teacher_controls as controls,teacher_replay as replay,teacher_approval as approval,teacher_data as data
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import frames,question,seal
from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import actor
from qwen3vl_local.sft_new_loop_phase4.sampling import plan,CapacityError


def controlled_frames(position=5.):
    fs=frames('vru',True)
    for f in fs:f['actors'].append(actor(50,**{'class':'stop_sign'},position=[position,0.,0.],extent=[1.,2.,.5],affects_ego=True))
    return fs


@pytest.mark.parametrize('case',['far','moving','lane','gap','initial'])
def test_stationary_is_not_automatically_a_completed_stop(case):
    fs=controlled_frames();tracker=controls.StopDutyTracker()
    if case=='far':
        for f in fs:f['actors'][-1]['position'][0]=15.
    if case=='moving':
        for f in fs:f['meta']['speed']=.5
    if case=='lane':
        for f in fs:f['meta']['lane_id']=999
    if case=='gap':fs=[fs[0],fs[2],fs[4]]
    if case=='initial':fs=fs[:3];[f.update(frame_id=i) for i,f in enumerate(fs)]
    for f in fs:result=tracker.observe(f)
    assert not result or not result['50']['satisfied']


def test_stop_duty_is_causal_persistent_and_id_bound():
    fs=controlled_frames();tracker=controls.StopDutyTracker()
    for f in fs:f['stop_duties']=tracker.observe(f)
    assert not fs[1]['stop_duties']['50']['satisfied']
    assert fs[2]['stop_duties']['50']['satisfied']
    assert fs[-1]['stop_duties']['50']['observations'][-1]['frame_id']==6
    later=deepcopy(fs[-1]);later['frame_id']+=1;later['actors'][-1]['id']=51
    assert not tracker.observe(later)['51']['satisfied']


def test_stop_completion_does_not_grant_clearance_to_crossing_vehicle():
    f=controlled_frames()[-1];stop=f['actors'][-1];f['meta']['ego_lane_width']=3.5
    f['actors']=[f['actors'][0],stop,actor(3,position=[8.,4.,0.],ego_velocity=[0.,-3.])]
    f['image_evidence']['3']=dict(quality_pass=True)
    assert controls.crossing_clear(f,stop) is False
    f['image_evidence']['3']['quality_pass']=False
    assert controls.crossing_clear(f,stop) is None
    f['actors'].pop()
    assert controls.crossing_clear(f,stop) is True


def test_first_clear_seed_is_not_an_episode(monkeypatch):
    fs=frames();mapping={f['frame_id']:f for f in fs}
    monkeypatch.setattr(replay.g,'load_frame',lambda root,s,r,n:mapping[n])
    monkeypatch.setattr(rules,'facts',lambda *args:({'event_restriction_observed':False,'release_ready':True,'corridor_clear':True,'priority_satisfied':True},[]))
    route=dict(scenario=fs[0]['scenario'],route_id=fs[0]['route_id'],physical_group='test',split='train',rgb_frames=list(mapping))
    rows=list(replay.route_records(None,route))
    assert not any(r['questions'] for r in rows)
    assert 'seed_without_observed_event_restriction' in rows[-1]['reasons']


def test_weak_policy_explicit_train_only_and_never_approved():
    q=question();registry=approval.weak_registry([q['rule_class']],'synthetic_experiment');a=approval.validate_registry(registry)
    assert not a['approved']
    assert approval.target(q,a)==('YES','weak_rule_experiment')
    for split in ('val','test'):
        assert approval.target(dict(q,split=split),a)==('UNKNOWN','weak_teacher_train_only')
    q['rule_target']='UNKNOWN';assert approval.target(q,a)==('UNKNOWN','rule_abstained')
    registry['independently_approved']=True
    with pytest.raises(ValueError):approval.validate_registry(registry)


def test_weak_provenance_cannot_be_called_approved():
    q=question();a=approval.validate_registry(approval.weak_registry([q['rule_class']],'synthetic_experiment'))
    proof=dict(question=q,registry_sha256=a['registry_sha256'],approval_sha256=a['weak_classes'][q['rule_class']])
    assert data.validate_proof(proof,a)==q
    row=dict(label_basis='approved_rule_teacher',teacher_provenance=proof)
    with pytest.raises(ValueError):data.validate_row(row,a)


def test_sparse_perfect_sample_cannot_pass_strict_budget():
    qs=[question(route=f'Town01_Rep0_{i}_0_route0') for i in range(12)]
    r=approval.best_case_feasibility(qs)
    assert not r['potentially_passable'] and r['answer_deficits']['YES']['routes']==8
    assert r['answer_deficits']['YES']['judgments']==61


def rows():
    from qwen3vl_local.sft_new_loop_phase4.tests.test_fourteenth_audit_fixes import row
    out=[]
    for e,event in enumerate(('U-E1','U-E4')):
        for i in range(26):
            q=row(e*100+i,event);q.update(edge='proceed',target='YES' if i<2 else 'NO');out.append(q)
    return out


def test_edge_answer_balance_and_resume_are_exact():
    pool=rows();history=None
    for epoch in range(3):
        selected,a=plan(pool,epoch=epoch,policy='event_edge_answer',budget=16,history=history);history=a['next_history']
        direct,b=plan(pool,epoch=epoch,policy='event_edge_answer',budget=16)
        assert selected==direct and history==b['next_history']
        assert set(a['event_counts'].values())=={8}
        assert set(a['edge_answer_counts'].values())=={4}
        assert a['max_frame_repeat']<=8
    with pytest.raises(ValueError):plan(pool,epoch=3,budget=16,history=history)


def test_sparse_yes_capacity_cannot_be_replaced_by_no():
    with pytest.raises(CapacityError) as exc:plan(rows(),epoch=0,policy='event_edge_answer',budget=16,cap=1)
    assert any(k.endswith('/YES') for k in exc.value.diagnostic['cell_deficits'])


def test_stop_control_proof_rejects_future_observations():
    q=question();q['control_evidence']={'50':dict(stop_id=50,satisfied=True,observations=[dict(frame_id=n,sources=[]) for n in (9,10,11)])}
    with pytest.raises(ValueError,match='future'):replay.validate_question(seal(q))


def test_true_stop_duty_and_red_light_are_independent_constraints():
    fs=controlled_frames();tracker=controls.StopDutyTracker()
    for f in fs:f['stop_duties']=tracker.observe(f)
    f=fs[-1];f['actors']=[a for a in f['actors'] if a['class']!='walker']
    assert rules.local_priority(f,{'review_horizon_m':18.}) is True
    f['meta']['light_hazard']=True
    assert rules.local_priority(f,{'review_horizon_m':18.}) is False
