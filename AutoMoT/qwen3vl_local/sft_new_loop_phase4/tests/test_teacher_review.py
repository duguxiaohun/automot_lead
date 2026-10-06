from copy import deepcopy
import json
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_review as review,teacher_rules as rules,teacher_replay as replay,teacher_approval as approval,dataset,candidate_pool
from qwen3vl_local.sft_new_loop_phase4.identity import digest,file_sha,write_json
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import question,seal,frames,approved_registry,bind_fixture,make_bundle


def snapshot(questions=None,purpose='development'):
    body=dict(policy=review.POLICY,teacher=rules.identity(),purpose=purpose,questions=questions or [question(),question('NO',number=20)],production_index_sha256='synthetic_population')
    body['public_sha256']=review.public_packet(body)['sha256']
    return review.seal(body)


def decisions(snap):
    public=review.public_packet(snap)
    return dict(policy=review.POLICY,public_sha256=public['sha256'],reviewer='synthetic_blind_reviewer',
        independence_attested=True,blind_to_rule_answers=True,frozen_before_review=True,
        decisions=[dict(card_id=c['id'],reference='UNKNOWN',event_identity_confirmed=False,state_confirmed=False,
                        current_input_resolves_answer=False,evidence='synthetic unresolved input') for c in public['cards']])


def test_public_packet_does_not_expose_teacher_or_source_hints():
    s=snapshot();text=json.dumps(review.public_packet(s))
    for secret in ('rule_target','rule_class','facts','actor_id','Scenario','Town01','near_transition','causal_sources','state_rule_target'):
        assert secret not in text
    assert all(len(c['images'])==2 for c in review.public_packet(s)['cards'])


def test_import_keeps_unknown_references_and_binds_complete_sample():
    snap=snapshot();imported=review.import_review(snap,decisions(snap))
    ref=next(iter(imported['references'].values()))
    assert len(ref['samples'])==2 and all(s['reference']=='UNKNOWN' for s in ref['samples'])
    assert approval.report(ref['samples'])['answers']['YES']['correct']==0
    review.validate_binding(ref,'development')
    ref['samples'].pop()
    with pytest.raises(ValueError,match='complete frozen'):review.validate_binding(ref,'development')


@pytest.mark.parametrize('case',['missing','duplicate','foreign','pending','blank','unblind','unfrozen','binary_unresolved','wrong_hash','author_missing'])
def test_incomplete_or_unbound_review_is_rejected(case):
    snap=snapshot();d=decisions(snap)
    if case=='missing':d['decisions'].pop()
    if case=='duplicate':d['decisions'].append(d['decisions'][0])
    if case=='foreign':d['decisions'][0]['card_id']='foreign'
    if case=='pending':d['decisions'][0]['reference']=None
    if case=='blank':d['decisions'][0]['evidence']=''
    if case=='unblind':d['blind_to_rule_answers']=False
    if case=='unfrozen':d['frozen_before_review']=False
    if case=='binary_unresolved':d['decisions'][0]['reference']='YES'
    if case=='wrong_hash':d['public_sha256']='wrong'
    if case=='author_missing':d['reviewer']=' '
    with pytest.raises(ValueError):review.import_review(snap,d)


def test_independent_review_requires_attestation_and_cannot_change_transition_strata():
    snap=snapshot(purpose='independent');d=decisions(snap);d['independence_attested']=False
    with pytest.raises(ValueError,match='independent'):review.import_review(snap,d)
    d['independence_attested']=True;result=review.import_review(snap,d)
    ref=next(iter(result['references'].values()));ref['samples'][0]['near_transition']=False
    with pytest.raises(ValueError,match='stratum'):review.validate_binding(ref,'independent')


def test_render_checks_sources_and_copies_only_actual_model_inputs(tmp_path):
    from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import write_source
    q=question()
    for n in range(4,11):write_source(tmp_path/'data',n)
    for s in q['causal_sources']:s['sha256']=file_sha(tmp_path/'data'/s['path'])
    q['input_sha256']=digest(q['input_sources']);seal(q);snap=snapshot([q]);out=tmp_path/'public'
    result=review.render(snap,tmp_path/'data',out)
    assert result['cards']==1 and len(list(out.glob('*.jpg')))==2
    assert 'rule_target' not in (out/'packet.json').read_text()
    assert json.loads((out/'decisions.json').read_text())['decisions'][0]['reference'] is None
    (tmp_path/'data'/q['input_sources'][0]['path']).write_bytes(b'wrong')
    with pytest.raises(ValueError,match='RGB source'):review.render(snap,tmp_path/'data',tmp_path/'changed')
    assert not (tmp_path/'changed').exists()


def test_per_answer_routes_cannot_be_filled_by_unknown_or_other_answer(monkeypatch):
    reg=approved_registry(monkeypatch);audit=reg['rules'][0]['independent']
    # Keep 100 YES judgments on one route and 100 NO judgments on 20 routes.
    for i,s in enumerate(audit['samples'][:100]):
        q=question(number=100+i,route='Town01_Rep0_0_0_route0');q['split']='val';seal(q)
        s.update({k:q[k] for k in ('question_id','rule_class','rule_target','input_sha256','scenario','route_id','physical_group')})
        s.update(question=q,teacher_record_sha256=digest(q))
    bind_fixture(reg);result=approval.validate_registry(reg)
    assert not result['approved']
    measured=next(iter(result['decisions'].values()))['measured']
    assert measured['physical_routes']==20 and measured['answers']['YES']['physical_routes']==1


def test_gap_growth_from_reversing_ego_is_not_lead_release():
    fs=frames(release=True)
    for f in fs:f['actors'][1].update(speed=0.,ego_velocity=[0.,0.])
    vals,_=rules.facts(fs,Episode('U-E1','e',longitudinal='HOLD'),2)
    assert vals['release_ready'] is False


@pytest.mark.parametrize('direction,expected',[((1.,0.),True),((-1.,0.),False),((0.,1.),False),((1.,1.),False)])
def test_execution_receipt_requires_forward_motion(direction,expected):
    fs=frames()
    for i,f in enumerate(fs):
        f['meta']['ego_matrix'][0][3]=direction[0]*i/4
        f['meta']['ego_matrix'][1][3]=direction[1]*i/4
        f['meta']['speed']=1.
    assert rules.execution_motion(fs) is expected


def test_turning_motion_not_used_to_confirm_straight_execution():
    fs=frames();fs[-1]['meta'].update(speed=1.,ego_matrix=[[0.,-1.,0.,1.],[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    assert rules.execution_motion(fs) is False


@pytest.mark.parametrize('bad',[0,1,'False',[],{}])
def test_non_boolean_facts_cannot_become_no(bad):
    with pytest.raises(ValueError,match='bool'):rules.answer(['x'],dict(x=bad))


def test_completed_actor_can_start_new_instance_after_visible_clear_interval():
    completed={('U-E1',2):dict(clear_observations=0,last_frame=3)};seen={('U-E1',2)}
    for f in frames()[:3]:replay.rearm_completed(completed,seen,[],f)
    assert not seen and not completed


@pytest.mark.parametrize('case',['missing','invisible','gap','still_conflicting'])
def test_unresolved_actor_cannot_rearm(case):
    completed={('U-E1',2):dict(clear_observations=0,last_frame=3)};seen={('U-E1',2)}
    for f in frames()[:3]:
        seeds=[]
        if case=='missing':f['actors'].pop()
        if case=='invisible':f['image_evidence']['2']['quality_pass']=False
        if case=='gap':f['frame_id']+=10
        if case=='still_conflicting':seeds=[dict(event='U-E1',actor_id=2)]
        replay.rearm_completed(completed,seen,seeds,f)
    assert seen


def test_selector_reports_scope_gaps_and_freezes_deterministic_sample(tmp_path,monkeypatch):
    pool,_,_,_=make_bundle(tmp_path)
    monkeypatch.setattr(candidate_pool,'validate',lambda pool:pool)
    monkeypatch.setattr(dataset,'groups',lambda:set())
    monkeypatch.setattr(dataset,'holdout_reservations',lambda:{})
    monkeypatch.setattr(dataset,'producer_check_reservations',lambda:{})
    monkeypatch.setattr(dataset,'split_for',lambda *a,**kw:'train')
    a=review.select(pool,tmp_path/'index.json',per_class=2,per_route=1)
    assert a==review.select(pool,tmp_path/'index.json',per_class=2,per_route=1)
    assert len(a['questions'])==1
    assert next(iter(a['statistics']['classes'].values()))['sample_route_deficit']==19
    assert review.select(pool,tmp_path/'index.json',purpose='independent')['questions']==[]


def test_imported_references_assemble_through_real_approval_path(monkeypatch):
    reg=approved_registry(monkeypatch);record=reg['rules'][0];cls=record['rule_class']
    dev=dict(policy=review.POLICY,teacher=rules.identity(),references={cls:record['development']})
    independent=dict(policy=review.POLICY,teacher=rules.identity(),references={cls:record['independent']})
    registry,report=review.assemble_registry(dev,independent,['synthetic_author'])
    assert cls in report['approval']['approved']
    with pytest.raises(ValueError,match='independent'):review.assemble_registry(dev,independent,['synthetic_other'])
