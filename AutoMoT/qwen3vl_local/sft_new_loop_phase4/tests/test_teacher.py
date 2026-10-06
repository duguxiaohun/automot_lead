"""Synthetic approval cases test mechanics; they are never real certifications."""
from copy import deepcopy
from collections import Counter
import json
import math
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_rules as t,teacher_replay as r,teacher_approval as a,teacher_data as d
from qwen3vl_local.sft_new_loop_phase4 import dataset,candidate_pool
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT,digest,file_sha,write_json
from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import snapshot,actor


def frames(kind='lead',release=False):
    out=[]
    for i in range(7):
        ac=actor(position=[8.+(.3*i if release else 0),0.,0.],speed=2. if release else 0.,ego_velocity=[2. if release else 0.,0.])
        if kind=='vru':ac=actor(**{'class':'walker'},position=[10.,-3.-.15*i if release else 0.,0.],extent=[.3,.3,.9],speed=1.)
        f=snapshot(i+4,[ac]);f['meta'].update(speed=0.,distance_to_next_junction=math.inf,route=[[0.,0.],[5.,0.],[10.,0.],[15.,0.],[20.,0.]],changed_route=False)
        f['sources']=[dict(path=f"{f['scenario']}/{f['route_id']}/{k}/{i+4:04d}.{ext}",kind=k,frame_id=i+4,sha256=digest([k,i])) for k,ext in [('metas','pkl'),('bboxes','pkl'),('rgb','jpg')]]
        f['image_evidence']['2'].update(crops=[dict(usable=True,bounds=[100+i*3,100,140+i*3,150])])
        out.append(f)
    return out


def question(target='YES',event='U-E1',number=10,route='Town01_Rep0_1_0_route0'):
    scenario='Scenario';ep=dict(event=event,instance_id='synthetic',state='YIELD',longitudinal='HOLD')
    sources=[dict(path=f'{scenario}/{route}/{k}/{f:04d}.{ext}',kind=k,frame_id=f,sha256=digest([route,f,k])) for f in range(number-6,number+1) for k,ext in [('metas','pkl'),('bboxes','pkl'),('rgb','jpg')]]
    inputs=[s for s in sources if s['kind']=='rgb' and s['frame_id'] in [number-4,number]]
    vals=dict(release_ready=target=='YES',corridor_clear=True,priority_satisfied=True)
    q=dict(teacher_sha256=digest(t.identity()),scenario=scenario,route_id=route,physical_group=dataset.source_route(dict(scenario=scenario,route_id=route)),split='train',frame_id=number,
        episode=ep,edge='proceed',rgb_mode=2,history_frames=[number-4,number],input_sources=inputs,input_sha256=digest(inputs),causal_sources=sources,
        ego_speed=0.,ego_motion='stationary',near_transition=True,facts=vals,rule_target=target,state_rule_target=target,slice='readiness',rule_class=t.rule_class(event,'proceed',2),reasons=[],instance=dict(actor_id=2))
    return seal(q)


def seal(q):
    q['question_id']=digest({k:v for k,v in q.items() if k!='question_id'});return q


def registry():return dict(policy=a.POLICY,criteria=a.CRITERIA,teacher=t.identity(),rules=[])


@pytest.mark.parametrize('kind,event',[('lead','U-E1'),('vru','U-E4')])
def test_teacher_has_positive_and_negative_readiness_paths(kind,event):
    ep=Episode(event,'actual',longitudinal='HOLD')
    for releasing,expected in [(False,'NO'),(True,'YES')]:
        values,_=t.facts(frames(kind,releasing),ep,2)
        assert t.answer(('release_ready','corridor_clear','priority_satisfied'),values)==expected


@pytest.mark.parametrize('distance',[None,float('nan'),-float('inf'),-1.])
def test_unknown_or_near_junction_is_not_local_priority(distance):
    fs=frames(release=True);fs[-1]['meta']['distance_to_next_junction']=distance
    vals,_=t.facts(fs,Episode('U-E1','e',longitudinal='HOLD'),2)
    assert vals['priority_satisfied'] is None
    assert t.answer(('release_ready','corridor_clear','priority_satisfied'),vals)=='UNKNOWN'


@pytest.mark.parametrize('case',['dark','missing','jump','invalid','short','gap'])
def test_bad_evidence_never_releases(case):
    fs=frames(release=True)
    if case=='dark':fs[-1]['image_evidence']['2']['quality_pass']=False
    if case=='missing':fs[-1]['actors'].pop()
    if case=='jump':fs[-1]['actors'][1]['position'][0]=70.
    if case=='invalid':fs[-1]['source_geometry_issues']=['negative_extent']
    if case=='short':fs.pop(0)
    if case=='gap':fs[1]['frame_id']=99
    vals,_=t.facts(fs,Episode('U-E1','e',longitudinal='HOLD'),2)
    assert t.answer(('release_ready','corridor_clear','priority_satisfied'),vals)=='UNKNOWN'


def test_local_unknown_static_occupancy_cannot_be_empty():
    fs=frames('vru',True)
    fs[-1]['actors'].append(actor(3,**{'class':'static'},position=[9.,0.,0.],visible_pixels=0))
    vals,_=t.facts(fs,Episode('U-E4','e',longitudinal='HOLD'),2)
    assert vals['corridor_clear'] is None
    fs[-1]['image_evidence']['3']=dict(quality_pass=True)
    vals,_=t.facts(fs,Episode('U-E4','e',longitudinal='HOLD'),2)
    assert vals['corridor_clear'] is False


def test_identity_hints_do_not_create_an_event_without_local_conflict():
    fs=frames();fs[-1]['actors'][1]['role_name']='scenario';fs[-1]['meta']['vehicle_affecting_id']=2
    assert t.seeds(fs)[0]['identity_sources']==['role_name','affecting_id']
    fs[-1]['actors'][1]['position'][1]=9
    assert t.seeds(fs)==[]


def test_three_valued_conjunction():
    assert t.conjunction(True,None) is None
    assert t.conjunction(False,None) is False
    assert t.conjunction(True,True) is True


def test_replay_questions_follow_real_state_and_motion_does_not_make_readiness(monkeypatch):
    fs=frames();all_frames={f['frame_id']:f for f in fs}
    for n in range(11,16):
        f=deepcopy(fs[-1]);f['frame_id']=n
        for s in f['sources']:
            s['frame_id']=n;s['path']=s['path'].replace('0010',f'{n:04d}')
        all_frames[n]=f
    monkeypatch.setattr(r.g,'load_frame',lambda root,scenario,route,frame:all_frames[frame])
    monkeypatch.setattr(t,'execution_motion',lambda fs:fs[-1]['frame_id']>=12)
    def facts(fs,ep,ident):
        n=fs[-1]['frame_id'];clear=n>=11
        return dict(event_restriction_observed=not clear,release_ready=False if ep.longitudinal=='FOLLOW' else clear,corridor_clear=clear,priority_satisfied=True,
            restriction_present=not clear,stationary_wait_required=not clear,restricted_progress_established=n>=13,
            stable_progress_established=ep.longitudinal=='FOLLOW',event_resolved=n>=13),[]
    monkeypatch.setattr(t,'facts',facts)
    route=dict(scenario=fs[0]['scenario'],route_id=fs[0]['route_id'],physical_group='test',split='train',rgb_frames=list(all_frames))
    rows=list(r.route_records(None,route));qs=[q for row in rows for q in row['questions'] if q['rgb_mode']==2]
    assert [(q['frame_id'],q['rule_target']) for q in qs if q['edge']=='proceed']==[(10,'NO'),(11,'YES'),(12,'YES')]
    assert next(q for q in qs if q['frame_id']==12 and q['edge']=='proceed')['slice']=='readiness'
    assert any(q['edge']=='recover_follow' and q['rule_target']=='YES' for q in qs)
    assert any(q['edge']=='complete' and q['rule_target']=='YES' for q in qs)
    assert rows[-2]['traces'][0]['finished']


@pytest.mark.parametrize('change',['future','facts','class','mode','target','catchup','input','path'])
def test_question_tampering_rejected_even_after_rehash(change):
    q=question()
    if change=='future':q['causal_sources'][-1]['frame_id']=11
    if change=='facts':q['facts']['release_ready']=False
    if change=='class':q['rule_class']=q['rule_class'].replace('U-E1','U-E4')
    if change=='mode':q['rgb_mode']=4
    if change=='target':q['rule_target']='NO'
    if change=='catchup':q['slice']='catchup';q['rule_class']=t.rule_class('U-E1','proceed',2,'catchup')
    if change=='input':q['input_sources'][0]['sha256']='wrong'
    if change=='path':q['causal_sources'][0]['path']='../secret'
    with pytest.raises(ValueError):r.validate_question(seal(q))


def test_empty_approval_defers_binary_rules_and_stale_source_rejected():
    approved=a.validate_registry(registry())
    assert a.target(question(),approved)==('UNKNOWN','rule_class_not_approved')
    bad=registry();bad['teacher']['version']='old'
    with pytest.raises(ValueError):a.validate_registry(bad)


def test_precision_uses_unknown_reference_as_error_and_requires_both_answers():
    rows=[dict(rule_target='YES',reference='YES',physical_group=str(i%20),near_transition=True) for i in range(100)]
    assert a.report(rows)['answers']['YES']['precision_lower_95']>.95
    rows[0]['reference']='UNKNOWN'
    assert a.report(rows)['answers']['YES']['correct']==99
    assert a.report(rows)['answers']['NO']['precision_lower_95']==0
    assert a.wilson_lower(0,0)==0


def approved_registry(monkeypatch):
    # Entirely synthetic population and manual judgments. No real registry file.
    monkeypatch.setattr(dataset,'groups',lambda:set())
    monkeypatch.setattr(dataset,'holdout_reservations',lambda:{})
    monkeypatch.setattr(dataset,'split_for',lambda group,*args,**kw:'train' if 'Town99' in group else 'val')
    references=[];ids=set()
    for i in range(200):
        q=question('YES' if i<100 else 'NO',number=10+i//20,route=f'Town01_Rep0_{i%20}_0_route0')
        q['split']='val';seal(q);ids.add(q['physical_group'])
        references.append(dict(**{k:q[k] for k in ('question_id','rule_class','rule_target','input_sha256','scenario','route_id','physical_group')},
            question=q,teacher_record_sha256=digest(q),reference=q['rule_target'],near_transition=True,rgb_review_evidence_id='synthetic_only',event_identity_confirmed=True,state_confirmed=True,current_input_resolves_answer=True))
    monkeypatch.setattr(dataset,'producer_check_reservations',lambda:{g:'val' for g in ids})
    dev=deepcopy(references[0]);q=question(route='Town99_Rep0_1_0_route0')
    dev.update({k:q[k] for k in ('question_id','rule_class','rule_target','input_sha256','scenario','route_id','physical_group')});dev.update(question=q,teacher_record_sha256=digest(q))
    reg=registry();reg['rules']=[dict(rule_class=q['rule_class'],teacher=t.identity(),rule_authors=['synthetic_author'],
        development=dict(source='development_rgb_review',samples=[dev]),independent=dict(source='independent_rgb_review',reviewer='synthetic_other',
        independence_attested=True,blind_to_rule_answers=True,frozen_before_review=True,samples=references))]
    bind_fixture(reg)
    return reg


def bind_fixture(reg):
    from qwen3vl_local.sft_new_loop_phase4 import teacher_review as review
    for record in reg['rules']:
        for purpose in ('development','independent'):
            ref=record[purpose];qs=[s['question'] for s in ref['samples']]
            ref['sample_plan']=review.binding(qs,purpose,review.public_packet(dict(purpose=purpose,questions=qs))['sha256'],'synthetic_population')


def test_approval_opens_only_measured_class(monkeypatch):
    reg=approved_registry(monkeypatch);approval=a.validate_registry(reg)
    assert a.target(question(),approval)[0]=='YES'
    assert a.target(question(event='U-E4'),approval)[0]=='UNKNOWN'


@pytest.mark.parametrize('case',['author','unblind','unfrozen','duplicates','wrong_input','unmeasured','no_negative','few_routes'])
def test_unreliable_or_nonindependent_approval_rejected(monkeypatch,case):
    reg=approved_registry(monkeypatch);audit=reg['rules'][0]['independent']
    if case=='author':audit['reviewer']='synthetic_author'
    if case=='unblind':audit['blind_to_rule_answers']=False
    if case=='unfrozen':audit['frozen_before_review']=False
    if case=='duplicates':audit['samples'].append(deepcopy(audit['samples'][0]))
    if case=='wrong_input':audit['samples'][0]['input_sha256']='wrong'
    if case=='unmeasured':audit['samples'][0]['reference']='UNKNOWN'
    if case=='no_negative':audit['samples']=audit['samples'][:100]
    if case=='few_routes':audit['samples']=[s for s in audit['samples'] if s['route_id']=='Town01_Rep0_0_0_route0']
    if case in ('no_negative','few_routes'):bind_fixture(reg)
    if case in ('unmeasured','no_negative','few_routes'):
        assert not a.validate_registry(reg)['approved'] if case!='unmeasured' else bool(a.validate_registry(reg)['decisions'])
    else:
        with pytest.raises(ValueError):a.validate_registry(reg)


def test_teacher_agreement_is_not_human_accuracy():
    from qwen3vl_local.sft_new_loop_phase4.evaluate import metrics
    row=dict(event='U-E1',edge='proceed',slice='readiness',target='YES',prediction='YES',reference_kind='rule_teacher')
    out=metrics([row,dict(row,reference_kind='reviewed_rgb',prediction='NO')])
    assert out['accuracy']==0 and out['count']==1
    assert out['teacher_consistency']['agreement']==1 and out['total_count']==2
    assert metrics([row])['event_macro_accuracy_observed'] is None


def make_bundle(tmp_path):
    q=question();route={k:q[k] for k in ('scenario','route_id','physical_group','split')};route['rgb_frames']=[10,11]
    pool=dict(sha256='synthetic',routes=[route]);header=dict(policy=r.POLICY,teacher=t.identity(),candidate_pool_sha256=pool['sha256'])
    path=tmp_path/'route.jsonl'
    rows=[dict(kind='header',**header),dict(kind='frame_disposition',**{k:q[k] for k in ('scenario','route_id','physical_group','split')},frame_id=10,disposition='questions',questions=[q],reasons=[]),
          dict(kind='frame_disposition',**{k:q[k] for k in ('scenario','route_id','physical_group','split')},frame_id=11,disposition='abstained',questions=[],reasons=['unsupported'])]
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    receipt=r.inspect_route(path,route,header)
    index=dict(**header,routes=[receipt]);index['sha256']=digest(index);write_json(tmp_path/'index.json',index)
    return pool,rows,path,header


def test_accounting_complete_is_calculated_and_does_not_mean_approved(tmp_path):
    pool,_,_,_=make_bundle(tmp_path)
    out=r.accounting(pool,tmp_path/'index.json')
    assert out['complete_causal_production'] and out['processed_frames']==2
    assert a.validate_registry(registry())['approved']=={}
    pool['routes'].append(dict(pool['routes'][0],route_id='unprocessed'))
    assert not r.accounting(pool,tmp_path/'index.json')['complete_causal_production']


@pytest.mark.parametrize('case',['missing','duplicate','foreign','no_reason','tamper'])
def test_ledger_rejects_incomplete_or_modified_artifacts(tmp_path,case):
    pool,rows,path,header=make_bundle(tmp_path)
    if case=='missing':rows.pop()
    if case=='duplicate':rows.append(rows[-1])
    if case=='foreign':rows[-1]['route_id']='elsewhere'
    if case=='no_reason':rows[-1]['reasons']=[]
    if case=='tamper':rows[-1]['reasons']=['modified_after_receipt']
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError):r.accounting(pool,tmp_path/'index.json')


@pytest.mark.parametrize('weak',[False,True])
def test_approved_teacher_compiles_builds_and_loads_with_distinct_provenance(tmp_path,monkeypatch,weak):
    from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import write_source
    for n in range(4,12):write_source(tmp_path/'data',n)
    import lzma,pickle
    for path in (tmp_path/'data').rglob('metas/*.pkl'):
        with lzma.open(path,'rb') as stream:meta=pickle.load(stream)
        meta['speed']=0.
        with lzma.open(path,'wb') as stream:pickle.dump(meta,stream)
    monkeypatch.setattr(dataset,'groups',lambda:set())
    monkeypatch.setattr(dataset,'holdout_reservations',lambda:{})
    monkeypatch.setattr(dataset,'producer_check_reservations',lambda:{})
    monkeypatch.setattr(dataset,'split_for',lambda *args,**kw:'train')
    pool=candidate_pool.scan(tmp_path/'data',tmp_path/'pool.json')
    q=question()
    for s in q['causal_sources']:s['sha256']=file_sha(tmp_path/'data'/s['path'])
    q['input_sha256']=digest(q['input_sources']);seal(q)
    bundle=tmp_path/'bundle';bundle.mkdir();path=bundle/'route.jsonl'
    header=dict(policy=r.POLICY,teacher=t.identity(),candidate_pool_sha256=pool['sha256'])
    route=pool['routes'][0]
    items=[dict(kind='header',**header)]
    for n in route['rgb_frames']:
        items.append(dict(kind='frame_disposition',**{k:route[k] for k in ('scenario','route_id','physical_group','split')},frame_id=n,
                          disposition='questions' if n==10 else 'abstained',questions=[q] if n==10 else [],reasons=['synthetic']))
    path.write_text(''.join(json.dumps(x)+'\n' for x in items))
    index=dict(**header,routes=[r.inspect_route(path,route,header)]);index['sha256']=digest(index);write_json(bundle/'index.json',index)
    reg=registry()
    # The real approval path is tested separately above with 200 independent
    # synthetic judgments; this test isolates ingestion and actual file checks.
    approved=dict(approved={q['rule_class']:'synthetic_approval'},registry_sha256=digest(reg),decisions={})
    if weak:
        reg=a.weak_registry([q['rule_class']],'synthetic_ingestion');approved=a.validate_registry(reg)
    else:
        monkeypatch.setattr(a,'validate_registry',lambda registry:approved)
        monkeypatch.setattr(d,'validate_registry',lambda registry:approved)
    anns,report=d.compile_bundle(pool,bundle/'index.json',reg,rgb_mode=2)
    assert len(anns)==1 and 'reviewer' not in anns[0]
    output=tmp_path/'dataset'
    dataset.build(anns,tmp_path/'data',output,rgb_mode=2,candidate_pool=pool,teacher_registry=reg,production_index=bundle/'index.json')
    loaded,manifest=dataset.load_dataset(output)
    row=loaded['train'][0]
    assert row['target']=='YES' and row['reference_kind']=='rule_teacher'
    assert row['label_basis']==('weak_rule_teacher' if weak else 'approved_rule_teacher')
    assert manifest['production_coverage']['complete_causal_production']
    if weak:
        from qwen3vl_local.sft_new_loop_phase4.preflight import inspect
        assert not inspect(output)['formal_data_ready']
        assert manifest['training_admission']['weak_supervision_rows']==1
    bad=deepcopy(row);bad['target']='NO'
    with pytest.raises(ValueError):d.validate_row(bad,approved)
    with pytest.raises(ValueError):d.validate_annotation(anns[0],approved,tmp_path/'data',4)
    (tmp_path/'data'/q['causal_sources'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed'):d.validate_annotation(anns[0],approved,tmp_path/'data',2)


def test_accounted_all_unknown_is_not_formally_ready(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4 import preflight
    monkeypatch.setattr(preflight,'load_dataset',lambda path: (dict(train=[],val=[],test=[]),dict(trainable=False,training_admission={},
        complete_coverage=True,coverage={},risk_review_check={},production_coverage=dict(complete_causal_production=True,approved_rule_classes=[]))))
    from qwen3vl_local.sft_new_loop_phase4 import admission
    monkeypatch.setattr(admission,'review_capacity_report',lambda data:dict(numerical_floor_met=True))
    assert not preflight.inspect(tmp_path)['formal_data_ready']


def test_vru_stopped_outside_margin_does_not_become_unknown_again():
    fs=frames('vru',True)
    for f in fs:f['actors'][1].update(position=[10.,-4.,0.],speed=0.)
    vals,_=t.facts(fs,Episode('U-E4','e',longitudinal='HOLD'),2)
    assert vals['release_ready'] is True
    for f in fs:f['actors'][1]['position'][1]=-2.6
    vals,_=t.facts(fs,Episode('U-E4','e',longitudinal='HOLD'),2)
    assert vals['release_ready'] is None


def test_only_closest_stopped_vehicle_is_seeded_as_lead():
    fs=frames()
    for f in fs:
        f['actors'].append(actor(3,position=[15.,0.,0.],speed=0.))
        f['image_evidence']['3']=dict(quality_pass=True)
    assert [s['actor_id'] for s in t.seeds(fs)]==[2]
    fs[-1]['meta']['scenario_obstacles_ids']=[2]
    assert t.seeds(fs)==[]  # The nearer obstacle occludes lead identity; do not seed the queue behind it.


def test_approval_does_not_count_same_observable_question_twice(monkeypatch):
    reg=approved_registry(monkeypatch);sample=deepcopy(reg['rules'][0]['independent']['samples'][0])
    sample['question']['episode']['instance_id']='different_hidden_id';seal(sample['question'])
    sample.update(question_id=sample['question']['question_id'],teacher_record_sha256=digest(sample['question']))
    reg['rules'][0]['independent']['samples'].append(sample)
    bind_fixture(reg)
    with pytest.raises(ValueError,match='observable'):a.validate_registry(reg)


def test_conflicting_instances_are_excluded_before_compilation(tmp_path,monkeypatch):
    pool,rows,path,header=make_bundle(tmp_path)
    q=rows[1]['questions'][0];other=deepcopy(q)
    other['episode']['instance_id']='another_participant';other['rule_target']=other['state_rule_target']='NO';other['facts']['release_ready']=False;seal(other)
    rows[1]['questions'].append(other)
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    index=dict(**header,routes=[r.inspect_route(path,pool['routes'][0],header)]);index['sha256']=digest(index);write_json(tmp_path/'index.json',index)
    approval=dict(approved={q['rule_class']:'synthetic'},registry_sha256='synthetic',decisions={})
    monkeypatch.setattr(d,'validate_registry',lambda _:approval)
    annotations,report=d.compile_bundle(pool,tmp_path/'index.json',{},rgb_mode=2)
    assert not annotations and report['counts']['conflicting_observable_questions_excluded']==2


def test_old_question_cannot_be_rebound_to_new_rule_sources():
    q=question();q['teacher_sha256']='old_source_digest';seal(q)
    with pytest.raises(ValueError,match='stale teacher'):r.validate_question(q)
