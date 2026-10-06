from collections import Counter
from copy import deepcopy
import json
import os
import subprocess
import pytest
from qwen3vl_local.sft_new_loop_phase4.sampling import plan,coverage_schedule,CapacityError
from qwen3vl_local.sft_new_loop_phase4.full_sampling import validate_dataset_scope
from qwen3vl_local.sft_new_loop_phase4.tests.test_fourteenth_audit_fixes import row


def pool():
    return [row(i,e,frame=1) for e,n in [('U-E1',37),('U-E2',2),('U-E4',1)] for i in range(n)]


@pytest.mark.parametrize('world',[1,3,4,8])
def test_every_question_equal_events_and_equal_ddp_lengths(world):
    rows=pool();indices,audit=plan(rows,epoch=0,policy='full_event_equal',world_size=world,cap=1,max_question_repeat=1)
    assert set(indices)==set(range(len(rows)))
    assert len(set(Counter(rows[i]['episode']['event'] for i in indices).values()))==1
    assert len({len(indices[r::world]) for r in range(world)})==1
    assert sum(len(indices[r::world]) for r in range(world))==len(indices)
    assert audit['max_question_repeat']>=37 and audit['cumulative_missing_questions']==0
    assert audit['unique_questions']==len(rows)


def test_full_plan_stable_under_input_order_and_mode_and_resumes():
    rows=pool();other=list(reversed(deepcopy(rows)))
    for r in other:r['model_input_sha256']='rgb4-'+r['id']
    history=None;history4=None;orders=[]
    for epoch in range(3):
        a,report=plan(rows,epoch=epoch,policy='full_event_equal',history=history)
        b,r4=plan(other,epoch=epoch,policy='full_event_equal',history=history4)
        assert [rows[i]['id'] for i in a]==[other[i]['id'] for i in b]
        orders.append(a);history=report['next_history'];history4=r4['next_history']
    assert orders[0]!=orders[1]
    replay,r=plan(rows,epoch=2,policy='full_event_equal')
    assert replay==orders[2] and r['next_history']==history
    assert coverage_schedule(rows,epochs=7,policy='full_event_equal')['complete']


@pytest.mark.parametrize('fault',['seed','world','target','count','next_epoch'])
def test_resume_mismatch_rejected(fault):
    rows=pool();_,a=plan(rows,epoch=0,policy='full_event_equal');h=a['next_history'];kwargs={}
    if fault=='seed':kwargs['seed']=5
    if fault=='world':kwargs['world_size']=4
    if fault=='target':rows[0]['target']='NO' if rows[0]['target']=='YES' else 'YES'
    if fault=='count':h['counts'][rows[0]['id']]=0
    if fault=='next_epoch':h['next_epoch']=3
    with pytest.raises(ValueError):plan(rows,epoch=1,policy='full_event_equal',history=h,**kwargs)


@pytest.mark.parametrize('budget',[10,110,112,0,-1])
def test_full_budget_never_silently_truncates(budget):
    with pytest.raises(CapacityError):plan(pool(),epoch=0,policy='full_event_equal',budget=budget)


@pytest.mark.parametrize('fault',['capped','incomplete','pair_drops'])
def test_full_scope_rejects_route_caps_or_pairing_loss(fault):
    m=dict(production_index_sha256='x',teacher_selection=dict(max_per_route=0,route_budget_excluded=0),production_coverage=dict(complete_causal_production=True))
    pairing=dict(pairs=10,original_train_counts={'2':10,'4':10})
    validate_dataset_scope(m,pairing)
    if fault=='capped':m['teacher_selection']['max_per_route']=100
    if fault=='incomplete':m['production_coverage']['complete_causal_production']=False
    if fault=='pair_drops':pairing['pairs']=9
    with pytest.raises(ValueError):validate_dataset_scope(m,pairing)


def test_zero_route_limit_compiles_every_admissible_candidate(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4 import teacher_data as d,teacher_review,dataset,risk_review
    from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import question
    q=question();questions=[]
    for i in range(137):
        x=deepcopy(q);x['frame_id']=i;x['question_id']=str(i);questions.append(x)
    (tmp_path/'route.jsonl').write_text(json.dumps({})+'\n'+json.dumps(dict(questions=questions))+'\n')
    (tmp_path/'index.json').write_text(json.dumps(dict(routes=[dict(artifact='route.jsonl')])))
    monkeypatch.setattr(d,'accounting',lambda *a:{})
    monkeypatch.setattr(d,'validate_registry',lambda *a:dict(approved={q['rule_class']:'fixture'},registry_sha256='fixture'))
    monkeypatch.setattr(d,'target',lambda *a:('YES','fixture'))
    monkeypatch.setattr(teacher_review,'observable',lambda q:q['frame_id'])
    monkeypatch.setattr(dataset,'holdout_reservations',lambda:set())
    monkeypatch.setattr(risk_review,'registry',lambda:{})
    capped,_=d.compile_bundle({},tmp_path/'index.json',{},rgb_mode=2,max_per_route=100)
    full,report=d.compile_bundle({},tmp_path/'index.json',{},rgb_mode=2,max_per_route=0)
    assert len(capped)==100 and len(full)==137
    assert report['counts']['route_budget_excluded']==0 and report['max_per_route']==0


@pytest.mark.parametrize('rgb',[2,4])
def test_full_shell_wires_policy_peer_and_accumulation(tmp_path,rgb):
    from qwen3vl_local.sft_new_loop_phase4.identity import ROOT
    fake=tmp_path/'python';log=tmp_path/'calls'
    fake.write_text('#!/usr/bin/env python3\nimport json,sys\nwith open('+repr(str(log))+',"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n');fake.chmod(0o755)
    r=subprocess.run(['bash',str(ROOT/'train_full.sh'),str(rgb),'host-preflight','--accumulation','16'],env=dict(os.environ,PYTHON=str(fake)),capture_output=True)
    assert r.returncode==0,r.stderr
    args=json.loads(log.read_text().splitlines()[-1])
    assert args[args.index('--sampling-policy')+1]=='full_event_equal'
    assert args[args.index('--paired-with')+1].endswith('data'+str(6-rgb))
    assert args[args.index('--accumulation')+1]=='16'
