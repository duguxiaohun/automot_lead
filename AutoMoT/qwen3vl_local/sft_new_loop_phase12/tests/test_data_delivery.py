import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from PIL import Image
import pytest
from qwen3vl_local.sft_new_loop_phase12 import data, prompts, evaluate, audit, finish, validation, CONTRACT, UPDATES
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof


def parents(tmp_path):
    rgb=tmp_path/'rgb';rows=[]
    facts={k:k=='RS1' for k in prompts.P1.ANSWER_KEYS}
    for rep,split in ((0,'train'),(1,'test')):
        route=f'Town01_Rep{rep}_123_route0_01_01_01_01_01';paths=[f's/{route}/rgb/{i:04}.jpg' for i in range(4)]
        for path in paths:
            p=rgb/path;p.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(16,16),'red').save(p)
        rows.append(dict(scenario='s',route_id=route,frame_id=3,split=split,rs='R1',answers=facts,
                         history_rgb_paths=paths,latest_rgb_path=paths[-1]))
    p1=tmp_path/'p1.jsonl';p1.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    second={**rows[0],'true_rs':'R1','question_domain':'ROAD_CORRIDOR','target_event_class':'UE1'}
    second['answers']={k:k in ('UE1','DOMAIN_ROAD_CORRIDOR') for k in prompts.P2.ANSWER_KEYS}
    second['answers'].update(DOMAIN_ROAD_CORRIDOR=True,DOMAIN_LOCAL_JUNCTION=False)
    invalid={**second,'question_domain':'LOCAL_JUNCTION','target_event_class':'INVALID',
        'answers':{k:k in ('INVALID_EVENT_CONTEXT','DOMAIN_LOCAL_JUNCTION') for k in prompts.P2.ANSWER_KEYS}}
    invalid['answers'].update(DOMAIN_ROAD_CORRIDOR=False,DOMAIN_LOCAL_JUNCTION=True)
    p2=tmp_path/'p2.jsonl';p2.write_text(json.dumps(second)+'\n'+json.dumps(invalid)+'\n')
    return p1,p2,rgb,rows


def test_join_protects_other_repetition_holdout_and_keeps_invalid_slot(tmp_path):
    p1,p2,rgb,rows=parents(tmp_path);before=p1.read_bytes(),p2.read_bytes()
    report=data.build(p1,p2,rgb,tmp_path/'joined')
    result=[json.loads(l) for l in (tmp_path/'joined/frame_index.jsonl').read_text().splitlines()]
    assert len(result)==2 and {r['split'] for r in result}=={'test'}
    assert result[0]['phase1_answers']==result[1]['phase1_answers']==rows[0]['answers']
    assert result[1]['answers']['INVALID_EVENT_CONTEXT'] is True
    assert report['split_transitions']=={'train->test':2}
    assert (p1.read_bytes(),p2.read_bytes())==before
    with pytest.raises(FileExistsError):data.build(p1,p2,rgb,tmp_path/'joined')


@pytest.mark.parametrize('problem',['missing','rs','history','boolean'])
def test_join_refuses_unreviewed_mismatch(tmp_path,problem):
    p1,p2,rgb,rows=parents(tmp_path)
    if problem=='missing':rows=rows[1:]
    elif problem=='rs':rows[0]['rs']='R2'
    elif problem=='history':rows[0]['history_rgb_paths'].reverse()
    else:rows[0]['answers']['VULNERABLE']='False'
    p1.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):data.build(p1,p2,rgb,tmp_path/'joined')


def evidence(root):
    facts={k:k=='RS1' for k in prompts.P1.ANSWER_KEYS}
    second={k:k in ('UE1','DOMAIN_ROAD_CORRIDOR') for k in prompts.P2.ANSWER_KEYS}
    second.update(DOMAIN_ROAD_CORRIDOR=True,DOMAIN_LOCAL_JUNCTION=False)
    key=['s','route',3];paths=[f's/route/rgb/{i:04}.jpg' for i in range(4)]
    spec2=prompts.P2.make_prompt_spec(variant='all_random_order',answers=second,seed_key='fixture')
    first=prompts.P1.make_prompt_spec(variant='all_random_order',answers=facts,seed_key='fixture')
    item=dict(key=key,domain='ROAD_CORRIDOR',answers=second,seed_key='fixture',output_keys=list(spec2.output_keys),balance_key='UE1')
    proof.write(root/'cpu/evaluation_members.json',{'val':[item],'test':[item]})
    proof.write(root/'cpu/fusion_labels.json',[dict(key=key,answers=facts,paths=paths)])
    refs={str(p):[{**item,'answers':facts if p==1 else second,'paths':paths,'original_asked_keys':list(first.output_keys if p==1 else spec2.output_keys)}] for p in (1,2)}
    proof.write(root/'cpu/reference_members.json',refs)
    proof.write(root/'cpu/guard_members.json',refs['1'])
    proof.write(root/'cpu/guard_coverage.json',{'coverage_adequate':True})
    text1,text2=prompts.targets(facts,first,spec2)
    for folder,phases in [('fusion_test',(1,2)),('original_phase1_test',(1,)),('original_phase2_test',(2,))]:
        out=root/'evaluation/4rgb'/folder;out.mkdir(parents=True)
        case=dict(case_index=0,key=key,balance_key='UE1',scenario='s',rgb=paths)
        for p,spec,raw in ((1,first,text1),(2,spec2,text2)):
            if p in phases:case[f'phase{p}']=dict(generated_tokens=1,token_ids=[p],stop_token_id=1,reached_token_limit=False,text=raw,gt=facts if p==1 else prompts.P2.spec_answers(spec),parsed=evaluate._parse(raw,spec,p))
        if folder=='fusion_test':case['performance']=dict(full_answer_equivalence_checked=True,independent_branches=[dict(case['phase1']),dict(case['phase2'])],paired_benchmark={'trials':[{'shared_seconds':1.,'independent_seconds':2.}]*2})
        (out/'cases.jsonl').write_text(json.dumps(case)+'\n')
        result=evaluate.metrics([case],phases)
        if folder=='fusion_test':result['performance']={'paired_benchmark':evaluate.benchmark_summary([case])}
        proof.write(out/'metrics.json',result)


    for step in [*range(1000,UPDATES,1000),UPDATES]:
        folder=root/'runs/4rgb'/f'val_step_{step}'
        for sub,source in [('', 'fusion_test'),('weak_guard','original_phase1_test')]:
            dest=folder/sub;dest.mkdir(parents=True)
            for filename in ('cases.jsonl','metrics.json'):
                (dest/filename).write_bytes((root/'evaluation/4rgb'/source/filename).read_bytes())
        review=validation.decision(folder,None if step==1000 else root/'runs/4rgb/val_step_1000',{'coverage_adequate':True})
        proof.write(folder/'selection_review.json',dict(**review,promoted=step==1000))
        if step==1000:proof.write(root/'runs/4rgb/selection.json',dict(step=step,score=review['score'],adapter='best_step_1000'))


def test_audit_reparses_text_and_rejects_missing_or_relabelled_case(tmp_path):
    evidence(tmp_path);assert audit.verify_evaluations(tmp_path,'4rgb')['evaluations']==25
    file=tmp_path/'evaluation/4rgb/fusion_test/cases.jsonl';original=file.read_text();case=json.loads(original)
    case['phase1']['text']='RS1: NO'
    file.write_text(json.dumps(case)+'\n')
    with pytest.raises(ValueError):audit.verify_evaluations(tmp_path,'4rgb')
    file.write_text(original+original)
    with pytest.raises(ValueError):audit.verify_evaluations(tmp_path,'4rgb')


def test_real_package_source_closure_is_standalone_and_tamper_rejected(tmp_path,monkeypatch):
    evidence(tmp_path);proof.write(tmp_path/'ready.json',{'contract':CONTRACT});proof.write(tmp_path/'execution.json',{})
    (tmp_path/'runs/4rgb').mkdir(parents=True,exist_ok=True)
    monkeypatch.setattr(finish.integrity,'verify_ready',lambda root:{'version':'explicit_fixture_no_model'})
    monkeypatch.setattr(finish.integrity,'verify_training',lambda *a:{'selected':'fixture_adapter'})
    archive=finish.package(tmp_path,'4rgb');assert archive.stat().st_size<30*1024**2
    bundle=tmp_path/'4rgb_audit_bundle'
    command='import sys;sys.path.insert(0,sys.argv[1]);from qwen3vl_local.sft_new_loop_phase12.audit import verify_bundle;print(verify_bundle(sys.argv[2]))'
    result=subprocess.run([sys.executable,'-I','-c',command,str(bundle/'code'),str(bundle)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    target=bundle/'evaluation/4rgb/fusion_test/cases.jsonl';target.write_text(target.read_text()+'\n')
    with pytest.raises(ValueError):audit.verify_bundle(bundle)
