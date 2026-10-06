from copy import deepcopy
import importlib
import json
import os
import subprocess
import sys
import pytest
from qwen3vl_local.sft_new_loop_phase4 import identity,risk_review
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT

@pytest.mark.parametrize('option',['--datas','--data-r','--model-d','--output-d'])
@pytest.mark.parametrize('equals',[True,False])
def test_shell_rejects_abbreviated_paths_before_python(tmp_path,option,equals):
    env=dict(os.environ,PYTHON='/nonexistent/python',DATASET=str(tmp_path/'missing'))
    args=[option+'=/tmp/override'] if equals else [option,'/tmp/override']
    p=subprocess.run(['bash',str(ROOT/'train.sh'),'check',*args],env=env,capture_output=True,text=True)
    assert p.returncode==2 and 'environment variables' in p.stderr

@pytest.mark.parametrize('module,extra',[('train',['--output-dir','/tmp/test']),('preflight',[]),('launch',[])])
@pytest.mark.parametrize('option',['--datas=/tmp/override','--model-d=/tmp/override'])
def test_direct_cli_rejects_abbreviations_before_data_or_model(module,extra,option,monkeypatch,capsys):
    m=importlib.import_module('qwen3vl_local.sft_new_loop_phase4.'+module)
    monkeypatch.setattr(sys,'argv',['test','--dataset','/nonexistent/dataset',*extra,option])
    with pytest.raises(SystemExit) as e:m.main()
    assert e.value.code==2 and 'unrecognized arguments' in capsys.readouterr().err

@pytest.mark.parametrize('mode',['check','preflight','host-preflight'])
def test_pipeline_checks_never_read_stale_best_or_evaluate(tmp_path,mode):
    stub=tmp_path/'python';log=tmp_path/'calls';run=tmp_path/'run';run.mkdir()
    (run/'best.json').write_text('{"adapter":"old-adapter"}')
    stub.write_text('''#!/usr/bin/env python3
import json,os,sys
with open(os.environ['CALL_LOG'],'a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
if '-c' in sys.argv:raise SystemExit(91)
''');stub.chmod(0o755)
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}');(model/'model.safetensors').touch()
    env=dict(os.environ,PYTHON=str(stub),MODEL_DIR=str(model),CALL_LOG=str(log),MODE=mode,OUTPUT_DIR=str(run),DATA_DIR=str(tmp_path/'data'),PIPELINE_ROOT=str(tmp_path/'pipeline'),CANDIDATE_POOL='/candidate')
    for key in ('PRODUCTION_INDEX','TEACHER_REGISTRY','CONDITION_STREAM'):env.pop(key,None)
    p=subprocess.run(['bash',str(ROOT/'run_full_pipeline.sh')],env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    calls=[json.loads(s) for s in log.read_text().splitlines()]
    assert not any('-c' in c for c in calls)
    assert [c[1].split('.')[-1] for c in calls]==['full_pipeline','launch','train' if mode=='check' else 'preflight']

def test_contract_error_identifies_changed_dependency(monkeypatch):
    current=identity.contract();recorded=deepcopy(current);name='sft_new_loop_phase3/build_dataset.py'
    recorded['dependencies'][name]='0'*64
    monkeypatch.setattr(identity,'contract',lambda:current)
    with pytest.raises(ValueError,match='dependencies/sft_new_loop_phase3/build_dataset.py') as e:identity.check_contract(recorded)
    assert 'recorded='+'0'*64 in str(e.value) and current['dependencies'][name] in str(e.value)

def window():
    return dict(risk_id='x',record=dict(manual_scope='causal_history_window',rgb_evidence=[dict(frame=24),dict(frame=25)]))

def row(frames):
    return dict(scenario='s',route_id='r',label_basis='reviewed_transition_band',target='YES',observation=dict(history_frames=frames),image_sha256=['a'*64]*len(frames))

def test_manual_window_checks_rgb_gap_and_preserves_later_frames():
    known={('s','r'):[window()]}
    with pytest.raises(ValueError,match='per-frame admission'):risk_review.validate_rows([row([22,26])],known)
    out=risk_review.validate_rows([row([43,47])],known)
    assert out['row_dispositions']=={'manual_outside_registered_windows':1}

@pytest.mark.parametrize('fault',['legacy','no_localization'])
def test_window_does_not_relax_legacy_or_unlocalized_risks(fault):
    r=window()
    if fault=='legacy':r['record'].pop('manual_scope')
    else:r['record']['rgb_evidence']=[]
    with pytest.raises(ValueError):risk_review.validate_rows([row([43,47])],{('s','r'):[r]})

def test_teacher_checks_full_causal_history_not_only_model_rgb():
    r=row([26,30]);r.update(label_basis='weak_rule_teacher',teacher_provenance=dict(question=dict(causal_sources=[dict(frame_id=n) for n in range(24,31)])))
    with pytest.raises(ValueError,match='teacher risk window'):risk_review.validate_rows([r],{('s','r'):[window()]})

@pytest.mark.parametrize('mode',[2,4])
def test_real_existing_manual_questions_outside_new_window_survive(tmp_path,mode):
    from qwen3vl_local.sft_new_loop_phase4.dataset import build,load_dataset
    risk=next(r for r in json.loads((ROOT/'thirty_fourth_audit_exposure_20261006.json').read_text())['calibration_risks'] if r['scenario']=='DynamicObjectCrossing')
    ann=[r for r in json.loads((ROOT/'reviewed_state_pairs_v9.json').read_text()) if r['route_id']==risk['route_id'] and r.get('transition_band',{}).get('review_start') in (47,48)]
    out=tmp_path/'data';build(ann,ROOT.parents[1]/'lead_data',out,rgb_mode=mode);data,_=load_dataset(out)
    assert {r['observation']['frame_id'] for r in data['train']}=={47,48}
    assert all(r['target']=='NO' and r.get('risk_review') is None for r in data['train'])

def test_exposure_excludes_selected_unviewed_validation_group():
    a=json.loads((ROOT/'thirty_fourth_audit_exposure_20261006.json').read_text())
    assert len(a['train_only_groups'])==13 and len(a['calibration_risks'])==11
    assert all(r['manual_scope']=='causal_history_window' and not r['training_label_approved'] for r in a['calibration_risks'])

def test_explicit_uncertain_review_is_not_lost_outside_window():
    r=row([30,34]);known={('s','r'):[window()]}
    proof=dict(policy=risk_review.POLICY,reviewer='synthetic',evidence_id='synthetic',risk_ids=['x'],frames={str(f):dict(status='uncertain',reason='synthetic ambiguous frame',observed_until=f,rgb_sha256='a'*64,**{k:True for k in risk_review.CHECKS}) for f in (30,34)})
    r['risk_review']=risk_review.row_review(r,proof,known)
    assert r['risk_review']['disposition']=='uncertain'
    with pytest.raises(ValueError,match='cannot supply'):risk_review.validate_rows([r],known)
    r['target']='UNKNOWN'
    assert risk_review.validate_rows([r],known)['row_dispositions']=={'uncertain':1}

@pytest.mark.parametrize('mode',[2,4])
def test_review_for_four_images_remains_valid_and_uncertain_for_two(tmp_path,mode):
    run=tmp_path/'s'/'r'/'rgb';run.mkdir(parents=True)
    proof=dict(policy=risk_review.POLICY,reviewer='synthetic',evidence_id='synthetic',risk_ids=['x'],frames={})
    for f in (24,26,28,30):
        p=run/f'{f:04d}.jpg';p.write_bytes(str(f).encode())
        proof['frames'][str(f)]=dict(status='uncertain',reason='synthetic ambiguity',observed_until=f,rgb_sha256=identity.file_sha(p),**{k:True for k in risk_review.CHECKS})
    ann=dict(scenario='s',route_id='r',start=30,end=30,label_basis='reviewed_transition_band',risk_review=proof)
    known={('s','r'):[window()]}
    assert risk_review.annotation_review(ann,tmp_path,mode,known)==proof
    frames=[26,30] if mode==2 else [24,26,28,30]
    r=row(frames);r['image_sha256']=[proof['frames'][str(f)]['rgb_sha256'] for f in frames]
    assert risk_review.row_review(r,proof,known)['disposition']=='uncertain'
