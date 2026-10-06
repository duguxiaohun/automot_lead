from copy import deepcopy
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT,contract,digest
from qwen3vl_local.sft_new_loop_phase4.branch_support import report
from qwen3vl_local.sft_new_loop_phase4 import paired_eval as paired,launch


def row(branch='default',returning=True,phase='readiness',edge='depart'):
    return dict(id='x',episode=dict(event='U-E2',branch=branch,return_required=returning),edge=edge,target='YES',slice=phase,physical_group='p',scenario='S',route_id='R',split='val',label_basis='reviewed_transition_band',observation=dict(frame_id=10,history_frames=[6,10],speed_mps=0.),images=['a','b'],image_sha256=['a'*64,'b'*64],image_rgb_sha256=['c'*64,'d'*64],model_input_sha256='e'*64,prompt_sha256='f'*64)


def result(r=None,prediction='YES'):
    r=r or row();return dict(id=r['id'],event=r['episode']['event'],edge=r['edge'],slice=r['slice'],target=r['target'],prediction=prediction,raw=prediction,reference_kind=r.get('reference_kind','reviewed_rgb'),paired_identity=paired.case_identity(r))


def test_zero_branches_and_readiness_are_not_hidden():
    r=row(phase='catchup');data=dict(train=[r])
    out=report(data)
    assert out['branches']['U-E2/default/no_return']['training_questions']==0
    cells=out['branches']['U-E2/default/return']['edges']
    assert cells['depart']['train']['readiness']['YES']['questions']==0
    assert cells['depart']['train']['catchup']['YES']['questions']==1
    assert cells['return']['train']['readiness']['YES']['questions']==0
    assert 'U-E4/cyclist_bypass/no_return' in out['branches']
    assert 'U-E4/cyclist_follow/return' in out['branches']
    assert 'U-E2' in out['missing_evaluation_events']['test']
    assert out['blocking'] is False


def test_return_free_branch_cannot_fill_returning_branch():
    out=report(dict(train=[row(returning=False)]))
    assert out['branches']['U-E2/default/no_return']['edges']['depart']['train']['readiness']['YES']['questions']==1
    assert out['branches']['U-E2/default/return']['edges']['depart']['train']['readiness']['YES']['questions']==0
    assert 'return' not in out['branches']['U-E2/default/no_return']['edges']


@pytest.mark.parametrize('field',['target','image_sha256','image_rgb_sha256','model_input_sha256','prompt_sha256','observation','episode','reference_kind'])
def test_paired_input_truth_or_reference_change_rejected(field):
    a=result();b=deepcopy(a);fields=b['paired_identity']['fields']
    if field=='target':fields[field]=b['target']='NO'
    elif field=='reference_kind':fields[field]=b[field]='rule_teacher'
    elif field=='observation':fields[field]['frame_id']=11
    elif field=='episode':fields[field]['branch']='in_lane_pass'
    elif isinstance(fields[field],list):fields[field][0]='0'*64
    else:fields[field]='0'*64
    b['paired_identity']['sha256']=digest(fields)
    with pytest.raises(ValueError):paired.compare({'x':a},{'x':b})


def test_paired_malformed_counted_wrong_teacher_separate():
    a=result(prediction='MALFORMED');b=result();r=row();r['id']='teacher';r['reference_kind']='rule_teacher'
    out=paired.compare({'x':a,'teacher':result(r)},{'x':b,'teacher':result(r,'NO')})
    assert out['reviewed_accuracy']['all']['delta']==1.
    assert out['teacher_consistency']['all']['delta']==-1.
    with pytest.raises(ValueError):paired.compare({'x':a},{})
    with pytest.raises(ValueError):paired.compare({}, {})


def test_paired_missing_and_duplicate_identity_rejected(tmp_path):
    p=tmp_path/'cases.jsonl';r=result();p.write_text(json.dumps(r)+'\n'+json.dumps(r)+'\n')
    with pytest.raises(ValueError,match='duplicate'):paired.load(p)
    r.pop('paired_identity');p.write_text(json.dumps(r)+'\n')
    with pytest.raises(ValueError,match='identity'):paired.load(p)


def test_launch_missing_paths_reports_both_without_loading_model(tmp_path):
    out=launch.check(tmp_path/'data',tmp_path/'model',tmp_path/'rgb')
    assert not out['configuration_valid'] and len(out['errors'])==3
    assert any('MODEL_DIR' in e for e in out['errors']) and any('DATASET' in e for e in out['errors'])
    assert out['downloads_performed'] is False


def test_launch_contract_rejects_old_data(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(contract={})))
    assert not launch.check(tmp_path)['configuration_valid']
    (tmp_path/'manifest.json').write_text(json.dumps(dict(contract=contract())))
    for name in ('train','val','test'):(tmp_path/(name+'.jsonl')).touch()
    assert launch.check(tmp_path)['configuration_valid']


def test_launch_cli_cannot_override_preflight_paths(tmp_path):
    env=dict(os.environ,PYTHON=sys.executable,DATASET=str(tmp_path/'missing'))
    r=subprocess.run(['bash',str(ROOT/'train.sh'),'check','--dataset=/tmp/another'],env=env,capture_output=True,text=True)
    assert r.returncode==2 and 'environment variables' in r.stderr


def test_pipeline_teacher_registry_required_before_inventory(tmp_path):
    env=dict(os.environ,PRODUCTION_INDEX='/missing/index.json',PYTHON='/missing/python');env.pop('TEACHER_REGISTRY',None)
    r=subprocess.run(['bash',str(ROOT/'run_full_pipeline.sh')],env=env,capture_output=True,text=True)
    assert r.returncode!=0 and 'source-bound teacher registry' in r.stderr


def test_seven_confirmed_risks_registered_without_withdrawn_candidates():
    v=json.loads((ROOT/'thirty_third_audit_exposure_20261006.json').read_text())
    assert len(v['calibration_risks'])==7 and len(v['train_only_groups'])==12
    assert all(len(r['rgb_evidence'])==2 and not r['training_label_approved'] for r in v['calibration_risks'])
    assert not any('Scenario3_27' in r['route_id'] for r in v['calibration_risks'])
