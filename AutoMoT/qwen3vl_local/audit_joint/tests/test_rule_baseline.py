from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
from qwen3vl_local.audit_joint import rule_baseline as rb
from qwen3vl_local.audit_joint.training_budget import COMPONENTS
from qwen3vl_local.sft_new_loop_phase4.tests.test_production_selection_v50 import pool
from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS


def test_first_epoch_measure_uses_actual_balanced_plans_and_reports_both_answers():
    data=pool('rule_teacher')
    for r in data['train']:
        r['edge']={'U-E2':'return','U-E7':'complete'}.get(r['episode']['event'],'proceed')
    evaluation=dict(train_ids=[],plans={},exclusions=[])
    report=rb.measure(data,evaluation)
    assert report['missing_critical_support']==[]
    assert report['schedule']['world1_world4_order_equal']
    assert report['selected_counts']['train']==20
    assert 'selected_ids' not in report and 'train_ids' not in report['evaluation']
    data['train']=[r for r in data['train'] if not (r['episode']['event']=='U-E7' and r['target']=='YES')]
    assert rb.measure(data,evaluation)['missing_critical_support']==['U-E7/complete/YES']


def budget():
    return dict(free_bytes=100*1024**3,reserve_bytes=20*1024**3,modes=2,retained_checkpoints_per_mode=1,
                components={k:dict(upper_bound_bytes=1,evidence='test only') for k in COMPONENTS})


def test_budget_rechecks_capacity_and_rejects_missing_or_wrong_scope(tmp_path,monkeypatch):
    monkeypatch.setattr(rb.shutil,'disk_usage',lambda _:SimpleNamespace(free=20*1024**3))
    assert rb.budget_check(budget(),tmp_path)['sufficient'] is False
    monkeypatch.setattr(rb.shutil,'disk_usage',lambda _:SimpleNamespace(free=25*1024**3))
    assert rb.budget_check(budget(),tmp_path)['sufficient'] is True
    q=budget();q['components']['optimizer_per_checkpoint']['upper_bound_bytes']=None
    assert rb.budget_check(q,tmp_path)['sufficient'] is None
    for change in (dict(modes=1),dict(retained_checkpoints_per_mode=7),dict(reserve_bytes=0)):
        with pytest.raises(ValueError):rb.budget_check(dict(budget(),**change),tmp_path)


def test_automatic_launcher_blocks_before_any_gpu_command(tmp_path):
    from qwen3vl_local.sft_new_loop_phase4.identity import ROOT
    stub=tmp_path/'python';log=tmp_path/'calls'
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\nexit 2\n');stub.chmod(0o755)
    env=dict(os.environ,PYTHON=str(stub),CALL_LOG=str(log),DATASET=str(tmp_path/'view'),
             TEACHER_REFERENCES=str(tmp_path/'refs'),FIRST_EPOCH_BUDGET=str(tmp_path/'budget'),OUTPUT_DIR=str(tmp_path/'run'))
    result=subprocess.run(['bash',str(ROOT/'rgb_short/run_automatic.sh')],env=env,capture_output=True)
    assert result.returncode==2
    calls=log.read_text().splitlines()
    assert len(calls)==1 and '-m qwen3vl_local.audit_joint.rule_baseline' in calls[0]
    assert 'torch.distributed' not in calls[0]
    result=subprocess.run(['bash',str(ROOT/'rgb_short/run_automatic.sh'),'--epoch-samples','40'],env=env,capture_output=True)
    assert result.returncode==2 and log.read_text().splitlines()==calls


@pytest.mark.parametrize('coverage',['strict','existing_data_development'])
def test_automatic_launcher_wires_one_epoch_reload_and_fixed_test_without_latest(tmp_path,coverage):
    from qwen3vl_local.sft_new_loop_phase4.identity import ROOT
    run=tmp_path/'qwen3vl_local/sft_new_loop_phase4/rgb_short/run.sh';run.parent.mkdir(parents=True)
    run.write_text('#!/bin/sh\nprintf "TRAIN %s\\n" "$*" >> "$CALL_LOG"\n')
    guard=tmp_path/'qwen3vl_local/audit_joint/run_guarded.sh';guard.parent.mkdir(parents=True)
    guard.write_text('#!/bin/sh\nprintf "GUARD %s\\n" "$*" >> "$CALL_LOG"\n')
    stub=tmp_path/'python';log=tmp_path/'calls'
    stub.write_text('#!/bin/sh\nprintf "CHECK %s\\n" "$*" >> "$CALL_LOG"\n');stub.chmod(0o755)
    env=dict(os.environ,PYTHON=str(stub),CALL_LOG=str(log),DATASET='/view',TEACHER_REFERENCES='/refs',
             FIRST_EPOCH_BUDGET='/budget',OUTPUT_DIR=str(tmp_path/'run'),RGB_MODE='4',RUN_TAG='fixture',
             AUTOMATIC_COVERAGE_POLICY=coverage)
    result=subprocess.run(['bash',str(ROOT/'rgb_short/run_automatic.sh')],cwd=tmp_path,env=env,capture_output=True)
    assert result.returncode==0,result.stderr
    calls=log.read_text().splitlines();assert len(calls)==3
    assert '--coverage-policy '+coverage in calls[0]
    assert '--selection-mode rule_consistency' in calls[1] and '--epochs 1' in calls[1]
    assert '--epoch-samples 10240' in calls[1]
    assert 'run_fixture_rgb4/epoch_000 --rgb-mode 4 --split test' in calls[2]
    assert 'latest' not in calls[2] and '--min-free-gib 20' in calls[2]


def diagnostic_pool(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import data
    from qwen3vl_local.sft_new_loop_phase4 import full_sampling
    raw=pool('rule_teacher')
    for row in raw['train']:
        row['edge']={'U-E2':'return','U-E7':'complete'}.get(row['episode']['event'],'proceed')
    raw['train']=[r for r in raw['train'] if not (r['episode']['event']=='U-E5' and r['target']=='YES')]
    monkeypatch.setattr(rb,'dataset_bindings',lambda _:dict(index='fixture'))
    monkeypatch.setattr(data,'load_dataset',lambda *a,**kw:(deepcopy(raw),{}))
    monkeypatch.setattr(full_sampling,'validate_dataset_scope',lambda _:None)
    monkeypatch.setattr(rb,'budget_check',lambda *a:dict(sufficient=True))
    return raw


@pytest.mark.parametrize('references',['missing','malformed'])
def test_unavailable_references_preserve_both_source_support_reports(tmp_path,monkeypatch,references):
    diagnostic_pool(tmp_path,monkeypatch)
    refs=tmp_path/'refs'
    if references=='malformed':
        refs.mkdir()
        for name in ('references.json','val.jsonl','test.jsonl'):(refs/name).write_text('{')
    report=rb.check(tmp_path,refs,budget=budget(),space_path=tmp_path)
    assert report['status']=='blocked' and not report['joint_g0_certified']
    assert set(report['modes'])=={'2','4'}
    for mode in report['modes'].values():
        assert mode['source_support']['splits']['train']['missing_events']==[]
        assert mode['source_automatic_support']['missing_critical_support']==['U-E5/proceed/YES']
        assert 'error' in mode and 'schedule' not in mode


@pytest.mark.parametrize('event',['U-E5','R-E5'])
@pytest.mark.parametrize('defect',['manual','hold','catchup','short_manual','holdout'])
def test_critical_support_requires_native_automatic_readiness_and_exact_edge(defect,event):
    row=next(r for r in pool('rule_teacher')['train'] if r['episode']['event']==event and r['target']=='YES')
    if defect=='manual':row['label_basis']='reviewed_transition_band'
    if defect=='hold':row['edge']='hold'
    if defect=='catchup':row['slice']='catchup'
    if defect=='short_manual':row['student_reference']={'original_label_basis':'reviewed_transition_band'}
    if defect=='holdout':row['split']='val'
    result=rb.critical_support([row])
    assert result['critical_support'][event+'/proceed/YES']==dict(rows=0,physical_groups=0)


@pytest.mark.parametrize('content',[None,'{','[]'])
def test_bad_budget_still_writes_blocked_report_with_source_evidence(tmp_path,monkeypatch,content):
    diagnostic_pool(tmp_path,monkeypatch)
    budget_path=tmp_path/'budget.json'
    if content is not None:budget_path.write_text(content)
    output=tmp_path/'report.json'
    monkeypatch.setattr('sys.argv',['rule_baseline','--dataset',str(tmp_path/'data'),
        '--teacher-references',str(tmp_path/'refs'),'--budget',str(budget_path),
        '--space-path',str(tmp_path),'--output',str(output)])
    assert rb.main()==2
    report=json.loads(output.read_text())
    assert report['status']=='blocked' and set(report['modes'])=={'2','4'}
    assert any('budget input' in error for error in report['errors'])


def test_source_drift_still_blocks_with_missing_references(tmp_path,monkeypatch):
    diagnostic_pool(tmp_path,monkeypatch)
    captures=iter([dict(index='before'),dict(index='after')])
    monkeypatch.setattr(rb,'dataset_bindings',lambda _:next(captures))
    report=rb.check(tmp_path,tmp_path/'refs',budget=budget(),space_path=tmp_path)
    assert report['status']=='blocked'
    assert 'baseline inputs changed during admission' in report['errors'][-1]


def test_complete_support_and_selection_can_still_be_admitted(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import evaluation_policy
    raw=diagnostic_pool(tmp_path,monkeypatch)
    raw['train'].append(next(r for r in pool('rule_teacher')['train']
                            if r['episode']['event']=='U-E5' and r['target']=='YES'))
    monkeypatch.setattr(evaluation_policy,'pack_binding',lambda _:dict(references='fixture'))
    monkeypatch.setattr(evaluation_policy,'prepare',lambda data,**kw:(data,dict(plans={},exclusions=[])))
    report=rb.check(tmp_path,tmp_path/'refs',budget=budget(),space_path=tmp_path)
    assert report['status']=='admitted_rule_baseline' and report['errors']==[]
    assert all(not m['missing_critical_support'] for m in report['modes'].values())
    assert all(m['schedule']['worlds']['4'][0]['event_counts']==dict.fromkeys(EVENTS,1024) for m in report['modes'].values())
