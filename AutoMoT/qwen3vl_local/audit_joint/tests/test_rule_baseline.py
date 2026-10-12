from copy import deepcopy
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
from qwen3vl_local.audit_joint import rule_baseline as rb
from qwen3vl_local.audit_joint.training_budget import COMPONENTS
from qwen3vl_local.sft_new_loop_phase4.tests.test_production_selection_v50 import pool


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


def test_automatic_launcher_wires_one_epoch_reload_and_fixed_test_without_latest(tmp_path):
    from qwen3vl_local.sft_new_loop_phase4.identity import ROOT
    run=tmp_path/'qwen3vl_local/sft_new_loop_phase4/rgb_short/run.sh';run.parent.mkdir(parents=True)
    run.write_text('#!/bin/sh\nprintf "TRAIN %s\\n" "$*" >> "$CALL_LOG"\n')
    guard=tmp_path/'qwen3vl_local/audit_joint/run_guarded.sh';guard.parent.mkdir(parents=True)
    guard.write_text('#!/bin/sh\nprintf "GUARD %s\\n" "$*" >> "$CALL_LOG"\n')
    stub=tmp_path/'python';log=tmp_path/'calls'
    stub.write_text('#!/bin/sh\nprintf "CHECK %s\\n" "$*" >> "$CALL_LOG"\n');stub.chmod(0o755)
    env=dict(os.environ,PYTHON=str(stub),CALL_LOG=str(log),DATASET='/view',TEACHER_REFERENCES='/refs',
             FIRST_EPOCH_BUDGET='/budget',OUTPUT_DIR=str(tmp_path/'run'),RGB_MODE='4',RUN_TAG='fixture')
    result=subprocess.run(['bash',str(ROOT/'rgb_short/run_automatic.sh')],cwd=tmp_path,env=env,capture_output=True)
    assert result.returncode==0,result.stderr
    calls=log.read_text().splitlines();assert len(calls)==3
    assert '--selection-mode rule_consistency' in calls[1] and '--epochs 1' in calls[1]
    assert '--epoch-samples 10240' in calls[1]
    assert 'run_fixture_rgb4/epoch_000 --rgb-mode 4 --split test' in calls[2]
    assert 'latest' not in calls[2] and '--min-free-gib 20' in calls[2]
