"""Exercise launched argument flow, including stale environment rejection, without GPUs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]


def launch(tmp_path,script,env,args=()):
    binary=tmp_path/'bin';binary.mkdir()
    capture=tmp_path/'calls.jsonl'
    fake=binary/'python'
    fake.write_text(f'#!{sys.executable}\nimport os,sys,json\n'
                    "with open(os.environ['CAPTURE'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n")
    fake.chmod(0o755)
    target=tmp_path/'qwen3vl_local/sft_new_loop_phase3/train.sh'
    target.parent.mkdir(parents=True)
    shutil.copyfile(ROOT/'sft_new_loop_phase3/train.sh',target)
    for path in ('sft_new_loop_phase4/rgb_short/run.sh','audit_joint/run_guarded.sh'):
        stub=tmp_path/'qwen3vl_local'/path;stub.parent.mkdir(parents=True,exist_ok=True)
        stub.write_text('#!/usr/bin/env bash\nexec python "$@"\n')
    clean={k:v for k,v in os.environ.items() if k not in (
        'BATCH_SIZE','GRAD_ACCUM','GRADIENT_CHECKPOINTING','HISTORY_RGB_MODE','ANCHOR_STRIDE',
        'WARMUP_STEPS','EVAL_STEPS','GENERATION_EVAL_STEPS','SAVE_STEPS','PROMPT_VARIANT','PYTHON')}
    entry=ROOT/script if '/' in script else ROOT/'audit_joint'/script
    result=subprocess.run(['bash',str(entry),*args],cwd=tmp_path,
        env={**clean,'PATH':str(binary)+os.pathsep+os.environ['PATH'],'CAPTURE':str(capture),'GPU_IDS':'0,1,2,3',**env},
        text=True,capture_output=True)
    calls=[json.loads(s) for s in capture.read_text().splitlines()] if capture.exists() else []
    return result,calls


@pytest.mark.parametrize('mode,stride',[('2rgb_short','2'),('4rgb','1')])
def test_phase3_profile_reaches_real_launcher_arguments(tmp_path,mode,stride):
    result,calls=launch(tmp_path,'train_phase3_batch.sh',
                        {'INDEX':'index with spaces.jsonl','HISTORY_RGB_MODE':mode},['sampling'])
    assert result.returncode==0,result.stderr
    args=calls[-1]
    for flag,value in {'--batch-size':'2','--grad-accum':'2','--anchor-stride':stride,
                       '--warmup-steps':'500','--eval-steps':'1000','--generation-eval-steps':'1000',
                       '--save-steps':'10000','--index':'index with spaces.jsonl',
                       '--prompt-variant':'v23_rgb_stage_candidate_20261006'}.items():
        assert args[args.index(flag)+1]==value
    assert '--gradient-checkpointing' in args and '--sampling-only' in args


@pytest.mark.parametrize('env',[{'INDEX':''},{'GPU_IDS':'0,1,2'}, {'GPU_IDS':'0,1,1,3'},
    {'BATCH_SIZE':'8'}, {'GRAD_ACCUM':'1'}, {'GRADIENT_CHECKPOINTING':'0'},
    {'HISTORY_RGB_MODE':'2rgb_endpoints'}, {'HISTORY_RGB_MODE':'4rgb','ANCHOR_STRIDE':'2'},
    {'ACTION_OUTPUT_MODE':'binary'}])
def test_phase3_profile_rejects_stale_incompatible_environment(tmp_path,env):
    result,calls=launch(tmp_path,'train_phase3_batch.sh',{'INDEX':'index.jsonl',**env},['sampling'])
    assert result.returncode!=0 and calls==[]


def test_phase3_profile_refuses_existing_run(tmp_path):
    output=tmp_path/'out';(output/'run_fixed_4rgb').mkdir(parents=True)
    result,calls=launch(tmp_path,'train_phase3_batch.sh',
        {'INDEX':'index.jsonl','RUN_TAG':'fixed','OUTPUT_DIR':str(output)},['sampling'])
    assert result.returncode!=0 and 'Refusing existing run' in result.stderr and calls==[]


def test_generic_phase3_launcher_defaults_to_ratio_not_fixed_steps(tmp_path):
    result,calls=launch(tmp_path,'sft_new_loop_phase3/train.sh',{'INDEX':'index.jsonl'},['sampling'])
    assert result.returncode==0,result.stderr
    args=calls[-1]
    assert '--warmup-steps' not in args
    assert args[args.index('--warmup-ratio')+1]=='0.1'


def test_phase4_profile_preserves_formal_update_budget(tmp_path):
    result,calls=launch(tmp_path,'benchmark_phase4_batch.sh',{'DATASET':'current view','OUTPUT_DIR':'new result','GPU_IDS':'0,1,2,3'})
    assert result.returncode==0,result.stderr
    args=calls[-1]
    for flag,value in {'--phase':'4','--dataset':'current view','--update-samples':'32',
                       '--samples-per-event':'32','--timing':'end_to_end','--seed':'20260929',
                       '--output':'new result'}.items():
        assert args[args.index(flag)+1]==value
    assert args[args.index('--batches')+1:args.index('--samples-per-event')]==['1','2','4']
    assert '--gradient-checkpointing' in args


@pytest.mark.parametrize('batch,accumulation',[('1','8'),('2','4'),('4','2'),('8','1')])
def test_phase4_formal_entry_keeps_update_samples_when_batch_changes(tmp_path,batch,accumulation):
    result,calls=launch(tmp_path,'sft_new_loop_phase4/rgb_short/run_automatic.sh',
        {'DATASET':'view','TEACHER_REFERENCES':'refs','FIRST_EPOCH_BUDGET':'budget',
         'OUTPUT_DIR':str(tmp_path/'result'),'BATCH_SIZE':batch,'UPDATE_SAMPLES':'32','ACCUMULATION':accumulation})
    assert result.returncode==0,result.stderr
    args=next(row for row in calls if '--selection-mode' in row)
    assert args[args.index('--batch-size')+1]==batch
    assert args[args.index('--accumulation')+1]==accumulation
    assert any('qwen3vl_local.audit_joint.rule_baseline' in row for row in calls)
    assert any('qwen3vl_local.sft_new_loop_phase4.rgb_short.evaluate' in row for row in calls)


@pytest.mark.parametrize('env',[{'BATCH_SIZE':'3'},{'UPDATE_SAMPLES':'0'},
                              {'BATCH_SIZE':'2','ACCUMULATION':'8'}])
def test_phase4_formal_entry_rejects_inconsistent_update_budget(tmp_path,env):
    result,calls=launch(tmp_path,'sft_new_loop_phase4/rgb_short/run_automatic.sh',
        {'DATASET':'view','TEACHER_REFERENCES':'refs','FIRST_EPOCH_BUDGET':'budget',
         'OUTPUT_DIR':str(tmp_path/'result'),**env})
    assert result.returncode!=0 and calls==[] and not (tmp_path/'result').exists()


def test_phase3_recovery_flags_reach_real_launcher(tmp_path):
    result,calls=launch(tmp_path,'train_phase3_batch.sh',{'INDEX':'index.jsonl','RESUME':'old state with spaces',
        'STOP_AFTER_EPOCH':'2','NUM_EPOCHS':'3'},['sampling'])
    assert result.returncode==0,result.stderr
    args=calls[-1]
    assert '--save-training-state' in args
    for flag,value in [('--resume','old state with spaces'),('--stop-after-epoch','2'),('--num-epochs','3')]:
        assert args[args.index(flag)+1]==value
