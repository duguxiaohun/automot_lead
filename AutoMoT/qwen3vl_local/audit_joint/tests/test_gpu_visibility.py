"""Actual shell launchers + actual selector; only hardware and training are faked."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
ENTRIES=('audit_joint/train_phase3_batch.sh','audit_joint/benchmark_phase4_batch.sh',
         'sft_new_loop_phase4/rgb_short/run.sh')


def launch(tmp_path,entry,rows,mask,pin=''):
    binary=tmp_path/'bin';binary.mkdir()
    capture=tmp_path/'calls.jsonl'
    smi=binary/'nvidia-smi'
    smi.write_text(f'#!{sys.executable}\nprint({rows!r})\n');smi.chmod(0o755)
    python=binary/'python'
    python.write_text(f'#!{sys.executable}\n'+f'''
import json,os,sys
if len(sys.argv)>1 and sys.argv[1]=='-c':
    os.execv({sys.executable!r},[{sys.executable!r},*sys.argv[1:]])
with open(os.environ['CAPTURE'],'a') as f:
    f.write(json.dumps(dict(args=sys.argv[1:],gpu_ids=os.environ.get('GPU_IDS'),mask=os.environ.get('CUDA_VISIBLE_DEVICES')))+'\\n')
''');python.chmod(0o755)
    generic=tmp_path/'qwen3vl_local/sft_new_loop_phase3/train.sh'
    generic.parent.mkdir(parents=True);shutil.copyfile(ROOT/'sft_new_loop_phase3/train.sh',generic)
    guard=tmp_path/'qwen3vl_local/audit_joint/run_guarded.sh';guard.parent.mkdir(parents=True)
    guard.write_text('#!/usr/bin/env bash\nset -eu\nshift 5\nexec "$@"\n')
    env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],PYTHONPATH=str(ROOT.parent),
        PYTHON=str(python),CAPTURE=str(capture),GPU_IDS=pin,INDEX='index.jsonl',DATASET='view',
        OUTPUT_DIR=str(tmp_path/'output'),RUN_TAG='check',PROMPT_VARIANT='baseline',
        BATCH_SIZE='2',GRAD_ACCUM='2',GRADIENT_CHECKPOINTING='1',ACTION_OUTPUT_MODE='choice',
        HISTORY_RGB_MODE='4rgb',ANCHOR_STRIDE='1')
    if mask is None:env.pop('CUDA_VISIBLE_DEVICES',None)
    else:env['CUDA_VISIBLE_DEVICES']=mask
    args=['sampling'] if entry.endswith('train_phase3_batch.sh') else []
    result=subprocess.run(['bash',str(ROOT/entry),*args],cwd=tmp_path,env=env,capture_output=True,text=True)
    calls=[json.loads(s) for s in capture.read_text().splitlines()] if capture.exists() else []
    return result,calls


@pytest.mark.parametrize('entry',ENTRIES)
@pytest.mark.parametrize('mask,busy',[('7',True),(None,True),('7',False),('',False)])
def test_busy_or_insufficient_visible_cards_never_launch(tmp_path,entry,mask,busy):
    rows='\n'.join(f'{i}, GPU-{i}, {70000 if busy else 0}, {99 if busy else 0}' for i in range(8))
    result,calls=launch(tmp_path,entry,rows,mask)
    assert result.returncode!=0 and calls==[]
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('entry',ENTRIES)
@pytest.mark.parametrize('mask',['4,5,6,7','GPU-4,GPU-5,GPU-6,GPU-7'])
def test_auto_selection_only_uses_idle_visible_cards(tmp_path,entry,mask):
    rows='\n'.join(f'{i}, GPU-{i}, {i*100}, 0' for i in range(8))
    result,calls=launch(tmp_path,entry,rows,mask)
    assert result.returncode==0,result.stderr
    if entry.endswith('benchmark_phase4_batch.sh'):
        args=calls[-1]['args'];assert args[args.index('--gpus')+1]=='4,5,6,7'
    elif entry.endswith('train_phase3_batch.sh'):
        assert calls[-1]['gpu_ids']=='4,5,6,7'
    else:
        train=next(c for c in calls if 'torch.distributed.run' in c['args'])
        assert train['mask']=='4,5,6,7'


@pytest.mark.parametrize('entry',ENTRIES)
def test_explicit_pin_remains_an_intentional_override(tmp_path,entry):
    result,calls=launch(tmp_path,entry,'invalid query must not be used','7',pin='0,1,2,3')
    assert result.returncode==0,result.stderr
    assert calls


@pytest.mark.parametrize('entry',ENTRIES)
def test_auto_selection_excludes_busy_card_even_if_more_memory_is_free(tmp_path,entry):
    rows='0, GPU-0, 0, 99\n1, GPU-1, 4096, 0\n'+ '\n'.join(f'{i}, GPU-{i}, 100, 0' for i in range(2,6))
    result,calls=launch(tmp_path,entry,rows,None)
    assert result.returncode==0,result.stderr
    if entry.endswith('benchmark_phase4_batch.sh'):
        args=calls[-1]['args'];assert args[args.index('--gpus')+1]=='2,3,4,5'
    elif entry.endswith('train_phase3_batch.sh'):assert calls[-1]['gpu_ids']=='2,3,4,5'
    else:assert next(c for c in calls if 'torch.distributed.run' in c['args'])['mask']=='2,3,4,5'
