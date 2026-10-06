"""Exercise the user's shell chain without GPUs; model quality is not mocked as evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from . import pipeline_support as support
from .prompt_candidate import CANDIDATE_NAME


@pytest.fixture
def shell_env(tmp_path):
    root = tmp_path/'AutoMoT'
    package = root/'qwen3vl_local/sft_new_loop_phase3'
    package.mkdir(parents=True)
    package.joinpath('run_full_pipeline.sh').write_bytes(Path(__file__).with_name('run_full_pipeline.sh').read_bytes())
    for name in ('train.sh', 'eval.sh'):
        package.joinpath(name).write_text('''#!/bin/bash
python fake_stage "$0" "$@"
''')
    bin_dir = tmp_path/'bin'; bin_dir.mkdir()
    bin_dir.joinpath('python').write_text(f'#!{sys.executable}\n'+r'''
import json,os,pathlib,sys
args=sys.argv[1:]
with open(os.environ['CALLS'],'a') as f: f.write(json.dumps(dict(args=args,env=dict(os.environ)))+'\n')
if args[0].endswith('build_dataset.py'):
    path=pathlib.Path(args[args.index('--output-dir')+1]);path.mkdir(parents=True)
    (path/'frame_index.jsonl').write_text('{}\n')
if args[0]=='fake_stage' and args[1].endswith('train.sh') and args[2]!='sampling':
    pathlib.Path(os.environ['OUTPUT_DIR']).mkdir(parents=True)
if args[:2]==['-m','qwen3vl_local.sft_new_loop_phase3.pipeline_support'] and args[2]=='adapter':
    print(os.environ['RUN_ROOT'])
''')
    bin_dir.joinpath('python').chmod(0o755)
    env = {'PATH':str(bin_dir)+os.pathsep+os.environ['PATH'], 'CALLS':str(tmp_path/'calls'),
           'TIMESTAMP':'fixed', 'HOME':os.environ['HOME']}
    return root, env


def run_shell(root, env, command):
    return subprocess.run(['bash','-c',command],cwd=root,env=env,capture_output=True,text=True)


def test_exact_four_group_chain_builds_once_and_propagates_candidate(shell_env):
    root, env = shell_env
    script='bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh'
    command=(f'ACTION_OUTPUT_MODE=binary {script} && '
        f'SKIP_BUILD=1 HISTORY_RGB_MODE=2rgb_endpoints ACTION_OUTPUT_MODE=binary {script} && '
        f'SKIP_BUILD=1 ACTION_OUTPUT_MODE=choice {script} && '
        f'SKIP_BUILD=1 HISTORY_RGB_MODE=2rgb_endpoints ACTION_OUTPUT_MODE=choice {script}')
    result=run_shell(root,env,command)
    assert result.returncode==0, result.stdout+result.stderr
    calls=[json.loads(line) for line in Path(env['CALLS']).read_text().splitlines()]
    builds=[c for c in calls if c['args'][0].endswith('build_dataset.py')]
    assert len(builds)==1 and '--development-groups-extra' in builds[0]['args']
    trains=[c for c in calls if c['args'][0]=='fake_stage' and c['args'][1].endswith('train.sh') and c['args'][2]=='ddp']
    assert [(c['env']['HISTORY_RGB_MODE'],c['env']['ACTION_OUTPUT_MODE']) for c in trains]==[(r,o) for r,o,_ in support.GROUPS]
    assert len({c['env']['OUTPUT_DIR'] for c in trains})==4
    assert len({c['env']['INDEX'] for c in trains})==1
    assert all(c['env']['PROMPT_VARIANT']==CANDIDATE_NAME for c in calls)
    evals=[c for c in calls if c['args'][0]=='fake_stage' and c['args'][1].endswith('eval.sh')]
    assert len(evals)==4
    assert all(c['env']['SPLIT']=='test' and c['env']['CASES_PER_BIN']=='0' for c in evals)
    assert len([c for c in calls if c['args'][-1]=='finish'])==4
    # Fixed stamp repeats cannot append over existing results.
    assert run_shell(root,env,f'ACTION_OUTPUT_MODE=binary {script}').returncode!=0


@pytest.mark.parametrize('override', ['SKIP_BUILD=1', 'MAX_STEPS=3', 'MAX_EVAL_FRAMES=1', 'SPLIT=val', 'PROMPT_VARIANT=bogus SKIP_BUILD=1'])
def test_bad_chain_inputs_stop_before_training(shell_env, override):
    root, env=shell_env
    result=run_shell(root,env,override+' bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh')
    assert result.returncode!=0
    assert not Path(env['CALLS']).exists()


@pytest.mark.parametrize('field,value', [('history_rgb_mode','2rgb_endpoints'), ('action_output_mode','choice'), ('prompt_variant','baseline')])
def test_wrong_adapter_rejected_before_eval(tmp_path, field, value):
    cfg=dict(history_rgb_mode='4rgb',action_output_mode='binary',prompt_variant=CANDIDATE_NAME,prompt_name=CANDIDATE_NAME)
    cfg[field]=value
    (tmp_path/'sft_new_loop_phase3_adapter_config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError): support.adapter_contract(tmp_path,'4rgb','binary',CANDIDATE_NAME)


@pytest.mark.parametrize('exit_code,report,expected', [(0,True,True),(2,False,False),(1,None,None),(2,None,None)])
def test_only_completed_quality_failure_can_continue(tmp_path, monkeypatch, exit_code, report, expected):
    monkeypatch.setattr(support,'replay_gpu',lambda env:'0')
    def run(argv,**kwargs):
        if 'qwen3vl_local.sft_new_loop_phase3.regression_gate' in argv:
            if report is not None: (tmp_path/'regression_gate.json').write_text(json.dumps({'candidate_gate_passed':report}))
            return SimpleNamespace(returncode=exit_code)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(support.subprocess,'run',run)
    args=(tmp_path/'cases',tmp_path/'adapter',tmp_path,CANDIDATE_NAME,tmp_path/'model',tmp_path)
    if expected is None:
        with pytest.raises((RuntimeError,FileNotFoundError)): support.replay_and_gate(*args)
    else:
        assert support.replay_and_gate(*args) is expected


def test_binding_detects_dataset_and_source_mutation(tmp_path, monkeypatch):
    index=tmp_path/'frame_index.jsonl';index.write_text('one')
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    monkeypatch.setattr(support,'source_identity',lambda:{'code':'one'})
    state={'data_identity':dict(index_sha256=support.sha(index),manifest_sha256=support.sha(manifest)), 'source_sha256':{'code':'one'}}
    support.assert_binding(index,state)
    manifest.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='index/manifest'): support.assert_binding(index,state)
    manifest.write_text('{}')
    monkeypatch.setattr(support,'source_identity',lambda:{'code':'two'})
    with pytest.raises(ValueError,match='source'): support.assert_binding(index,state)
