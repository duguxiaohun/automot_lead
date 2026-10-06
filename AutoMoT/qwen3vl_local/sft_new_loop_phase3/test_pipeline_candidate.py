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
if args[0]=='-':
    code=sys.stdin.read()
    if 'adapter_prompt_variant' in code: print('v23_rgb_stage_candidate_20261006')
    elif 'history_rgb_mode' in code: print('4rgb')
    elif 'action_output_mode' in code: print('binary')
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
    assert all(c['env']['RUN_REGRESSION']=='0' and c['env']['RUN_AUDITS']=='0' for c in calls)
    assert not any('audit' in Path(c['args'][0]).name for c in calls)
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


@pytest.mark.parametrize('kind', ['missing', 'archive_only', 'empty_directory'])
def test_missing_history_explains_sync_and_keeps_regression_required(tmp_path, kind):
    cases=tmp_path/'old_audit_bundle/lora_production'
    archive=tmp_path/'old_audit_bundle.tar.gz'
    if kind=='archive_only': archive.write_bytes(b'archive placeholder; must never be unpacked automatically')
    if kind=='empty_directory': cases.mkdir(parents=True)
    with pytest.raises(support.HistoricalInputsMissing) as exc:
        support.validate_history(cases,CANDIDATE_NAME,tmp_path)
    message=str(exc.value)
    assert str(cases.resolve()) in message
    assert '不随 Git' in message
    assert ('请先解压' if kind=='archive_only' else 'AUDIT_ROOT') in message
    assert 'RUN_REGRESSION=0' in message and '不能视为无退化验收' in message
    if kind=='archive_only':
        assert str(archive.resolve()) in message and not cases.exists()


def test_eval_without_audits_keeps_base_lora_and_paired_test(shell_env):
    root, env=shell_env
    script=root/'qwen3vl_local/sft_new_loop_phase3/eval.sh'
    script.write_bytes(Path(__file__).with_name('eval.sh').read_bytes())
    adapter=root/'adapter';adapter.mkdir()
    (adapter/'sft_new_loop_phase3_adapter_config.json').write_text('{}')
    env.update(ADAPTER_DIR=str(adapter),OUTPUT_ROOT=str(root/'evaluation'),GPU_IDS='0',
               RUN_AUDITS='0',RUN_VISUAL_AUDIT='1',RUN_AUDIT_PROMPT_EVAL='1')
    result=run_shell(root,env,'bash qwen3vl_local/sft_new_loop_phase3/eval.sh')
    assert result.returncode==0,result.stdout+result.stderr
    calls=[json.loads(line)['args'] for line in Path(env['CALLS']).read_text().splitlines()]
    evaluations=[args for args in calls if args[0].endswith('/eval.py')]
    assert len(evaluations)==2
    assert all('--no-audit-prompt' in args and args[args.index('--cases-per-bin')+1]=='0' for args in evaluations)
    assert sum(args[0].endswith('/paired_eval.py') for args in calls)==1
    assert not any('audit' in Path(args[0]).name for args in calls)
    assert len([args for args in calls if args[0]=='-'])==3  # config reads only; no bundle builder
    assert 'audit bundle:' not in result.stdout


def test_start_without_regression_does_not_read_old_cases(tmp_path,monkeypatch):
    from qwen3vl_local.qwen35 import preflight
    import torch
    env=dict(PIPELINE_ROOT=str(tmp_path),INDEX=str(tmp_path/'index.jsonl'),
             HISTORY_RGB_MODE='4rgb',ACTION_OUTPUT_MODE='binary',PROMPT_VARIANT=CANDIDATE_NAME,
             RUN_ROOT=str(tmp_path/'new_train'),SKIP_TRAIN='0',SKIP_EVAL='0',RUN_REGRESSION='0',
             RUN_AUDITS='0',MODEL_DIR=str(tmp_path/'model'),GPU_IDS='0',TRAIN_MODE='single')
    for key,value in env.items(): monkeypatch.setenv(key,value)
    for key in ('AUDIT_ROOT','REPLAY_AUTOMOT_ROOT'): monkeypatch.delenv(key,raising=False)
    monkeypatch.setattr(sys,'argv',['pipeline_support','start'])
    monkeypatch.setattr(preflight,'check',lambda path:dict(ready=True,errors=[]))
    monkeypatch.setattr(support.importlib,'import_module',lambda name:None)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    def forbidden(*args): raise AssertionError('historical case access forbidden')
    monkeypatch.setattr(support,'validate_history',forbidden)
    support.main()
    state=json.loads((tmp_path/'pipeline_manifest.json').read_text())
    assert state['historical_regression_enabled'] is False
    assert state['audits_enabled'] is False and 'historical_input' not in state
