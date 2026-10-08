"""Full-pool orchestration: process boundaries, failure propagation, immutable reuse."""
import json
import os
from pathlib import Path
import subprocess

import pytest
from qwen3vl_local.sft_new_loop_phase4 import full_pipeline as pipeline
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT


def shell(tmp_path, extra=None, args=()):
    stub=tmp_path/'python';log=tmp_path/'calls'
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}');(model/'model.safetensors').touch()
    stub.write_text('''#!/usr/bin/env python3
import json,os,sys
with open(os.environ['CALL_LOG'],'a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
if os.environ.get('FAIL_MODULE') in sys.argv:raise SystemExit(17)
if '-c' in sys.argv:
 if 'select_gpus' in sys.argv[2]:print('0,1' if 'select_gpus(4)' in sys.argv[2] else '0')
 else:print(os.environ['OUTPUT_DIR']+'/epoch_000')
''');stub.chmod(0o755)
    env={k:v for k,v in os.environ.items() if k not in (
        'DATASET','PAIRED_WITH','PRODUCTION_INDEX','TEACHER_REGISTRY','CANDIDATE_POOL','CONDITION_STREAM',
        'RUN_ROOT','OUTPUT_DIR','MODE','TRAIN_MODE','HISTORY_RGB_MODE','RGB_MODE','SKIP_BUILD','SKIP_TRAIN','SKIP_EVAL')}
    env.update(PYTHON=str(stub),MODEL_DIR=str(model),CALL_LOG=str(log),DATA_DIR=str(tmp_path/'data'),
               PIPELINE_ROOT=str(tmp_path/'pipeline'),OUTPUT_DIR=str(tmp_path/'run'),TRAIN_MODE='single')
    env.update(extra or {})
    result=subprocess.run(['bash',str(ROOT/'run_full_pipeline.sh'),*args],env=env,capture_output=True,text=True)
    calls=[json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
    return result,calls


@pytest.mark.parametrize('history,rgb',[('4rgb',4),('2rgb_endpoints',2)])
def test_default_runs_full_build_train_test_and_package(tmp_path,history,rgb):
    result,calls=shell(tmp_path,dict(HISTORY_RGB_MODE=history,SKIP_BUILD='1',EPOCHS='3',ACCUMULATION='16'))
    assert result.returncode==0,result.stderr
    modules=[c[1].split('.')[-1] for c in calls if c[0]=='-m']
    assert modules==['full_pipeline','launch','preflight','train','evaluate','audit_bundle']
    assert '--skip-build' in calls[0]
    train=next(c for c in calls if 'qwen3vl_local.sft_new_loop_phase4.train' in c)
    for key,value in (('--dataset',str(tmp_path/f'data/data{rgb}')),('--paired-with',str(tmp_path/f'data/data{6-rgb}')),
                      ('--sampling-policy','phase3_balanced'),('--epochs','3'),('--accumulation','16')):
        assert train[train.index(key)+1]==value
    evaluation=next(c for c in calls if 'qwen3vl_local.sft_new_loop_phase4.evaluate' in c)
    assert evaluation[evaluation.index('--split')+1]=='test'


@pytest.mark.parametrize('extra',[
    {'HISTORY_RGB_MODE':'wrong'},{'HISTORY_RGB_MODE':'4rgb','RGB_MODE':'2'},
    {'ACTION_OUTPUT_MODE':'choice'},{'SKIP_BUILD':'true'},{'TRAIN_MODE':'wrong'},
    {'MAX_TEACHER_QUESTIONS_PER_ROUTE':'100'}, {'DATASET':'/another/data'},
    {'PAIRED_WITH':'/another/data'}, {'TRAIN_MODE':'single','MODE':'ddp'},
    {'TEACHER_REGISTRY':'/missing'}, {'PRODUCTION_INDEX':'/missing'},
    {'SKIP_TRAIN':'1','OUTPUT_DIR':'','RUN_ROOT':''},
])
def test_invalid_configuration_rejected_before_build(tmp_path,extra):
    result,calls=shell(tmp_path,extra)
    assert result.returncode==2 and not calls


@pytest.mark.parametrize('option',['--datas=x','--sampling-policy=event_equal','--paired-with=/wrong'])
def test_cli_cannot_override_full_pair_or_paths(tmp_path,option):
    result,calls=shell(tmp_path,args=[option]);assert result.returncode==2 and not calls


@pytest.mark.parametrize('failed', ['full_pipeline','preflight','train','evaluate'])
def test_failures_stop_downstream_stages(tmp_path,failed):
    result,calls=shell(tmp_path,{'FAIL_MODULE':'qwen3vl_local.sft_new_loop_phase4.'+failed})
    assert result.returncode==17
    assert calls[-1][1]=='qwen3vl_local.sft_new_loop_phase4.'+failed


def test_skip_train_evaluates_explicit_existing_run(tmp_path):
    result,calls=shell(tmp_path,dict(SKIP_TRAIN='1',SKIP_BUILD='1',EVAL_DEVICE='cpu'))
    assert result.returncode==0,result.stderr
    assert not any('select_gpus' in ' '.join(c) for c in calls)
    assert [c[1].split('.')[-1] for c in calls if c[0]=='-m']==['full_pipeline','evaluate','audit_bundle']


def test_skip_eval_still_packages_training_records(tmp_path):
    result,calls=shell(tmp_path,dict(SKIP_EVAL='1'))
    assert result.returncode==0
    assert not any('qwen3vl_local.sft_new_loop_phase4.evaluate' in c for c in calls)
    assert calls[-1][1].endswith('.audit_bundle')


def test_ddp_receives_full_sampling_and_actual_world(tmp_path):
    result,calls=shell(tmp_path,dict(TRAIN_MODE='ddp'))
    assert result.returncode==0,result.stderr
    preflight=next(c for c in calls if 'qwen3vl_local.sft_new_loop_phase4.preflight' in c)
    assert preflight[preflight.index('--world-size')+1]=='2'
    ddp=next(c for c in calls if 'torch.distributed.run' in c)
    assert '--nproc_per_node=2' in ddp and 'phase3_balanced' in ddp


def test_missing_build_cannot_skip(tmp_path):
    ann=tmp_path/'annotations.json';ann.write_text('[]')
    with pytest.raises(ValueError,match='SKIP_BUILD=1'):
        pipeline.prepare(tmp_path/'absent',tmp_path,ann,skip_build=True)


def test_changed_source_cannot_resume(tmp_path):
    ann=tmp_path/'annotations.json';ann.write_text('[]');base=tmp_path/'data';base.mkdir()
    (base/'build_request.json').write_text('{}')
    with pytest.raises(ValueError,match='source/configuration changed'):
        pipeline.prepare(base,tmp_path,ann)


def test_foreign_directory_never_overwritten(tmp_path):
    ann=tmp_path/'annotations.json';ann.write_text('[]');base=tmp_path/'data';base.mkdir()
    (base/'keep').write_text('evidence')
    with pytest.raises(ValueError,match='not an owned'):
        pipeline.prepare(base,tmp_path,ann)
    assert (base/'keep').read_text()=='evidence'


def test_build_lock_prevents_second_writer(tmp_path):
    with pipeline.build_lock(tmp_path):
        with pytest.raises(ValueError,match='another pipeline'):
            with pipeline.build_lock(tmp_path):pass
    with pipeline.build_lock(tmp_path):pass


def test_registry_keeps_weak_scope_and_excludes_restrict():
    from qwen3vl_local.sft_new_loop_phase4.teacher_approval import validate_registry
    registry=pipeline.default_registry();report=validate_registry(registry)
    assert report['supervision_policy']=='weak_experiment_train_only' and not report['approved']
    assert not any('/restrict/' in c for c in registry['rule_classes'])


def test_ready_checks_source_and_content_before_reuse(tmp_path,monkeypatch):
    record={'request':{'revision':1},'artifacts':{'train':'sha'}}
    (tmp_path/'pipeline_ready.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='source/configuration changed'):
        pipeline.validate_ready(tmp_path,{'revision':2})
    monkeypatch.setattr(pipeline,'artifact_hashes',lambda _: {'train':'changed'})
    with pytest.raises(ValueError,match='content changed'):
        pipeline.validate_ready(tmp_path,{'revision':1})


def test_missing_model_fails_before_expensive_build(tmp_path):
    result,calls=shell(tmp_path,dict(MODEL_DIR=str(tmp_path/'missing')))
    assert result.returncode==2 and 'local base model missing' in result.stderr and not calls
