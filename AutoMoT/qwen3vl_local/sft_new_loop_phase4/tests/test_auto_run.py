import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from qwen3vl_local.sft_new_loop_phase4 import auto_run as auto


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_reuses_matching_complete_dataset_without_rebuilding(tmp_path):
    expected={'source':'current'}
    save(tmp_path/'checkpoints/phase4_v38_full/pipeline_ready.json',{'request':expected})
    save(tmp_path/'checkpoints/phase4_v99_full/build_request.json',expected)
    assert auto.choose_data_dir(tmp_path,expected)==tmp_path/'checkpoints/phase4_v38_full'


def test_resumes_matching_partial_build(tmp_path):
    expected={'source':'current'};out=tmp_path/'checkpoints/phase4_auto_full/partial'
    save(out/'build_request.json',expected)
    assert auto.choose_data_dir(tmp_path,expected)==out


def test_stale_data_gets_new_automatic_directory(tmp_path):
    old=tmp_path/'checkpoints/phase4_v38_full/pipeline_ready.json';save(old,{'request':{'source':'old'}})
    request={'source':'new'}
    out=auto.choose_data_dir(tmp_path,request)
    assert out.parent==tmp_path/'checkpoints/phase4_auto_full'
    assert out==auto.choose_data_dir(tmp_path,request)
    assert json.loads(old.read_text())=={'request':{'source':'old'}}


def test_malformed_receipt_not_treated_as_matching(tmp_path):
    f=tmp_path/'checkpoints/phase4_v38_full/pipeline_ready.json';f.parent.mkdir(parents=True);f.write_text('{')
    assert auto.choose_data_dir(tmp_path,{})!=f.parent


def test_sequence_builds_once_then_reuses_separate_runs(tmp_path):
    calls=[]
    def runner(args,**kw):calls.append((args,kw))
    config={'data_dir':str(tmp_path/'auto_data'),'data_root':str(tmp_path/'rgb'),'request':{}}
    polluted=dict(SKIP_TRAIN='1',SKIP_EVAL='1',SKIP_BUILD='1',RGB_MODE='2',MODE='check',
                  OUTPUT_DIR='/shared',RUN_ROOT='/shared',DATASET='/wrong',ANNOTATIONS='/wrong',
                  MAX_TEACHER_QUESTIONS_PER_ROUTE='100',ACTION_OUTPUT_MODE='choice',GPU_IDS='0,1')
    output=auto.execute(config,tmp_path,environ=polluted,runner=runner,model_resolver=lambda *_:tmp_path/'model')
    assert len(calls)==2
    first,second=[kw['env'] for _,kw in calls]
    assert first['HISTORY_RGB_MODE']=='4rgb' and second['HISTORY_RGB_MODE']=='2rgb_endpoints'
    assert first['SKIP_BUILD']=='0' and second['SKIP_BUILD']=='1'
    for env in (first,second):
        assert env['DATA_DIR']==config['data_dir'] and env['TRAIN_MODE']=='ddp'
        assert env['ACTION_OUTPUT_MODE']=='binary' and env['EPOCHS']=='7'
        assert env['GPU_IDS']=='0,1' and env['HF_HUB_OFFLINE']=='1'
        assert not {'SKIP_TRAIN','SKIP_EVAL','OUTPUT_DIR','RUN_ROOT','RGB_MODE','MODE','ANNOTATIONS'}&env.keys()
    assert first['PIPELINE_ROOT']!=second['PIPELINE_ROOT']
    record=json.loads((output/'automatic_run.json').read_text())
    assert record['sampling_policy']=='phase3_balanced' and record['rgb_order']==[4,2]


def test_first_failure_never_runs_second_experiment(tmp_path):
    calls=[]
    def runner(args,**kw):
        calls.append(args);raise subprocess.CalledProcessError(2,args)
    with pytest.raises(subprocess.CalledProcessError):
        auto.execute({'data_dir':'data','data_root':'rgb','request':{}},tmp_path,environ={},runner=runner,model_resolver=lambda *_:'model')
    assert len(calls)==1


def test_existing_model_avoids_download(tmp_path,monkeypatch):
    model=tmp_path/'checkpoints/Qwen3.5-4B'
    monkeypatch.setattr(auto,'model_ready',lambda p: p==model)
    monkeypatch.setattr(auto.subprocess,'run',lambda *a,**k:pytest.fail('must stay offline'))
    assert auto.resolve_model(tmp_path)==model


def test_explicit_invalid_model_is_not_silently_replaced(tmp_path,monkeypatch):
    monkeypatch.setattr(auto,'model_ready',lambda _:False)
    monkeypatch.setattr(auto.subprocess,'run',lambda *a,**k:pytest.fail('unexpected download'))
    with pytest.raises(ValueError,match='explicit MODEL_DIR'):auto.resolve_model(tmp_path,'/invalid')


def test_missing_model_download_is_isolated_and_checked(tmp_path,monkeypatch):
    ready=set();calls=[]
    monkeypatch.setattr(auto,'model_ready',lambda p:p in ready)
    def run(args,**kw):
        calls.append((args,kw));ready.add(Path(args[-1]))
    monkeypatch.setattr(auto.subprocess,'run',run)
    model=auto.resolve_model(tmp_path)
    assert model in ready and len(calls)==1
    assert calls[0][1]['env']['HF_HUB_OFFLINE']=='0'
    assert calls[0][1]['env']['HF_HUB_DISABLE_IMPLICIT_TOKEN']=='1'
    assert auto.resolve_model(tmp_path)==model and len(calls)==1


def test_download_uses_pinned_official_assets_only(tmp_path,monkeypatch):
    import huggingface_hub
    calls=[]
    def snapshot(repo,**kw):
        calls.append((repo,kw))
        if kw.get('local_files_only'):raise FileNotFoundError('no cache')
        tmp_path.mkdir(exist_ok=True);return str(tmp_path)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',snapshot)
    monkeypatch.setattr(auto,'model_ready',lambda _:True)
    auto.download_model(tmp_path)
    assert all(repo=='Qwen/Qwen3.5-4B' for repo,_ in calls)
    assert all(kw['revision']==auto.MODEL_REVISION and kw['token'] is False for _,kw in calls)
    assert calls[-1][1]['allow_patterns']==auto.MODEL_FILES
    assert '*.py' not in auto.MODEL_FILES


def test_complete_hub_cache_is_copied_without_network(tmp_path,monkeypatch):
    import huggingface_hub
    cached=tmp_path/'cached';cached.mkdir();(cached/'config.json').write_text('{}')
    blob=tmp_path/'blob';blob.write_bytes(b'weights');(cached/'model.safetensors').symlink_to(blob)
    (cached/'remote_code.py').write_text('never copied')
    def snapshot(*args,**kw):
        assert kw['local_files_only'];return str(cached)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',snapshot)
    monkeypatch.setattr(auto,'model_ready',lambda p:(p/'model.safetensors').is_file())
    target=tmp_path/'target';auto.download_model(target)
    assert (target/'model.safetensors').read_bytes()==b'weights' and not (target/'model.safetensors').is_symlink()
    assert not (target/'remote_code.py').exists()


def test_script_takes_no_parameters():
    result=subprocess.run(['bash',str(auto.ROOT/'run.sh'),'unexpected'],capture_output=True,text=True)
    assert result.returncode==2 and 'no parameters needed' in result.stderr


@pytest.mark.parametrize('probe_exit',[0,1])
def test_shell_uses_compatible_explicit_environment(tmp_path,probe_exit):
    executable=tmp_path/'python';log=tmp_path/'calls'
    executable.write_text('''#!/usr/bin/env python3
import json,os,sys
with open(os.environ['CALL_LOG'],'a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1]=='-c':raise SystemExit(int(os.environ['PROBE_EXIT']))
assert sys.argv[1:]==['-m','qwen3vl_local.sft_new_loop_phase4.auto_run']
assert os.environ['HF_HUB_DISABLE_TELEMETRY']=='1'
''');executable.chmod(0o755)
    result=subprocess.run(['bash',str(auto.ROOT/'run.sh')],capture_output=True,text=True,
                          env=dict(os.environ,PYTHON=str(executable),CALL_LOG=str(log),PROBE_EXIT=str(probe_exit)))
    calls=[json.loads(s) for s in log.read_text().splitlines()]
    assert len(calls)==(2 if probe_exit==0 else 1)
    assert result.returncode==(0 if probe_exit==0 else 2)


def test_actual_plan_only_reuses_current_pair_without_download():
    result=subprocess.run([sys.executable,'-m','qwen3vl_local.sft_new_loop_phase4.auto_run','--plan-only'],
                          cwd=auto.ROOT.parents[1],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    result=json.loads(result.stdout)
    expected=auto.configuration(auto.ROOT.parents[1],os.environ)
    assert result=={k:v for k,v in expected.items() if k!='request'}
    receipt=Path(result['data_dir'])/'pipeline_ready.json'
    if receipt.exists():assert json.loads(receipt.read_text())['request']==expected['request']
