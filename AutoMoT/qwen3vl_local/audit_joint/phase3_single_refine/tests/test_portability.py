import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from .. import bootstrap,pipeline,preflight
from ..state import atomic
from .test_pipeline import lifecycle


def test_training_entry_and_status_do_not_import_ijson(tmp_path):
    script='''
import sys,runpy,importlib.abc
class Reject(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname == 'ijson' or fullname.startswith('ijson.'):
   raise ModuleNotFoundError('ijson intentionally unavailable',name='ijson')
sys.meta_path.insert(0,Reject())
sys.argv=['pipeline','status','--root',sys.argv[1]]
runpy.run_module('qwen3vl_local.audit_joint.phase3_single_refine.pipeline',run_name='__main__')
'''
    completed=subprocess.run([sys.executable,'-B','-c',script,str(tmp_path/'unused')],capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    assert json.loads(completed.stdout)['state'] is None
    assert not (tmp_path/'unused').exists()


def test_bootstrap_reports_all_missing_and_broken_imports(monkeypatch):
    seen=[]
    def check(name):
        seen.append(name)
        if name=='numpy':raise ModuleNotFoundError('no numpy',name='numpy')
        if name=='torchvision':raise RuntimeError('binary operator mismatch')
    monkeypatch.setattr(bootstrap.importlib,'import_module',check)
    with pytest.raises(RuntimeError) as e:bootstrap.check('all')
    assert 'numpy' in str(e.value) and 'binary operator mismatch' in str(e.value)
    assert sys.executable in str(e.value) and 'pip install numpy' in str(e.value)
    assert 'peft' in seen and 'ijson' not in seen


def test_cpu_bootstrap_does_not_require_gpu_stack(monkeypatch):
    seen=[];monkeypatch.setattr(bootstrap.importlib,'import_module',seen.append)
    bootstrap.check('cpu');assert seen==['numpy','PIL']


@pytest.mark.parametrize('selection',['explicit','virtual','conda','project'])
def test_launcher_interpreter_precedence_from_unrelated_cwd(tmp_path,selection):
    app=tmp_path/'app with space';folder=app/'qwen3vl_local/audit_joint/phase3_single_refine';folder.mkdir(parents=True)
    script=folder/'run.sh';shutil.copy2(Path(__file__).parents[1]/'run.sh',script)
    locations={'project':app/'.venv','virtual':tmp_path/'active env','conda':tmp_path/'conda env','explicit':tmp_path/'explicit env'}
    for label,base in locations.items():
        exe=base/'bin/python';exe.parent.mkdir(parents=True)
        exe.write_text('#!/bin/bash\nprintf "%s\\n" '+label+' "$@"\n');exe.chmod(0o755)
    env=dict(os.environ)
    for name in ('PYTHON','VIRTUAL_ENV','CONDA_PREFIX'):env.pop(name,None)
    if selection in ('explicit','virtual'):env['VIRTUAL_ENV']=str(locations['virtual'])
    if selection!='project':env['CONDA_PREFIX']=str(locations['conda'])
    if selection=='explicit':env['PYTHON']=str(locations['explicit']/'bin/python')
    result=subprocess.run(['bash',str(script),'status'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.splitlines()==[selection,'-B',str(folder/'bootstrap.py'),'status']


def test_rejected_cpu_all_retries_preserving_previous_report(lifecycle,monkeypatch):
    root,native,events,_=lifecycle
    atomic(root/'ready_1280.json',dict(status='rejected',error='prior mount failure'))
    calls=[]
    def prepare(*args):calls.append(args);atomic(root/'ready_1280.json',dict(status='ready'));return dict(dataset='dataset',dataset_sha256='sha')
    monkeypatch.setattr(preflight,'prepare',prepare)
    pipeline.run('all',root,native,root.parent/'model')
    assert len(calls)==1 and events==['probe','train','validate','evaluate','package']
    old=list((root/'cpu_attempts').glob('*.json'));assert len(old)==1
    assert json.loads(old[0].read_text())['error']=='prior mount failure'


def test_stale_ready_is_not_replaced_on_reuse_failure(lifecycle,monkeypatch):
    root,native,events,_=lifecycle;p=root/'ready_1280.json';atomic(p,dict(status='ready',code='old'))
    before=p.read_bytes()
    def reject(*_a,**_kw):raise ValueError('candidate_source_changed_rerun_cpu')
    monkeypatch.setattr(preflight,'load_ready',reject)
    with pytest.raises(ValueError,match='source_changed'):pipeline.run('all',root,native,root.parent/'model')
    assert p.read_bytes()==before and not events


@pytest.mark.parametrize('stage',['probe','run'])
def test_cpu_cannot_overwrite_ready_of_started_job(lifecycle,stage):
    root,native,events,_=lifecycle;(root/stage).mkdir();p=root/'ready_1280.json';atomic(p,dict(status='ready'))
    before=p.read_bytes()
    with pytest.raises(ValueError,match='cpu_reprepare_refused'):pipeline.run('cpu',root,native,root.parent/'model')
    assert p.read_bytes()==before and not events


def test_custom_native_root_is_passed_to_builder(lifecycle,monkeypatch):
    root,_,events,_=lifecycle;native=root.parent/'custom-native/ready_1280.json'
    seen=[]
    def execute(cmd,*_a):
        seen.extend(cmd);native.parent.mkdir();atomic(native,dict(status='ready'))
    monkeypatch.setattr(pipeline,'execute',execute)
    import importlib
    original=importlib.import_module
    monkeypatch.setattr(importlib,'import_module',lambda name:None if name=='carla' else original(name))
    monkeypatch.setattr(preflight,'prepare',lambda *_a:dict(status='ready'))
    pipeline.run('cpu',root,native,root.parent/'model')
    assert seen[-2:]==['--study-root',str(native.parent)] and not events


def test_worker_sanitizes_inherited_distributed_environment(tmp_path,monkeypatch):
    keys=('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_WORLD_SIZE','MASTER_PORT','TORCHELASTIC_RUN_ID')
    for k in keys:monkeypatch.setenv(k,'stale')
    captured=[];monkeypatch.setattr(pipeline,'execute',lambda cmd,log,env:captured.append((cmd,env)))
    pipeline.worker('train',tmp_path,dict(dataset='data.json'),tmp_path/'model',['1','3','5','7'],tmp_path/'run')
    cmd,env=captured[0];assert not set(keys)&set(env)
    assert env['CUDA_VISIBLE_DEVICES']=='1,3,5,7' and cmd[0]==sys.executable
