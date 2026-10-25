"""Public launcher portability without modifying bound legacy training implementations."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from qwen3vl_local import runtime_dependencies as deps

ROOT=Path(__file__).resolve().parents[1]
ENTRIES=['audit_joint/phase3_compare.sh','audit_joint/geometry_study_abc.sh',
         'audit_joint/phase3_abc_refine/run.sh','action_expert_ablation/geometry_abc/run.sh']


@pytest.mark.parametrize('entry',ENTRIES)
@pytest.mark.parametrize('selection',['explicit','virtual','conda','project'])
def test_public_launchers_keep_active_python_and_project_cwd(tmp_path,entry,selection):
    app=tmp_path/'application with spaces';package=app/'qwen3vl_local'
    target=package/entry;target.parent.mkdir(parents=True);shutil.copy2(ROOT/entry,target)
    shutil.copy2(ROOT/'runtime_env.sh',package/'runtime_env.sh')
    locations={'explicit':tmp_path/'explicit env','virtual':tmp_path/'virtual env',
               'conda':tmp_path/'conda env','project':app/'.venv'}
    for label,folder in locations.items():
        exe=folder/'bin/python';exe.parent.mkdir(parents=True)
        exe.write_text('#!/bin/bash\nprintf "CHOICE='+label+' CWD=%s\\n" "$PWD"\nprintf "%s\\n" "$@"\n')
        exe.chmod(0o755)
    env=dict(os.environ)
    for k in ('PYTHON','VIRTUAL_ENV','CONDA_PREFIX'):env.pop(k,None)
    if selection in ('explicit','virtual'):env['VIRTUAL_ENV']=str(locations['virtual'])
    if selection!='project':env['CONDA_PREFIX']=str(locations['conda'])
    if selection=='explicit':env['PYTHON']=str(locations['explicit']/'bin/python')
    result=subprocess.run(['bash',str(target),'--help'],env=env,cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    choices=[l for l in result.stdout.splitlines() if l.startswith('CHOICE=')]
    assert choices and all(l=='CHOICE='+selection+' CWD='+str(app) for l in choices)


def test_legacy_environment_wrapper(tmp_path):
    app=tmp_path/'app';package=app/'qwen3vl_local';package.mkdir(parents=True)
    for file in ('runtime_env.sh','run_with_env.sh','runtime_version.py'):shutil.copy2(ROOT/file,package/file)
    legacy=package/'legacy.sh';legacy.write_text('#!/bin/bash\nprintf "%s\\n" "$PYTHON" "$PWD" "$@"\n')
    env=dict(os.environ,PYTHON=sys.executable)
    result=subprocess.run(['bash',str(package/'run_with_env.sh'),'qwen3vl_local/legacy.sh','status'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    lines=result.stdout.splitlines();assert lines[-3]==sys.executable and lines[-1]=='status'
    assert Path(lines[-2]).parent.parent==app/'checkpoints/runtime_versions'
    assert Path(lines[-2]).name=='app'


@pytest.mark.parametrize('operation,needed',[('prepare',True),('all',True),('replay',False),('pack',False),('--help',False)])
def test_abc_stream_dependency_only_when_used(monkeypatch,operation,needed):
    seen=[]
    def load(name):
        seen.append(name)
        if name=='ijson':raise ModuleNotFoundError('no ijson',name='ijson')
    monkeypatch.setattr(deps.importlib,'import_module',load)
    if needed:
        with pytest.raises(RuntimeError,match='pip install ijson'):deps.check('abc_refine',operation)
    else:deps.check('abc_refine',operation)
    assert ('ijson' in seen)==needed


def test_trajectory_missing_carla_does_not_suggest_wrong_pypi_version(monkeypatch):
    def load(name):
        if name=='carla':raise ModuleNotFoundError('no carla',name='carla')
    monkeypatch.setattr(deps.importlib,'import_module',load)
    with pytest.raises(RuntimeError) as e:deps.check('trajectory','prepare')
    assert 'CARLA_0915/PythonAPI/carla/dist' in str(e.value) and 'pip install carla' not in str(e.value)


def test_no_virtual_environment_uses_system_python3(tmp_path):
    app=tmp_path/'app';app.mkdir()
    env=dict(os.environ,PATH='/usr/bin:/bin')
    for k in ('PYTHON','VIRTUAL_ENV','CONDA_PREFIX'):env.pop(k,None)
    result=subprocess.run(['/bin/bash','-c','source "$1" "$2"; printf "%s\\n" "$PYTHON"','launcher',str(ROOT/'runtime_env.sh'),str(app)],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip() in ('/usr/bin/python','/bin/python','/usr/bin/python3','/bin/python3')


def test_no_virtual_environment_uses_path_python(tmp_path):
    app=tmp_path/'app';app.mkdir();bin_dir=tmp_path/'bin';bin_dir.mkdir()
    executable=bin_dir/'python';executable.write_text('#!/bin/sh\nexit 0\n');executable.chmod(0o755)
    env=dict(os.environ,PATH=str(bin_dir)+':/usr/bin:/bin')
    for k in ('PYTHON','VIRTUAL_ENV','CONDA_PREFIX'):env.pop(k,None)
    result=subprocess.run(['/bin/bash','-c','source "$1" "$2"; printf "%s\\n" "$PYTHON"','launcher',str(ROOT/'runtime_env.sh'),str(app)],env=env,capture_output=True,text=True)
    assert result.returncode==0 and result.stdout.strip()==str(executable)
