import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qwen3vl_local.audit_joint import phase3_matched_image_launch as launch

LAUNCHER = Path(launch.__file__).with_suffix('.sh').with_name('phase3_matched_image_eval.sh')


def test_shell_preserves_arguments_and_nonzero_exit(tmp_path):
    fake=tmp_path/'python'
    capture=tmp_path/'args.json'
    fake.write_text('#!/usr/bin/env python3\nimport sys,json,os\njson.dump(sys.argv[1:],open(os.environ["CAPTURE"],"w"))\nsys.exit(7)\n')
    fake.chmod(0o755)
    args=['--job','old job.json','--output','new output']
    got=subprocess.run(['bash',str(LAUNCHER),*args],cwd=tmp_path,
        env=dict(os.environ,PYTHON=str(fake),CAPTURE=str(capture)),capture_output=True,text=True)
    assert got.returncode==7
    assert json.loads(capture.read_text())==['-m','qwen3vl_local.audit_joint.phase3_matched_image_launch',*args]


def test_gpu_default_four_and_explicit_pin(monkeypatch):
    calls=[]
    def fake(*a,**kw):calls.append(a);return '0, 100, 0\n2, 10, 20\n1, 10, 0\n3, 20, 0\n4, 200, 0\n'
    monkeypatch.setattr(subprocess,'check_output',fake)
    assert launch.gpu_ids('')==['1','2','3','0']
    assert launch.gpu_ids('3,2,1,0')==['3','2','1','0']
    assert launch.gpu_ids('0')==['0']
    assert len(calls)==1


@pytest.mark.parametrize('ids',['0,0','01,1','0,','-1','a'])
def test_gpu_invalid(ids):
    with pytest.raises(ValueError):launch.gpu_ids(ids)


def test_default_never_silently_downgrades_to_single_gpu(monkeypatch):
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:'0, 0, 0\n')
    with pytest.raises(ValueError,match='four GPUs'):launch.gpu_ids('')


def test_actual_original_source_export_keeps_current_geometry_untouched(tmp_path):
    source=launch.APP/'qwen3vl_local/leadmot/config.py'
    before=source.read_bytes()
    release=launch.APP/'qwen3vl_local/audit_joint/phase3_base_trial_sources.json'
    runtime,note=launch.snapshot(tmp_path/'runtime',dict(release_sha256=launch.diag.file_sha(release)))
    assert note['verified_native_sources']==261
    assert source.read_bytes()==before
    expected=json.loads(release.read_text())['files']['qwen3vl_local/leadmot/config.py']
    assert launch.diag.file_sha(runtime/'qwen3vl_local/leadmot/config.py')==expected
    # Exercise native verification in a real fresh process, not mocked hashes.
    cmd=[sys.executable,'-c','from qwen3vl_local.audit_joint.phase3_base_trial import verified_sources; print(verified_sources())']
    got=subprocess.run(cmd,cwd=runtime,env=dict(os.environ,PYTHONPATH=str(runtime)),capture_output=True,text=True)
    assert got.returncode==0,got.stderr
    assert got.stdout.strip()==launch.diag.file_sha(release)


def test_snapshot_rejects_foreign_release(tmp_path):
    with pytest.raises(ValueError,match='supported v2 runtime'):
        launch.snapshot(tmp_path/'runtime',dict(release_sha256='bad'))


def test_failed_worker_terminates_sibling(tmp_path):
    commands=[[sys.executable,'-c','import sys; print("precise failure evidence",flush=True); sys.exit(9)'],
              [sys.executable,'-c','import time; time.sleep(120)']]
    import time
    start=time.monotonic()
    with pytest.raises(RuntimeError,match='exited 9') as caught:
        launch.processes(commands,tmp_path,[dict(os.environ)]*2,[tmp_path/'a.log',tmp_path/'b.log'])
    assert 'precise failure evidence' in str(caught.value)
    assert time.monotonic()-start<10


@pytest.mark.parametrize('status',['failed','completed','preparing_runtime'])
def test_existing_output_read_only_before_assets_or_gpu(tmp_path,monkeypatch,status):
    output=tmp_path/'existing';output.mkdir()
    (output/'receipt.json').write_text(json.dumps(dict(status=status,error='prior root cause')))
    (output/'preflight.log').write_text('x'*10000+'\nSOURCE_FAILURE\n')
    (output/'rank_0.log').write_text('rank evidence')
    before={p.name:p.read_bytes() for p in output.iterdir()}
    monkeypatch.setattr(launch,'gpu_ids',lambda x:pytest.fail('Must not select GPU'))
    with pytest.raises(launch.ExistingOutputError) as caught:
        launch.launch(tmp_path/'missing_job.json',output)
    report=json.loads(str(caught.value))
    assert report['receipt']['status']==status
    assert 'SOURCE_FAILURE' in report['log_tails']['preflight.log']
    assert len(report['log_tails']['preflight.log'])<=6000
    assert report['live_process_status']=='not_checked'
    assert {p.name:p.read_bytes() for p in output.iterdir()}==before


def test_inspect_existing_cli_does_not_require_assets_or_gpu(tmp_path):
    output=tmp_path/'existing';output.mkdir()
    (output/'receipt.json').write_text('{broken')
    env=dict(os.environ,PYTHON=sys.executable,GPU_IDS='invalid')
    result=subprocess.run(['bash',str(LAUNCHER),'--job',str(tmp_path/'missing.json'),
        '--output',str(output),'--inspect-existing'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert 'receipt_error' in report and report['files_modified'] is False
    assert (output/'receipt.json').read_text()=='{broken'
    assert list(output.iterdir())==[output/'receipt.json']


def test_existing_output_cli_shows_evidence_without_traceback(tmp_path):
    output=tmp_path/'existing';output.mkdir()
    (output/'preflight.log').write_text('previous failure evidence')
    result=subprocess.run(['bash',str(LAUNCHER),'--job',str(tmp_path/'missing.json'),
        '--output',str(output)],env=dict(os.environ,PYTHON=sys.executable),capture_output=True,text=True)
    assert result.returncode==2
    assert 'previous failure evidence' in result.stderr
    assert 'Traceback' not in result.stderr


@pytest.mark.parametrize('layout',['flat_without_git','nested_unrelated_git'])
def test_synced_runtime_without_original_git_history(tmp_path,monkeypatch,layout):
    import shutil
    app=tmp_path/('server' if layout=='flat_without_git' else 'unrelated/AutoMoT')
    app.mkdir(parents=True)
    package=Path('qwen3vl_local/audit_joint/phase3_behavior_diagnostic_v1')
    manifest=json.loads((launch.APP/package/'runtime_sources.json').read_text())
    for name,entry in manifest['files'].items():
        relative=Path(name) if entry['source']=='project' else package/'compat_sources'/name
        target=app/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(launch.APP/relative,target)
    (app/package).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(launch.APP/package/'runtime_sources.json',app/package/'runtime_sources.json')
    for name in launch.OVERLAY:
        shutil.copyfile(launch.APP/'qwen3vl_local/audit_joint'/name,app/'qwen3vl_local/audit_joint'/name)
    if layout=='nested_unrelated_git':(app/'.git').write_text('not the publication repository')
    with monkeypatch.context() as patch:
        patch.setattr(subprocess,'check_output',lambda *a,**k:pytest.fail('Runtime must not call git'))
        patch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('Runtime must not invoke external commands'))
        runtime,note=launch.snapshot(tmp_path/'runtime',dict(release_sha256=launch.diag.file_sha(
            app/'qwen3vl_local/audit_joint/phase3_base_trial_sources.json')),app=app)
    assert note['remote_git_required'] is False and note['verified_runtime_sources']==len(manifest['files'])
    assert runtime==tmp_path/'runtime'
    assert not (runtime/'.git').exists()
    expected={p.name:launch.diag.file_sha(p) for p in (runtime/'qwen3vl_local/audit_joint/phase3_single_frame').glob('*.py')}
    cmd=[sys.executable,'-c','import json; from qwen3vl_local.audit_joint.phase3_single_frame import dataset,worker,runtime; print(json.dumps(dataset.sources()))']
    result=subprocess.run(cmd,cwd=runtime,env=dict(os.environ,PYTHONPATH=str(runtime)),capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['experiment_sources']==expected
    compat=next(n for n,v in manifest['files'].items() if v['source']=='compat_sources')
    (app/package/'compat_sources'/compat).write_text('changed')
    with pytest.raises(ValueError,match='Synced runtime file missing or changed'):
        launch.snapshot(tmp_path/'must_not_exist',dict(release_sha256='bad'),app=app)
    assert not (tmp_path/'must_not_exist').exists()
