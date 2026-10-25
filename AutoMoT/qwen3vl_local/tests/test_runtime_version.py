import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest
from qwen3vl_local import runtime_version as version


def app(tmp_path):
    root=tmp_path/'origin/AutoMoT';q=root/'qwen3vl_local';q.mkdir(parents=True)
    (q/'__init__.py').write_text('')
    (q/'late.py').write_text('VALUE="old"\n')
    (root/'checkpoints').mkdir();(root/'lead_data').mkdir()
    (root/'lead_data/item.txt').write_text('data')
    return root


def test_content_version_reuse_and_source_isolation(tmp_path):
    root=app(tmp_path);one=version.prepare(root)
    assert version.prepare(root)==one
    (root/'qwen3vl_local/late.py').write_text('VALUE="new"\n')
    two=version.prepare(root)
    assert one!=two
    assert (one/'app/qwen3vl_local/late.py').read_text()=='VALUE="old"\n'
    version.verify(one);version.verify(two)
    assert (one/'app/lead_data').is_symlink()
    assert (one/'app/checkpoints').resolve()==root/'checkpoints'


@pytest.mark.parametrize('change',['modify','add','symlink','asset_link'])
def test_tampered_snapshot_rejected(tmp_path,change):
    root=app(tmp_path);one=version.prepare(root);p=one/'app/qwen3vl_local/late.py'
    if change=='modify':p.chmod(0o644);p.write_text('bad')
    elif change=='add':(p.parent/'injected.py').write_text('bad')
    elif change=='symlink':p.unlink();p.symlink_to(root/'qwen3vl_local/late.py')
    else:
        link=one/'app/lead_data';link.unlink();link.symlink_to(tmp_path)
    with pytest.raises(ValueError):version.verify(one)


def test_changed_source_during_capture_retries_before_any_execution(tmp_path,monkeypatch):
    root=app(tmp_path);original=version.inventory;count=0
    def inventory(path):
        nonlocal count
        count+=1
        if count==2:(root/'qwen3vl_local/late.py').write_text('VALUE="changed"\n')
        return original(path)
    monkeypatch.setattr(version,'inventory',inventory)
    one=version.prepare(root)
    assert count>=4 and (one/'app/qwen3vl_local/late.py').read_text()=='VALUE="changed"\n'
    assert not list(one.parent.glob('.preparing-*'))


def test_subprocess_lazy_import_survives_hot_update_and_shared_output(tmp_path):
    root=app(tmp_path);q=root/'qwen3vl_local'
    (q/'job.py').write_text('''from pathlib import Path
import subprocess,sys,time
Path('checkpoints/started').write_text('yes')
while not Path('checkpoints/continue').exists():time.sleep(.02)
from qwen3vl_local.late import VALUE
child=subprocess.check_output([sys.executable,'-B','-c','from qwen3vl_local.late import VALUE; print(VALUE)'],text=True).strip()
Path('checkpoints/result').write_text(VALUE+':'+child)
''')
    cmd=[sys.executable,'-B',version.__file__,'--app',str(root),'--','-m','qwen3vl_local.job']
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        until=time.monotonic()+20
        while not (root/'checkpoints/started').exists() and time.monotonic()<until:
            if process.poll() is not None:pytest.fail(str(process.communicate()))
            time.sleep(.02)
        assert (root/'checkpoints/started').exists()
        (q/'late.py').write_text('VALUE="new"\n');(root/'checkpoints/continue').write_text('yes')
        stdout,stderr=process.communicate(timeout=20)
        assert process.returncode==0,(stdout,stderr)
        assert (root/'checkpoints/result').read_text()=='old:old'
        receipts=list((root/'checkpoints/runtime_executions').glob('*.json'))
        assert json.loads(receipts[0].read_text())['status']=='completed'
    finally:
        if process.poll() is None:process.kill();process.wait()


def test_nonzero_exit_is_recorded_not_hidden(tmp_path):
    root=app(tmp_path);(root/'qwen3vl_local/job.py').write_text('raise SystemExit(2)')
    p=subprocess.run([sys.executable,'-B',version.__file__,'--app',str(root),'--','-m','qwen3vl_local.job'],capture_output=True)
    assert p.returncode==2
    r=json.loads(next((root/'checkpoints/runtime_executions').glob('*.json')).read_text())
    assert r['exit_code']==2 and r['status']=='nonzero_exit'


def test_saved_abc_run_selects_original_version_after_source_update(tmp_path):
    root=app(tmp_path);entry=root/'qwen3vl_local/action_expert_ablation/geometry_abc/run.sh';entry.parent.mkdir(parents=True)
    entry.write_text('#!/bin/bash\nprintf old > checkpoints/version_result\n')
    one=version.prepare(root);run=root/'checkpoints/run';run.mkdir()
    (run/'execution_version.json').write_text(json.dumps(dict(directory=str(one))))
    entry.write_text('#!/bin/bash\nprintf new > checkpoints/version_result\n')
    p=subprocess.run([sys.executable,'-B',version.__file__,'--app',str(root),'--',
        'qwen3vl_local/action_expert_ablation/geometry_abc/run.sh','pack','--run-dir',str(run)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert (root/'checkpoints/version_result').read_text()=='old'


def test_sigterm_stops_only_own_child_and_records_interruption(tmp_path):
    import signal
    root=app(tmp_path);(root/'qwen3vl_local/job.py').write_text("from pathlib import Path\nimport time\nPath('checkpoints/started').write_text('yes')\ntime.sleep(60)\n")
    p=subprocess.Popen([sys.executable,'-B',version.__file__,'--app',str(root),'--','-m','qwen3vl_local.job'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        until=time.monotonic()+20
        while not (root/'checkpoints/started').exists() and time.monotonic()<until:time.sleep(.02)
        assert (root/'checkpoints/started').exists()
        p.send_signal(signal.SIGTERM);p.communicate(timeout=10);assert p.returncode==143
        r=json.loads(next((root/'checkpoints/runtime_executions').glob('*.json')).read_text());assert r['exit_code']==143
    finally:
        if p.poll() is None:p.kill();p.wait()


def frozen_release(root):
    """Use the actual native verifier, not a replacement for its path guard."""
    import shutil
    from qwen3vl_local.action_prior import phase3_release
    base=root/'qwen3vl_local/action_prior';base.mkdir()
    shutil.copyfile(phase3_release.__file__,base/'phase3_release.py')
    release=base/'phase3_releases/v23';release.mkdir(parents=True)
    files={}
    for name in ('__init__.py','build_dataset.py','prompts.py','sampling.py','source_mapping.py',
                 'trajectory_action.py','choice_semantics.py'):
        (release/name).write_text('# frozen\n');files[name]=version.sha(release/name)
    dependencies={}
    for name in ('keyframe_filter/collection_output/audit/table.json','keyframe_filter/evidence.py',
                 'lead_video_tools/filter.py','qwen3vl_local/audit/decisions.jsonl'):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}\n')
        dependencies[name]=version.sha(p)
    other=root/'keyframe_filter/collection_output/raw';other.mkdir();(other/'large.json').write_text('shared')
    (release/'release.json').write_text(json.dumps(dict(schema='action_phase3_release_v1',release='v23',
        files=files,external_dependencies=dependencies)))
    return dependencies


def verify_native(tree):
    return subprocess.run([sys.executable,'-B','-c',
        'from qwen3vl_local.action_prior.phase3_release import verify_release; verify_release("v23")'],
        cwd=tree,env=dict(os.environ,PYTHONPATH=str(tree)),capture_output=True,text=True)


def test_frozen_external_dependencies_pass_actual_native_path_guard(tmp_path):
    root=app(tmp_path);dependencies=frozen_release(root)
    assert verify_native(root).returncode==0
    one=version.prepare(root);tree=one/'app'
    result=verify_native(tree);assert result.returncode==0,result.stderr
    for name in dependencies:
        p=tree/name
        assert not p.is_symlink() and p.resolve().is_relative_to(tree)
        assert p.read_bytes()==(root/name).read_bytes()
    shared=tree/'keyframe_filter/collection_output/raw'
    assert shared.is_symlink() and shared.resolve()==root/'keyframe_filter/collection_output/raw'
    assert (tree/'checkpoints').is_symlink() and (tree/'lead_data').is_symlink()
    (root/next(iter(dependencies))).write_text('changed')
    assert verify_native(tree).returncode==0
    two=version.prepare(root)
    result=verify_native(two/'app')
    assert result.returncode!=0 and 'Phase3 stable dependency changed' in result.stderr
    version.verify(one)


def test_frozen_dependency_ancestor_link_tampering_rejected(tmp_path):
    import shutil
    root=app(tmp_path);frozen_release(root);one=version.prepare(root)
    parent=one/'app/keyframe_filter';shutil.rmtree(parent);parent.symlink_to(root/'keyframe_filter')
    with pytest.raises(ValueError,match='execution_snapshot_changed'):version.verify(one)


def test_missing_frozen_dependency_is_not_fabricated(tmp_path):
    root=app(tmp_path);dependencies=frozen_release(root);name=next(iter(dependencies));(root/name).unlink()
    one=version.prepare(root);version.verify(one)
    assert not (one/'app'/name).exists()
    assert verify_native(one/'app').returncode!=0


@pytest.mark.parametrize('name',['../outside.json','/absolute.json'])
def test_unsafe_frozen_dependency_rejected(tmp_path,name):
    root=app(tmp_path);frozen_release(root)
    manifest=root/'qwen3vl_local/action_prior/phase3_releases/v23/release.json'
    r=json.loads(manifest.read_text());r['external_dependencies'][name]='invalid';manifest.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='invalid_frozen_dependency_path'):version.prepare(root)
