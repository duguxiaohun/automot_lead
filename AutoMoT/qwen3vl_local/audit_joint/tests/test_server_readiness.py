import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from qwen3vl_local.audit_joint import server_readiness as sr


def test_inventory_reports_legacy_and_invalid_without_certifying(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(contract={'task':'old'},counts={'train':8})))
    sub=tmp_path/'broken';sub.mkdir();(sub/'manifest.json').write_text('{')
    (tmp_path/'adapter_model.safetensors').write_bytes(b'weight bytes not read')
    (tmp_path/'train.jsonl').write_bytes(b'not parsed as rows')
    report=sr.scan(tmp_path)
    rows={r['logical_path']:r for r in report['records']}
    assert rows[str(tmp_path/'manifest.json')]['declared_task']=='old'
    assert all(r['metadata_verified'] is False for r in rows.values())
    assert rows[str(sub/'manifest.json')]['status']=='unreadable_or_invalid'
    assert rows[str(tmp_path/'adapter_model.safetensors')]['status']=='exists_contents_not_read'
    assert 'sha256' not in rows[str(tmp_path/'train.jsonl')]


def test_inventory_bounds_and_does_not_follow_links(tmp_path,monkeypatch):
    monkeypatch.setattr(sr,'METADATA_LIMIT',16)
    external=tmp_path/'external';external.mkdir();(external/'manifest.json').write_text('{}')
    root=tmp_path/'root';root.mkdir();(root/'linked').symlink_to(external,target_is_directory=True)
    (root/'manifest.json').write_text('x'*17)
    (root/'view.json').symlink_to(external/'manifest.json')
    r=sr.scan(root)
    assert len(r['records'])==2
    assert {x['status'] for x in r['records']}=={'oversized_metadata_not_read','symlink_not_read'}
    assert sr.scan(root,max_records=1)['truncated']


def test_inventory_current_view_action_and_production_names(tmp_path):
    for name in ('student_view.json', 'effective_pool_manifest.json', 'index.json', 'candidates.json'):
        sr.write(tmp_path/name, dict(policy='fixture', routes=[{'split':'train'},{'split':'val'}]))
    records=sr.scan(tmp_path)['records']
    assert len(records)==4
    assert all(r['declared_route_count']==2 and r['declared_route_splits']=={'train':1,'val':1} for r in records)
    assert all(r['status']=='parsed_unverified_metadata' for r in records)


def test_source_binding_changed_missing_and_unsafe(tmp_path):
    source=tmp_path/'code.py';source.write_text('pass')
    spec=tmp_path/'expected.json';sr.write(spec,dict(files={'code.py':sr.file_sha(source)}))
    assert sr.source_check(tmp_path,spec)['status']=='matched'
    source.write_text('other');assert sr.source_check(tmp_path,spec)['status']=='mismatch'
    source.unlink();assert sr.source_check(tmp_path,spec)['files'][0]['status']=='missing'
    sr.write(spec,dict(files={'../escape':'anything'}))
    with pytest.raises(ValueError,match='unsafe'):sr.source_check(tmp_path,spec)


def test_handoff_includes_only_diagnostics_and_capped_logs(tmp_path,monkeypatch):
    monkeypatch.setattr(sr,'LOG_LIMIT',10)
    app=tmp_path/'app';app.mkdir();(app/'x.py').write_text('pass')
    expected=tmp_path/'expected.json';sr.write(expected,dict(files={'x.py':sr.file_sha(app/'x.py')}))
    checkpoints=app/'checkpoints';checkpoints.mkdir()
    (checkpoints/'adapter_model.safetensors').write_bytes(b'do not package weights')
    log=tmp_path/'tests.log';log.write_bytes(b'01234567890123456789')
    out=tmp_path/'report'
    r=sr.collect(checkpoints,out,app=app,expected=expected,logs=[log])
    assert r['training_ready'] is None and r['source_status']=='matched'
    assert sr.verify(r['package'])['status']=='verified'
    with zipfile.ZipFile(r['package']) as z:
        assert set(z.namelist())=={'summary.json','source_check.json','inventory.json','tests.log','receipt.json'}
        assert z.read('tests.log')==b'0123456789'
        assert json.loads(z.read('summary.json'))['logs'][0]['truncated']
    with pytest.raises(FileExistsError):sr.collect(checkpoints,out,app=app,expected=expected)
    with pytest.raises(ValueError,match='logs'):sr.collect(checkpoints,tmp_path/'bad',app=app,expected=expected,logs=[checkpoints/'adapter_model.safetensors'])


def test_handoff_rejects_tampered_bytes(tmp_path):
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('summary.json','changed')
        z.writestr('receipt.json',json.dumps(dict(schema='server_readiness_v1',files={'summary.json':{'bytes':7,'sha256':'wrong'}})))
    with pytest.raises(ValueError,match='content mismatch'):sr.verify(path)


@pytest.mark.parametrize('extra', ['test_capacity_diagnostic.py', 'nested/conftest.py', 'orphan.pyc'])
def test_source_check_reports_overlay_python_without_deleting(tmp_path, extra):
    name='qwen3vl_local/sft_new_loop_phase3'
    root=tmp_path/name;root.mkdir(parents=True)
    code=root/'sampling.py';code.write_text('pass')
    expected=tmp_path/'release.json';sr.write(expected,dict(files={name+'/sampling.py':sr.file_sha(code)}))
    assert sr.source_check(tmp_path,expected)['status']=='matched'
    other=root/extra;other.parent.mkdir(parents=True,exist_ok=True);other.write_text('do not execute')
    result=sr.source_check(tmp_path,expected)
    assert result['status']=='matched'
    assert result['unexpected_python_files']==[name+'/'+extra]
    assert result['files'][0]['status']=='matched' and other.read_text()=='do not execute'


def test_source_check_does_not_treat_probe_artifacts_as_runtime(tmp_path):
    name='qwen3vl_local/sft_new_loop_phase3'
    root=tmp_path/name;root.mkdir(parents=True)
    code=root/'sampling.py';code.write_text('pass')
    old=root/'probe_output'/'old.py';old.parent.mkdir();old.write_text('old diagnostic')
    expected=tmp_path/'release.json';sr.write(expected,dict(files={name+'/sampling.py':sr.file_sha(code)}))
    assert sr.source_check(tmp_path,expected)['status']=='matched'


def run_fixture(app, expected, env=None):
    code = ('from pathlib import Path; from qwen3vl_local.audit_joint.server_readiness import run_tests; '
            'import sys; sys.exit(run_tests(Path(sys.argv[1]), Path(sys.argv[2])))')
    environ = dict(os.environ, PYTHONPATH=str(Path(sr.__file__).resolve().parents[2]))
    environ.update(env or {})
    return subprocess.run([sys.executable, '-c', code, str(app), str(expected)],
                          env=environ, text=True, capture_output=True, timeout=60)


def test_overlay_executes_published_tests_and_preserves_unrelated_files(tmp_path):
    app=tmp_path/'app';root=app/'qwen3vl_local/sft_new_loop_phase3';root.mkdir(parents=True)
    test=root/'test_published.py';test.write_text('def test_current():\n    assert True\n')
    expected=app/'release.json';sr.write(expected,dict(files={str(test.relative_to(app)):sr.file_sha(test)}))
    for p in (root/'test_capacity_diagnostic.py', root/'conftest.py', app/'conftest.py'):
        p.write_text('raise RuntimeError("retained local file must not execute")\n')
    (app/'pytest.ini').write_text('[pytest]\naddopts = --not-a-pytest-option\n')
    result=run_fixture(app,expected,dict(PYTEST_ADDOPTS='--also-invalid',PYTEST_PLUGINS='nonexistent_local_plugin'))
    assert result.returncode==0, result.stdout+result.stderr
    assert '1 passed' in result.stdout
    assert (root/'test_capacity_diagnostic.py').exists() and (root/'conftest.py').exists()


@pytest.mark.parametrize('bytecode', [False,True])
def test_overlay_blocks_actual_unlisted_dependency_import_before_execution(tmp_path,bytecode):
    import py_compile
    app=tmp_path/'app';root=app/'qwen3vl_local/sft_new_loop_phase3';root.mkdir(parents=True)
    test=root/'test_published.py';test.write_text('import unlisted_dependency\ndef test_current(): pass\n')
    expected=app/'release.json';sr.write(expected,dict(files={str(test.relative_to(app)):sr.file_sha(test)}))
    marker=tmp_path/'executed';extra=root/'unlisted_dependency.py'
    extra.write_text('from pathlib import Path\nPath('+repr(str(marker))+').write_text("executed")\n')
    if bytecode:
        py_compile.compile(str(extra),cfile=str(extra.with_suffix('.pyc')),doraise=True)
        extra.unlink()
    result=run_fixture(app,expected)
    assert result.returncode!=0 and 'unlisted local runtime module' in result.stdout, result.stdout+result.stderr
    assert not marker.exists()


def test_overlay_rejects_changed_published_test(tmp_path):
    root=tmp_path/'qwen3vl_local/sft_new_loop_phase3';root.mkdir(parents=True)
    test=root/'test_published.py';test.write_text('def test_current(): assert True\n')
    expected=tmp_path/'release.json';sr.write(expected,dict(files={str(test.relative_to(tmp_path)):sr.file_sha(test)}))
    test.write_text('raise RuntimeError("changed release must not execute")\n')
    result=run_fixture(tmp_path,expected)
    assert result.returncode==2 and 'source_mismatch' in result.stdout


def test_overlay_rejects_checked_package_from_another_checkout(tmp_path):
    root=tmp_path/'qwen3vl_local/sft_new_loop_phase3';root.mkdir(parents=True)
    test=root/'test_published.py'
    test.write_text('import qwen3vl_local.sft_new_loop_phase3.sampling\ndef test_current(): pass\n')
    expected=tmp_path/'release.json'
    sr.write(expected,dict(files={str(test.relative_to(tmp_path)):sr.file_sha(test)}))
    result=run_fixture(tmp_path,expected)
    assert result.returncode!=0 and 'checked runtime module loaded outside application' in result.stdout, result.stdout+result.stderr
