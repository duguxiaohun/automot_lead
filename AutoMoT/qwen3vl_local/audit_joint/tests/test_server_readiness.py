import json
from pathlib import Path
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
