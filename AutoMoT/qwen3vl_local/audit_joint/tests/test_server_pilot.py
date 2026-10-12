import hashlib
import json
import zipfile

import pytest
from qwen3vl_local.audit_joint import server_pilot as p


def row(split, scenario, route, frames=100):
    return dict(split=split, scenario=scenario, route_id=route, rgb_frames=list(range(frames)))


def test_selection_is_bounded_stratified_deterministic():
    rows = [row(s, c, str(n)) for s in ('train','val','test') for c in ('a','b') for n in range(10)]
    rows += [row('val','long','l',401),row('val','empty','e',0)]
    selected = p.select(rows)
    assert selected == p.select(list(reversed(rows)))
    assert len(selected) == len({(r['split'],r['scenario'],r['route_id']) for r in selected}) == 30
    assert {s:sum(r['split']==s for r in selected) for s in ('train','val','test')} == dict(train=10,val=10,test=10)
    assert all(len(r['rgb_frames']) == 100 for r in selected)
    assert {r['scenario'] for r in selected[:6]} == {'a','b'}


def test_selection_does_not_fallback_to_all_routes():
    assert p.select([row('train','a','long',401)]) == []
    assert p.select([row('train','a','one')]) == [row('train','a','one')]


def test_package_only_explicit_members_and_exact_sha(tmp_path):
    (tmp_path/'adapter_model.safetensors').write_bytes(b'private weight')
    receipt = p.package(tmp_path, {'summary.json': {'status':'partial'}})
    assert receipt['bytes'] < p.LIMIT
    assert receipt['zip_sha256'] == hashlib.sha256((tmp_path/'pilot_report.zip').read_bytes()).hexdigest()
    with zipfile.ZipFile(tmp_path/'pilot_report.zip') as z:
        assert z.namelist() == ['summary.json']
        assert hashlib.sha256(z.read('summary.json')).hexdigest() == receipt['files']['summary.json']
    with pytest.raises(FileExistsError):
        p.package(tmp_path, {})


def test_package_rejects_oversize_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'LIMIT', 100)
    with pytest.raises(ValueError, match='exceed'):
        p.package(tmp_path, {'summary.json':'x'*200})
    assert not (tmp_path/'pilot_report.zip').exists()


def test_missing_or_interrupted_production_keeps_diagnostics(tmp_path):
    (tmp_path/'pilot.log').write_text('worker interrupted\n')
    receipt = p.report(tmp_path, 75)
    assert receipt['status'] == 'verified'
    summary = json.loads((tmp_path/'pilot_summary.json').read_text())
    assert summary['status'] == 'incomplete'
    assert summary['command_exit_code'] == 75
    assert not summary['training_ready']
    assert summary['sufficient_for_full_build'] is None
    assert summary['errors']
