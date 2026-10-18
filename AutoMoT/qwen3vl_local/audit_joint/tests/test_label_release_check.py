import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from qwen3vl_local.audit_joint import label_release_check as check
from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS


def fixture(tmp_path, monkeypatch, all_events=True):
    root = tmp_path/'data4'; root.mkdir()
    rows = [dict(id=event, episode=dict(event=event, instance_id=event), scenario='scene',
        route_id=event, physical_group=event, split='train', observation=dict(frame_id=6),
        edge='return' if event == 'U-E2' else 'proceed', target='YES', slice='readiness',
        label_basis='weak_rule_teacher') for event in (EVENTS if all_events else ['U-E2'])]
    raw = ''.join(json.dumps(row)+'\n' for row in rows).encode()
    (root/'train.jsonl').write_bytes(raw)
    (root/'manifest.json').write_text(json.dumps(dict(contract={'old': True},
        files={'train.jsonl': hashlib.sha256(raw).hexdigest()}, counts={'train': len(rows)})))
    monkeypatch.setattr(check, 'contract_differences', lambda value: ['old source'])
    monkeypatch.setattr(check, 'source_check', lambda *a: dict(status='matched'))
    original = check.plan
    monkeypatch.setattr(check, 'plan', lambda rows, **kw: original(rows, budget=8*len(EVENTS), **kw))
    return root


def test_old_pool_inventory_does_not_grant_admission_or_modify_inputs(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    out = tmp_path/'out'
    assert check.run(root, out) == 0
    report = json.loads((out/'label_release_report.json').read_text())
    assert report['training_authorized'] is False
    inv = report['inventory']
    assert inv['contract_status'] == 'different'
    for mode in ('2', '4'):
        p = inv['plans'][mode]
        assert p['event_counts'] == dict.fromkeys(EVENTS, 8)
        assert p['action_minimums']['U-E2']['return/YES']['actual'] >= 1
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    receipt = json.loads((out/'label_release_package.json').read_text())
    with zipfile.ZipFile(out/receipt['archive']) as bundle:
        assert bundle.testzip() is None
        assert hashlib.sha256(bundle.read('label_release_report.json')).hexdigest() == receipt['members']['label_release_report.json']
    with pytest.raises(FileExistsError): check.run(root, out)


@pytest.mark.parametrize('damage', ['bytes', 'manifest', 'count', 'split', 'duplicate', 'source'])
def test_bad_input_packaged_as_blocked(tmp_path, monkeypatch, damage):
    root = fixture(tmp_path, monkeypatch)
    path = root/'manifest.json'; m = json.loads(path.read_text())
    if damage == 'bytes': (root/'train.jsonl').write_bytes(b'{}\n')
    elif damage == 'manifest': path.write_text('{')
    elif damage == 'count': m['counts']['train'] += 1; path.write_text(json.dumps(m))
    elif damage in ('split', 'duplicate'):
        rows = [json.loads(x) for x in (root/'train.jsonl').read_text().splitlines()]
        if damage == 'split': rows[0]['split'] = 'val'
        else: rows[1]['id'] = rows[0]['id']
        raw = ''.join(json.dumps(row)+'\n' for row in rows).encode()
        (root/'train.jsonl').write_bytes(raw)
        m['files']['train.jsonl'] = hashlib.sha256(raw).hexdigest(); path.write_text(json.dumps(m))
    else: monkeypatch.setattr(check, 'source_check', lambda *a: dict(status='mismatch'))
    out = tmp_path/'out'; assert check.run(root, out) == 2
    report = json.loads((out/'label_release_report.json').read_text())
    assert report['status'] == 'blocked' and report['errors']
    assert 'inventory' not in report
    assert (out/'label_release_package.json').is_file()


def test_incomplete_event_pool_is_not_sampled(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch, all_events=False)
    monkeypatch.setattr(check, 'plan', lambda *a, **kw: pytest.fail('must not sample incomplete pool'))
    report = check.inventory(root)
    assert report['sampling_status'] == 'blocked_missing_events'
    assert len(report['missing_events']) == 9
    with pytest.raises(ValueError, match='outside'): check.run(root, root/'output')


def test_symlink_retarget_during_plan_rejected(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch)
    path = root/'train.jsonl'; first = root/'old.jsonl'; path.rename(first)
    second = root/'new.jsonl'; second.write_bytes(first.read_bytes()); path.symlink_to(first)
    original = check.plan
    def retarget(*args, **kw):
        path.unlink(); path.symlink_to(second)
        return original(*args, **kw)
    monkeypatch.setattr(check, 'plan', retarget)
    with pytest.raises(ValueError, match='changed'): check.inventory(root)
