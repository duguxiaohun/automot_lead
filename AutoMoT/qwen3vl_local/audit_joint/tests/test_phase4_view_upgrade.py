from copy import deepcopy
import json
from pathlib import Path

import pytest
from qwen3vl_local.audit_joint import phase4_view_upgrade as upgrade
from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import contract,POLICY


@pytest.mark.parametrize('release',['v61_batch_audit_20261015','v70_selected_b2_20261016'])
def test_known_old_view_sources_allowed_but_unknown_or_semantic_drift_refused(tmp_path,release):
    old=json.loads(Path(upgrade.__file__).with_name('release_sources_'+release+'.json').read_text())['files']
    saved=contract();prefix='qwen3vl_local/sft_new_loop_phase4/rgb_short/'
    for name in list(saved['sources']):
        path=Path(prefix)/name;parts=[]
        for part in path.parts:
            if part=='..':parts.pop()
            else:parts.append(part)
        key='/'.join(parts)
        if key in old:saved['sources'][name]=old[key]
        else:del saved['sources'][name]
    for name in saved['automatic_admission_sources']:
        saved['automatic_admission_sources'][name]=old['qwen3vl_local/audit_joint/'+name]
    descriptor=dict(schema=POLICY,source=saved)
    path=tmp_path/'student_view.json';path.write_text(json.dumps(descriptor))
    _,current,changes=upgrade.inspect_view(tmp_path)
    assert current==contract() and 'runtime_data.py (added)' in changes
    for name in ('data.py','model.py'):
        bad=deepcopy(descriptor);bad['source']['sources'][name]='unknown'
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):upgrade.inspect_view(tmp_path)


def test_upgrade_restores_original_inspector_on_failure(monkeypatch,tmp_path):
    original=upgrade.phase4_load.inspect_view
    def fail(*a,**kw):
        assert upgrade.phase4_load.inspect_view is upgrade.inspect_view
        raise ValueError('failed membership')
    monkeypatch.setattr(upgrade.phase4_load,'migrate_view',fail)
    with pytest.raises(ValueError):upgrade.upgrade(tmp_path,tmp_path/'out',tmp_path)
    assert upgrade.phase4_load.inspect_view is original
