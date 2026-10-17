from copy import deepcopy
import json
from pathlib import Path

import pytest
from qwen3vl_local.audit_joint import phase4_view_upgrade as upgrade
from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import contract,POLICY


@pytest.mark.parametrize('release',['v61_batch_audit_20261015','v70_selected_b2_20261016','v71_sampling_runtime_20261016'])
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
    assert current==contract()
    assert ('runtime_data.py' if release.startswith('v71') else 'runtime_data.py (added)') in changes
    for name in ('data.py','model.py','runtime_data.py'):
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


from qwen3vl_local.sft_new_loop_phase4.tests.test_short_rgb import native_view


def test_v71_real_view_upgrade_preserves_members_and_rejects_exposure_drift(native_view,tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import data
    from qwen3vl_local.audit_joint.io import file_sha
    parent,view=native_view;data.create(parent,view)
    expected={m:data.load_dataset(view,rgb_mode=m)[0] for m in (2,4)}
    path=view/'student_view.json';saved=json.loads(path.read_text())
    old=json.loads(Path(upgrade.__file__).with_name('release_sources_v71_sampling_runtime_20261016.json').read_text())['files']
    saved['source']['sources']['../../sequence_batch.py']=old['qwen3vl_local/sequence_batch.py']
    saved['source']['sources']['runtime_data.py']=old['qwen3vl_local/sft_new_loop_phase4/rgb_short/runtime_data.py']
    path.write_text(json.dumps(saved));before=file_sha(path)
    output=tmp_path/'upgraded'
    result=upgrade.upgrade(view,output,tmp_path)
    assert result['status']=='view_migrated' and file_sha(path)==before
    monkeypatch.setenv('P4_VALIDATED_DATA',str(tmp_path/'prepared'))
    for mode in (2,4):assert data.load_dataset(output,rgb_mode=mode)[0]==expected[mode]
    groups,files=data.exposure_inventory()
    monkeypatch.setattr(data,'exposure_inventory',lambda:(groups,dict(files,unexpected='sha')))
    with pytest.raises(ValueError,match='exposure inventory changed'):
        upgrade.upgrade(view,tmp_path/'rejected',tmp_path)
    assert not (tmp_path/'rejected/student_view.json').exists()
