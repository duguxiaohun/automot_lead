from copy import deepcopy
import json
from pathlib import Path
import pytest
from qwen3vl_local.audit_joint import phase4_load as fast
from qwen3vl_local.sft_new_loop_phase4 import dataset as native
from qwen3vl_local.sft_new_loop_phase4.rgb_short import data as student
from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import contract, POLICY, observation_contract


def test_reservations_validate_twice_not_per_row_and_restore(tmp_path,monkeypatch):
    (tmp_path/'plan').write_text('protocol')
    monkeypatch.setattr(native,'ROOT',tmp_path);monkeypatch.setattr(native,'PRODUCER_CHECK_PLAN','plan')
    calls=[]
    def original():calls.append(1);return {'reserved':'val'}
    monkeypatch.setattr(native,'producer_check_reservations',original)
    with fast.validated_reservations():
        for _ in range(1000):native.check_evaluation_protocol({'physical_group':'train'},{})
        with pytest.raises(ValueError,match='model selection'):
            native.check_evaluation_protocol({'physical_group':'reserved'},{})
        value=native.producer_check_reservations();value.clear()
        assert native.producer_check_reservations()=={'reserved':'val'}
    assert len(calls)==2 and native.producer_check_reservations is original


@pytest.mark.parametrize('change',['plan','owners','source','exception'])
def test_invalid_or_mutated_protocol_never_returns_success(tmp_path,monkeypatch,change):
    plan=tmp_path/'plan';plan.write_text('before')
    monkeypatch.setattr(native,'ROOT',tmp_path);monkeypatch.setattr(native,'PRODUCER_CHECK_PLAN','plan')
    state={'value':{}}
    def original():
        if state.get('source_changed'):raise ValueError('frozen source changed')
        return state['value']
    monkeypatch.setattr(native,'producer_check_reservations',original)
    with pytest.raises(ValueError):
        with fast.validated_reservations():
            if change=='plan':plan.write_text('after')
            elif change=='owners':state['value']={'other':'val'}
            elif change=='source':state['source_changed']=True
            else:raise ValueError('row invalid')
    assert native.producer_check_reservations is original


def test_isolation_and_nested_scope_rejected(monkeypatch):
    with fast.validated_reservations():
        with pytest.raises(RuntimeError,match='nested'):
            with fast.validated_reservations():pass
    monkeypatch.setattr(fast.threading,'active_count',lambda:2)
    with pytest.raises(RuntimeError,match='isolated'):
        with fast.validated_reservations():pass


def descriptor_fixture(tmp_path):
    v=tmp_path/'old';v.mkdir()
    saved=contract()
    release=json.loads(Path(fast.__file__).with_name('release_sources_v61_batch_audit_20261015.json').read_text())['files']
    for name in ('evaluate.py','reload_check.py','run.sh'):
        saved['sources'][name]=release['qwen3vl_local/sft_new_loop_phase4/rgb_short/'+name]
    saved['sources']['../../sequence_batch.py']=release['qwen3vl_local/sequence_batch.py']
    d=dict(schema=POLICY,source=saved)
    (v/'student_view.json').write_text(json.dumps(d))
    return v,d


def test_only_reviewed_runtime_changes_migrate(tmp_path):
    v,d=descriptor_fixture(tmp_path)
    _,current,changed=fast.inspect_view(v)
    assert set(changed)=={'../../sequence_batch.py','evaluate.py','reload_check.py','run.sh'}
    for field in ('data.py','model.py','train.py'):
        bad=deepcopy(d);bad['source']['sources'][field]='changed'
        (v/'student_view.json').write_text(json.dumps(bad))
        with pytest.raises(ValueError,match='beyond'):fast.inspect_view(v)
    d['source']['sources']['evaluate.py']='unknown'
    (v/'student_view.json').write_text(json.dumps(d))
    with pytest.raises(ValueError,match='unrecognized'):fast.inspect_view(v)


def test_migration_revalidates_both_modes_and_preserves_old_view(tmp_path,monkeypatch):
    v,d=descriptor_fixture(tmp_path)
    parent=tmp_path/'native';parent.mkdir();(parent/'manifest.json').write_text('{}')
    (v/'rgb_identities.json').write_text('{}')
    d.update(parent=dict(logical_path=str(parent),resolved_path=str(parent.resolve()),files={'binding':1}),
        observation_contracts={str(m):observation_contract(m) for m in (2,4)},exposure_files={},
        rgb_identities_sha256=fast.file_sha(v/'rgb_identities.json'),counts={'train':1},
        selected_ids={'train':native.digest(['one'])},exclusions=[])
    (v/'student_view.json').write_text(json.dumps(d));before=(v/'student_view.json').read_bytes()
    monkeypatch.setattr(student,'parent_binding',lambda *a:{'binding':1})
    monkeypatch.setattr(student,'exposure_inventory',lambda:(set(),{}))
    calls=[]
    monkeypatch.setattr(native,'load_dataset',lambda *a:({'train':[{'id':'one'}]},{}))
    def paired(data,mode,*a,**kw):calls.append(mode);return data,[]
    monkeypatch.setattr(student,'paired_view',paired)
    output=tmp_path/'new'
    result=fast.migrate_view(v,output,tmp_path)
    assert result['status']=='view_migrated' and calls==[2,4]
    assert (v/'student_view.json').read_bytes()==before
    new=json.loads((output/'student_view.json').read_text())
    assert new['source']==contract() and new['selected_ids']==d['selected_ids']
    assert new['runtime_migration']['old_checkpoint_resume_allowed'] is False
    monkeypatch.setattr(student,'paired_view',lambda *a,**kw:({'train':[]},[]))
    with pytest.raises(ValueError,match='membership'):fast.migrate_view(v,tmp_path/'bad',tmp_path)
    assert not (tmp_path/'bad').exists()


def test_real_frozen_file_mutation_is_rejected_at_transaction_end(tmp_path,monkeypatch):
    source=tmp_path/'source.py';source.write_text('old')
    (tmp_path/'plan').write_text(json.dumps(dict(policy='prospective_producer_manual_check_v1',
        frozen_sources={'source.py':fast.file_sha(source)},reservations=[])))
    monkeypatch.setattr(native,'ROOT',tmp_path);monkeypatch.setattr(native,'PRODUCER_CHECK_PLAN','plan')
    original=native.producer_check_reservations
    with pytest.raises(ValueError,match='sources changed'):
        with fast.validated_reservations():
            assert native.producer_check_reservations()=={}
            source.write_text('new')
    assert native.producer_check_reservations is original


def test_identity_sidecar_corruption_rejected_before_native_load(tmp_path,monkeypatch):
    v,d=descriptor_fixture(tmp_path)
    parent=tmp_path/'native';parent.mkdir();(parent/'manifest.json').write_text('{}')
    d.update(parent=dict(logical_path=str(parent),resolved_path=str(parent.resolve()),files={}),
        observation_contracts={str(m):observation_contract(m) for m in (2,4)},exposure_files={},
        rgb_identities_sha256='wrong')
    (v/'student_view.json').write_text(json.dumps(d));(v/'rgb_identities.json').write_text('{}')
    monkeypatch.setattr(student,'parent_binding',lambda *a:{})
    monkeypatch.setattr(student,'exposure_inventory',lambda:(set(),{}))
    monkeypatch.setattr(native,'load_dataset',lambda *a:pytest.fail('invalid sidecar must fail before native loading'))
    with pytest.raises(ValueError,match='sidecar'):fast.migrate_view(v,tmp_path/'new',tmp_path)
