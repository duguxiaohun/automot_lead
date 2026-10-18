import json
from types import SimpleNamespace
import pytest
from qwen3vl_local.audit_joint import build_storage, replay_recovery, server_finish
from qwen3vl_local.sft_new_loop_phase4 import teacher_replay as replay
from qwen3vl_local.sft_new_loop_phase4.tests.test_production_storage import batched_pool, snapshot


@pytest.mark.parametrize('free', [21, 40, 100, 300])
def test_reserve_is_absolute_not_twelve_gib_write_cap(tmp_path, monkeypatch, free):
    monkeypatch.setattr(build_storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=free*2**30))
    result=build_storage.plan(tmp_path)
    assert result['reserve_bytes']==20*2**30
    assert result['writable_before_reserve_bytes']==(free-20)*2**30
    assert result['can_start'] and result['sufficient_for_complete_build'] is None


@pytest.mark.parametrize('reserve', ['nan','inf','-1','0','19.9','bad'])
def test_invalid_or_unsafe_reserve_rejected(tmp_path,reserve):
    with pytest.raises(ValueError):build_storage.plan(tmp_path,reserve)


def test_actual_low_space_still_blocks(tmp_path,monkeypatch):
    monkeypatch.setattr(build_storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=19*2**30))
    assert not build_storage.plan(tmp_path)['can_start']


def fail_replay(*args,**kwargs):
    pytest.fail('Recovery must never compute a route')


def test_recover_missing_index_native_receipts_and_reload(batched_pool,monkeypatch):
    raw,pool,out=batched_pool
    original=replay.produce(pool,raw,out,workers=1,compression='gzip')
    index_bytes=(out/'index.json').read_bytes()
    (out/'index.json').unlink()
    (out/'index.json.pending').write_text('interrupted write retained')
    before=snapshot(out)
    monkeypatch.setattr(replay,'route_records',fail_replay)
    monkeypatch.setattr(replay,'produce',fail_replay)
    report=replay_recovery.recover(pool,out)
    assert report['complete_causal_production'] and report['recomputed_routes']==0
    assert report['index_sha256']==original['index_sha256']
    assert (out/'index.json').read_bytes()==index_bytes
    assert before=={k:v for k,v in snapshot(out).items() if k!='index.json'}
    assert replay_recovery.recover(pool,out)['recovery']=='validated_existing_index'


@pytest.mark.parametrize('damage',['missing','truncated','stale_header','published_missing','partial_index','bad_index'])
def test_recovery_refuses_damaged_evidence_without_replacing_it(batched_pool,monkeypatch,damage):
    from qwen3vl_local.sft_new_loop_phase4.production_storage import write_route
    from qwen3vl_local.sft_new_loop_phase4.identity import digest
    raw,pool,out=batched_pool
    replay.produce(pool,raw,out,workers=1,compression='gzip')
    index=out/'index.json';record=json.loads(index.read_text())
    if damage=='partial_index':
        record['routes']=record['routes'][:-1]
        record['sha256']=digest({k:v for k,v in record.items() if k!='sha256'})
        index.write_text(json.dumps(record))
    elif damage=='bad_index':index.write_text('{}')
    else:
        index.unlink();route=out/record['routes'][0]['artifact']
        if damage=='missing':route.unlink()
        elif damage=='truncated':route.write_bytes(route.read_bytes()[:-8])
        elif damage=='stale_header':
            route.unlink()
            with write_route(route,compressed=True) as stream:stream.write('{"kind":"header","teacher":"old"}\n')
        elif damage=='published_missing':(out/'published.json').write_text('{}')
    before=snapshot(out)
    monkeypatch.setattr(replay,'route_records',fail_replay)
    monkeypatch.setattr(replay,'produce',fail_replay)
    with pytest.raises((ValueError,FileNotFoundError,EOFError)):
        replay_recovery.recover(pool,out)
    assert snapshot(out)==before


def test_build_recovery_failure_packaged_without_falling_back_to_replay(batched_pool,tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint import first_epoch_budget
    raw,pool,production=batched_pool
    pilot=tmp_path/'pilot';pilot.mkdir()
    (pilot/'candidates.json').write_text(json.dumps(pool))
    out=tmp_path/'new';out.mkdir()
    monkeypatch.setattr(first_epoch_budget,'profile',lambda _: {})
    monkeypatch.setattr(replay,'produce',fail_replay)
    result=server_finish.build(pilot,raw,out,tmp_path/'model',reuse_complete_replay=True)
    assert result['status']=='blocked' and result['stage']=='complete_replay'
    assert 'production directory missing' in result['errors'][0]
    server_finish.package(out,2)
    assert (out/'handoff/fullflow_report.zip').is_file()
