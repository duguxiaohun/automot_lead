import json
import os
from pathlib import Path
import sys

import pytest

from qwen3vl_local.audit_joint import phase4_production_cache as cache
from qwen3vl_local.audit_joint.benchmark_runtime import run_owned
from qwen3vl_local.sft_new_loop_phase4 import teacher_replay as replay, teacher_rules
from qwen3vl_local.sft_new_loop_phase4.identity import digest, file_sha, write_json


def fixture(tmp_path):
    native=tmp_path/'native';production=native/'production';production.mkdir(parents=True)
    routes=[dict(scenario='scenario',route_id=str(n),physical_group=str(n),split='train',rgb_frames=[4,5])
            for n in range(2)]
    pool=dict(routes=routes);pool['sha256']=digest(pool)
    write_json(native/'candidate_pool.json',pool)
    header=dict(policy=replay.POLICY,teacher=teacher_rules.identity(),candidate_pool_sha256=pool['sha256'])
    receipts=[]
    for route in routes:
        path=production/(route['route_id']+'.jsonl')
        frames=[dict(kind='frame_disposition',frame_id=f,disposition='irrelevant',reasons=['no_event'],
                     questions=[],**{k:v for k,v in route.items() if k!='rgb_frames'}) for f in route['rgb_frames']]
        path.write_text('\n'.join(json.dumps(r) for r in [dict(kind='header',**header),*frames])+'\n')
        receipts.append(replay.inspect_route(path,route,header))
    index=dict(**header,routes=receipts);index['sha256']=digest(index)
    write_json(production/'index.json',index)
    write_json(native/'manifest.json',dict(candidate_pool_sha256=file_sha(native/'candidate_pool.json'),
                                         production_index_sha256=file_sha(production/'index.json')))
    return native,tmp_path/'cache',pool


def run(native,directory,pool):
    with cache.production_routes(native,directory) as c:
        result=replay.accounting(pool,c.index_path)
    return result,c


def test_completed_routes_reused_with_identical_accounting(tmp_path,monkeypatch):
    native,directory,pool=fixture(tmp_path)
    expected=replay.accounting(pool,native/'production/index.json')
    actual,first=run(native,directory,pool)
    assert actual==expected and first.fresh==2 and first.reused==0
    def forbidden(*a):pytest.fail('previously checked route must not be reparsed')
    monkeypatch.setattr(replay,'inspect_route',forbidden)
    actual,second=run(native,directory,pool)
    assert actual==expected and second.fresh==0 and second.reused==2


@pytest.mark.parametrize('kind',['checksum','receipt','binding','artifact','index','source'])
def test_changed_cache_input_or_source_rejected(tmp_path,monkeypatch,kind):
    native,directory,pool=fixture(tmp_path);run(native,directory,pool)
    path=next((directory/'routes').glob('*.json'));saved=json.loads(path.read_text())
    if kind=='checksum':saved['sha256']='broken';write_json(path,saved)
    elif kind in ('receipt','binding'):
        if kind=='receipt':saved['receipt']['frames']=999
        else:saved['binding']['route_sha256']='wrong'
        saved['sha256']=digest({k:v for k,v in saved.items() if k!='sha256'});write_json(path,saved)
    elif kind=='artifact':
        with (native/'production/0.jsonl').open('a') as f:f.write('{}\n')
    elif kind=='index':
        with (native/'production/index.json').open('a') as f:f.write(' ')
    else:
        original=cache.source_state
        def changed():return dict(original(),adapters={'changed':'source'})
        monkeypatch.setattr(cache,'source_state',changed)
    with pytest.raises(ValueError):run(native,directory,pool)


def test_source_changes_during_validation_never_publish_route(tmp_path,monkeypatch):
    native,directory,pool=fixture(tmp_path);original=replay.inspect_route
    state=cache.source_state();changed=False
    monkeypatch.setattr(cache,'source_state',lambda:dict(state,changed=True) if changed else state)
    def inspect(*a):
        nonlocal changed
        result=original(*a);changed=True;return result
    monkeypatch.setattr(replay,'inspect_route',inspect)
    with pytest.raises(ValueError,match='sources changed'):run(native,directory,pool)
    assert not list((directory/'routes').glob('*.json'))
    assert replay.inspect_route is inspect


def test_partial_failure_preserves_only_completed_routes(tmp_path,monkeypatch):
    native,directory,pool=fixture(tmp_path);original=replay.inspect_route
    def inspect(path,*args):
        if Path(path).name=='1.jsonl':raise ValueError('bad second route')
        return original(path,*args)
    monkeypatch.setattr(replay,'inspect_route',inspect)
    with pytest.raises(ValueError,match='bad second'):run(native,directory,pool)
    assert len(list((directory/'routes').glob('*.json')))==1
    monkeypatch.setattr(replay,'inspect_route',original)
    _,result=run(native,directory,pool)
    assert result.reused==1 and result.fresh==1


def test_real_timeout_preserves_receipt_and_releases_lock(tmp_path):
    native,directory,pool=fixture(tmp_path)
    script=f'''
import time
from pathlib import Path
from qwen3vl_local.audit_joint.phase4_production_cache import production_routes
from qwen3vl_local.sft_new_loop_phase4 import teacher_replay as replay
with production_routes(Path({str(native)!r}),Path({str(directory)!r})) as c:
    row=c.index['routes'][0]
    replay.inspect_route(c.index_path.parent/row['artifact'],c.pool['routes'][0],c.header)
    time.sleep(30)
'''
    code=run_owned([sys.executable,'-u','-c',script],tmp_path/'worker.log',dict(os.environ),2)
    assert code==124 and len(list((directory/'routes').glob('*.json')))==1
    _,result=run(native,directory,pool)
    assert result.reused==1 and result.fresh==1


def test_nested_cache_foreign_route_and_native_output_rejected(tmp_path):
    native,directory,pool=fixture(tmp_path);original=replay.inspect_route
    with cache.production_routes(native,directory) as c:
        with pytest.raises(RuntimeError,match='nested'):
            with cache.production_routes(native,directory):pass
        with pytest.raises(ValueError,match='outside cache'):
            replay.inspect_route(c.index_path.parent/'0.jsonl',pool['routes'][1],c.header)
    assert replay.inspect_route is original
    with pytest.raises(ValueError,match='outside native'):
        with cache.production_routes(native,native/'cache'):pass
    assert not (native/'cache').exists()
