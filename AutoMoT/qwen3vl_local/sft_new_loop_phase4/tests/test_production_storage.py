import gzip
import json
import shutil
import pytest
from qwen3vl_local.sft_new_loop_phase4 import production_storage as s
from qwen3vl_local.sft_new_loop_phase4.identity import file_sha


def fixture(tmp_path):
    src=tmp_path/'source';src.mkdir()
    with s.write_route(src/'route.jsonl.gz',compressed=True) as f:
        f.write('{"record":"测试"}\n')
    (src/'index.json').write_text(json.dumps(dict(routes=[dict(artifact='route.jsonl.gz',sha256=file_sha(src/'route.jsonl.gz'))])))
    out=tmp_path/'dataset';out.mkdir()
    sha=s.install(src/'index.json',out)
    manifest=dict(production_index_sha256=file_sha(src/'index.json'),production_reference_sha256=sha)
    return src,out,manifest


def test_shared_replay_reloads_across_cwd_without_copy(tmp_path,monkeypatch):
    src,out,m=fixture(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    assert s.resolve(out,m)==src/'index.json'
    assert list((out/'production').iterdir())==[out/'production/index.json']
    with s.open_route(src/'route.jsonl.gz') as f:
        assert json.load(f)=={'record':'测试'}


@pytest.mark.parametrize('name',['index.json','route.jsonl.gz'])
@pytest.mark.parametrize('damage',['delete','bytes','same_bytes_retarget'])
def test_source_drift_is_rejected(tmp_path,name,damage):
    src,out,m=fixture(tmp_path)
    p=src/name
    if damage=='delete':p.unlink()
    elif damage=='bytes':p.write_bytes(b'changed')
    else:
        target=tmp_path/'replacement';shutil.copyfile(p,target);p.unlink();p.symlink_to(target)
    with pytest.raises((ValueError,FileNotFoundError)):
        s.resolve(out,m)


def test_copy_is_independent_but_reference_requires_source(tmp_path):
    src,out,m=fixture(tmp_path)
    copy=tmp_path/'copy';copy.mkdir()
    assert s.install(src/'index.json',copy,mode='copy') is None
    shutil.rmtree(src)
    assert s.resolve(copy,dict(m,production_reference_sha256=None))==copy/'production/index.json'
    with pytest.raises(FileNotFoundError):s.resolve(out,m)


def test_gzip_stream_deterministic_roundtrip_and_exclusive_write(tmp_path):
    data='中文 causal evidence\n'*10000
    for name in ('a.gz','b.gz'):
        with s.write_route(tmp_path/name,compressed=True) as f:
            f.write(data)
    assert (tmp_path/'a.gz').read_bytes()==(tmp_path/'b.gz').read_bytes()
    assert gzip.decompress((tmp_path/'a.gz').read_bytes()).decode()==data
    with pytest.raises(FileExistsError):
        with s.write_route(tmp_path/'a.gz',compressed=True):pass
    p=tmp_path/'a.gz';p.write_bytes(p.read_bytes()[:-6])
    with pytest.raises(EOFError):
        with s.open_route(p) as f:f.read()


def test_invalid_route_paths_not_installed(tmp_path):
    src,out,m=fixture(tmp_path)
    index=src/'index.json';r=json.loads(index.read_text());r['routes'][0]['artifact']='../escape'
    index.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='artifact'):s.capture(index)


@pytest.mark.parametrize('damage',['encoding','contract','partial_encoding'])
def test_resume_never_replaces_legacy_or_partial_evidence(tmp_path,monkeypatch,damage):
    from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import make_bundle
    from qwen3vl_local.sft_new_loop_phase4 import teacher_replay, candidate_pool
    pool,_,_,_=make_bundle(tmp_path)
    monkeypatch.setattr(candidate_pool,'validate',lambda _:None)
    if damage=='contract':
        index=tmp_path/'index.json';r=json.loads(index.read_text());r['teacher']['version']='old'
        index.write_text(json.dumps(r))
    if damage=='partial_encoding':(tmp_path/'index.json').unlink()
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError):
        teacher_replay.produce(pool,tmp_path,tmp_path,workers=1,compression='none' if damage=='contract' else 'gzip')
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir() if p.name!='.production.lock'}


@pytest.fixture
def batched_pool(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4 import candidate_pool,teacher_replay
    root=tmp_path/'raw'
    for i in range(5):
        route=root/'Scenario'/f'Town01_Rep0_{i}_0_route0'
        for kind,ext in [('rgb','jpg'),('metas','pkl')]:
            (route/kind).mkdir(parents=True)
            (route/kind/f'0004.{ext}').touch()
    pool=candidate_pool.scan(root,tmp_path/'pool.json')
    def records(root,route):
        yield dict(kind='frame_disposition',**{k:route[k] for k in ('scenario','route_id','physical_group','split')},
                   frame_id=4,disposition='irrelevant',questions=[],reasons=['synthetic mechanics test'])
    monkeypatch.setattr(teacher_replay,'route_records',records)
    return root,pool,tmp_path/'production'


def keys(pool,selection):
    return {(r['scenario'],r['route_id']) for r in pool['routes'][selection]}


def snapshot(path):
    return {p.name:p.read_bytes() for p in path.iterdir() if p.is_file() and p.name!='.production.lock'}


@pytest.mark.parametrize('compression',['none','gzip'])
def test_batches_accumulate_in_pool_order_and_published_subsets_are_noop(batched_pool,tmp_path,compression):
    from qwen3vl_local.sft_new_loop_phase4.teacher_replay import produce
    root,pool,out=batched_pool
    first=produce(pool,root,out,route_keys=keys(pool,slice(2,None)),workers=1,compression=compression)
    assert first['processed_routes']==3 and not first['complete_causal_production']
    second=produce(pool,root,out,route_keys=keys(pool,slice(0,2)),workers=1,compression=compression)
    assert second['processed_routes']==5 and second['complete_causal_production']
    index=json.loads((out/'index.json').read_text())
    assert [r['route_id'] for r in index['routes']]==[r['route_id'] for r in pool['routes']]
    dataset=tmp_path/'data';dataset.mkdir()
    sha=s.install(out/'index.json',dataset)
    manifest=dict(production_index_sha256=file_sha(out/'index.json'),production_reference_sha256=sha)
    before=snapshot(out)
    for selection in (slice(0,2),slice(2,None),slice(None)):
        result=produce(pool,root,out,route_keys=keys(pool,selection),workers=1,compression=compression)
        assert result['processed_routes']==5 and snapshot(out)==before
        assert s.resolve(dataset,manifest)==out/'index.json'


def test_published_partial_pool_cannot_expand(batched_pool,tmp_path):
    from qwen3vl_local.sft_new_loop_phase4.teacher_replay import produce
    root,pool,out=batched_pool
    produce(pool,root,out,route_keys=keys(pool,slice(0,2)),workers=1)
    dataset=tmp_path/'data';dataset.mkdir();s.install(out/'index.json',dataset)
    before=snapshot(out)
    with pytest.raises(ValueError,match='immutable'):
        produce(pool,root,out,workers=1)
    assert before==snapshot(out)


def test_previous_batch_corruption_rejected_without_index_replacement(batched_pool):
    from qwen3vl_local.sft_new_loop_phase4.teacher_replay import produce
    root,pool,out=batched_pool
    produce(pool,root,out,route_keys=keys(pool,slice(0,2)),workers=1)
    index=out/'index.json';before=index.read_bytes()
    name=json.loads(before)['routes'][0]['artifact']
    (out/name).write_text('broken')
    with pytest.raises(ValueError):produce(pool,root,out,workers=1)
    assert index.read_bytes()==before


def test_builder_and_reference_install_share_exclusive_lock(batched_pool,tmp_path):
    from qwen3vl_local.sft_new_loop_phase4.teacher_replay import produce
    root,pool,out=batched_pool;produce(pool,root,out,workers=1)
    dataset=tmp_path/'data';dataset.mkdir();before=snapshot(out)
    with s.production_lock(out):
        with pytest.raises(ValueError,match='active'):produce(pool,root,out,workers=1)
        with pytest.raises(ValueError,match='active'):s.install(out/'index.json',dataset)
    assert snapshot(out)==before and not list(dataset.iterdir())
