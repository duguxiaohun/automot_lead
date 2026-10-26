import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from .test_core import record, tiny, inputs
from .test_coverage_probe import fixture_processor, supported_rows
from qwen3vl_local.audit_joint.abc_delivery import sha, write
from qwen3vl_local.action_expert_ablation.geometry_abc import indexed_rows as indexed
from qwen3vl_local.action_expert_ablation.geometry_abc import contracts, runtime, worker, probe_inputs, diagnostic_coverage, planning


def rows_file(path, records, split='train'):
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    return indexed.Rows(indexed.Store(path,sha(path),split,'/unused'))


def test_rows_views_pickle_and_cache_are_bounded(tmp_path):
    records=[record(i) for i in range(25)]
    rows=rows_file(tmp_path/'train.jsonl',records)
    assert list(rows)==[runtime.sample(r,'/unused') for r in records]
    assert len(rows.store.cache)==8
    view=rows.ordered([records[i]['id'] for i in (24,1,24,0)])
    assert [r['abc_record']['id'] for r in view[1::2]]==['A1','A0']
    copied=pickle.loads(pickle.dumps(view))
    assert not copied.store.cache
    assert list(copied)==list(view)
    assert list(rows[::-1])==list(reversed(rows))
    with pytest.raises(IndexError):rows[25]


@pytest.mark.parametrize('change',['same_length','truncate','replace'])
def test_consumption_rehashes_indexed_bytes(tmp_path,change):
    p=tmp_path/'train.jsonl';rows=rows_file(p,[record()])
    raw=p.read_bytes()
    altered=raw.replace(b'"A0"',b'"B0"') if change!='truncate' else raw[:10]
    if change=='replace':
        other=tmp_path/'replacement';other.write_bytes(altered);other.replace(p)
    else:p.write_bytes(altered)
    with pytest.raises(ValueError,match='indexed_record_changed'):rows[0]['abc_record']


@pytest.mark.parametrize('change',['hash','duplicate','split'])
def test_index_rejects_bad_input(tmp_path,change):
    records=[record()];p=tmp_path/'train.jsonl'
    if change=='duplicate':records*=2
    if change=='split':records[0]['split']='test'
    p.write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError):indexed.Store(p,'bad' if change=='hash' else sha(p),'train','/unused')


def test_manifest_and_route_checks_match_eager_loader(tmp_path):
    from qwen3vl_local.audit_joint import label_quarantine
    from qwen3vl_local.action_expert_ablation.geometry_abc import data
    p=tmp_path/'train.jsonl';records=[record()];rows_file(p,records)
    receipt=tmp_path/'routes/r/receipt.json';write(receipt,{'bindings':{}})
    manifest=dict(code=contracts.source_identity(data_only=True),ledger_sha256=sha(label_quarantine.LEDGER),
        scoped_smoke_only=False,decoded=True,splits={'train':sha(p)},routes={'r':sha(receipt)})
    write(tmp_path/'manifest.json',manifest)
    rows,m=indexed.load(tmp_path,data_root='/unused',manifest_sha256=sha(tmp_path/'manifest.json'))
    eager,em=data.load(tmp_path)
    assert m==em and list(rows['train'])==[runtime.sample(r,'/unused') for r in eager['train']]
    receipt.write_text('{}')
    with pytest.raises(ValueError,match='route_receipt_changed'):
        indexed.load(tmp_path,data_root='/unused',manifest_sha256=sha(tmp_path/'manifest.json'))


def test_all_variants_probe_members_identical_with_metadata_only_index(tmp_path):
    records,processor=fixture_processor();rows={}
    for split in ('train','val','test'):
        rows[split]=rows_file(tmp_path/(split+'.jsonl'),[r for r in records if r['split']==split],split)
    for variant in 'abc':
        original=probe_inputs.selection(records,processor,variant)
        assert probe_inputs.selection((r for values in rows.values() for r in values.probe_records()),processor,variant)==original
    assert all(not v.store.cache for v in rows.values())


def test_seven_plans_preserve_every_rank_order_and_tail(tmp_path):
    from qwen3vl_local.action_prior import event_balance
    records=[record(i) for i in range(65)];rows=rows_file(tmp_path/'train.jsonl',records)
    projection=planning.project(records,quotas={'A':65},epochs=7,cap=1,world=1,accumulation=16)
    with worker.planned_sampler(projection,2026):
        for epoch,plan in enumerate(projection['plans']):
            selected,_=event_balance.build_balanced_epoch(rows,mode='abc_regions',total=65,seed=2026+epoch)
            assert isinstance(selected,indexed.Rows)
            for rank in range(4):
                assert [r['abc_record']['id'] for r in selected[rank::4]]==plan['ordered_ids'][rank::4]


def test_equal_event_selection_and_condition_coverage_unchanged(tmp_path):
    from qwen3vl_local.action_prior.event_balance import equal_event_evaluation
    from qwen3vl_local.action_prior.build_event_balance_index import CONTEXT_TO_BUCKET
    records=[record(i,e,'val') for e in contracts.EVENTS for i in range(20)]
    rows=rows_file(tmp_path/'val.jsonl',records,'val');eager=[runtime.sample(r,'/unused') for r in records]
    merged={};lookup={}
    for r in eager:
        e=r['abc_record']['event'];key=(r['scenario'],r['run_id'],r['anchor']);bucket=CONTEXT_TO_BUCKET[e]
        lookup[key,bucket]=r
        if key not in merged:merged[key]=dict(r,event_balance_all_special_buckets=[])
        merged[key]['event_balance_all_special_buckets'].append(bucket)
    selected,plan=equal_event_evaluation(list(merged.values()),seed=2097,max_samples=0)
    expected=[lookup[(r['scenario'],r['run_id'],r['anchor']),r['_evaluation_event']] for r in selected]
    actual,actual_plan=worker.validation_members(rows,2097)
    assert actual==expected and actual_plan==plan
    coverage_records=supported_rows();cov=rows_file(tmp_path/'coverage.jsonl',coverage_records,'val')
    for variant in 'abc':
        old,report=diagnostic_coverage.select([runtime.sample(r,'/unused') for r in coverage_records],variant)
        new,new_report=diagnostic_coverage.select(cov,variant)
        assert old==new and report==new_report


def test_mapped_checkpoint_real_fm_optimizer_ema_reload(tmp_path):
    from qwen3vl_local.action_prior.flow_matching import make_training_flow,flow_matching_loss
    from qwen3vl_local.action_expert_ablation.common import leadmot_train
    records=[record()];rows=rows_file(tmp_path/'train.jsonl',records);m=tiny()
    optimizer=torch.optim.AdamW(m.parameters(),lr=.001);ema=leadmot_train()._DecoderEMA(m,.9)
    flow=make_training_flow(torch.randn(1,10,2),torch.randn(1,8,2),m.flow_config)
    result=m(**inputs(m,rows[0]['abc_record']),flow_state=flow['flow_state'],flow_time=flow['flow_time'])
    loss=flow_matching_loss(result,flow['flow_target_velocity'],10,.5,1.)[0];loss.backward();optimizer.step();ema.update(m)
    path=tmp_path/'state.pt';torch.save(dict(decoder=m.state_dict(),optimizer=optimizer.state_dict(),ema=ema.state_dict(),step=1),path)
    before=sha(path);state=indexed.checkpoint(path);restored=tiny();restored.load_state_dict(state['decoder'])
    new_ema=leadmot_train()._DecoderEMA(restored,.9);new_ema.load_state_dict(state['ema'])
    new_optimizer=torch.optim.AdamW(restored.parameters(),lr=.001);new_optimizer.load_state_dict(state['optimizer'])
    del state
    for a,b in zip(m.parameters(),restored.parameters()):assert torch.equal(a,b)
    for key in ema.shadow:assert torch.equal(ema.shadow[key],new_ema.shadow[key])
    assert new_optimizer.state_dict()['param_groups']==optimizer.state_dict()['param_groups']
    assert sha(path)==before


def test_route_receipt_cache_bounded_and_evicted_receipt_reverified(tmp_path,monkeypatch):
    from collections import OrderedDict
    from qwen3vl_local.audit_joint.abc_delivery import digest
    from qwen3vl_local.action_expert_ablation.geometry_abc import consumption
    reader=consumption.Sources.__new__(consumption.Sources)
    reader.directory=tmp_path;reader.receipts=OrderedDict();reader.manifest={'routes':{}}
    records=[{'scenario':'s','route_id':f'r{i}'} for i in range(40)]
    for r in records:
        key=digest([r['scenario'],r['route_id']]);path=tmp_path/'routes'/key/'receipt.json'
        write(path,{'bindings':{'item':key}});reader.manifest['routes'][key]=sha(path)
        assert reader.receipt(r)['bindings']['item']==key
        assert len(reader.receipts)<=32
    first=digest(['s','r0']);assert first not in reader.receipts
    (tmp_path/'routes'/first/'receipt.json').write_text('{}')
    with pytest.raises(ValueError,match='consumed_source_changed'):reader.receipt(records[0])


@pytest.mark.parametrize('workers',[0,2])
def test_probe_retains_configured_real_spawn_workers_through_save(tmp_path,monkeypatch,workers):
    from .test_consumption import fixture
    from qwen3vl_local.action_expert_ablation.geometry_abc import probe_loader
    r,args,_=fixture(tmp_path,monkeypatch)
    args.num_workers=workers;args.persistent_workers=False;args.prefetch_factor=1;args.seed=2026
    chosen=runtime.sample(r,args.data_root);loader=None
    try:
        loader,prepared,proof=probe_loader.prepare(chosen,args)
        assert proof['decoded_copies']==max(1,workers) and proof['formal_training_exposures']==0
        assert prepared['sample']['abc_record']==r
        if workers:
            processes=loader._iterator._workers
            assert len(processes)==workers and all(p.is_alive() for p in processes)
        path=tmp_path/'state.pt';torch.save({'decoder':torch.arange(5)},path)
        assert torch.equal(indexed.checkpoint(path)['decoder'],torch.arange(5))
        if workers:assert all(p.is_alive() for p in processes)
    finally:
        if loader is not None:probe_loader.close(loader)
    if workers:assert all(not p.is_alive() for p in processes)


def test_probe_loader_failure_closes_its_workers(monkeypatch):
    from qwen3vl_local.action_expert_ablation.geometry_abc import probe_loader,loader as native
    closed=[]
    class Broken:
        _iterator=SimpleNamespace(_shutdown_workers=lambda:closed.append(True))
        def __iter__(self):return iter([{'_error':'bad source'}])
    monkeypatch.setattr(native,'make_loader',lambda *a,**kw:(Broken(),None))
    with pytest.raises(ValueError,match='bad source'):
        probe_loader.prepare({},SimpleNamespace(num_workers=2,persistent_workers=False,seed=2026))
    assert closed==[True]
