"""Bounded-memory full-parent verification and unchanged-record dataset copy."""
import hashlib
import json
from pathlib import Path
import ijson
from ..abc_delivery import sha,write
from ..geometry_abc_v2 import dataset,admission,planning,regions
from ..geometry_v1 import dataset as native
from . import runtime,prompt,VERSION


def header(path):
    builder=ijson.common.ObjectBuilder()
    with Path(path).open('rb') as stream:
        for prefix,event,value in ijson.parse(stream,use_float=True):
            if prefix=='records' or prefix.startswith('records.'):continue
            if prefix=='' and event=='map_key' and value=='records':continue
            builder.event(event,value)
    return builder.value


def records(path):
    with Path(path).open('rb') as stream:yield from ijson.items(stream,'records.item',use_float=True)


def planning_record(r):
    result={k:r[k] for k in ('id','scenario','route_id','frame','physical_group','split','event','sampling_mass')}
    def target(t):return {k:t[k] for k in ('valid','token_id','target') if k in t}
    result['targets']=[target(t) for t in r['targets']]
    result['representations']={v:{'targets':[target(t) for t in r['representations'][v]['targets']]} for v in 'abc'}
    return result


def verify_native(path,expected_sha):
    path=Path(path);receipt=json.loads(path.with_name('receipt.json').read_text())
    if sha(path)!=expected_sha or receipt['dataset_sha256']!=expected_sha:raise ValueError('native_parent_changed')
    h=header(path)
    # The original verifier consumes the full record stream for split validation,
    # then checks every directory member, resolved mount and source-content hash.
    native.verify(dict(h,records=records(path)))
    if sha(path)!=expected_sha:raise ValueError('native_parent_changed_during_verification')


def prepare(parent,output):
    parent=Path(parent).resolve(strict=True);output=Path(output).resolve()
    expected=json.loads(parent.with_name('receipt.json').read_text())['dataset_sha256']
    if sha(parent)!=expected:raise ValueError('baseline_dataset_changed')
    h=header(parent)
    if h['code']!=dataset.code_identity() or h['contract']!={v:regions.contract(v) for v in 'abc'}:
        raise ValueError('baseline_contract_or_code_changed')
    for protected in (parent.parent,Path(h['data_root']).resolve()):
        if output.is_relative_to(protected) or protected.is_relative_to(output):raise ValueError('output_overlaps_input')
    print('[refine CPU] checking complete native parent and raw sources',flush=True)
    verify_native(h['parent_geometry'],h['parent_sha256'])
    for path,value in h['map_proof'].items():
        if sha(path)!=value:raise ValueError('baseline_map_changed')
    output.mkdir(parents=True,exist_ok=False)
    original_code=h['code'];original_contract=h['contract'];code=runtime.identity()
    h['code']=code;h['contract']={v:runtime.contract(v) for v in 'abc'}
    compact=[];record_hash=hashlib.sha256();count=0
    with (output/'dataset.json').open('w') as stream,(output/'members.jsonl').open('w') as members:
        stream.write(json.dumps(h,ensure_ascii=False,separators=(',',':'),allow_nan=False)[:-1]+',"records":[')
        for r in records(parent):
            prompt.verify_no_target_leak(dict(r,variant='a'))
            if count:stream.write(',')
            json.dump(r,stream,ensure_ascii=False,separators=(',',':'),allow_nan=False)
            record_hash.update(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode());record_hash.update(b'\n')
            compact.append(planning_record(r))
            members.write(json.dumps([r['id'],r['split'],r['event'],r['physical_group'],
                {v:[t['token_id'] if t['valid'] else -1 for t in r['representations'][v]['targets']] for v in 'abc'}],separators=(',',':'))+'\n')
            count+=1
            if count%25000==0:print(f'[refine CPU] unchanged records={count}',flush=True)
        stream.write(']}\n')
    # Compare every output record, including all original geometric evidence.
    output_hash=hashlib.sha256()
    for r in records(output/'dataset.json'):
        output_hash.update(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode());output_hash.update(b'\n')
    if output_hash.digest()!=record_hash.digest() or sha(parent)!=expected:raise ValueError('record_copy_or_parent_changed')
    with runtime.activated():
        report=admission.require(admission.inspect(dict(h,records=compact)))
        projection=planning.project(compact,per_event=1280,epochs=3,world_size=4,seed=20260904)
    if code!=runtime.identity():raise ValueError('refine_source_changed_during_prepare')
    dataset_sha=sha(output/'dataset.json')
    write(output/'receipt.json',dict(dataset_sha256=dataset_sha,parent_sha256=expected,
        baseline_code=original_code,baseline_contract=original_contract,records_sha256=record_hash.hexdigest(),
        records_hash_format='sorted_compact_JSON_each_record_then_LF',labels_unchanged=True,full_parent_verified=True))
    write(output/'admission.json',report);write(output/'plan.json',projection)
    state=dict(schema=VERSION,status='data_plan_ready',dataset=str(output/'dataset.json'),dataset_sha256=dataset_sha,
               records=count,data_ready=True,plan_ready=True,runtime_ready=False,package_ready=False,prompt_benefit_measured=False)
    write(output/'status.json',state);return state
