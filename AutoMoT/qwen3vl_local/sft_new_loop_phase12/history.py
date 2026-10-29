"""Portable, explicit historical test contract. No model is loaded by this module."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from qwen3vl_local.sft_new_loop_phase1.training_policy import identity,physical_group
from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices
from .data import relative_rgb,resolve_rgb

SCHEMA='phase12_historical_members_v1'
RELEASE=Path(__file__).with_name('history_release.json')
CONTRACT_STAT_FIELDS=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')


def _read_contract_once(path):
    """Bind the bytes parsed to one file descriptor and both pathname checks."""
    snapshot=lambda s:tuple(getattr(s,k) for k in CONTRACT_STAT_FIELDS)
    before=snapshot(path.stat())
    if before[2]>64*1024*1024:raise ValueError('Historical contract exceeds size limit')
    with path.open('rb') as handle:
        opened=snapshot(os.fstat(handle.fileno()))
        content=handle.read(64*1024*1024+1)
        closed=snapshot(os.fstat(handle.fileno()))
    after=snapshot(path.stat())
    if len(content)>64*1024*1024:raise ValueError('Historical contract exceeds size limit')
    return content,(before,opened,closed,after)


def _contract_bytes(path,expected):
    content,states=_read_contract_once(path)
    def check_digest(data):
        if hashlib.sha256(data).hexdigest()!=expected:
            raise ValueError('Changed historical contract content; released SHA does not match: '+str(path))
    check_digest(content)
    if len(set(states))==1:return content
    changed=[k for i,k in enumerate(CONTRACT_STAT_FIELDS) if len({s[i] for s in states})!=1]
    if changed!=['st_ctime_ns']:
        raise ValueError(f'Historical contract changed while reading: {path}; fields={changed}')
    # Publishing/unlinking a hard link changes ctime even when content is fixed.
    # Only this pinned metadata file gets a bounded recovery; generic proof.sha
    # and all dataset/adapter/snapshot checks remain strict and unchanged.
    print(f'[history] ctime-only metadata drift: {path}; verifying two stable reads against released SHA',file=sys.stderr,flush=True)
    previous=None
    for _ in range(2):
        time.sleep(.05)
        fresh,stable=_read_contract_once(path)
        check_digest(fresh)
        if (len(set(stable))!=1 or stable[0][:-1]!=states[0][:-1]
                or (previous is not None and stable[0]!=previous)):
            raise ValueError('Historical contract did not stabilize across two verified reads: '+str(path))
        previous=stable[0];content=fresh
    return content


def boolean(v):
    if type(v) is bool:return v
    if v in ('YES','NO'):return v=='YES'
    raise ValueError('Invalid historical label')


def import_inputs(path,*,primary=False):
    path=Path(path)
    source_sha=proof.sha(path);completion_sha=proof.sha(path.parent/'evaluation_complete.json')
    obj=proof.read(path);complete=proof.read(path.parent/'evaluation_complete.json')
    if (obj['arguments']['split']!='test' or complete['status']!='verified'
            or complete['inputs_sha256']!=source_sha or complete['phase']!=obj['phase']
            or complete['total_cases']!=obj['expected_total'] or len(obj['items'])!=obj['expected_total']):
        raise ValueError('Historical input is not a complete bound TEST evaluation')
    phase=obj['phase'];mode=obj['history_rgb_mode'];cases=[]
    if phase not in (1,2) or mode not in ('2rgb_endpoints','4rgb'):raise ValueError('Unsupported historical phase/mode')
    for name,sha in complete['case_files_sha256'].items():
        if Path(name).name!=name or proof.sha(path.parent/name)!=sha:raise ValueError('Historical case shard changed')
        cases.extend(json.loads(l) for l in (path.parent/name).open() if l.strip())
    cases.sort(key=lambda r:r['case_index'])
    if [r['case_index'] for r in cases]!=list(range(len(obj['items']))):raise ValueError('Incomplete historical members')
    hashes={}
    for name,sha in obj['image_sha256'].items():
        name=relative_rgb(name)
        if name in hashes and hashes[name]!=sha:raise ValueError('Conflicting historical RGB aliases')
        hashes[name]=sha
    records=[];source=source_sha
    for number,(item,case) in enumerate(zip(obj['items'],cases)):
        row=item['row'];paths=[relative_rgb(p) for p in row['history_rgb_paths']]
        if len(paths)!=4 or any(tuple(Path(p).parts[:2])!=identity(row)[:2] for p in paths):
            raise ValueError('Historical image path does not identify its physical member')
        rgb=[paths[i] for i in history_rgb_indices(mode)]
        if row['split']!='test' or tuple(identity(row))!=tuple(identity(case)):
            raise ValueError('Historical member differs from recorded case')
        if rgb!=[relative_rgb(p) for p in case['history_rgb_paths_used']] or any(p not in hashes for p in rgb):
            raise ValueError('Unbound historical RGB member')
        gt={k:boolean(v) for k,v in case['gt'].items()}
        if any(k in row['answers'] and row['answers'][k] is not v for k,v in gt.items()):
            raise ValueError('Historical labels conflict with case')
        spec=item['spec'];seed=spec['phase2_spec']['seed_key'] if phase==1 else spec['seed_key']
        records.append(dict(case_index=number,historical_case_id=f'{source}:{number}',key=list(identity(row)),
            scenario=row['scenario'],answers=row['answers'],paths=paths,rgb=rgb,
            balance_key=item['balance_key'],seed_key=seed,original_asked_keys=list(gt),
            baseline_gt=gt,baseline_parsed={k:(boolean(v) if v in (True,False,'YES','NO') else None) for k,v in case['parsed'].items()},
            baseline_raw=case['raw_output'],historical_spec=spec,source_split='test'))
    bindings={path:source_sha,path.parent/'evaluation_complete.json':completion_sha}
    bindings.update({path.parent/name:sha for name,sha in complete['case_files_sha256'].items()})
    if any(proof.sha(p)!=sha for p,sha in bindings.items()):
        raise ValueError('Historical evidence changed during import')
    return dict(phase=phase,mode=mode,primary=primary,source_name=path.parent.parent.name,
        inputs_sha256=source,completion_sha256=completion_sha,
        case_shards=complete['case_files_sha256'],records=records,image_sha256=hashes)


def build(primary_paths,holdout_paths,output):
    sources=[import_inputs(p,primary=True) for p in primary_paths]
    sources.extend(import_inputs(p) for p in holdout_paths)
    primary=[(s['phase'],s['mode']) for s in sources if s['primary']]
    if sorted(primary)!=[(1,'2rgb_endpoints'),(1,'4rgb'),(2,'2rgb_endpoints'),(2,'4rgb')]:
        raise ValueError('Exactly one actual historical primary set for each phase/mode required')
    if len({s['inputs_sha256'] for s in sources})!=len(sources):raise ValueError('Duplicate historical source')
    groups=sorted({physical_group(tuple(r['key'])) for s in sources for r in s['records']})
    obj=dict(schema=SCHEMA,sources=sources,holdout_groups=groups,
        scope='Only explicitly imported verified historical tests; not all possible historical experiments')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(obj,f,ensure_ascii=False,separators=(',',':'));f.write('\n')
    return dict(schema=SCHEMA,contract_sha256=proof.sha(output),holdout_groups=len(groups),
        primary_counts={f"phase{s['phase']}_{s['mode']}":len(s['records']) for s in sources if s['primary']},
        source_names=[s['source_name'] for s in sources])


def load(path,*,expected_sha=None):
    path=Path(path)
    expected_sha=expected_sha or proof.read(RELEASE)['contract_sha256']
    if not path.is_file():raise ValueError('Missing historical contract; prepare verified local history, never re-sample historical tests')
    obj=json.loads(_contract_bytes(path,expected_sha))
    if obj['schema']!=SCHEMA:raise ValueError('Wrong historical contract')
    groups=sorted({physical_group(tuple(r['key'])) for s in obj['sources'] for r in s['records']})
    if groups!=[tuple(g) for g in obj['holdout_groups']]:raise ValueError('Historical holdout registry mismatch')
    return obj


def primary(obj,phase,mode):
    found=[s for s in obj['sources'] if s['primary'] and s['phase']==phase and s['mode']==mode]
    if len(found)!=1:raise ValueError('Missing/ambiguous historical primary set')
    return found[0]


def groups(obj):return {tuple(g) for g in obj['holdout_groups']}


def verify_images(obj,data_root):
    checked={}
    for source in obj['sources']:
        for name,sha in source['image_sha256'].items():
            if name in checked and checked[name]!=sha:raise ValueError('Historical sources disagree on RGB')
            if name not in checked and proof.sha(resolve_rgb(data_root,name))!=sha:raise ValueError('Historical RGB changed: '+name)
            checked[name]=sha
    return checked


def assert_isolation(index,obj):
    forbidden=groups(obj);counts=Counter()
    for line in Path(index).open():
        row=json.loads(line)
        if physical_group(identity(row)) in forbidden:
            counts[row['split']]+=1
            if row['split']!='test':raise ValueError('Historical test route leaked into '+row['split']+': '+str(identity(row)))
    return dict(status='historical_routes_test_only',counts=dict(counts))


def main():
    p=argparse.ArgumentParser();p.add_argument('--primary',type=Path,action='append',required=True)
    p.add_argument('--holdout',type=Path,action='append',default=[]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.primary,a.holdout,a.output),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
