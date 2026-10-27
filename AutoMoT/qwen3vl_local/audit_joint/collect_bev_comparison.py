#!/usr/bin/env python3
"""Read-only collection of existing BEV/ABC evidence. Python standard library only.
No torch imports, inference, checkpoint deserialization, contract rewriting or GPU work.
The published directory (ZIP plus receipt) is strictly below 30,000,000 bytes.
"""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
import zipfile

LIMIT = 30_000_000
COORDS = ('pred_route', 'pred_waypoints', 'gt_route', 'gt_waypoints')
READ_SIGNATURES = {}

def compact(x):
    return (json.dumps(x, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n').encode()

def signature(p):
    s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def digest(p):
    before=signature(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if signature(p)!=before:raise RuntimeError('input changed while reading: '+str(p))
    return h.hexdigest()

def read(p):
    before=signature(p);b=p.read_bytes()
    if signature(p)!=before:raise RuntimeError('input changed while reading: '+str(p))
    prior=READ_SIGNATURES.setdefault(str(p),before)
    if prior!=before:raise RuntimeError('input changed between reads: '+str(p))
    return b

def optional_source_snapshot(p, attempts=3):
    """Capture a stable optional reference, never bind a live source to run validity."""
    errors=[]
    for attempt in range(1,attempts+1):
        try:
            before=signature(p)
            first=p.read_bytes();middle=signature(p)
            second=p.read_bytes();after=signature(p)
            if before==middle==after and first==second:
                return first,dict(status='captured_snapshot',attempts=attempt,
                    signature=list(after),scope='collection_time_reference_not_live_training_source')
            errors.append('source_changed_during_snapshot')
        except OSError as exc:
            errors.append(type(exc).__name__+': '+str(exc))
        if attempt<attempts:time.sleep(0.05)
    return None,dict(status='omitted_unstable_or_unavailable',attempts=attempts,
                     errors=errors,scope='optional_reference_only')

def flatten(x):
    if isinstance(x,list):
        if not x:return [0],[]
        sub=[flatten(v) for v in x];shape=sub[0][0]
        if any(s!=shape for s,_ in sub):raise ValueError('ragged coordinates')
        return [len(x)]+shape,[v for _,values in sub for v in values]
    if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x):
        raise ValueError('nonfinite/non-numeric coordinates')
    return [],[float(x)]

def coordinate_bytes(value):
    shape,values=flatten(value)
    try:
        floats=array.array('f',values)
        exact=all(float(a)==b for a,b in zip(floats,values))
    except OverflowError:exact=False
    if not exact:floats=array.array('d',values)
    if sys.byteorder!='little':floats.byteswap()
    return floats.tobytes(),dict(shape=shape,dtype='<f4' if exact else '<f8',count=len(values))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',required=True,help='Existing old BEV run; no training is performed')
    ap.add_argument('--abc-run',help='Optional existing ABC suite on this same server')
    ap.add_argument('--output',default='checkpoints/audit_exports')
    ap.add_argument('--no-current-source-reference',action='store_true',help='Skip optional current-code snapshots; retain original contracts and all saved results')
    ap.add_argument('--data-dir',help='Optional relocated old data directory on THIS server')
    ap.add_argument('--lead-bev-ckpt',help='Optional relocated BEV pretrained file on THIS server')
    a=ap.parse_args();app=Path.cwd().resolve();old=Path(a.run).expanduser().resolve()
    if not (old/'config.json').is_file():ap.error('old run/config.json does not exist: '+str(old))
    outbase=Path(a.output).expanduser().resolve();outbase.mkdir(parents=True,exist_ok=True)
    label='bev_comparison_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
    final=outbase/label
    if final.exists():ap.error('output already exists')
    roots={'old_bev':old}
    if a.abc_run:
        ar=Path(a.abc_run).expanduser().resolve()
        for v in 'abc':
            p=ar/(v+'_bev_only')
            if not (p/'config.json').is_file():ap.error('ABC group missing: '+str(p))
            roots[v+'_bev_only']=p
    if any(outbase==r or outbase.is_relative_to(r) for r in roots.values()):
        ap.error('output must be outside all input run directories')
    manifest=dict(schema='bev_historical_comparison_collection_v1',limit_bytes=LIMIT,
                  project_root=str(app),runs={k:str(v) for k,v in roots.items()},
                  inputs=[],missing=[],warnings=[],case_sets={},external_resources={},
                  collection_only=True,weights_included=False,model_loaded=False,
                  current_source_is_not_certified_training_source=True)
    if not a.abc_run:manifest['missing'].append('ABC full training histories not supplied; existing ABC final package remains a separate input')
    tracked={}
    with tempfile.TemporaryDirectory(prefix='.bev_compare_collect_',dir=outbase) as tmp:
        tmp=Path(tmp);stage=tmp/'stage';stage.mkdir()
        def put(name,b):
            p=stage/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        def record(p,b,kind):
            key=str(p)
            if key not in tracked:
                tracked[key]=READ_SIGNATURES[key]
                manifest['inputs'].append(dict(path=key,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),kind=kind))
        def json_file(p,name,required=False):
            if not p.is_file():
                if required:manifest['missing'].append(str(p))
                return None
            b=read(p);x=json.loads(b);put(name,b);record(p,b,'original_json');return x
        def stream_json_files(files,name):
            target=stage/name;target.parent.mkdir(parents=True,exist_ok=True)
            count=0
            with target.open('wb') as f:
                for p in files:
                    b=read(p);obj=json.loads(b);record(p,b,'merged_json')
                    f.write(compact(dict(source=str(p),record=obj)));count+=1
            return count
        for group,run in roots.items():
            print('[collect]',group,run,flush=True)
            cfg=None
            for fn in ('config.json','condition_contract.json','training_plan.json','action_token_initial.json',
                       'metrics.json','training_complete.json','completion.json','checkpoint_receipt.json'):
                x=json_file(run/fn,group+'/'+fn,fn in ('config.json','condition_contract.json','training_plan.json') and group=='old_bev')
                if fn=='config.json':cfg=x
            scopes=['eval_test'] if group=='old_bev' else ['test','']
            evaluation=None
            for scope in scopes:
                d=run/scope
                if (d/'metrics.json').is_file():
                    evaluation=json_file(d/'metrics.json',group+'/evaluation/'+(scope or 'root')+'/metrics.json')
            if evaluation is None:manifest['missing'].append(group+': test metrics not found')
            evaluation_files=[]
            for dirname in ('eval_test','test'):
                d=run/dirname
                if d.is_dir():evaluation_files.extend(p for p in d.rglob('*.json') if 'cases' not in p.relative_to(d).parts)
            stream_json_files(sorted(set(evaluation_files)),group+'/evaluation_evidence.jsonl')
            for dirname in ('training_audit','validation','performance','progress'):
                files=sorted((run/dirname).rglob('*.json')) if (run/dirname).is_dir() else []
                count=stream_json_files(files,group+'/'+dirname+'.jsonl')
                if not count:manifest['missing'].append(group+': '+dirname+' JSON records absent')
            # Include original compact audit ZIP as fallback, not weights or raw TensorBoard.
            zpath=run/'training_audit.zip'
            if zpath.is_file():
                b=read(zpath);record(zpath,b,'existing_training_audit_zip');put(group+'/training_audit.zip',b)
            sampling=sorted((run/'sampling').rglob('*.json')) if (run/'sampling').is_dir() else []
            small=[]
            for p in sampling:
                if p.stat().st_size<=16_000_000:small.append(p)
                else:
                    manifest['warnings'].append('Large sampling detail omitted (budget/epoch records retained): '+str(p))
                    manifest['external_resources'][str(p)]=dict(bytes=p.stat().st_size,sha256=digest(p))
                    tracked[str(p)]=signature(p)
            stream_json_files(small,group+'/sampling.jsonl')
            log=run/'train.log'
            if log.is_file():
                before=signature(log)
                with log.open('rb') as f:f.seek(max(0,before[2]-256_000));tail=f.read()
                if signature(log)!=before:raise RuntimeError('train.log is changing; collect after run stops')
                put(group+'/train_log_tail.txt',tail)
                manifest['warnings'].append(group+': train.log contains only final 256000 bytes; raw TB excluded, structured metric history collected')
            # Inventory all files without loading weight objects.
            inventory=[]
            for root,dirs,files in os.walk(run,followlinks=False):
                dirs[:]=sorted(d for d in dirs if not (Path(root)/d).is_symlink())
                for fn in sorted(files):
                    p=Path(root)/fn;s=p.stat()
                    inventory.append(dict(path=str(p.relative_to(run)),bytes=s.st_size,symlink=p.is_symlink()))
            put(group+'/file_inventory.json',compact(inventory))
            for fn in ('best.pt','latest.pt'):
                p=run/fn
                if p.is_file():
                    print('[hash only]',group,fn,flush=True);h=digest(p);tracked[str(p)]=signature(p)
                    manifest['external_resources'][group+'/'+fn]=dict(path=str(p),bytes=p.stat().st_size,sha256=h,deserialized=False)
            # Keep all case metrics, identities and condition records; numeric trajectories use lossless binary storage.
            case_dir=run/('eval_test/cases' if group=='old_bev' else 'test/cases')
            cases=sorted(case_dir.glob('rank*_case*.json')) if case_dir.is_dir() else []
            json_file(case_dir/'evaluation_plan.json',group+'/evaluation_plan.json')
            if not cases:
                manifest['missing'].append(group+': no saved rank*_case*.json; no new inference was run')
                # New ABC uses NPZ; copy its original compact evidence if available.
                for p in (run/'cases.npz',run/'test/cases.npz'):
                    if p.is_file():b=read(p);record(p,b,'original_npz');put(group+'/existing_'+p.parent.name+'_cases.npz',b)
                continue
            meta=stage/group/'cases.jsonl';binary=stage/group/'case_coordinates.bin';offset=0;count=0
            with meta.open('wb') as fm,binary.open('wb') as fb:
                for p in cases:
                    b=read(p);obj=json.loads(b);tracked[str(p)]=READ_SIGNATURES[str(p)]
                    provenance=dict(file=p.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
                    coordinates={}
                    for key in COORDS:
                        if key not in obj:continue
                        raw,schema=coordinate_bytes(obj.pop(key));schema['offset_bytes']=offset;schema['bytes']=len(raw)
                        fb.write(raw);offset+=len(raw);coordinates[key]=schema
                    # Envelope avoids collisions with fields in the original case.
                    fm.write(compact(dict(source=provenance,record=obj,coordinates=coordinates)));count+=1
                    if count%5000==0:print('[cases]',group,count,'/',len(cases),flush=True)
            expected=(evaluation or {}).get('metrics',evaluation or {}).get('samples')
            if expected is None:expected=(evaluation or {}).get('all',{}).get('cases')
            manifest['case_sets'][group]=dict(count=count,expected_metrics_samples=expected,complete=(count==expected if expected is not None else None),coordinate_format='little endian, exact float32 when representable, otherwise float64; offset/shape/dtype per case',source_directory=str(case_dir))
            if expected is not None and count!=expected:manifest['warnings'].append(group+': case count differs from metrics samples; do not treat as full test')
        # Resolve only explicit/current-machine paths. Never silently pick a different run or model.
        cfg=json.loads(read(old/'config.json'))
        def local_resource(text,override=None):
            raw=override or text
            if not raw:return None
            p=Path(raw).expanduser()
            if not p.is_absolute():p=app/p
            if p.exists():return p.resolve()
            if not override and '/checkpoints/' in str(p):
                q=app/'checkpoints'/str(p).split('/checkpoints/',1)[1]
                if q.exists():manifest['warnings'].append('Relocated configured resource onto current project root: '+str(q));return q.resolve()
            return None
        bev=local_resource(cfg.get('lead_bev_ckpt'),a.lead_bev_ckpt)
        if bev and bev.is_file():
            print('[hash only] current BEV pretrained file',flush=True)
            h=digest(bev);tracked[str(bev)]=signature(bev)
            contract=json.loads(read(old/'condition_contract.json')) if (old/'condition_contract.json').is_file() else {}
            recorded=contract.get('identity_payload',{}).get('bev')
            manifest['external_resources']['current_bev_file']=dict(path=str(bev),sha256=h,training_recorded_sha256=recorded,matches_training_record=(h==recorded),tensor_equality_not_checked=True)
        else:manifest['missing'].append('configured BEV pretrained file not found on this server')
        data=local_resource(cfg.get('data_dir'),a.data_dir)
        # Membership supplement is useful even if the old evaluation did not save per-case predictions.
        if data and (data/'test.jsonl').is_file():
            p=data/'test.jsonl';before=signature(p);h=hashlib.sha256();count=0
            target=stage/'old_bev/test_membership.jsonl'
            with p.open('rb') as src,target.open('wb') as dst:
                for line in src:
                    h.update(line)
                    if not line.strip():continue
                    x=json.loads(line)
                    keys=[k for k in x if k in ('id','scenario','run_id','route_id','anchor','frame','physical_group','physical_route_key','split','action_token','action_token_id','high_level_action','high_level_action_token') or k.startswith('event_balance')]
                    dst.write(compact({k:x[k] for k in keys}));count+=1
            if signature(p)!=before:raise RuntimeError('test index changed during collection')
            tracked[str(p)]=before
            expected=cfg.get('event_balance_source_identity',{}).get('action_dataset_hashes',{}).get('test')
            manifest['external_resources']['current_test_index']=dict(path=str(p),sha256=h.hexdigest(),records=count,training_recorded_sha256=expected,matches_training_record=(h.hexdigest()==expected if expected else None),membership_not_evaluated_predictions=True)
        else:manifest['missing'].append('old test.jsonl membership index not available')
        # Current sources are only a reference; never label them as original training code without matching the contract.
        contract=json.loads(read(old/'condition_contract.json')) if (old/'condition_contract.json').is_file() else {}
        code=contract.get('identity_payload',{}).get('execution',{}).get('code',{})
        for rel in ('qwen3vl_local/action_prior/flow_matching.py','qwen3vl_local/action_prior/metrics.py','qwen3vl_local/action_prior/training_core.py','qwen3vl_local/action_expert_ablation/common.py'):
            p=app/rel
            if a.no_current_source_reference:
                b=None;info=dict(status='omitted_by_request',scope='optional_reference_only')
            else:
                b,info=optional_source_snapshot(p)
            info['training_recorded']=code.get(rel)
            if b is not None:
                put('current_source_reference/'+rel,b)
                info.update(current=hashlib.sha256(b).hexdigest(),matches=hashlib.sha256(b).hexdigest()==code.get(rel))
            else:
                info.update(current=None,matches=None)
                manifest['warnings'].append('Optional current source reference omitted: '+rel+' ('+info['status']+'); original run evidence retained')
            manifest.setdefault('source_comparison',{})[rel]=info
        put('collector.py',Path(__file__).read_bytes())
        def build(mode):
            manifest['trajectory_coordinates_included']=mode=='all'
            if mode!='all':manifest['warnings'].append('Size cap: all existing case metrics and identities retained, coordinate binary omitted; no case sampling')
            put('COLLECTION_MANIFEST.json',compact(manifest))
            p=tmp/'comparison.zip'
            with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
                members=[]
                for f in sorted(stage.rglob('*')):
                    if not f.is_file() or (mode!='all' and f.name=='case_coordinates.bin'):continue
                    name=str(f.relative_to(stage));z.write(f,name)
                    members.append(dict(path=name,bytes=f.stat().st_size,sha256=digest(f)))
                z.writestr('PACKAGE_MANIFEST.json',compact(dict(schema='sha256_members_v1',members=members)))
            return p
        print('[pack] compressing complete saved evidence',flush=True)
        package=build('all')
        if package.stat().st_size>LIMIT-64_000:
            print('[cap] retrying with all case metrics/identities, without coordinate binary',flush=True)
            package=build('metrics_only')
        if package.stat().st_size>LIMIT-64_000:
            raise RuntimeError('Core metrics/identities exceed 30MB; nothing published. Do not truncate cases. Send the console size/error for a narrower lossless collection.')
        for path,sig in READ_SIGNATURES.items():
            tracked.setdefault(path,sig)
        for path,sig in tracked.items():
            if signature(Path(path))!=sig:raise RuntimeError('input changed before publication: '+path)
        with zipfile.ZipFile(package) as z:
            if z.testzip():raise RuntimeError('ZIP CRC failure')
        receipt=dict(directory=str(final),package=str(final/'bev_comparison_review.zip'),bytes=package.stat().st_size,sha256=digest(package),missing=manifest['missing'],warnings=manifest['warnings'],case_sets=manifest['case_sets'],coordinates_included=manifest['trajectory_coordinates_included'])
        receipt_bytes=compact(receipt)
        if len(receipt_bytes)+package.stat().st_size>LIMIT:raise RuntimeError('delivery directory exceeds 30MB')
        final.mkdir();package.replace(final/'bev_comparison_review.zip');(final/'receipt.json').write_bytes(receipt_bytes)
        print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
