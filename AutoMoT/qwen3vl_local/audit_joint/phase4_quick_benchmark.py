"""Bounded, ten-event Phase4 runtime comparison; NOT full-pool admission.

Read bounded windows of native train once, retain a small candidate pool, and use
the formal encoder plus the existing fixed-update DDP benchmark worker. No
student view, teacher replay, checkpoint or production batch recommendation.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

from . import batch_benchmark as benchmark
from .benchmark_runtime import progress, run_owned, trace_waits
from qwen3vl_local.sequence_batch import phase4_records

SCOPE = ('bounded native train windows, diagnostic only; full pool, teacher proof, '
         'split isolation and model quality NOT admitted; not a production batch recommendation')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_binding():
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import contract
    root = Path(__file__).parent
    return dict(student=contract(), diagnostic={name:file_sha(root/name) for name in
        ('phase4_quick_benchmark.py', 'batch_benchmark.py', 'benchmark_runtime.py')})


def stamp(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def scan_train(path, *, per_event, candidates_per_event, max_rows, max_bytes):
    from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
    pools = {event:[] for event in EVENTS}
    seen = set(); scanned = 0; consumed = 0; scanned_hash = hashlib.sha256(); windows = []
    before = stamp(path)
    size = before[2]
    # Large libraries are ordered by route/scenario: a prefix alone can miss
    # entire events. Deterministic spaced windows keep local sequence runs while
    # reaching later scenarios without reading/parsing the whole library.
    offsets = [0] if size<=max_bytes else [size*k//32 for k in range(32)]
    window_rows = (max_rows+len(offsets)-1)//len(offsets)
    with Path(path).open('rb') as stream:
        for window_index,offset in enumerate(offsets):
            if consumed>=max_bytes or scanned>=max_rows:break
            window_limit=consumed+(max_bytes-consumed)//(len(offsets)-window_index)
            stream.seek(max(0,offset-1))
            if offset:
                partial=stream.readline(min(8*1024*1024+1,max_bytes-consumed))
                consumed+=len(partial);scanned_hash.update(partial)
                if not partial.endswith(b'\n'):raise ValueError('window boundary exceeds line/byte budget')
            start=stream.tell()
            for _ in range(min(window_rows,max_rows-scanned)):
                # Never materialize an unbounded line or read past the byte budget.
                raw = stream.readline(min(8*1024*1024+1, max(0,window_limit-consumed)))
                if not raw:break
                consumed += len(raw); scanned += 1; scanned_hash.update(raw)
                if len(raw)>8*1024*1024:raise ValueError('diagnostic row exceeds line budget')
                if not raw.endswith(b'\n') and stream.tell()<size:break
                row = json.loads(raw)
                if row['split']!='train':raise ValueError('non-training row in diagnostic input')
                event = row['episode']['event']
                if event not in pools:raise ValueError('unknown diagnostic event: '+event)
                if (row['target'] in ('YES','NO') and row['id'] not in seen
                        and len(pools[event])<candidates_per_event):
                    pools[event].append(row); seen.add(row['id'])
                if all(len(rows)>=candidates_per_event for rows in pools.values()):break
                if consumed>=max_bytes:break
            windows.append(dict(start=start,end=stream.tell()))
            progress(f'window={len(windows)} scanned={scanned} bytes={consumed} counts='+str({e:len(v) for e,v in pools.items()}))
            if all(len(rows)>=candidates_per_event for rows in pools.values()):break
    if stamp(path)!=before:raise ValueError('native train changed during window read')
    counts = {e:len(rows) for e,rows in pools.items()}
    if min(counts.values())<per_event:
        raise ValueError(f'bounded windows lack {per_event} distinct questions/event: {counts}; '
                         f'scanned={scanned}, bytes={consumed}; no GPU trial started')
    return [r for rows in pools.values() for r in rows], dict(
        rows_scanned=scanned, bytes_read=consumed, scanned_bytes_sha256=scanned_hash.hexdigest(), windows=windows,
        train_stat=before, candidate_counts=counts, full_train_sha_verified=False)


def prepare(a):
    from qwen3vl_local.sft_new_loop_phase4.identity import check_contract
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import data
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import observation_contract
    manifest_path = a.native/'manifest.json'
    manifest_sha = file_sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    check_contract(manifest['contract'])
    if manifest['rgb_mode']!=4:raise ValueError('native RGB4 dataset required')
    sources = source_binding()
    progress('bounded train windows START; no full-pool load')
    rows, provenance = scan_train(a.native/'train.jsonl', per_event=a.samples_per_event,
        candidates_per_event=a.candidates_per_event, max_rows=a.max_scan_rows,
        max_bytes=a.max_scan_mib*1024*1024)
    progress(f'windows DONE candidates={len(rows)}; paired RGB identities START')
    native = dict(train=rows, val=[], test=[])
    identities = data.image_identities(native, a.data_root)
    selected, exclusions = data.paired_view(native, a.rgb_mode, set(), identities=identities)
    items = selected['train']; records = phase4_records(items)
    from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
    counts = Counter(r['event'] for r in records)
    if any(counts[e]<a.samples_per_event for e in EVENTS):
        raise ValueError('too few diagnostic candidates after paired-input exclusions: '+str(counts))
    updates, selection = benchmark.schedule(records, per_event=a.samples_per_event,
        update_samples=a.update_samples, world=len(a.gpus.split(',')), batch=a.batches[0],
        mode=a.rgb_mode, seed=a.seed)
    if selection['unique_questions']!=selection['presentations']:
        raise ValueError('quick comparison requires distinct measured questions')
    used = sorted({i for block in updates for i in block}); remap = {i:j for j,i in enumerate(used)}
    if (file_sha(manifest_path)!=manifest_sha or stamp(a.native/'train.jsonl')!=provenance['train_stat']
            or source_binding()!=sources):raise ValueError('diagnostic inputs changed during preparation')
    provenance.update(scope=SCOPE, full_pool_admitted=False, native=str(a.native.resolve()),
        manifest_sha256=manifest_sha, data_root=str(a.data_root.resolve()), sources=sources,
        exclusions=exclusions, selected_row_sha256=benchmark.digest([items[i] for i in used]))
    payload = dict(schema='phase4_benchmark_selection_v1', mode=a.rgb_mode, seed=a.seed,
        per_event=a.samples_per_event, update_samples=a.update_samples, world=len(a.gpus.split(',')),
        items=[items[i] for i in used], records=[records[i] for i in used],
        updates=[[remap[i] for i in block] for block in updates], selection=selection,
        manifest={'observation_contract':observation_contract(a.rgb_mode)}, diagnostic=provenance)
    benchmark.write(a.output/'prepared.json', payload)
    progress(f'prepared DONE unique={len(used)} timed_presentations={selection["presentations"]}')


def check_prepared(path, a):
    payload = json.loads(Path(path).read_text()); d = payload['diagnostic']
    expected = dict(schema='phase4_benchmark_selection_v1', mode=a.rgb_mode, seed=a.seed,
        per_event=a.samples_per_event, update_samples=a.update_samples, world=len(a.gpus.split(',')))
    if any(payload.get(k)!=v for k,v in expected.items()):raise ValueError('selection configuration mismatch')
    if (d.get('full_pool_admitted') is not False or d['sources']!=source_binding()
            or d['data_root']!=str(a.data_root.resolve()) or d['native']!=str(a.native.resolve())
            or d['manifest_sha256']!=file_sha(a.native/'manifest.json')
            or d['train_stat']!=stamp(a.native/'train.jsonl')
            or d['selected_row_sha256']!=benchmark.digest(payload['items'])):
        raise ValueError('diagnostic selection source/input binding mismatch; prepare a new selection')
    if payload['records']!=phase4_records(payload['items']):
        raise ValueError('diagnostic records differ from selected questions')
    return payload


def worker(a):
    # Reuse the real benchmark loop without pretending that a student view exists.
    b = benchmark.parser().parse_args(['--phase','4','--data-root',str(a.data_root),
        '--model-dir',str(a.model_dir),'--output',str(a.output)])
    for key in ('gpus','seed','samples_per_event','update_samples','batches'):
        setattr(b,key,getattr(a,key))
    b.modes=[a.rgb_mode]; b.gradient_checkpointing=True; b.timing='end_to_end'
    b.prepared_samples=a.selection; b.prepared_sha256=a.selection_sha; b.ddp_timeout=30
    rank=int(os.environ.get('RANK',0))
    try:
        if file_sha(a.selection)!=a.selection_sha:raise ValueError('selection SHA mismatch')
        check_prepared(a.selection,a)
        benchmark.worker(b)
        path=a.output/f'rank{rank}.json'; result=json.loads(path.read_text())
        result.update(scope=SCOPE,full_pool_admitted=False)
        benchmark.write(path,result)
    except Exception as exc:
        benchmark.write(a.output/f'rank{rank}.json',dict(status='error',rank=rank,
            error_type=type(exc).__name__,error=str(exc),scope=SCOPE,full_pool_admitted=False))
        raise


def summary(trials):
    result=benchmark.summarize(trials,1)
    result['provisional_subset_candidates']=result.pop('fastest_measured_candidates')
    result.update(scope=SCOPE,full_pool_admitted=False,
        note='One repeat, bounded window pool, fixed samples/update: exploratory timing only. '
             'Native full-pool validation remains pending; do not use as production batch approval.')
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for key in ('native','data-root','model-dir','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--rgb-mode',type=int,choices=(2,4),default=2)
    p.add_argument('--batches',type=int,nargs='+',default=[1,2])
    p.add_argument('--samples-per-event',type=int,default=8)
    p.add_argument('--update-samples',type=int,default=8)
    p.add_argument('--candidates-per-event',type=int,default=32)
    p.add_argument('--max-scan-rows',type=int,default=20000)
    p.add_argument('--max-scan-mib',type=int,default=256)
    p.add_argument('--prepare-timeout',type=int,default=60)
    p.add_argument('--trial-timeout',type=int,default=180)
    p.add_argument('--seed',type=int,default=20260904)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--selection',type=Path,help='Reuse this mode\'s prepared.json, rechecking source/input bindings')
    p.add_argument('--selection-sha',help=argparse.SUPPRESS)
    p.add_argument('--prepare-worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    return p


def main():
    from .guard import no_core
    no_core(); p=parser(); a=p.parse_args()
    cards=a.gpus.split(',')
    if (any(not c.isdigit() for c in cards) or len(cards)!=len(set(cards))
            or min(a.prepare_timeout,a.trial_timeout,a.samples_per_event,a.update_samples,
                   a.max_scan_rows,a.max_scan_mib,*a.batches)<1
            or a.candidates_per_event<a.samples_per_event
            or a.samples_per_event%a.update_samples
            or any(a.update_samples%(len(cards)*b) for b in a.batches)):
        p.error('invalid GPU or sampling/time budget')
    if not (a.native/'train.jsonl').is_file() or not (a.native/'manifest.json').is_file():p.error('native train/manifest missing')
    if not a.data_root.is_dir():p.error('data root missing')
    if not (a.prepare_only or a.prepare_worker) and not (a.model_dir/'config.json').is_file():p.error('model config missing')
    if a.prepare_worker:trace_waits();prepare(a);return
    if a.worker:trace_waits();worker(a);return
    a.output.mkdir(parents=True,exist_ok=False)
    benchmark.write(a.output/'request.json',dict(
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        scope=SCOPE,full_pool_admitted=False))
    common=[]
    for key in ('native','data_root','model_dir','gpus','rgb_mode','samples_per_event','update_samples',
                'candidates_per_event','max_scan_rows','max_scan_mib','seed'):
        common+=['--'+key.replace('_','-'),str(getattr(a,key))]
    module='qwen3vl_local.audit_joint.phase4_quick_benchmark'
    selected=a.selection or a.output/'prepared.json'
    if not a.selection:
        command=[sys.executable,'-u','-m',module,'--prepare-worker',*common,'--output',str(a.output),
                 '--batches',str(a.batches[0])]
        progress(f'CPU preparation limit={a.prepare_timeout}s output={a.output}')
        code=run_owned(command,a.output/'prepare.log',dict(os.environ,CUDA_VISIBLE_DEVICES=''),a.prepare_timeout)
        if code:
            benchmark.write(a.output/'summary.json',dict(status='preparation_failed',returncode=code,
                full_pool_admitted=False,scope=SCOPE))
            raise SystemExit(code)
    check_prepared(selected,a)
    sha=file_sha(selected)
    benchmark.write(a.output/'selection_binding.json',dict(path=str(selected.resolve()),sha256=sha))
    if a.prepare_only:
        benchmark.write(a.output/'summary.json',dict(status='prepared_cpu_only',scope=SCOPE,full_pool_admitted=False))
        return
    trials=[]
    for batch in sorted(set(a.batches)):
        folder=a.output/f'rgb{a.rgb_mode}_batch{batch}';folder.mkdir()
        command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(cards)}',
            '--monitor-interval=1','-m',module,'--worker',*common,'--output',str(folder),
            '--batches',str(batch),'--selection',str(selected),'--selection-sha',sha]
        progress(f'GPU RGB={a.rgb_mode} batch/card={batch} limit={a.trial_timeout}s')
        code=run_owned(command,folder/'worker.log',dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,
            OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'),a.trial_timeout)
        try:result=benchmark.collect(folder,len(cards),code)
        except (ValueError,OSError,KeyError,TypeError) as exc:
            result=dict(status='error',returncode=code,error=str(exc))
        result.update(rgb_mode=a.rgb_mode,batch_size=batch,repeat=0,folder=str(folder),
            full_pool_admitted=False,scope=SCOPE)
        trials.append(result); benchmark.write(folder/'trial.json',result)
        benchmark.write(a.output/'summary.json',summary(trials))
        print(json.dumps({k:v for k,v in result.items() if k not in ('ranks','selection')}),flush=True)
        if result['status']!='measured':raise SystemExit(code or 2)


if __name__=='__main__':main()
