"""Local, source/input-bound student snapshots prepared once before torchrun.

Workers still parse student rows, but never repeat native proof validation.
Only trusted local preparation receipts are accepted, not external attestations.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

from ..identity import write_json
from .contract import contract
from qwen3vl_local.audit_joint.io import file_sha

POLICY='phase4_validated_student_snapshot_v1'


def stamp(path):
    path=Path(path).absolute();resolved=path.resolve(strict=True);s=resolved.stat()
    return dict(path=str(path),resolved=str(resolved),device=s.st_dev,inode=s.st_ino,
                bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def inputs(view,mode):
    """Full hashes are checked by the native loader at preparation.

    Reuse guards every declared source artifact with path/device/inode/size/
    mtime/ctime and hashes the prepared payload. RGBs are checked by encoders.
    """
    view=Path(view).absolute()
    descriptor=json.loads((view/'student_view.json').read_text())
    parent=Path(descriptor['parent']['logical_path'])
    manifest=json.loads((parent/'manifest.json').read_text())
    # Enumerate without hashing huge raw JSONL on every rank. Native preparation
    # performs parent_binding's full validation; no stat-only initial acceptance.
    names={'manifest.json','review_queue.jsonl',*manifest['files']}
    for field,name in [('candidate_pool_sha256','candidate_pool.json'),
        ('teacher_registry_sha256','teacher_registry.json'),
        ('production_reference_sha256','production_reference.json'),
        ('production_index_sha256','production/index.json')]:
        if manifest.get(field):names.add(name)
    paths={view/'student_view.json',view/'rgb_identities.json',*(parent/n for n in names)}
    if manifest.get('production_reference_sha256'):
        record=json.loads((parent/'production_reference.json').read_text())
        paths.update(Path(r['logical_path']) for r in [record['index'],*record['routes'].values()])
    elif manifest.get('production_index_sha256'):
        index=parent/'production/index.json'
        for r in json.loads(index.read_text())['routes']:
            name=r['artifact']
            if Path(name).name!=name or name in ('.','..'):raise ValueError('invalid production artifact')
            paths.add(index.parent/name)
    adapter=Path(__file__).resolve().parents[2]/'audit_joint'
    return dict(policy=POLICY,rgb_mode=mode,view=str(view),source=contract(),
                read_adapters={n:file_sha(adapter/n) for n in ('phase4_load.py','phase4_production_cache.py')},
                files=[stamp(p) for p in sorted(paths)])


@contextmanager
def native_read():
    """Route all native reads in one preparation through the verified adapter."""
    from .. import dataset as native
    from qwen3vl_local.audit_joint.phase4_load import validated_reservations
    from qwen3vl_local.audit_joint.phase4_production_cache import production_routes
    original=native.load_dataset
    if getattr(native.producer_check_reservations,'_reservation_snapshot',False):
        # Existing benchmark/migration preparation already owns the transaction.
        yield
        return
    if getattr(original,'_runtime_preparation',False):raise RuntimeError('nested runtime data preparation')
    def load(path,*args,**kwargs):
        cache=os.environ.get('P4_PRODUCTION_CACHE') or None
        with production_routes(path,cache),validated_reservations():
            return original(path,*args,**kwargs)
    load._runtime_preparation=True
    native.load_dataset=load
    try:yield
    finally:native.load_dataset=original


def load(view,mode,loader,*,require_trainable=False,require_complete_coverage=False):
    if require_complete_coverage:
        raise ValueError('student view has no independent complete-coverage certification')
    directory=os.environ.get('P4_VALIDATED_DATA')
    if not directory:
        if int(os.environ.get('WORLD_SIZE','1'))>1:
            raise ValueError('DDP requires CPU-prepared P4_VALIDATED_DATA; use rgb_short/run.sh')
        with native_read():
            return loader(view,rgb_mode=mode,require_trainable=require_trainable,
                          require_complete_coverage=False,_runtime_raw=True)
    directory=Path(directory).absolute()
    descriptor=json.loads((Path(view)/'student_view.json').read_text())
    parent=Path(descriptor['parent']['logical_path'])
    if any(directory.resolve().is_relative_to(p.resolve()) for p in (Path(view),parent)):
        raise ValueError('runtime snapshot must be outside source view/native data')
    worker=int(os.environ.get('WORLD_SIZE','1'))>1
    if worker and not directory.is_dir():
        raise ValueError('DDP requires CPU-prepared P4_VALIDATED_DATA before torchrun')
    directory.mkdir(parents=True,exist_ok=True)
    payload=directory/f'rgb{mode}.json';receipt=directory/f'rgb{mode}.receipt.json'
    # Lock only preparation/publication, never serialize four worker JSON reads.
    if not receipt.exists():
        if worker:raise ValueError('DDP snapshot missing; prepare before torchrun')
        with (directory/f'rgb{mode}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not receipt.exists():
                print(f'[phase4 data] RGB={mode} single-process native preparation START',flush=True)
                before=inputs(view,mode)
                with native_read():
                    data,manifest=loader(view,rgb_mode=mode,require_trainable=False,
                                         require_complete_coverage=False,_runtime_raw=True)
                if inputs(view,mode)!=before:raise ValueError('runtime inputs changed during preparation')
                pending=payload.with_suffix('.pending')
                print(f'[phase4 data] RGB={mode} writing verified student snapshot',flush=True)
                from shutil import disk_usage
                # Check periodically while streaming, keeping the same 20 GiB
                # reserve as formal training; no giant serialized memory copy.
                with pending.open('w') as stream:
                    chunks=[];size=0
                    for chunk in json.JSONEncoder(separators=(',',':')).iterencode(dict(data=data,manifest=manifest)):
                        chunks.append(chunk);size+=len(chunk)
                        if size>=8*1024**2:
                            if disk_usage(directory).free<20*1024**3+size:
                                raise ValueError('insufficient disk reserve for runtime snapshot')
                            stream.write(''.join(chunks));chunks=[];size=0
                    if disk_usage(directory).free<20*1024**3+size:
                        raise ValueError('insufficient disk reserve for runtime snapshot')
                    stream.write(''.join(chunks))
                pending.replace(payload)
                sha=file_sha(payload)
                if inputs(view,mode)!=before:raise ValueError('runtime inputs changed before publication')
                write_json(receipt,dict(status='validated_student_snapshot',binding=before,
                    payload_sha256=sha,counts={k:len(v) for k,v in data.items()}))
                del data,manifest
    before=inputs(view,mode);receipt_sha=file_sha(receipt)
    saved=json.loads(receipt.read_text())
    if saved.get('status')!='validated_student_snapshot' or saved.get('binding')!=before:
        raise ValueError('stale runtime snapshot source/input binding; use a new preparation directory')
    if file_sha(payload)!=saved['payload_sha256']:raise ValueError('runtime snapshot payload hash mismatch')
    value=json.loads(payload.read_text())
    if (inputs(view,mode)!=before or file_sha(receipt)!=receipt_sha
            or file_sha(payload)!=saved['payload_sha256']):
        raise ValueError('runtime snapshot or inputs changed during read')
    data,manifest=value['data'],value['manifest']
    if {k:len(v) for k,v in data.items()}!=saved['counts']:raise ValueError('runtime snapshot counts mismatch')
    if require_trainable and not manifest['training_admission']['trainable']:
        raise ValueError('student view training prerequisites failed: '+
                         '; '.join(manifest['training_admission']['errors']))
    print(f'[phase4 data] RGB={mode} verified snapshot read; rank={os.environ.get("RANK","CPU")} native_validation=not_repeated',flush=True)
    return data,manifest


def main():
    from qwen3vl_local.audit_joint.guard import no_core
    from .data import load_dataset
    no_core()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--rgb-mode',type=int,choices=(2,4),required=True)
    a=p.parse_args()
    if not os.environ.get('P4_VALIDATED_DATA'):p.error('set P4_VALIDATED_DATA for shared preparation')
    load_dataset(a.dataset,rgb_mode=a.rgb_mode)


if __name__=='__main__':main()
