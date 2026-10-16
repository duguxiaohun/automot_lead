"""Resumable, bounded validation of existing native teacher route artifacts.

Cache receipts are local checkpoints, not signed attestations or full-pool
admission. Reuse requires unchanged source, index, route input and artifact bytes.
No teacher replay, training, native dataset modification or persistent row cache.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import sys
import threading

from .benchmark_runtime import progress, run_owned, trace_waits
from .io import file_sha
from qwen3vl_local.sft_new_loop_phase4.identity import digest, write_json

POLICY = 'phase4_verified_production_routes_v1'


def source_state():
    from qwen3vl_local.sft_new_loop_phase4 import teacher_rules as rules
    teacher = deepcopy(rules.identity())
    # Read bytes directly even inside the isolated source-snapshot adapter.
    teacher['sources'] = {name: file_sha(rules.ROOT/name) for name in rules.SOURCES}
    return dict(teacher=teacher, adapters={name: file_sha(Path(__file__).with_name(name))
                for name in ('phase4_production_cache.py', 'phase4_load.py')})


class RouteCache:
    def __init__(self, native, directory):
        from qwen3vl_local.sft_new_loop_phase4 import production_storage as storage
        from qwen3vl_local.sft_new_loop_phase4 import teacher_replay as replay
        self.native = Path(native).resolve(strict=True)
        self.directory = Path(directory).resolve()
        if self.directory.is_relative_to(self.native):
            raise ValueError('cache must be outside native input')
        self.sources = source_state()
        self.manifest_path = self.native/'manifest.json'
        self.pool_path = self.native/'candidate_pool.json'
        self.manifest_sha = file_sha(self.manifest_path)
        manifest = json.loads(self.manifest_path.read_text())
        self.pool_sha = file_sha(self.pool_path)
        if self.pool_sha != manifest['candidate_pool_sha256']:
            raise ValueError('candidate pool hash mismatch')
        self.pool = json.loads(self.pool_path.read_text())
        if self.pool.get('sha256') != digest({k:v for k,v in self.pool.items() if k!='sha256'}):
            raise ValueError('candidate pool self hash mismatch')
        progress('production input binding START (artifact SHA checks; CPU only)')
        self.index_path = storage.resolve(self.native, manifest)
        self.index_sha = file_sha(self.index_path)
        self.index = json.loads(self.index_path.read_text())
        self.header = dict(policy=replay.POLICY, teacher=self.sources['teacher'],
                           candidate_pool_sha256=self.pool['sha256'])
        if (self.index.get('sha256') != digest({k:v for k,v in self.index.items() if k!='sha256'})
                or any(self.index.get(k)!=v for k,v in self.header.items())):
            raise ValueError('teacher production identity mismatch')
        self.routes = {(r['scenario'],r['route_id']):r for r in self.pool['routes']}
        if len(self.routes)!=len(self.pool['routes']):raise ValueError('duplicate candidate route')
        self.receipts = {}
        seen = set()
        for receipt in self.index['routes']:
            name = receipt['artifact']; key = receipt['scenario'],receipt['route_id']
            if (name in self.receipts or Path(name).name!=name or name in ('.','..','index.json')
                    or key in seen or key not in self.routes):
                raise ValueError('invalid/duplicate/foreign production artifact')
            self.receipts[name] = receipt; seen.add(key)
        self.binding = dict(policy=POLICY, native=str(self.native), sources=self.sources,
            manifest_sha256=self.manifest_sha, pool_sha256=self.pool_sha,
            index_path=str(self.index_path.absolute()), index_resolved=str(self.index_path.resolve()),
            index_sha256=self.index_sha)
        self.fresh = self.reused = 0
        progress(f'production input binding DONE routes={len(self.receipts)}')

    def recheck(self):
        if (source_state()!=self.sources or file_sha(self.manifest_path)!=self.manifest_sha
                or file_sha(self.pool_path)!=self.pool_sha or file_sha(self.index_path)!=self.index_sha
                or str(self.index_path.resolve())!=self.binding['index_resolved']):
            raise ValueError('production inputs or validator sources changed during validation')

    def inspect(self, original, path, route, header):
        from qwen3vl_local.sft_new_loop_phase4.production_storage import binding
        path = Path(path)
        expected = self.receipts.get(path.name)
        key = route['scenario'],route['route_id']
        if (expected is None or path.absolute()!=self.index_path.absolute().parent/path.name
                or route!=self.routes.get(key) or header!=self.header
                or key!=(expected['scenario'],expected['route_id'])):
            raise ValueError('route outside cache-bound production index')
        artifact = binding(path)
        if artifact['sha256']!=expected['sha256']:raise ValueError('production artifact hash mismatch')
        item_binding = dict(artifact=artifact, route_sha256=digest(route),
                            transaction_sha256=digest(self.binding))
        dest = self.directory/'routes'/(digest(path.name)+'.json')
        if dest.exists():
            saved = json.loads(dest.read_text())
            if (saved.get('sha256')!=digest({k:v for k,v in saved.items() if k!='sha256'})
                    or saved.get('binding')!=item_binding or saved.get('receipt')!=expected
                    or saved.get('status')!='route_validated'):
                raise ValueError('invalid/stale route cache receipt')
            actual = deepcopy(saved['receipt']); self.reused += 1
        else:
            actual = original(path,route,header)
            if actual!=expected:raise ValueError('teacher route receipt mismatch')
            if binding(path)!=artifact:raise ValueError('route changed during validation')
            # Persist only a fully checked route, with current source/input SHA.
            # The index/pool objects are immutable snapshots already bound above;
            # rehash their files at transaction end, not thousands of times.
            if source_state()!=self.sources:
                raise ValueError('validator sources changed during route validation')
            saved = dict(status='route_validated', binding=item_binding, receipt=actual)
            saved['sha256'] = digest(saved)
            write_json(dest,saved); self.fresh += 1
        count = self.fresh+self.reused
        write_json(self.directory/'progress.json',dict(status='in_progress',
            checked_this_invocation=count,total_routes=len(self.receipts),
            newly_validated=self.fresh,reused=self.reused,full_pool_admitted=False))
        if count==1 or count%100==0 or count==len(self.receipts):
            progress(f'production routes checked={count}/{len(self.receipts)} new={self.fresh} reused={self.reused}')
        return actual


@contextmanager
def production_routes(native, directory):
    if directory is None:
        yield None
        return
    from qwen3vl_local.sft_new_loop_phase4 import teacher_replay as replay
    if threading.current_thread() is not threading.main_thread() or threading.active_count()!=1:
        raise RuntimeError('route cache requires an isolated single-threaded process')
    original = replay.inspect_route
    if getattr(original,'_route_cache',False):raise RuntimeError('nested route cache')
    directory = Path(directory)
    if directory.resolve().is_relative_to(Path(native).resolve()):
        raise ValueError('cache must be outside native input')
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as ex:raise ValueError('route cache already in use') from ex
        cache = RouteCache(native,directory)
        meta = directory/'binding.json'
        if meta.exists():
            if json.loads(meta.read_text())!=cache.binding:
                raise ValueError('cache source/input binding changed; use a new cache directory')
        else:write_json(meta,cache.binding)
        def inspect(path,route,header):return cache.inspect(original,path,route,header)
        inspect._route_cache = True
        replay.inspect_route = inspect
        try:
            yield cache
            cache.recheck()
        finally:
            replay.inspect_route = original


def main():
    from .guard import no_core
    no_core()
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True,help='Keep the same directory to resume verified routes')
    p.add_argument('--output',type=Path,required=True,help='New log directory for each bounded invocation')
    p.add_argument('--timeout',type=int,default=240)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.timeout<1:p.error('timeout must be positive')
    if (a.output.resolve().is_relative_to(a.native.resolve())
            or a.cache.resolve().is_relative_to(a.native.resolve())
            or a.output.resolve().is_relative_to(a.cache.resolve())
            or a.cache.resolve().is_relative_to(a.output.resolve())):
        p.error('native, cache and output directories must be separate')
    if a.worker:
        from .phase4_load import validated_reservations
        from qwen3vl_local.sft_new_loop_phase4 import teacher_replay as replay
        trace_waits()
        with production_routes(a.native,a.cache) as cache, validated_reservations():
            accounting = replay.accounting(cache.pool,cache.index_path)
        result = dict(status='production_routes_validated',full_pool_admitted=False,
            total_routes=len(cache.receipts),newly_validated=cache.fresh,reused=cache.reused,
            accounting=accounting,scope='Existing production artifacts only; native rows/view/admission not certified')
        write_json(a.output/'result.json',result)
        write_json(a.cache/'progress.json',{k:v for k,v in result.items() if k!='accounting'})
        print(json.dumps({k:v for k,v in result.items() if k!='accounting'}),flush=True)
        return
    a.output.mkdir(parents=True,exist_ok=False)
    args = [sys.executable,'-u','-m','qwen3vl_local.audit_joint.phase4_production_cache',
            '--worker','--native',str(a.native),'--cache',str(a.cache),'--output',str(a.output)]
    progress(f'output={a.output}; wall limit={a.timeout}s; route receipts survive timeout')
    code = run_owned(args,a.output/'load.log',dict(os.environ,CUDA_VISIBLE_DEVICES=''),a.timeout)
    write_json(a.output/'status.json',dict(returncode=code,completed=code==0,
        cache=str(a.cache),log=str(a.output/'load.log'),full_pool_admitted=False))
    raise SystemExit(code)


if __name__=='__main__':main()
