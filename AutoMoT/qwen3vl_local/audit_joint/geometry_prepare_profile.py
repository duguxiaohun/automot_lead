"""Observe the native geometry preparation without changing cache identities.

Reports append to a separate JSONL, never to cached payloads or their contracts.
Stage durations are inclusive and can overlap; they must not be summed.
"""
import argparse
from contextlib import contextmanager
import fcntl
import functools
import json
import os
from pathlib import Path
import sys
import time
import uuid

SCHEMA = 'geometry_prepare_timing_v1'


class Profile:
    def __init__(self):
        self.started=time.perf_counter()
        self.stages={}
        self.stack=[]
        self.hash_calls=0
        self.hash_bytes=0
        self.actions=[]
        self.selections=[]

    @contextmanager
    def stage(self,name):
        start=time.perf_counter()
        row=self.stages.setdefault(name,dict(calls=0,failures=0,inclusive_seconds=0.,hash_calls=0,hash_bytes=0))
        row['calls']+=1;self.stack.append(name)
        print('[prepare timing] start '+name,flush=True)
        try:
            yield
        except BaseException:
            row['failures']+=1
            raise
        finally:
            elapsed=time.perf_counter()-start
            row['inclusive_seconds']+=elapsed
            self.stack.pop()
            print(f'[prepare timing] finish {name}: {elapsed:.3f}s',flush=True)

    def wrap(self,name,fn):
        @functools.wraps(fn)
        def call(*args,**kwargs):
            with self.stage(name):
                result=fn(*args,**kwargs)
                if name=='geometry.prepare':
                    self.selections.append({k:result[k] for k in
                        ('action','dataset','selection','candidate_completeness_verified') if k in result})
                return result
        return call

    def report(self,status,error=None):
        return dict(schema=SCHEMA,run_id=uuid.uuid4().hex,status=status,error=error,
            total_seconds=time.perf_counter()-self.started,stages=self.stages,
            cache_actions=self.actions,selections=self.selections,
            adapter_hashing=dict(calls=self.hash_calls,bytes=self.hash_bytes),
            interpretation='Stage times are inclusive and overlap. Hash counters cover successful versioned_data.bind calls only, count repeated reads, and exclude hashing internal to native geometry validators. No throughput guarantee.',
            training_performed=False)

    @contextmanager
    def instrument(self):
        from . import data_prepare,versioned_data as cache
        from .geometry_v1 import dataset
        replaced=[]
        def replace(module,name,fn):
            replaced.append((module,name,getattr(module,name)));setattr(module,name,fn)
        original_bind=cache.bind
        def bind(path):
            value=original_bind(path)
            self.hash_calls+=1;self.hash_bytes+=value['bytes']
            # Attribute to the innermost measured stage only (not every parent).
            if self.stack:
                row=self.stages[self.stack[-1]];row['hash_calls']+=1;row['hash_bytes']+=value['bytes']
            return value
        original_ensure=cache.ensure
        original_routes=dataset.annotation_routes
        def annotation_routes(*args,**kwargs):
            with self.stage('candidates.enumerate'):
                yield from original_routes(*args,**kwargs)
        def ensure(root,family,version,contract,build,validate,**kwargs):
            with self.stage(family+'.cache'):
                result=original_ensure(root,family,version,
                    self.wrap(family+'.dependencies',contract),self.wrap(family+'.build',build),
                    self.wrap(family+'.validate',validate),**kwargs)
                self.actions.append(dict(family=family,action=result['action'],key=result['key']))
                return result
        try:
            replace(cache,'bind',bind);replace(cache,'ensure',ensure)
            replace(dataset,'annotation_routes',annotation_routes)
            for module,name,label in (
                (cache,'bindings','dependencies.hash_files'),(cache,'outputs','outputs.hash_files'),
                (cache,'discover','discovery.enumerate'),(cache,'unique_compatible','discovery.validate_candidates'),
                (dataset,'build','native.build'),(dataset,'load','native.validate'),
                (data_prepare,'geometry','geometry.prepare'),(data_prepare,'geometry_plan','geometry.plan')):
                replace(module,name,self.wrap(label,getattr(module,name)))
            yield
        finally:
            for module,name,fn in reversed(replaced):setattr(module,name,fn)


def append_report(path,report):
    path=Path(path)
    if path.suffix!='.jsonl' or path.is_symlink():
        raise ValueError('Timing report must be a separate regular .jsonl file')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        stream.seek(0)
        for line in stream:
            if not line.strip() or json.loads(line).get('schema')!=SCHEMA:
                raise ValueError('Existing timing file is not this profiler JSONL; preserved')
        stream.seek(0,os.SEEK_END)
        stream.write(json.dumps(report,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False,add_help=False)
    p.add_argument('--timing-report',type=Path,required=True)
    p.add_argument('--method',choices=['geometry'],default='geometry')
    args,remaining=p.parse_known_args()
    if args.timing_report.suffix!='.jsonl' or args.timing_report.is_symlink():
        p.error('--timing-report must be a separate regular .jsonl file')
    from . import data_prepare
    profile=Profile();status='failed';error=None;original_argv=sys.argv
    try:
        sys.argv=[original_argv[0],'--method','geometry',*remaining]
        with profile.instrument():data_prepare.main()
        status='completed'
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        sys.argv=original_argv
        report=profile.report(status,error)
        try:append_report(args.timing_report,report)
        except Exception as exc:
            # An observational write error must not hide the native failure.
            if error is None:raise
            print(f'[prepare timing] report write failed: {exc}',file=sys.stderr,flush=True)
        print('[prepare timing] '+json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
