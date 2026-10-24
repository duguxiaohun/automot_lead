"""Bounded cooperative startup checks and coordinated, single-reader source audit."""
import math
import time
import torch.distributed as dist

from . import dataset


class Budget:
    def __init__(self, hours):
        if not math.isfinite(hours) or hours <= 0: raise ValueError('invalid_startup_budget')
        self.start = time.monotonic(); self.limit = hours*3600
        self.stages = []; self.stage = 'initialization'

    def check(self):
        if time.monotonic()-self.start > self.limit:
            raise TimeoutError('startup_time_budget_exceeded: '+self.stage)

    def run(self, stage, fn):
        self.stage = stage; self.check(); before = time.monotonic()
        print(f'[geometry startup] {stage} started', flush=True)
        result = fn()
        self.stages.append(dict(stage=stage, seconds=time.monotonic()-before))
        self.check()
        print(f'[geometry startup] {stage} completed in {self.stages[-1]["seconds"]:.2f}s', flush=True)
        return result

    def report(self):
        return dict(elapsed_seconds=time.monotonic()-self.start, limit_seconds=self.limit,
                    stages=self.stages, enforcement='cooperative checks; blocking library calls checked on return')


def primary(fn, *, rank, world):
    """Return a small result or the same failure on every rank."""
    result = [None]
    if rank == 0:
        try: result[0] = dict(value=fn(), error=None)
        except Exception as exc: result[0] = dict(error=f'{type(exc).__name__}: {exc}')
    if world > 1: dist.broadcast_object_list(result, src=0)
    if result[0]['error']: raise RuntimeError('startup primary failed: '+result[0]['error'])
    return result[0]['value']


def all_ranks(fn, *, world):
    value = None; error = None
    try: value = fn()
    except Exception as exc: error = f'{type(exc).__name__}: {exc}'
    errors = [error]
    if world > 1:
        errors = [None]*world; dist.all_gather_object(errors, error)
    if any(errors): raise RuntimeError('startup rank failed: '+str(errors))
    return value


def load_dataset(path, *, rank, world, budget, variant):
    holder = {}
    def verified():
        holder['artifact'], sha = budget.run('full_source_verification_rank0',
            lambda: dataset.load(path, variant=variant, check_budget=budget.check, with_sha256=True))
        return sha
    sha = primary(verified, rank=rank, world=world)
    def local():
        budget.check()
        if rank == 0: return holder['artifact']
        # Shared-node torchrun: workers bind the exact verified artifact bytes,
        # but do not re-hash all raw meta/RGB files. No reusable bypass receipt.
        return budget.run('verified_artifact_read', lambda: dataset.load(path, variant=variant,
            check_sources=False, expected_sha256=sha, check_budget=budget.check))
    return all_ranks(local, world=world), sha


def probe_records(work, validation):
    """At most one per event per train/val; no full-pool processor pass."""
    chosen = []
    for rows in (work, validation):
        seen = set()
        for row in rows:
            if row['event'] in seen: continue
            seen.add(row['event']); chosen.append(row)
    return chosen
