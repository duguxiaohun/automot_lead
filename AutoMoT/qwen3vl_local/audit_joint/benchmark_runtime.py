"""Bounded benchmark subprocesses and compact, validated Phase4 selections."""
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def progress(stage):
    print(f'[benchmark stage] pid={os.getpid()} {stage}', flush=True)


def trace_waits():
    faulthandler.enable()
    faulthandler.dump_traceback_later(30, repeat=True)


def descendants(root, known):
    """Remember start ticks too: never signal a reused PID or unrelated job."""
    entries = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            parts = path.read_text().rsplit(')', 1)[1].split()
            entries[int(path.parent.name)] = (int(parts[1]), parts[19])
        except (OSError, ValueError, IndexError):
            continue
    roots = {root} if root in entries and (root not in known or entries[root][1] == known[root]) else set()
    parents = roots | {pid for pid, start in known.items()
                               if pid in entries and entries[pid][1] == start}
    while True:
        added = {pid for pid, (ppid, _) in entries.items() if ppid in parents} - parents
        if not added:
            break
        parents.update(added)
    for pid in parents:
        if pid in entries:
            known[pid] = entries[pid][1]


def signal_known(known, number):
    for pid, start in list(known.items()):
        try:
            parts = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            if parts[19] == start:
                os.kill(pid, number)
        except (OSError, IndexError):
            pass


def run_owned(command, log, env, timeout):
    """Stream logs, bound wall time, and clean descendants in separate sessions."""
    stopped = []
    handlers = {n: signal.signal(n, lambda number, frame: stopped.append(number))
                for n in (signal.SIGINT, signal.SIGTERM)}
    process = None
    known = {}
    started = time.monotonic()
    code = 2
    try:
        with Path(log).open('xb') as stream, Path(log).open() as reader:
            process = subprocess.Popen(command, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            def relay():
                text = reader.read()
                if text:
                    print(text, end='', flush=True)
            while True:
                descendants(process.pid, known)
                relay()
                if stopped:
                    code = 128 + stopped[0]
                    break
                if time.monotonic() - started >= timeout:
                    code = 124
                    break
                result = process.poll()
                if result is not None:
                    code = result if result >= 0 else 128 - result
                    break
                time.sleep(.1)
            relay()
    finally:
        if process is not None:
            descendants(process.pid, known)
            signal_known(known, signal.SIGTERM)
            # torchrun workers can create their own sessions; killpg alone misses them.
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                process.poll()
                live = []
                for pid, start in known.items():
                    try:
                        parts = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                        if parts[19] == start and parts[0] != 'Z':
                            live.append(pid)
                    except (OSError, IndexError):
                        pass
                if not live:
                    break
                time.sleep(.05)
            descendants(process.pid, known)
            signal_known(known, signal.SIGKILL)
            process.wait()
        for number, handler in handlers.items():
            signal.signal(number, handler)
    return code


def prepare_phase4(args):
    from .batch_benchmark import load_pool, schedule, write
    progress('load_validate_full_pool START (CPU only, once for this RGB mode)')
    items, records, manifest = load_pool(args, args.modes[0])
    progress(f'load_validate_full_pool DONE rows={len(items)}; sampling START')
    updates, selection = schedule(records, per_event=args.samples_per_event,
        update_samples=args.update_samples, world=len(args.gpus.split(',')),
        batch=args.batches[0], mode=args.modes[0], seed=args.seed)
    used = sorted({i for block in updates for i in block})
    remap = {old: new for new, old in enumerate(used)}
    payload = dict(schema='phase4_benchmark_selection_v1', mode=args.modes[0],
        seed=args.seed, per_event=args.samples_per_event, update_samples=args.update_samples,
        world=len(args.gpus.split(',')), items=[items[i] for i in used],
        records=[records[i] for i in used], updates=[[remap[i] for i in block] for block in updates],
        selection=selection, manifest={'observation_contract': manifest['observation_contract']})
    write(args.output, payload)
    progress(f'sampling DONE; prepared unique={len(used)} presentations={selection["presentations"]}')


def read_prepared(args):
    raw = args.prepared_samples.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.prepared_sha256:
        raise ValueError('prepared sample digest mismatch')
    data = json.loads(raw)
    expected = dict(schema='phase4_benchmark_selection_v1', mode=args.modes[0], seed=args.seed,
        per_event=args.samples_per_event, update_samples=args.update_samples,
        world=len(args.gpus.split(',')))
    if any(data.get(k) != v for k, v in expected.items()):
        raise ValueError('prepared sample configuration mismatch')
    selection = dict(data['selection'], accumulation=args.update_samples // (data['world'] * args.batches[0]))
    return data['items'], data['records'], data['manifest'], data['updates'], selection
