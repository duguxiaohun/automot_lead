"""Read-only host/cgroup evidence, including when a child cannot catch SIGKILL."""
import os
from pathlib import Path
import time


def snapshot(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
    def read(path):
        try:
            return path.read_text()[:16000]
        except OSError as exc:
            return {'unavailable': str(exc)}
    status = read(proc/'self/status')
    metrics = {}
    if isinstance(status, str):
        for line in status.splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'VmHWM', 'VmSize', 'VmSwap'):
                try:
                    metrics[key+'_bytes'] = int(value.split()[0])*1024
                except (ValueError, IndexError):
                    metrics[key+'_unavailable'] = value
    membership = read(proc/'self/cgroup')
    roots = {cgroup, cgroup/'memory'}
    if isinstance(membership, str):
        for line in membership.splitlines():
            parts = line.split(':', 2)
            if len(parts) == 3 and (parts[1] == '' or 'memory' in parts[1].split(',')):
                base = cgroup if parts[1] == '' else cgroup/'memory'
                # Some containers mount their own cgroup at the mount root;
                # others expose the absolute membership path beneath it.
                roots.add(base/parts[2].lstrip('/'))
    values = {}
    for root in sorted(roots):
        for name in ('memory.max','memory.current','memory.peak','memory.events',
                     'memory.limit_in_bytes','memory.usage_in_bytes','memory.max_usage_in_bytes',
                     'memory.failcnt','memory.oom_control'):
            path = root/name
            try:
                if path.is_file():
                    values[str(path)] = read(path)
            except OSError as exc:
                values[str(path)] = {'unavailable': str(exc)}
    return dict(timestamp=time.time(),pid=os.getpid(),process=metrics,
                meminfo=read(proc/'meminfo'),membership=membership,cgroup=values,
                counters_scope='cumulative; compare before/after, not a per-run OOM verdict')


def run_logged(command, log):
    from .launch import run_logged as run
    from qwen3vl_local.audit_joint.abc_delivery import write
    log = Path(log)
    target = log.parent.parent/'memory'/log.stem
    def record(suffix):
        try:
            write(target.with_suffix(suffix), snapshot())
        except OSError as exc:
            print(f'[host memory evidence unavailable] {exc}', flush=True)
    record('.before.json')
    try:
        return run(command, log)
    finally:
        record('.after.json')
