"""CPU checkpoint coordination: peers cannot enqueue the next CUDA collective."""
from __future__ import annotations
import copy
from datetime import timedelta
import os
from pathlib import Path
import tempfile
import time

from .progress import report

_CONTROL = {}


def control_group(world):
    import torch.distributed as dist
    if world == 1:
        return None
    if not dist.is_initialized() or dist.get_world_size() != world:
        raise RuntimeError('checkpoint_process_group_mismatch')
    if dist.get_backend() == 'gloo':
        return dist.group.WORLD
    # Keep checkpoint disk waits off the NCCL communicator and CUDA streams.
    parent = dist.group.WORLD
    if parent not in _CONTROL:
        _CONTROL[parent] = dist.new_group(backend='gloo', timeout=timedelta(seconds=600))
    return _CONTROL[parent]


def cpu_snapshot(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        result = copy.copy(value)  # Preserve OrderedDict state_dict metadata.
        for key, child in value.items():
            result[key] = cpu_snapshot(child)
        return result
    if isinstance(value, list):
        return [cpu_snapshot(child) for child in value]
    if isinstance(value, tuple):
        children = [cpu_snapshot(child) for child in value]
        return type(value)(*children) if hasattr(value, '_fields') else tuple(children)
    return value


def save(path, payload_factory, rank, world):
    """All ranks return after atomic publication, or receive the same write error."""
    import torch
    import torch.distributed as dist
    group = control_group(world)
    path = Path(path)
    result = [None]
    if rank == 0:
        temporary = None
        started = time.monotonic()
        try:
            report('checkpoint/cpu_snapshot', announce=True, checkpoint=str(path))
            payload = cpu_snapshot(payload_factory())
            staged = time.monotonic()
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
            temporary = Path(name)
            report('checkpoint/write', announce=True, checkpoint_temporary=str(temporary),
                   checkpoint_cpu_seconds=staged-started)
            with os.fdopen(fd, 'wb') as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            size = temporary.stat().st_size
            temporary.replace(path)
            result[0] = dict(ok=True, bytes=size, cpu_seconds=staged-started,
                             write_seconds=time.monotonic()-staged)
        except Exception as exc:
            result[0] = dict(ok=False, error=f'{type(exc).__name__}: {exc}',
                             temporary=str(temporary) if temporary else None)
    else:
        report('checkpoint/wait_for_writer', announce=True, checkpoint=str(path))
    if world > 1:
        try:
            dist.broadcast_object_list(result, src=0, group=group, device=torch.device('cpu'))
        except Exception as exc:
            raise RuntimeError(f'checkpoint CPU coordination failed for {path}; '
                               'inspect rank0 checkpoint progress and storage I/O') from exc
    if not result[0] or not result[0]['ok']:
        raise RuntimeError(f'checkpoint write failed for {path}: {result[0]}')
    report('checkpoint/saved', announce=True, checkpoint=str(path),
           checkpoint_bytes=result[0]['bytes'], checkpoint_cpu_seconds=result[0]['cpu_seconds'],
           checkpoint_write_seconds=result[0]['write_seconds'])
    return result[0]
