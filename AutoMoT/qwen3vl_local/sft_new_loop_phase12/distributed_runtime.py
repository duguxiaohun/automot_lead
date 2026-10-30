"""Bounded CPU coordination for work that only rank zero performs.

Never leave NCCL work pending while other ranks wait for model probes or
validation. The torchrun store is independent of CUDA collectives.
"""
import faulthandler
import json
import math
import os
from pathlib import Path
import threading
import time
import traceback
from datetime import timedelta
import torch
import torch.distributed as dist
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof


def initialize(local):
    device=torch.device('cuda',local)
    torch.cuda.set_device(device)
    # Match the explicit rank/device binding used by independent Phase2.
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=5))
    return device


class Stages:
    def __init__(self,store,rank,world,output):
        self.store=store;self.rank=rank;self.world=world;self.output=Path(output);self.sequence=0

    def run(self,name,fn,*,rank0_only=False,timeout=900):
        if not math.isfinite(timeout) or timeout<=0:raise ValueError('Finite positive stage deadline required')
        self.sequence+=1;key=f'{self.sequence}:{name}';started=time.monotonic()
        path=self.output/f'{self.sequence:03d}_{name}_rank{self.rank}.json'
        record=dict(stage=name,rank=self.rank,world=self.world,rank0_only=rank0_only,timeout_seconds=timeout)
        lock=threading.Lock();finished=threading.Event()
        def publish(status,**extra):
            item=dict(record,status=status,elapsed_seconds=time.monotonic()-started,**extra)
            proof.write(path,item)
            self.store.set(f'{key}/{self.rank}',json.dumps(item))
        def expired():
            with lock:
                if finished.is_set():return
                message=f'Stage deadline exceeded: {name} rank={self.rank} timeout={timeout}s'
                print('[fusion-stage:FAILED] '+message,flush=True)
                faulthandler.dump_traceback()
                # Do not enter destroy_process_group with hung peers/kernels.
                try:publish('failed',error=message)
                finally:os._exit(124)
        timer=threading.Timer(timeout,expired);timer.daemon=True
        print(f'[fusion-stage] rank={self.rank} START {name} timeout={timeout}s',flush=True)
        publish('running');timer.start()
        try:
            # Everyone must finish the preceding work before rank zero reads
            # per-rank completion files or starts a solo evaluation/save.
            keys=[f'{key}/{rank}' for rank in range(self.world)]
            while not self.store.check(keys):
                for peer in keys:
                    if self.store.check([peer]):
                        status=json.loads(self.store.get(peer))
                        if status['status']=='failed':raise RuntimeError(f"{name}: rank {status['rank']}: {status['error']}")
                time.sleep(.1)
            value=fn() if not rank0_only or self.rank==0 else None
            publish('done')
            next_message=time.monotonic()+30
            while True:
                pending=[]
                for rank in range(self.world):
                    peer=f'{key}/{rank}'
                    if not self.store.check([peer]):pending.append(rank);continue
                    status=json.loads(self.store.get(peer))
                    if status['status']=='failed':raise RuntimeError(f"{name}: rank {rank}: {status['error']}")
                    if status['status']!='done':pending.append(rank)
                if not pending:break
                if time.monotonic()>=next_message:
                    print(f'[fusion-stage] rank={self.rank} {name} waiting_for={pending} elapsed={time.monotonic()-started:.1f}s',flush=True)
                    next_message=time.monotonic()+30
                time.sleep(.1)
            print(f'[fusion-stage] rank={self.rank} DONE {name}',flush=True)
            return value
        except BaseException as exc:
            publish('failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
            raise
        finally:
            with lock:finished.set();timer.cancel()
