import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from qwen3vl_local.sft_new_loop_phase12 import distributed_runtime as runtime


def test_nccl_device_binding_and_short_communication_timeout(monkeypatch):
    calls=[]
    monkeypatch.setattr(runtime.torch.cuda,'set_device',lambda device:calls.append(device))
    monkeypatch.setattr(runtime.dist,'init_process_group',lambda *a,**k:calls.append((a,k)))
    assert runtime.initialize(3)==torch.device('cuda:3')
    assert calls[0]==torch.device('cuda:3')
    assert calls[1][1]['device_id']==torch.device('cuda:3')
    assert calls[1][1]['timeout'].total_seconds()==300


WORKER='''
import json,os,sys,time
from pathlib import Path
from contextlib import nullcontext
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from qwen3vl_local.sft_new_loop_phase12.distributed_runtime import Stages
from qwen3vl_local.sft_new_loop_phase1.run_integrity import finite_loss
torch.set_num_threads(1)
dist.init_process_group('gloo')
rank=dist.get_rank();world=dist.get_world_size();root=Path(sys.argv[1]);mode=sys.argv[2]
store=dist.PrefixStore(mode,dist.distributed_c10d._get_default_store())
stages=Stages(store,rank,world,root/'diagnostics')
if mode=='uncaught':
    from torch.distributed.elastic.multiprocessing.errors import record
    @record
    def entry():
        def fail():raise ValueError('original_rank0_probe_error')
        stages.run('probe_error',fail,rank0_only=True,timeout=10)
    entry()
if mode=='hang':
    stages.run('stuck_probe',lambda:time.sleep(60),rank0_only=True,timeout=2)
    raise AssertionError('deadline did not fire')
if mode=='error':
    def fail():raise ValueError('original_rank0_probe_error')
    try:stages.run('probe_error',fail,rank0_only=True,timeout=10)
    except (ValueError,RuntimeError) as exc:
        assert 'original_rank0_probe_error' in str(exc)
    else:raise AssertionError('swallowed rank0 failure')
else:
    # Other ranks wait on CPU while rank0 executes an uneven solo stage.
    value=stages.run('probe',lambda:(time.sleep(.3),17)[1],rank0_only=True,timeout=10)
    assert value==(17 if rank==0 else None)
    torch.manual_seed(3);model=torch.nn.Linear(2,1,bias=False);ref=torch.nn.Linear(2,1,bias=False)
    ref.load_state_dict(model.state_dict());model=DDP(model)
    opt=torch.optim.SGD(model.parameters(),lr=.01);ref_opt=torch.optim.SGD(ref.parameters(),lr=.01)
    for step in range(2):
        opt.zero_grad();ref_opt.zero_grad()
        for branch in range(2):
            x=torch.tensor([[rank+1.,branch+1.]])
            with model.no_sync() if branch==0 else nullcontext():
                loss=(model(x)-(rank-branch)) .square().mean()
                finite_loss(loss,distributed=dist);(loss*.5).backward()
            for peer in range(world):
                xx=torch.tensor([[peer+1.,branch+1.]])
                ((ref(xx)-(peer-branch)).square().mean()*.5/world).backward()
        torch.testing.assert_close(model.module.weight.grad,ref.weight.grad)
        opt.step();ref_opt.step();torch.testing.assert_close(model.module.weight,ref.weight)
    # Finalization must not race writes from slower ranks.
    if rank==world-1:time.sleep(.3)
    (root/f'optimizer_{rank}').write_text('2')
    def finalize():assert len(list(root.glob('optimizer_*')))==world
    stages.run('finalize',finalize,rank0_only=True,timeout=10)
dist.barrier();dist.destroy_process_group()
print('WORKER_PASSED',rank,flush=True)
'''


@pytest.mark.parametrize('mode',['success','error','hang','uncaught'])
def test_four_real_gloo_ranks_gradient_equivalence_errors_and_deadlines(tmp_path,mode):
    worker=tmp_path/'worker.py';worker.write_text(WORKER)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'PYTHONPATH':str(Path(__file__).resolve().parents[3])}
    result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
        '--max_restarts=0','--log-dir',str(tmp_path/'torchrun'),str(worker),str(tmp_path),mode],env=env,capture_output=True,text=True,timeout=60)
    log=result.stdout+result.stderr
    if mode=='hang':
        assert result.returncode!=0 and 'Stage deadline exceeded: stuck_probe' in log,log
        assert 'NCCL' not in log
    elif mode=='uncaught':
        assert result.returncode!=0 and 'original_rank0_probe_error' in log,log
        errors=list((tmp_path/'torchrun').rglob('error.json'))
        assert errors and any('original_rank0_probe_error' in p.read_text() for p in errors)
    else:
        assert result.returncode==0 and log.count('WORKER_PASSED')==4,log
