"""Real four-rank Gloo/Linear optimizer and save/reload smoke, not a 4B GPU claim."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from qwen3vl_local.audit_joint.phase3_single_refine.training import train,objective,regions
from qwen3vl_local.audit_joint.phase3_single_refine.audit import sha


def worker(rank,root,bad):
    root=Path(root);torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(root/'rendezvous'),rank=rank,world_size=4)
    try:
        torch.manual_seed(23)
        model=torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2,6))
        bundle=SimpleNamespace(model=model,device='cpu')
        objective.readout=lambda b,p:b.model(p['x'])
        target=dict(valid=True,target=[0.,0.,0.,0.,0.,1.])
        work=[dict(id=str(i),targets=[target]*4) for i in range(12)]
        def pack(r):
            if bad and rank==2 and r['id']=='2':raise ValueError('bad image on rank 2')
            return dict(x=torch.ones(4,2),choices=list(range(6)))
        opt=torch.optim.AdamW(model.parameters(),lr=.01);scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        exposure={};total=0
        try:
            for _ in range(3):
                result=train.train_epoch(bundle,work,opt,scheduler,pack,rank=rank,world=4,exposure=exposure)
                exposure=result['actual_exposure'];total+=result['steps']
            assert all(v==3 for v in exposure.values()) and len(exposure)==12 and total==9
            artifact=dict(contract=regions.contract('six'),code={'test':'Gloo_Linear'})
            ds=root/'dataset.json';binding=dict(contract=artifact['contract'],code=artifact['code'],dataset_sha256=sha(ds))
            if rank==0:train.save_checkpoint(root/'checkpoint',bundle,opt,scheduler,binding=binding,epoch=3,step=total,exposure=exposure,status='epoch_complete')
            dist.barrier();before=model.module.weight.detach().clone()
            with torch.no_grad():model.module.weight.add_(1)
            train.restore(bundle,root/'checkpoint',artifact=artifact,dataset_path=ds)
            torch.testing.assert_close(before,model.module.weight,rtol=0,atol=0)
            payload=dict(status='passed',rank=rank,loss=result['loss'],steps=total,exposure=exposure)
        except ValueError as exc:
            if not bad:raise
            payload=dict(status='coordinated_input_failure',rank=rank,error=str(exc))
        (root/f'rank_{rank}.json').write_text(json.dumps(payload))
    finally:dist.destroy_process_group()

if __name__=='__main__':
    root=Path(sys.argv[1]).resolve();root.mkdir();(root/'dataset.json').write_text('{}')
    mp.spawn(worker,args=(str(root),sys.argv[2]=='bad'),nprocs=4,join=True)
