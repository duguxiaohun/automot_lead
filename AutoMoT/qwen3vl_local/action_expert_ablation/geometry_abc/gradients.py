"""Observe successful optimizer steps before zero_grad; never change the loop."""
from collections import Counter
import torch
from .model import embedding_diagnostics


class Evidence:
    def __init__(self,model,optimizer,rank):
        self.model=model;self.rank=rank;self.steps=0;self.pending=Counter();self.exposure=Counter()
        self.sum_norms=torch.zeros(10,dtype=torch.float64);self.last=None
        self.handle=optimizer.register_step_post_hook(self.after_step)

    def observe(self,record,variant):
        self.pending['samples']+=1
        for role,key in (('future','targets'),('history','history')):
            for h,t in enumerate(record['representations'][variant][key]):
                token=t['token_id'] if t['valid'] else 9
                self.pending[f'{role}/{h}/{token}']+=1

    def after_step(self,optimizer,args,kwargs):
        grad=self.model.conditioner.geometry_embedding.token.weight.grad
        if grad is None or not torch.isfinite(grad).all():raise ValueError('missing_or_nonfinite_token_gradient')
        if not self.pending['samples']:raise ValueError('token_gradient_without_consumed_slots')
        norms=grad.detach().float().norm(dim=1).cpu().double()
        self.last=norms.tolist();self.sum_norms+=norms;self.steps+=1
        self.exposure.update(self.pending);self.pending.clear()

    def snapshot(self,step):
        if self.steps!=step or self.pending:raise ValueError('token_gradient_step_or_exposure_mismatch')
        report=embedding_diagnostics(self.model)
        report.update(gradient_norms=self.last,gradient_norm_sums=self.sum_norms.tolist(),
            optimizer_steps=self.steps,slot_exposure=dict(self.exposure),rank=self.rank,
            gradient_scope='DDP reduced, clipped, after successful optimizer step before zero_grad',
            exposure_scope='local rank successful training presentations; future/history slot counts separate',
            gradient_roles='shared embedding combined gradient; not a causal role decomposition')
        return report

    def close(self):self.handle.remove()
