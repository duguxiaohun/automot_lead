"""Explicit five-method comparison contract; independent eligible sample pools."""
import math

SEED = 20260904

def contract():
    return dict(version='phase3_five_method_comparison_v1', methods=['base','single','a','b','c'],
        per_event=1280, events=10, epochs=3, world_size=4, batch_size_per_rank=1,
        accumulation=1, presentations_per_epoch=12800, optimizer_steps_per_epoch=3200,
        total_optimizer_steps=9600, warmup_steps=2500, learning_rate=1e-5,
        seed=SEED, scheduler='warmup_then_cosine_native_step_index',
        selection='final_completed_epoch', sample_alignment='independent pools and sampling allowed',
        abc_alignment='shared geometric members, masks and plan; different labels',
        intentional_differences=['four images versus one','single versus four queries',
            'native action versus spatial region targets','offline action versus causal position history',
            'native full vocabulary versus nine region loss'])

def require(*,per_event,epochs,seed,learning_rate):
    if (per_event,epochs,seed,learning_rate)!=(1280,3,SEED,1e-5):
        raise ValueError('comparison_requires_1280_three_epochs_seed20260904_lr1e-5')

def lr_multiplier(step,total,warmup):
    if not 0 < warmup < total: raise ValueError('invalid_schedule_budget')
    if step < warmup: return step/max(1,warmup)
    progress=(step-warmup)/max(1,total-warmup)
    return max(0.,.5*(1.+math.cos(math.pi*progress)))

def make_scheduler(optimizer,total,warmup):
    import torch
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:lr_multiplier(s,total,warmup))
