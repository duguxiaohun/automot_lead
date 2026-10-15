import math
import pytest
from qwen3vl_local.audit_joint.batch_probe import capacity_summary
from qwen3vl_local.sft_new_loop_phase3.training_schedule import resolve_warmup


@pytest.mark.parametrize('batch,accumulation',[(1,1),(2,1),(4,1),(8,1),(2,2),(2,3)])
def test_warmup_uses_optimizer_budget_and_scheduler_reaches_peak(batch,accumulation):
    import torch
    from qwen3vl_local.sft_v2.train import make_scheduler
    total=math.ceil((10240//(4*batch)*3)/accumulation)
    config=resolve_warmup(total)
    warmup=config['warmup_updates']
    assert warmup==math.ceil(total*.1) and warmup<total
    parameter=torch.nn.Parameter(torch.tensor(1.))
    optimizer=torch.optim.SGD([parameter],lr=1.)
    scheduler=make_scheduler(optimizer,total_steps=total,warmup_steps=warmup)
    rates=[]
    for _ in range(total):
        optimizer.step();scheduler.step();rates.append(optimizer.param_groups[0]['lr'])
    assert max(rates)==pytest.approx(1.) and rates[-1]<rates[warmup-1]


def test_explicit_profile_and_short_run_warmup():
    assert resolve_warmup(1920,steps=500)['warmup_updates']==500
    assert resolve_warmup(1)['warmup_updates']==0
    assert resolve_warmup(5,steps=0)['warmup_updates']==0
    for value in (5,2000,-1):
        with pytest.raises(ValueError,match='warmup'):resolve_warmup(5,steps=value)
    for ratio in (-1,1,float('nan')):
        with pytest.raises(ValueError):resolve_warmup(100,ratio=ratio)


def trials():
    return [dict(gpu=g,rgb_mode=m,batch_size=b,status='measured',fits_with_headroom=True)
            for g in ('0','1') for m in (2,4) for b in range(1,8)]


def test_probe_separates_capacity_from_formal_launch_compatibility():
    result=capacity_summary(trials(),gpus=['0','1'],modes=[2,4],phase='4')
    assert result['conservative_measured_candidates']=={'2':7,'4':7}
    for mode in ('2','4'):
        assert result['training_compatible_candidates'][mode]==[
            dict(batch_size=1,accumulation=8),dict(batch_size=2,accumulation=4),dict(batch_size=4,accumulation=2)]
        assert set(result['training_exclusions'][mode])=={'3','5','6','7'}
    # Custom quota cannot be mistaken for compatibility with the fixed P4 entry.
    result=capacity_summary(trials(),gpus=['0'],modes=[2],phase='4',per_event=1032,update_samples=24)
    assert result['training_compatible_candidates']['2']==[]


def test_probe_requires_same_batch_actually_measured_on_every_gpu():
    rows=[r for r in trials() if not (r['gpu']=='1' and r['batch_size']==4)]
    for r in rows:
        if r['batch_size']>4:r['fits_with_headroom']=False
    result=capacity_summary(rows,gpus=['0','1'],modes=[2],phase='3')
    assert result['conservative_measured_candidates']=={'2':3}
    assert [r['batch_size'] for r in result['training_compatible_candidates']['2']]==[1,2]
    assert capacity_summary(rows,gpus=['0','missing'],modes=[2],phase='3')['conservative_measured_candidates']=={'2':0}
