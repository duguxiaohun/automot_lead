"""仅小矩阵 CUDA 验证；无 Qwen/BEV/数据，双卡时额外核验 NCCL 累积更新。"""
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from io import BytesIO
import math
from pathlib import Path
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_prior.optimization import orthogonalized_update, optimizer_step_with_metrics
from qwen3vl_local.action_prior.tests.test_optimization import SmallDecoder, DistributedMatrix, make, advance

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required; CPU does not verify BF16 kernels')


def spectral_reference(gradient, steps):
    """独立 FP64 SVD 参考：在奇异值上求五次多项式，避免重用矩阵 NS 实现。

    多项式见 https://github.com/KellerJordan/Muon/blob/master/muon.py 。
    FP64 理想算术与 CUDA BF16 有舍入差异，比较相对 Frobenius 误差。
    """
    value = gradient.detach().cpu().double()
    u, singular, vh = torch.linalg.svd(value, full_matrices=False)
    singular = singular / value.norm().clamp_min(1e-7)
    for _ in range(steps):
        singular = 3.4445 * singular - 4.775 * singular**3 + 2.0315 * singular**5
    return (u * singular) @ vh


@pytest.mark.parametrize('shape', [(17, 61), (61, 17), (32, 32)])
def test_cuda_bf16_against_spectral_reference(shape):
    torch.manual_seed(42)
    gradient = torch.randn(shape, device='cuda')
    actual = orthogonalized_update(gradient, 5)
    expected = spectral_reference(gradient, 5)
    error = (actual.cpu().double() - expected).norm() / expected.norm()
    assert torch.isfinite(actual).all() and actual.dtype == torch.float32
    assert error < .08, f'BF16 NS relative Frobenius error={error.item()}'
    assert torch.count_nonzero(orthogonalized_update(torch.zeros_like(gradient), 5)) == 0
    assert torch.isfinite(orthogonalized_update(gradient * 1e-10, 5)).all()


@pytest.mark.parametrize('cut', [2, 3, 4])
def test_cuda_restart_resume_and_monitor(cut):
    """18步、warmup1、默认短周期2/5/10；在低谷前/低谷/下一峰值恢复。"""
    model = SmallDecoder().cuda()
    opt, scheduler = make(model, total=18)
    assert opt.action_contract['schedule']['cycle_steps'] == [2, 5, 10]
    for index in range(cut):
        advance(model, opt, scheduler, index)
    data = BytesIO()
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scheduler=scheduler.state_dict()), data)
    data.seek(0)
    saved = torch.load(data, map_location='cpu', weights_only=False)
    resumed = SmallDecoder().cuda()
    resumed.load_state_dict(saved['model'])
    opt2, scheduler2 = make(resumed, total=18)
    opt2.load_state_dict(saved['opt']); scheduler2.load_state_dict(saved['scheduler'])
    for index in range(cut, 18):
        advance(model, opt, scheduler, index)
        # 监控必须不改变更新结果；同时覆盖 CUDA event 计时。
        for i, p in enumerate(resumed.parameters()):
            if p.requires_grad:
                p.grad = torch.sin(torch.arange(p.numel()).reshape(p.shape) + index + i).to(p)
        metrics = optimizer_step_with_metrics(opt2, monitor=True)
        scheduler2.step(); opt2.zero_grad(set_to_none=True)
        assert all(math.isfinite(v) for v in metrics.values())
        assert metrics['optimizer_time_ms/muon'] >= 0 and metrics['optimizer_time_ms/adamw'] >= 0
        assert scheduler.get_last_lr() == scheduler2.get_last_lr()
        for p, q in zip(model.parameters(), resumed.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
    for p, q in zip(model.parameters(), resumed.parameters()):
        for key, value in opt2.state[q].items():
            torch.testing.assert_close(opt.state[p][key].cpu(), value.cpu(), rtol=0, atol=0)
            if key != 'step':
                assert value.dtype == torch.float32 and value.device.type == 'cuda'


def _nccl_worker(rank, rendezvous):
    """两个累积 micro 的 DDP 更新与相同全局 batch 的本地更新比较。"""
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method='file://' + rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(19)
        model = DistributedMatrix().cuda(rank)
        reference = deepcopy(model)
        decoder = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        opt, scheduler = make(model, total=18)
        ref, ref_scheduler = make(reference, total=18)
        # 全局样本固定，两个 rank 各取两条，梯度平均口径与累积窗口一致。
        device = torch.device('cuda', rank)
        x = torch.tensor([[1., 2.], [2., -1.], [-1., 3.], [2., 2.]], device=device)
        target = torch.tensor([[1.], [0.], [-1.], [2.]], device=device)
        for _ in range(8):
            opt.zero_grad(set_to_none=True); ref.zero_grad(set_to_none=True)
            for micro in range(2):
                with decoder.no_sync() if micro == 0 else nullcontext():
                    i = rank * 2 + micro
                    ((decoder(x[i:i+1]) - target[i:i+1]).square().mean() / 2).backward()
            (reference(x) - target).square().mean().backward()
            opt.step(); scheduler.step(); ref.step(); ref_scheduler.step()
            for p, q in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(p, q, rtol=2e-4, atol=2e-5)
                copies = [torch.empty_like(p) for _ in range(2)]
                dist.all_gather(copies, p.detach())
                torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_two_cuda_ranks_nccl_accumulation(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip('two CUDA GPUs required for NCCL validation')
    mp.spawn(_nccl_worker, args=(str(tmp_path / 'nccl_init'),), nprocs=2, join=True)
