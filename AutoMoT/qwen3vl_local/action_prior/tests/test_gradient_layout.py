"""CPU 回归：BEV 拼接反向布局、广播等价性与 DDP 累积梯度。"""

from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import sys
import warnings

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.leadmot.config import LeadMoTPlanningDecoderConfig
from qwen3vl_local.leadmot.projectors import LeadBEVProjector


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
def test_bev_cat_backward_matches_original(batch_size, compute_dtype, tmp_path):
    """真实 projector 经 142 token 拼接后，DDP 累积/清零均保持布局和值正确。"""
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("需要 CPU Gloo 支持")
    # 只保留本问题涉及的 projector；不加载 Qwen/BEV 权重或 CARLA。
    config = LeadMoTPlanningDecoderConfig(bev_channels=4)
    model = LeadBEVProjector(config)
    reference = deepcopy(model)
    assert model.pos_embed.shape == (1, 120, 1024)
    reference.load_state_dict(model.state_dict(), strict=True)
    dist.init_process_group(
        "gloo", init_method=(tmp_path / "rendezvous").as_uri(),
        rank=0, world_size=1, timeout=timedelta(seconds=30),
    )
    try:
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        generator = torch.Generator().manual_seed(42)
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            for micro in range(2):
                bev = torch.randn(batch_size, 4, 10, 12, generator=generator)
                extra = torch.randn(batch_size, 22, 1024, generator=generator)
                # 先 no_sync 累积，再同步；同时检查 set_to_none 后的新一轮梯度。
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with ddp.no_sync() if micro == 0 else nullcontext():
                        with torch.autocast("cpu", dtype=compute_dtype,
                                            enabled=compute_dtype != torch.float32):
                            actual = torch.cat([ddp(bev), extra], dim=1)
                            original = reference.proj(bev.flatten(2).transpose(1, 2))
                            expected = torch.cat([original + reference.pos_embed, extra], dim=1)
                            actual.square().mean().backward()
                        # 参考分支使用修改前的三维参数相加公式。
                        expected.square().mean().backward()
                assert not any("Grad strides do not match" in str(w.message) for w in caught)
                assert model.pos_embed.grad.stride() == model.pos_embed.stride()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for (name, param), (ref_name, ref_param) in zip(
                    model.named_parameters(), reference.named_parameters()
                ):
                    assert name == ref_name
                    torch.testing.assert_close(param.grad, ref_param.grad, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
