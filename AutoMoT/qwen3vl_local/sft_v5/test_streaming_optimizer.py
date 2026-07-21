"""SFT v5 流式 optimizer 窗口的纯逻辑回归测试。

不加载 Qwen；除纯逻辑检查外，还会用两个 CPU gloo 进程验证真实 collective、
frame 平均梯度修正和缺失本地梯度补零。远端改 DDP 训练循环前应先跑本文件。
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_AUTOMOT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTOMOT_ROOT))

from qwen3vl_local.sft_v5.train import (  # noqa: E402
    OptimizerWindow,
    _gradient_rescale_factor,
    _save_adapter,
    _streaming_update_reason,
    _sync_trainable_grads_by_global_frames,
)


def test_streaming_update_reason() -> None:
    """frame 阈值优先，低吞吐尾段由 timestep 上限兜底。"""

    assert _streaming_update_reason(
        global_frames=511,
        timesteps=31,
        target_global_frames=512,
        max_timesteps=32,
    ) is None
    assert _streaming_update_reason(
        global_frames=512,
        timesteps=16,
        target_global_frames=512,
        max_timesteps=32,
    ) == "target_frames"
    assert _streaming_update_reason(
        global_frames=120,
        timesteps=32,
        target_global_frames=512,
        max_timesteps=32,
    ) == "max_timesteps"
    assert _streaming_update_reason(
        global_frames=8,
        timesteps=100,
        target_global_frames=512,
        max_timesteps=0,
    ) is None


def test_four_gpu_eight_route_window() -> None:
    """四卡每卡 8 个有效 frame 时，应在第 16 个 timestep 达到默认 512 frame。"""

    window = OptimizerWindow()
    for timestep in range(1, 17):
        window.local_frames += 8
        window.global_frames += 4 * 8
        window.timesteps += 1
        reason = _streaming_update_reason(
            global_frames=window.global_frames,
            timesteps=window.timesteps,
            target_global_frames=512,
            max_timesteps=32,
        )
        if timestep < 16:
            assert reason is None
        else:
            assert reason == "target_frames"
    assert (window.local_frames, window.global_frames, window.timesteps) == (128, 512, 16)


def test_gradient_rescale_factor() -> None:
    """固定 normalizer 的累计梯度应能还原成实际 global frame 平均。"""

    assert _gradient_rescale_factor(backward_normalizer=512, global_frames=512) == 1.0
    assert _gradient_rescale_factor(backward_normalizer=512, global_frames=640) == 0.8
    assert _gradient_rescale_factor(backward_normalizer=512, global_frames=128) == 4.0


def test_single_process_gradient_sync_and_window_reset() -> None:
    """单进程路径也必须应用 frame 修正，并能清空窗口计数。"""

    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    param.grad = torch.tensor([0.25, 0.5])
    _sync_trainable_grads_by_global_frames(
        [param],
        global_frames=128,
        backward_normalizer=512,
        bucket_cap_mb=0.001,
    )
    assert torch.allclose(param.grad, torch.tensor([1.0, 2.0]))

    window = OptimizerWindow(local_frames=32, global_frames=128, timesteps=4)
    window.reset()
    assert (window.local_frames, window.global_frames, window.timesteps) == (0, 0, 0)


def test_adapter_metadata_records_effective_window() -> None:
    """checkpoint 配置必须直接写出 GRAD_ACCUM 放大后的真实流式窗口。"""

    class _FakeModel:
        """只实现 adapter 保存接口，避免测试加载真实 Qwen。"""

        def save_pretrained(self, _path: str) -> None:
            """模拟 PEFT 保存；元数据写入由被测函数完成。"""

            return None

    class _FakeBundle:
        """提供 _save_adapter 需要的 LoRA 配置和 unwrap 接口。"""

        lora_vision_scope = "off"
        lora_target_modules = ["q_proj"]

        def unwrap(self) -> _FakeModel:
            """返回可接受 save_pretrained 调用的 fake model。"""

            return _FakeModel()

    args = SimpleNamespace(
        max_new_tokens_q1=1024,
        max_new_tokens_q2=1024,
        temperature=1.0,
        parallel_kl=True,
        parallel_kl_microbatch_size=2,
        update_mode="streaming_frames",
        target_global_frames_per_step=512,
        max_timesteps_per_step=32,
        grad_accum=2,
        learning_rate=1e-5,
        checkpoint_probe=True,
        checkpoint_probe_num_cases=24,
        checkpoint_probe_num_routes=1,
        checkpoint_probe_with_teacher=True,
        checkpoint_probe_sample_mode="random",
        checkpoint_probe_context_radius=8,
        checkpoint_probe_sequence_length=24,
        checkpoint_probe_artifact_level="review",
    )
    with tempfile.TemporaryDirectory(prefix="sft_v5_adapter_meta_") as tmp:
        output_dir = pathlib.Path(tmp) / "adapter"
        _save_adapter(_FakeBundle(), output_dir, args)
        meta = json.loads((output_dir / "sft_v5_adapter_config.json").read_text(encoding="utf-8"))
    assert meta["effective_target_global_frames_per_step"] == 1024
    assert meta["effective_max_timesteps_per_step"] == 64
    assert meta["parallel_kl_microbatch_size"] == 2
    assert meta["checkpoint_probe_enabled"] is True
    assert meta["checkpoint_probe_num_cases"] == 24
    assert meta["checkpoint_probe_num_routes"] == 1
    assert meta["checkpoint_probe_with_teacher"] is True
    assert meta["checkpoint_probe_sample_mode"] == "random"
    assert meta["checkpoint_probe_context_radius"] == 8
    assert meta["checkpoint_probe_sequence_length"] == 24
    assert meta["checkpoint_probe_artifact_level"] == "review"
    assert meta["gradient_sync"] == "bucketed_sum_allreduce_then_global_frame_average"
    assert meta["memory_curriculum"] == {
        "rs_error_patience": 4,
        "event_error_patience": 3,
        "rs_repair_interval": 2,
        "event_repair_interval": 1,
        "rs_memory_corrupt_prob": 0.06,
        "rs_memory_unknown_prob": 0.02,
        "event_memory_corrupt_prob": 0.10,
        "event_memory_unknown_prob": 0.05,
        "rs_initial_gt_prob": 0.5,
        "event_initial_gt_prob": 0.5,
        "q1_abnormal_direct_event_reset": False,
    }


def test_memory_tensorboard_tags_are_complete() -> None:
    """显存审计必须同时保留当前值和历史峰值，且区分 allocated/reserved。"""

    train_source = pathlib.Path(__file__).with_name("train.py").read_text(encoding="utf-8")
    expected_tags = (
        "memory/allocated_gb",
        "memory/reserved_gb",
        "memory/max_allocated_gb",
        "memory/max_reserved_gb",
        "progress/cuda_max_allocated_gb",
        "progress/cuda_max_reserved_gb",
    )
    for tag in expected_tags:
        assert tag in train_source, f"missing TensorBoard memory tag: {tag}"


def _distributed_gradient_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    """两进程 gloo worker：验证 SUM 后按 global frame 修正，而不是 rank 等权。"""

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        param = torch.nn.Parameter(torch.tensor([0.0]))
        missing_local_grad = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
        # 模拟两个 rank 在除以 backward_normalizer 后分别得到 1 和 3 的梯度。
        param.grad = torch.tensor([1.0 if rank == 0 else 3.0])
        # 第二个参数只在 rank1 的动态分支中产生梯度。rank0 必须补零并参加相同 bucket
        # collective，最终两个 rank 都拿到一致的全局结果。
        if rank == 1:
            missing_local_grad.grad = torch.tensor([2.0, 4.0])
        _sync_trainable_grads_by_global_frames(
            [param, missing_local_grad],
            global_frames=8,
            backward_normalizer=4,
            bucket_cap_mb=0.00001,
        )
        # SUM=4，再乘 4/8，两个 rank 都应得到严格一致的 2，而不是 rank 平均造成
        # 其它额外缩放。
        assert torch.allclose(param.grad, torch.tensor([2.0]))
        assert torch.allclose(missing_local_grad.grad, torch.tensor([1.0, 2.0]))
    finally:
        dist.destroy_process_group()


def test_distributed_gradient_sync() -> None:
    """用 CPU gloo 真实跑一次两 rank collective，提前发现同步公式回归。"""

    with tempfile.TemporaryDirectory(prefix="sft_v5_streaming_ddp_") as tmp:
        rendezvous_path = str(pathlib.Path(tmp) / "rdzv")
        mp.spawn(_distributed_gradient_worker, args=(2, rendezvous_path), nprocs=2, join=True)


def main() -> None:
    """无需 pytest 的直接运行入口。"""

    test_streaming_update_reason()
    test_four_gpu_eight_route_window()
    test_gradient_rescale_factor()
    test_single_process_gradient_sync_and_window_reset()
    test_adapter_metadata_records_effective_window()
    test_memory_tensorboard_tags_are_complete()
    test_distributed_gradient_sync()
    print("[ok] streaming optimizer policy")


if __name__ == "__main__":
    main()
