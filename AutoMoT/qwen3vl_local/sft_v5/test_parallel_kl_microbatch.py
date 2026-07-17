"""SFT v5 parallel-KL 微批与 OOM 二分的轻量回归测试。

不加载 Qwen、不需要 CUDA。测试通过替换 scoring runner 模拟长上下文 OOM，验证：

1. 8 路 rollout 可以按 4+4 KL 微批逐批 backward；
2. 4 路 forward OOM 时会安全拆成 2+2；
3. 拆分前后累计梯度仍等价于 8 个 frame 的 loss sum / normalizer。
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Sequence

import torch

_AUTOMOT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTOMOT_ROOT))

import qwen3vl_local.sft_v5.train as train_module  # noqa: E402


def _run_fake_microbatch(*, oom_above: int) -> tuple[train_module.ParallelKLMicrobatchResult, float]:
    """运行一个 8-frame 假 chunk，并返回拆分信息与标量参数梯度。"""

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    original_runner = train_module._run_chunk_parallel_kl

    def fake_runner(
        _bundle: Any,
        chunk: Sequence[int],
        **_kwargs: Any,
    ) -> tuple[torch.Tensor, list[int]]:
        """用 chunk 大小触发模拟 OOM，并返回可反传的线性标量 loss。"""

        if len(chunk) > int(oom_above):
            raise torch.cuda.OutOfMemoryError("simulated parallel KL forward OOM")
        # 每个 frame 对 loss 的贡献都是 parameter；8 frame / normalizer 8 后梯度应为 1。
        return parameter * float(len(chunk)), list(chunk)

    train_module._run_chunk_parallel_kl = fake_runner  # type: ignore[assignment]
    try:
        result = train_module._run_parallel_kl_microbatches(
            bundle=object(),
            chunk=list(range(8)),  # type: ignore[arg-type]
            q1_rollouts=[object()] * 8,  # type: ignore[list-item]
            q2_rollouts=[object()] * 8,  # type: ignore[list-item]
            temperature=1.0,
            backward_normalizer=8,
            microbatch_size=4,
        )
    finally:
        train_module._run_chunk_parallel_kl = original_runner
    assert parameter.grad is not None
    return result, float(parameter.grad.item())


def test_fixed_four_way_microbatch() -> None:
    """无 OOM 时，8 frame 应按两个 4 路 KL 微批完成。"""

    result, grad = _run_fake_microbatch(oom_above=4)
    assert result.microbatch_sizes == [4, 4]
    assert result.oom_splits == 0
    assert result.frame_results == list(range(8))
    assert abs(result.detached_loss_sum - 8.0) < 1e-6
    assert abs(grad - 1.0) < 1e-6


def test_adaptive_oom_split_preserves_gradient() -> None:
    """4 路 forward OOM 后拆成 2+2，梯度和样本覆盖不能改变。"""

    result, grad = _run_fake_microbatch(oom_above=2)
    assert result.microbatch_sizes == [2, 2, 2, 2]
    assert result.oom_splits == 2
    assert result.frame_results == list(range(8))
    assert abs(result.detached_loss_sum - 8.0) < 1e-6
    assert abs(grad - 1.0) < 1e-6


def main() -> None:
    """无需 pytest 的直接运行入口。"""

    test_fixed_four_way_microbatch()
    test_adaptive_oom_split_preserves_gradient()
    print("[ok] parallel KL microbatch policy")


if __name__ == "__main__":
    main()
