#!/usr/bin/env python3
"""三条 action 入口共用的小矩阵 CUDA 检查；自动选卡，不加载模型或训练数据。"""
import argparse
from pathlib import Path
import resource
import sys


def main():
    """导入 torch 前选卡；显式要求的 GPU/BF16/NCCL 条件不足直接失败。"""
    resource.setrlimit(resource.RLIMIT_CORE, (0, resource.getrlimit(resource.RLIMIT_CORE)[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from qwen3vl_local.action_prior.launch import ensure_gpu
    count = ensure_gpu(args.gpus)
    if count != args.gpus:
        parser.error('GPU_IDS count must equal --gpus')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != args.gpus:
        raise RuntimeError('requested CUDA devices unavailable')
    for index in range(args.gpus):
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(f'GPU {index} does not support BF16')
            print(f'GPU {index}: {torch.cuda.get_device_name(index)}', flush=True)
    if args.gpus == 2 and not torch.distributed.is_nccl_available():
        raise RuntimeError('NCCL unavailable')
    print(f'torch={torch.__version__}, CUDA={torch.version.cuda}; GPUs={args.gpus}', flush=True)
    import pytest
    test_path = Path(__file__).parent / 'tests/test_optimization_cuda.py'
    filters = ['-k', 'not two_cuda_ranks'] if args.gpus == 1 else []
    raise SystemExit(pytest.main(['-q', '-rs', str(test_path), *filters]))


if __name__ == '__main__':
    main()
