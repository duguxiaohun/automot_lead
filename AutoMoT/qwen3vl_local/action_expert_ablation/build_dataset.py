#!/usr/bin/env python3
# 构建共享索引（在 AutoMoT/ 下执行；只读 lead_data，不加载模型）：
#   python qwen3vl_local/action_expert_ablation/build_dataset.py --data-root lead_data --output-dir checkpoints/action_prior_data
"""Build the shared action expert ablation dataset index."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3vl_local.action_prior.build_dataset import main


if __name__ == "__main__":
    main()
