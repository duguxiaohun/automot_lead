#!/usr/bin/env python3
# 单卡运行示例（在 AutoMoT/ 下执行；多卡请用同目录 train.sh）：
#   python qwen3vl_local/action_expert_ablation/qwen_simple/train.py
#   GPU_IDS=0 python qwen3vl_local/action_expert_ablation/qwen_simple/train.py
"""Train the Qwen simple-prompt action expert ablation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3vl_local.action_expert_ablation.common import train_main


if __name__ == "__main__":
    train_main("qwen_simple")
