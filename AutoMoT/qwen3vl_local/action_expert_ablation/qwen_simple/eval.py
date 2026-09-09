#!/usr/bin/env python3
# 单卡运行示例（在 AutoMoT/ 下执行）：
#   python qwen3vl_local/action_expert_ablation/qwen_simple/eval.py --checkpoint checkpoints/action_expert_ablation/qwen_simple/latest/best.pt
#   GPU_IDS=0 python qwen3vl_local/action_expert_ablation/qwen_simple/eval.py --checkpoint checkpoints/action_expert_ablation/qwen_simple/latest/best.pt
"""Evaluate the Qwen simple-prompt action expert ablation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3vl_local.action_expert_ablation.common import eval_main


if __name__ == "__main__":
    eval_main("qwen_simple")
