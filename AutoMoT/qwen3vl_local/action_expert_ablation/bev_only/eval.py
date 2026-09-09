#!/usr/bin/env python3
"""Evaluate the BEV-only action expert ablation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3vl_local.action_expert_ablation.common import eval_main


if __name__ == "__main__":
    eval_main("bev_only")
