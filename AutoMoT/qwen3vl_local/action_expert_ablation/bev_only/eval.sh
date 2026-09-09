#!/usr/bin/env bash
# Run from AutoMoT/:
#   bash qwen3vl_local/action_expert_ablation/bev_only/eval.sh --checkpoint checkpoints/action_expert_ablation/bev_only/latest/best.pt
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
export PYTHONUNBUFFERED=1

exec python qwen3vl_local/action_expert_ablation/launch.py eval --variant bev_only "$@"
