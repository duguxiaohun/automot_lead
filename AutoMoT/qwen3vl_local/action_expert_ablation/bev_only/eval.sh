#!/usr/bin/env bash
# 运行示例（在 AutoMoT/ 下执行；默认自动选一张空闲 GPU）：
#   bash qwen3vl_local/action_expert_ablation/bev_only/eval.sh --checkpoint checkpoints/action_expert_ablation/bev_only/latest/best.pt
#   GPU_IDS=0 bash qwen3vl_local/action_expert_ablation/bev_only/eval.sh --checkpoint checkpoints/action_expert_ablation/bev_only/latest/best.pt
# 可追加 --dump-cases 保存逐例预测，--split val 切换到验证集。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
export PYTHONUNBUFFERED=1

exec python qwen3vl_local/action_expert_ablation/launch.py eval --variant bev_only "$@"
