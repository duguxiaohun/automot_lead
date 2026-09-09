#!/usr/bin/env bash
# 运行示例（在 AutoMoT/ 下执行）：
# 构建/复用索引 → 训练 → 用本次 best.pt 做 test；默认自动选四张空闲 GPU。
#   bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
source qwen3vl_local/action_expert_ablation/pipeline_common.sh
action_ablation_run_full_pipeline qwen_simple checkpoints/action_expert_ablation/qwen_simple "$@"
