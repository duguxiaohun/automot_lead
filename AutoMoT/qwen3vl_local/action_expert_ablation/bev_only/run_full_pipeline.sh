#!/usr/bin/env bash
# 运行示例（在 AutoMoT/ 下执行）：
# 构建/复用索引 → 训练 → 用本次 best.pt 做 test；默认自动选四张空闲 GPU。
#   bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh
# 事件均衡（full pipeline 自动准备；train.sh 需显式 EVENT_BALANCE_INDEX 或 --event-balance-index）：
#   bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced
# 兼容 EVENT_BALANCED=1；--no-event-balanced 关闭。课程/比例直接引用 action_prior，详见 ../run.md。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
source qwen3vl_local/action_expert_ablation/pipeline_common.sh
action_ablation_run_full_pipeline bev_only checkpoints/action_expert_ablation/bev_only "$@"
