#!/usr/bin/env bash
# 优化细节统一默认；每个epoch训练/验证后自动更新run目录的 training_audit.zip，无需审计开关。
# 优化器/LR 共用 action_prior Python 配置：默认 muon_adamw + cosine_restarts。
# 可追加 --optimizer adamw --lr-scheduler cosine 作基线；环境变量 OPTIMIZER/LR_SCHEDULER 同样生效，CLI 优先。
# 续训恢复原配置并严格校验；完整参数和三组对照见 action_prior/OPTIMIZATION.md。
# 运行示例（在 AutoMoT/ 下执行）：
# 构建/复用索引 → 训练 → 用本次 best.pt 做 test；默认自动选四张空闲 GPU。
#   bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh
# 事件均衡（full pipeline 自动准备；train.sh 需显式 EVENT_BALANCE_INDEX 或 --event-balance-index）：
#   bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced
# 兼容 EVENT_BALANCED=1；--no-event-balanced 关闭。课程/比例直接引用 action_prior，详见 ../run.md。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
source qwen3vl_local/action_expert_ablation/pipeline_common.sh
action_ablation_run_full_pipeline qwen_simple checkpoints/action_expert_ablation/qwen_simple "$@"
