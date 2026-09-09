#!/usr/bin/env bash
# 运行示例（在 AutoMoT/ 下执行；需已有共享数据索引）：
# 默认四次 optimizer update，保持正式训练的每 step 样本预算。
#   bash qwen3vl_local/action_expert_ablation/qwen_simple/smoke.sh
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/qwen_simple/smoke.sh
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-4}"
VAL_STEPS="${VAL_STEPS:-2}" \
SAVE_STEPS="${SAVE_STEPS:-2}" \
LOGGING_STEPS="${LOGGING_STEPS:-1}" \
NUM_EPOCHS="${NUM_EPOCHS:-1}" \
NO_RUN_SUBDIR="${NO_RUN_SUBDIR:-}" \
bash qwen3vl_local/action_expert_ablation/qwen_simple/train.sh --max-train-steps "$MAX_TRAIN_STEPS" "$@"
