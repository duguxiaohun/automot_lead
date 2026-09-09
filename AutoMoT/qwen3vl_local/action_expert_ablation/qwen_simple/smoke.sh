#!/usr/bin/env bash
# Four optimizer updates with the same per-step case budget as the full run.
set -euo pipefail
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-4}"
VAL_STEPS="${VAL_STEPS:-2}" \
SAVE_STEPS="${SAVE_STEPS:-2}" \
LOGGING_STEPS="${LOGGING_STEPS:-1}" \
NUM_EPOCHS="${NUM_EPOCHS:-1}" \
NO_RUN_SUBDIR="${NO_RUN_SUBDIR:-}" \
bash qwen3vl_local/action_expert_ablation/qwen_simple/train.sh --max-train-steps "$MAX_TRAIN_STEPS" "$@"
