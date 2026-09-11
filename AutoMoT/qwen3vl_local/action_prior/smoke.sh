#!/usr/bin/env bash
# 已准备好真实模型与索引后的四步 smoke；本入口不自动构建数据。
#   DATASET_PRIORS=1 bash qwen3vl_local/action_prior/smoke.sh
#   GPU_IDS=0 DATASET_PRIORS=1 bash qwen3vl_local/action_prior/smoke.sh
# 首次使用请走自动全流程：bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced
# 最终 best 验证仍遍历完整 val。日常说明见 run.md。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 用真实模型/真实索引跑四个更新；显式区别于 61 epoch 正式训练。
DDP_GPU_COUNT="${DDP_GPU_COUNT:-1}" GRAD_ACCUM=1 NUM_WORKERS=0 VAL_STEPS=2 \
 bash "$HERE/train.sh" --max-train-steps 4 --val-max-samples 4 --logging-steps 1 "$@"
