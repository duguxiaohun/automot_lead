#!/usr/bin/env bash
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 用真实模型/真实索引跑四个更新；显式区别于 61 epoch 正式训练。
DDP_GPU_COUNT="${DDP_GPU_COUNT:-1}" GRAD_ACCUM=1 NUM_WORKERS=0 VAL_STEPS=2 \
 bash "$HERE/train.sh" --max-train-steps 4 --val-max-samples 4 --logging-steps 1 "$@"
