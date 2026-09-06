#!/usr/bin/env bash
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
# 参数用数组传递，路径包含空格时也不会被拆开。
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
args=(--data-root "${DATA_ROOT:-lead_data}" --data-dir "${DATA_DIR:-checkpoints/action_prior_data}"
 --checkpoint-root "${CHECKPOINT_ROOT:-checkpoints}" --selection-policy "${SELECTION_POLICY:-available}"
 --model-dir "${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
 --lead-bev-ckpt "${LEAD_BEV_CKPT:-checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth}"
 --num-epochs "${NUM_EPOCHS:-61}" --learning-rate "${LR:-0.0002}"
 --grad-accum-steps "${GRAD_ACCUM:-16}" --val-steps "${VAL_STEPS:-250}"
 --save-steps "${SAVE_STEPS:-1000}" --num-workers "${NUM_WORKERS:-8}")
[[ -z "${PHASE1_ADAPTER:-}" ]] || args+=(--phase1-adapter "$PHASE1_ADAPTER")
[[ -z "${PHASE2_ADAPTER:-}" ]] || args+=(--phase2-adapter "$PHASE2_ADAPTER")
exec python "$HERE/launch.py" "${ACTION_MODE:-train}" "${args[@]}" "$@"
