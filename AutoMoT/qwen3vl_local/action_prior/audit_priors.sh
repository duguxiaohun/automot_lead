#!/usr/bin/env bash
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 只运行一个模型进程；ensure_gpu 自动选卡，GPU_IDS 可显式指定。
exec python "$HERE/audit_priors.py" --data-root "${DATA_ROOT:-lead_data}" \
 --data-dir "${DATA_DIR:-checkpoints/action_prior_data}" \
 --model-dir "${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}" \
 --lead-bev-ckpt "${LEAD_BEV_CKPT:-checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth}" \
 --phase1-adapter "${PHASE1_ADAPTER:-}" --phase2-adapter "${PHASE2_ADAPTER:-}" "$@"
