#!/usr/bin/env bash
# Compare against current Phase4 B1 * 4 GPUs * accumulation8 = 32 samples/update.
# DATASET must be the current student view, not the native data4 directory.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
: "${DATASET:?Set DATASET to the current Phase4 student/view directory}"
if [[ -z "${GPU_IDS:-}" ]]; then
  GPU_IDS="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, '{gsub(/ /,"",$0); print $2, $3, $1}' | sort -n -k1,1 -k2,2 \
    | head -n 4 | awk '{print $3}' | paste -sd, -)"
fi
IFS=, read -r -a P4_BATCH_GPUS <<< "$GPU_IDS"
[[ ${#P4_BATCH_GPUS[@]} == 4 ]] || { echo 'This comparison profile requires four GPUs' >&2; exit 2; }
exec "${PYTHON:-python}" -m qwen3vl_local.audit_joint.batch_benchmark \
  --phase 4 --dataset "$DATASET" --data-root "${DATA_ROOT:-lead_data}" \
  --model-dir "${MODEL_DIR:-checkpoints/Qwen3.5-4B}" \
  --gradient-checkpointing --gpus "$GPU_IDS" \
  --modes 2 4 --batches 1 2 4 --samples-per-event 32 --update-samples 32 \
  --seed 20260929 --repeats 2 --timing end_to_end \
  --output "${OUTPUT_DIR:-checkpoints/p4_batch_e2e_$(date +%Y%m%d_%H%M%S_%N)}" "$@"
