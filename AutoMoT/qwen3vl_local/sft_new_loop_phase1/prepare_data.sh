#!/usr/bin/env bash
# Independent Phase1 CPU preparation. No GPU training, no other phase invocation.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
read -r -a MODES <<< "${HISTORY_RGB_MODES:-4rgb}"
exec python -B -m qwen3vl_local.sft_new_loop_phase1.data_stage prepare \
  --phase 1 --audit-dir "${DATA_AUDIT_DIR:-checkpoints/sft_new_loop_phase1_data_cpu/coverage_v2}" \
  --run-build "${RUN_BUILD:-auto}" --timeout "${DATA_STAGE_TIMEOUT_SECONDS:-7200}" \
  --world-size "${DDP_GPU_COUNT:-4}" --modes "${MODES[@]}" \
  --focus-balance-count "${TRAIN_FOCUS_BALANCE_COUNT:-768}" \
  --num-epochs "${TRAIN_NUM_EPOCHS:-3}" \
  --epoch-step-limit "${EPOCH_STEP_LIMIT:-4096}" --total-step-limit "${TOTAL_STEP_LIMIT:-12288}" \
  --epoch-hours-limit "${EPOCH_HOURS_LIMIT:-10}" \
  --action-index "${ACTION_INDEX:-auto}" -- \
  --output-dir "${DATA_OUTPUT_DIR:-checkpoints/sft_new_loop_phase1_data}" \
  --data-root "${DATA_ROOT:-lead_data}" \
  --collection-dir "${COLLECTION_DIR:-keyframe_filter/collection_output}" "$@"
