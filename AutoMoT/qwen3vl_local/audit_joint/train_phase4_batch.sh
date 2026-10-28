#!/usr/bin/env bash
# User-selected four-card B2 profile for both RGB modes. No batch search.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P4_APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$P4_APP_ROOT"
[[ $# == 0 ]] || { echo 'Configure this fixed B2 entry through environment variables, not extra arguments.' >&2; exit 2; }
: "${PREPARED:?Set PREPARED to the existing Phase4 bundle containing data4 and references4}"
: "${DATASET:?Set DATASET to an existing student/view; known runtime-only old versions are upgraded into the new output}"
: "${P4_PRODUCTION_CACHE:?Set P4_PRODUCTION_CACHE to a local route-cache path; first validated use may create it}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new Phase4 workflow directory}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export ACCUMULATION="${ACCUMULATION:-4}"
export UPDATE_SAMPLES="${UPDATE_SAMPLES:-32}"
[[ "$BATCH_SIZE/$ACCUMULATION/$UPDATE_SAMPLES" == 2/4/32 ]] || {
  echo 'Selected Phase4 profile requires BATCH_SIZE=2 ACCUMULATION=4 UPDATE_SAMPLES=32; clear conflicting old values.' >&2; exit 2;
}
[[ "${GRADIENT_CHECKPOINTING:-1}" == 1 ]] || {
  echo 'Selected Phase4 profile requires gradient checkpointing, enabled by the formal model loader.' >&2; exit 2;
}
P4_TOTAL_EPOCHS="${EPOCHS:-1}"
[[ "$P4_TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo 'EPOCHS must be a positive absolute target' >&2; exit 2; }
if [[ "$P4_TOTAL_EPOCHS" != 1 || -n "${RESUME_RGB2:-}" || -n "${RESUME_RGB4:-}" ]]; then
  [[ -z "${FIRST_EPOCH_BUDGET:-}" ]] || { echo 'Explicit FIRST_EPOCH_BUDGET is currently supported by the fresh one-epoch entry only.' >&2; exit 2; }
  P4_RECOVERY_ARGS=()
  [[ -z "${RESUME_RGB2:-}" ]] || P4_RECOVERY_ARGS+=(--resume-rgb2 "$RESUME_RGB2")
  [[ -z "${RESUME_RGB4:-}" ]] || P4_RECOVERY_ARGS+=(--resume-rgb4 "$RESUME_RGB4")
  exec "${PYTHON:-python}" -m qwen3vl_local.audit_joint.phase4_training \
    --prepared "$PREPARED" --view "$DATASET" --output "$OUTPUT_DIR" --epochs "$P4_TOTAL_EPOCHS" \
    --data-root "${DATA_ROOT:-lead_data}" --model-dir "${MODEL_DIR:-checkpoints/Qwen3.5-4B}" \
    "${P4_RECOVERY_ARGS[@]}"
fi
echo '[phase4-batch] RGB2/RGB4 batch/card=2 world=4 forward_samples=8 accumulation=4 update_samples=32 checkpointing=1'
P4_BUDGET_ARGS=()
[[ -z "${FIRST_EPOCH_BUDGET:-}" ]] || P4_BUDGET_ARGS+=(--budget "$FIRST_EPOCH_BUDGET")
exec "${PYTHON:-python}" -m qwen3vl_local.audit_joint.phase4_first_epoch \
  --prepared "$PREPARED" --view "$DATASET" --output "$OUTPUT_DIR" \
  --data-root "${DATA_ROOT:-lead_data}" --model-dir "${MODEL_DIR:-checkpoints/Qwen3.5-4B}" \
  --batch-size 2 --batch-size-rgb2 2 --batch-size-rgb4 2 "${P4_BUDGET_ARGS[@]}"
