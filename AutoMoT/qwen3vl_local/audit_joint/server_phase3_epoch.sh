#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
: "${PREPARED_ROOT:?Set PREPARED_ROOT to the successful Phase3 preparation directory}"
EPOCH_PYTHON="${PYTHON:-python}"
EPOCH_OUT="${OUTPUT_DIR:-checkpoints/phase3_epoch_$(date +%Y%m%d_%H%M%S_%N)}"
mkdir "$EPOCH_OUT"
EPOCH_EXTRA=(--epochs "${EPOCHS:-1}")
if [[ -n "${COMPLETED_EPOCH_ROOT:-}" ]]; then EPOCH_EXTRA+=(--completed-epoch "$COMPLETED_EPOCH_ROOT"); fi
if [[ -n "${REUSE_EPOCH_ROOT:-}" ]]; then EPOCH_EXTRA+=(--reuse-epoch "$REUSE_EPOCH_ROOT"); fi
set +e
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHON="$EPOCH_PYTHON" \
 bash qwen3vl_local/audit_joint/run_guarded.sh --space-path "$EPOCH_OUT" --min-free-gib 20 -- \
 "$EPOCH_PYTHON" -m qwen3vl_local.audit_joint.server_phase3_epoch run \
 --prepared "$PREPARED_ROOT" --output "$EPOCH_OUT" "${EPOCH_EXTRA[@]}" \
 --data-root "${DATA_ROOT:-lead_data}" --model-dir "${MODEL_DIR:-checkpoints/Qwen3.5-4B}" \
 > "$EPOCH_OUT/epoch.log" 2>&1
EPOCH_RC=$?
"$EPOCH_PYTHON" -m qwen3vl_local.audit_joint.server_phase3_epoch package \
 --output "$EPOCH_OUT" --exit-code "$EPOCH_RC" --epochs "${EPOCHS:-1}"
EPOCH_PACK_RC=$?
set -e
printf 'Return: %s/phase3_epoch_report.zip and %s/phase3_epoch_package.json\n' "$EPOCH_OUT" "$EPOCH_OUT"
if [[ "$EPOCH_RC" -ne 0 ]]; then exit "$EPOCH_RC"; fi
exit "$EPOCH_PACK_RC"
