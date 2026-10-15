#!/usr/bin/env bash
# Fixed first-epoch automatic-reference path; the original reviewed entry stays available.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
: "${DATASET:?Set DATASET to the short view}"
: "${TEACHER_REFERENCES:?Set TEACHER_REFERENCES to the RGB4 automatic holdout pack}"
: "${FIRST_EPOCH_BUDGET:?Set FIRST_EPOCH_BUDGET to the completed budget request}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new experiment output base}"
[[ $# == 0 ]] || { echo 'This fixed baseline takes configuration through environment variables, not extra training flags.' >&2; exit 2; }
P4_AUTO_BATCH="${BATCH_SIZE:-1}"
P4_AUTO_UPDATE="${UPDATE_SAMPLES:-32}"
[[ "$P4_AUTO_BATCH" =~ ^[1-9][0-9]*$ && "$P4_AUTO_UPDATE" =~ ^[1-9][0-9]*$ ]] || {
  echo 'BATCH_SIZE and UPDATE_SAMPLES must be positive integers' >&2; exit 2;
}
(( P4_AUTO_UPDATE % (4 * P4_AUTO_BATCH) == 0 )) || {
  echo 'UPDATE_SAMPLES must be divisible by 4 * BATCH_SIZE' >&2; exit 2;
}
P4_AUTO_ACCUMULATION="$((P4_AUTO_UPDATE / (4 * P4_AUTO_BATCH)))"
[[ "${ACCUMULATION:-$P4_AUTO_ACCUMULATION}" == "$P4_AUTO_ACCUMULATION" ]] || {
  echo 'ACCUMULATION conflicts with UPDATE_SAMPLES / (4 * BATCH_SIZE)' >&2; exit 2;
}
echo "[phase4-automatic] batch/card=$P4_AUTO_BATCH accumulation=$P4_AUTO_ACCUMULATION update_samples=$P4_AUTO_UPDATE"
P4_AUTO_PYTHON="${PYTHON:-python}"
export RUN_TAG="${RUN_TAG:-automatic_$(date +%Y%m%d_%H%M%S_%N)}"
mkdir -p "$OUTPUT_DIR"
P4_AUTO_REPORT="$OUTPUT_DIR/admission_${RGB_MODE:-2}_$(date +%Y%m%d_%H%M%S_%N).json"
"$P4_AUTO_PYTHON" -m qwen3vl_local.audit_joint.rule_baseline \
  --dataset "$DATASET" --teacher-references "$TEACHER_REFERENCES" \
  --data-root "${DATA_ROOT:-lead_data}" --batch-size "${BATCH_SIZE:-1}" \
  --budget "$FIRST_EPOCH_BUDGET" --space-path "$OUTPUT_DIR" --output "$P4_AUTO_REPORT" \
  --coverage-policy "${AUTOMATIC_COVERAGE_POLICY:-strict}"
bash qwen3vl_local/sft_new_loop_phase4/rgb_short/run.sh \
  --selection-mode rule_consistency --teacher-references "$TEACHER_REFERENCES" \
  --epochs 1 --batch-size "$P4_AUTO_BATCH" --accumulation "$P4_AUTO_ACCUMULATION" \
  --sampling-policy contiguous_event_equal --epoch-samples 10240 --seed 20260929
P4_AUTO_RUN="$OUTPUT_DIR/run_${RUN_TAG}_rgb${RGB_MODE:-2}"
# Use this invocation's checkpoint, never a mutable latest link from another run.
bash qwen3vl_local/audit_joint/run_guarded.sh --space-path "$OUTPUT_DIR" --min-free-gib 20 -- \
  "$P4_AUTO_PYTHON" -m qwen3vl_local.sft_new_loop_phase4.rgb_short.evaluate \
  --dataset "$DATASET" --data-root "${DATA_ROOT:-lead_data}" \
  --model-dir "${MODEL_DIR:-checkpoints/Qwen3.5-4B}" \
  --adapter "$P4_AUTO_RUN/epoch_000" --rgb-mode "${RGB_MODE:-2}" \
  --split test --output-dir "$P4_AUTO_RUN/test_fixed"
