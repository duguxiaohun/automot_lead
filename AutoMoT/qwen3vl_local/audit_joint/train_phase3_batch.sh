#!/usr/bin/env bash
# Current default: restored independent 4rgb choice base (B1/acc1, 7680 updates).
# The historical usage notes below apply only with LEGACY_PHASE3_SEQUENCE=1.
# Four-H20 Phase3 profile selected by fixed-sample end-to-end measurements.
# Run from AutoMoT/. The generic train.sh remains available for other profiles.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true

# Default entry is the restored independent base; the code below is legacy-only.
if [[ "${LEGACY_PHASE3_SEQUENCE:-0}" != 1 ]]; then
  exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/phase3_default.sh" "$@"
fi
: "${INDEX:?Set INDEX to the existing Phase3 frame_index.jsonl}"
[[ $# -le 1 ]] || { echo 'Use train_phase3_batch.sh [ddp|sampling]' >&2; exit 2; }
P3_BATCH_MODE="${1:-ddp}"
[[ "$P3_BATCH_MODE" == ddp || "$P3_BATCH_MODE" == sampling ]] || { echo 'Use ddp or sampling' >&2; exit 2; }
if [[ -z "${GPU_IDS:-}" ]]; then
  GPU_IDS="$(python -c 'from qwen3vl_local.sft_new_loop_phase4.devices import select_gpus; print(select_gpus(4))')"
fi
export GPU_IDS
IFS=, read -r -a P3_BATCH_GPUS <<< "$GPU_IDS"
[[ ${#P3_BATCH_GPUS[@]} == 4 ]] || { echo 'This profile requires four idle visible GPUs, or four explicit GPU_IDS' >&2; exit 2; }
declare -A P3_BATCH_SEEN=()
for P3_BATCH_GPU in "${P3_BATCH_GPUS[@]}"; do
  [[ "$P3_BATCH_GPU" =~ ^[0-9]+$ && ! -v P3_BATCH_SEEN[$P3_BATCH_GPU] ]] || { echo 'This profile requires four distinct GPU IDs' >&2; exit 2; }
  P3_BATCH_SEEN[$P3_BATCH_GPU]=1
done
export NPROC_PER_NODE=4 DDP_GPU_COUNT=4
export SAVE_TRAINING_STATE="${SAVE_TRAINING_STATE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-2}" GRAD_ACCUM="${GRAD_ACCUM:-2}" GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
[[ "$BATCH_SIZE/$GRAD_ACCUM/$GRADIENT_CHECKPOINTING" == 2/2/1 ]] || {
  echo 'Measured profile requires BATCH_SIZE=2 GRAD_ACCUM=2 GRADIENT_CHECKPOINTING=1; use train.sh for other configurations.' >&2; exit 2;
}
export HISTORY_RGB_MODE="${HISTORY_RGB_MODE:-4rgb}"
case "$HISTORY_RGB_MODE" in
  2rgb_short) P3_BATCH_STRIDE=2 ;;
  4rgb) P3_BATCH_STRIDE=1 ;;
  *) echo 'Use HISTORY_RGB_MODE=2rgb_short or 4rgb' >&2; exit 2 ;;
esac
export ANCHOR_STRIDE="${ANCHOR_STRIDE:-$P3_BATCH_STRIDE}"
[[ "$ANCHOR_STRIDE" == "$P3_BATCH_STRIDE" ]] || { echo 'Anchor stride must match the current temporal contract' >&2; exit 2; }
export PROMPT_VARIANT="${PROMPT_VARIANT:-v23_rgb_stage_candidate_20261006}"
export ACTION_OUTPUT_MODE="${ACTION_OUTPUT_MODE:-choice}"
[[ "$ACTION_OUTPUT_MODE" == choice ]] || { echo 'This profile uses ACTION_OUTPUT_MODE=choice' >&2; exit 2; }
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S_%N)}"
export OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/phase3_batch_runs}/run_${RUN_TAG}_${HISTORY_RGB_MODE}"
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || { echo "Refusing existing run: $OUTPUT_DIR" >&2; exit 2; }
# Preserve sample-based schedule from the old four-card B1/accum1 defaults:
# warmup: 2000 * 4 = 500 * 16; evaluation: 2000 * 4 = 1000 * 8.
export WARMUP_STEPS="${WARMUP_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-1000}" GENERATION_EVAL_STEPS="${GENERATION_EVAL_STEPS:-1000}"
export SAVE_STEPS="${SAVE_STEPS:-10000}"
echo "[phase3-batch] batch/card=2 accumulation=2 checkpointing=1 GPUs=${GPU_IDS:-auto4} forward_samples=8 update_samples=16 rgb=$HISTORY_RGB_MODE stride=$ANCHOR_STRIDE warmup_updates=$WARMUP_STEPS"
exec bash qwen3vl_local/sft_new_loop_phase3/train.sh "$P3_BATCH_MODE"
