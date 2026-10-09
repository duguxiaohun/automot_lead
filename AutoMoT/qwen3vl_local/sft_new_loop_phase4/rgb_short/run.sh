#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P4_PYTHON="${PYTHON:-python}"
: "${DATASET:?Set DATASET to the prepared view directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the experiment output base}"
P4_MODE="${RGB_MODE:-2}"
[[ "$P4_MODE" == 2 || "$P4_MODE" == 4 ]] || { echo 'RGB_MODE must be 2 or 4' >&2; exit 2; }
P4_GPUS="$(env -u CUDA_VISIBLE_DEVICES "$P4_PYTHON" -c 'from qwen3vl_local.sft_new_loop_phase4.devices import select_gpus; print(select_gpus(4))')"
IFS=',' read -ra P4_GPU_ARRAY <<< "$P4_GPUS"
[[ "${#P4_GPU_ARRAY[@]}" == 4 ]] || { echo 'This experiment requires four GPUs; set GPU_IDS=0,1,2,3 to pin.' >&2; exit 2; }
P4_RUN="$OUTPUT_DIR/run_${RUN_TAG:-$(date +%Y%m%d_%H%M%S_%N)}_rgb${P4_MODE}"
for P4_ARG in "$@"; do
  P4_OPTION="${P4_ARG%%=*}"
  for P4_FIXED in --dataset --data-root --model-dir --output-dir --rgb-mode; do
    [[ "$P4_OPTION" != --?* || "$P4_FIXED" != "$P4_OPTION"* ]] || { echo 'Set paths and RGB mode through environment variables.' >&2; exit 2; }
  done
done
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$P4_GPUS" PYTHON="$P4_PYTHON" \
bash qwen3vl_local/audit_joint/run_guarded.sh --space-path "$OUTPUT_DIR" --min-free-gib 20 -- \
  "$P4_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m qwen3vl_local.sft_new_loop_phase4.rgb_short.train \
  --dataset "$DATASET" --data-root "${DATA_ROOT:-lead_data}" \
  --model-dir "${MODEL_DIR:-checkpoints/Qwen3.5-4B}" --output-dir "$P4_RUN" \
  --rgb-mode "$P4_MODE" "$@"
GPU_IDS="${P4_GPU_ARRAY[0]}" PYTHON="$P4_PYTHON" \
bash qwen3vl_local/audit_joint/run_guarded.sh --space-path "$OUTPUT_DIR" --min-free-gib 20 -- \
  "$P4_PYTHON" -m qwen3vl_local.sft_new_loop_phase4.rgb_short.reload_check \
  --run "$P4_RUN" --dataset "$DATASET" --data-root "${DATA_ROOT:-lead_data}" \
  --model-dir "${MODEL_DIR:-checkpoints/Qwen3.5-4B}"
ln -sfn "$(basename "$P4_RUN")" "$OUTPUT_DIR/latest"
