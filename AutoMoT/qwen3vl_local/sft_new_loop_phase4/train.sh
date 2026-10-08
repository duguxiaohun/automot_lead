#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P4_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${P4_DIR}/../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
MODE="${1:-ddp}"
if [[ $# -gt 0 ]]; then shift; fi
PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-checkpoints/phase4_v40_full/data${RGB_MODE:-4}}"
MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3.5-4B}"
DATA_ROOT="${DATA_ROOT:-lead_data}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft_new_loop_phase4_runs/$(date +%Y%m%d_%H%M%S_%N)}"
[[ "$MODE" == single || "$MODE" == ddp || "$MODE" == check || "$MODE" == preflight || "$MODE" == host-preflight ]] || { echo 'mode must be single/ddp/check/preflight/host-preflight' >&2; exit 2; }
for P4_ARG in "$@"; do
  P4_OPTION="${P4_ARG%%=*}"
  for P4_PATH_OPTION in --dataset --model-dir --data-root --output-dir; do
    if [[ "$P4_OPTION" == --?* && "$P4_PATH_OPTION" == "$P4_OPTION"* ]]; then
      echo 'Set DATASET/MODEL_DIR/DATA_ROOT/OUTPUT_DIR as environment variables so preflight and training use the same paths.' >&2; exit 2
    fi
  done
done
P4_CONFIG=(--dataset "$DATASET" --data-root "$DATA_ROOT")
if [[ "$MODE" == single || "$MODE" == ddp || "$MODE" == host-preflight ]]; then
  P4_CONFIG+=(--model-dir "$MODEL_DIR")
fi
"$PYTHON" -m qwen3vl_local.sft_new_loop_phase4.launch "${P4_CONFIG[@]}"
if [[ "$MODE" == check ]]; then
  exec "$PYTHON" -m qwen3vl_local.sft_new_loop_phase4.train --dataset "$DATASET" --output-dir "$OUTPUT_DIR" --sampling-only "$@"
fi
if [[ "$MODE" == preflight ]]; then
  exec "$PYTHON" -m qwen3vl_local.sft_new_loop_phase4.preflight --dataset "$DATASET" --data-root "$DATA_ROOT" --require-trainable "$@"
fi
# Forward the exact effective sampling flags, preserving order and both CLI forms.
# Other trainer flags (LoRA, optimizer, resume) are not preflight arguments.
P4_SAMPLING_ARGS=()
P4_PENDING=0
for P4_ARG in "$@"; do
  if [[ "$P4_PENDING" == 1 ]]; then
    P4_SAMPLING_ARGS+=("$P4_ARG"); P4_PENDING=0; continue
  fi
  case "$P4_ARG" in
    --paired-with|--accumulation|--cap|--seed|--epochs|--epoch-samples|--max-question-repeat|--sampling-policy)
      P4_SAMPLING_ARGS+=("$P4_ARG"); P4_PENDING=1 ;;
    --paired-with=*|--accumulation=*|--cap=*|--seed=*|--epochs=*|--epoch-samples=*|--max-question-repeat=*|--sampling-policy=*|--require-complete-coverage)
      P4_SAMPLING_ARGS+=("$P4_ARG") ;;
  esac
done
P4_WORLD=1
if [[ "$MODE" != host-preflight ]]; then
  GPU_IDS="$("$PYTHON" -c 'from qwen3vl_local.sft_new_loop_phase4.devices import select_gpus; print(select_gpus(4))')"
  if [[ "$MODE" == single ]]; then GPU_IDS="${GPU_IDS%%,*}"; fi
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  IFS=',' read -r -a P4_GPUS <<< "$GPU_IDS"
  P4_WORLD="${#P4_GPUS[@]}"
fi
"$PYTHON" -m qwen3vl_local.sft_new_loop_phase4.preflight --dataset "$DATASET" --model-dir "$MODEL_DIR" --data-root "$DATA_ROOT" --require-ready --world-size "$P4_WORLD" "${P4_SAMPLING_ARGS[@]}"
if [[ "$MODE" == host-preflight ]]; then exit 0; fi
ARGS=(--dataset "$DATASET" --model-dir "$MODEL_DIR" --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR")
if [[ "$MODE" == single ]]; then
  exec "$PYTHON" -m qwen3vl_local.sft_new_loop_phase4.train "${ARGS[@]}" "$@"
fi
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${#P4_GPUS[@]}" --module qwen3vl_local.sft_new_loop_phase4.train "${ARGS[@]}" "$@"
