#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P4_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${P4_DIR}/../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
MODE="${1:-ddp}"
if [[ $# -gt 0 ]]; then shift; fi
PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-checkpoints/sft_new_loop_phase4_data_v7}"
MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3.5-4B}"
DATA_ROOT="${DATA_ROOT:-lead_data}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft_new_loop_phase4_runs/$(date +%Y%m%d_%H%M%S_%N)}"
if [[ "$MODE" == check ]]; then
  exec "$PYTHON" -m qwen3vl_local.sft_new_loop_phase4.train --dataset "$DATASET" --output-dir "$OUTPUT_DIR" --sampling-only "$@"
fi
[[ "$MODE" == single || "$MODE" == ddp ]] || { echo 'mode must be single/ddp/check' >&2; exit 2; }
"$PYTHON" -m qwen3vl_local.sft_new_loop_phase4.preflight --dataset "$DATASET" --model-dir "$MODEL_DIR" --data-root "$DATA_ROOT" --require-ready
GPU_IDS="$("$PYTHON" -c 'from qwen3vl_local.sft_new_loop_phase4.devices import select_gpus; print(select_gpus(4))')"
if [[ "$MODE" == single ]]; then GPU_IDS="${GPU_IDS%%,*}"; fi
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
IFS=',' read -r -a P4_GPUS <<< "$GPU_IDS"
ARGS=(--dataset "$DATASET" --model-dir "$MODEL_DIR" --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR")
if [[ "$MODE" == single ]]; then
  exec "$PYTHON" -m qwen3vl_local.sft_new_loop_phase4.train "${ARGS[@]}" "$@"
fi
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${#P4_GPUS[@]}" --module qwen3vl_local.sft_new_loop_phase4.train "${ARGS[@]}" "$@"
