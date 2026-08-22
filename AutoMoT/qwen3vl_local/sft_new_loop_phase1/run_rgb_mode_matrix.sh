#!/usr/bin/env bash
# Run the fused Phase1+Phase2 4rgb / 2rgb_endpoints comparison matrix:
# base production/audit -> train -> LoRA production/audit for each RGB mode.
#
# From the AutoMoT/ directory:
#   bash qwen3vl_local/sft_new_loop_phase1/run_rgb_mode_matrix.sh
#
# Useful overrides:
#   GPU_IDS=0
#   DATA_ROOT=/path/to/lead_data
#   CASES_PER_BIN=64
#   MATRIX_TAG=my_probe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
INDEX="${INDEX:-checkpoints/sft_new_loop_phase1_data/frame_index.jsonl}"
DATA_ROOT="${DATA_ROOT:-lead_data}"
SPLIT="${SPLIT:-test}"
CASES_PER_BIN="${CASES_PER_BIN:-64}"
MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MATRIX_TAG="${MATRIX_TAG:-$(date +%Y%m%d_%H%M%S)}"
EVAL_ROOT="${EVAL_ROOT:-checkpoints/sft_new_loop_phase1_eval_matrix/${MATRIX_TAG}}"
TRAIN_ROOT="${TRAIN_ROOT:-checkpoints/sft_new_loop_phase1_runs}"
export GPU_IDS
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
NPROC="$(awk -F',' '{print NF}' <<< "${GPU_IDS}")"

find_free_master_port() {
  python -c 'import socket
s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null || echo "$((20000 + RANDOM % 20000))"
}

run_eval() {
  local title="$1"
  local output_dir="$2"
  shift 2
  echo
  echo "========== ${title} =========="
  local args=(
    --model-dir "${MODEL_DIR}"
    --index "${INDEX}"
    --data-root "${DATA_ROOT}"
    --split "${SPLIT}"
    --cases-per-bin "${CASES_PER_BIN}"
    --max-frames "${MAX_EVAL_FRAMES}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --output-dir "${output_dir}"
    "$@"
  )
  if [[ "${NPROC}" -gt 1 ]]; then
    torchrun --nproc_per_node="${NPROC}" \
      --master_addr="${MASTER_ADDR:-127.0.0.1}" \
      --master_port="$(find_free_master_port)" \
      qwen3vl_local/sft_new_loop_phase1/eval.py "${args[@]}"
  else
    python qwen3vl_local/sft_new_loop_phase1/eval.py "${args[@]}"
  fi
}

run_train() {
  local title="$1"
  local history_rgb_mode="$2"
  local run_tag="${MATRIX_TAG}_${history_rgb_mode}"
  echo
  echo "========== ${title} =========="
  if [[ "${NPROC}" -gt 1 ]]; then
    RUN_TAG="${run_tag}" HISTORY_RGB_MODE="${history_rgb_mode}" DATA_ROOT="${DATA_ROOT}" \
      MODEL_DIR="${MODEL_DIR}" INDEX="${INDEX}" DDP_GPU_COUNT="${NPROC}" \
      bash qwen3vl_local/sft_new_loop_phase1/train.sh ddp
  else
    RUN_TAG="${run_tag}" HISTORY_RGB_MODE="${history_rgb_mode}" DATA_ROOT="${DATA_ROOT}" \
      MODEL_DIR="${MODEL_DIR}" INDEX="${INDEX}" \
      bash qwen3vl_local/sft_new_loop_phase1/train.sh single
  fi
}

adapter_dir_for_mode() {
  local history_rgb_mode="$1"
  local run_root="${TRAIN_ROOT}/run_${MATRIX_TAG}_${history_rgb_mode}_combined_phase1_phase2_${history_rgb_mode}"
  if [[ -d "${run_root}/best_generation" ]]; then
    echo "${run_root}/best_generation"
  elif [[ -d "${run_root}/best_val" ]]; then
    echo "${run_root}/best_val"
  else
    echo "${run_root}/final"
  fi
}

echo "[matrix] AutoMoT root: ${AUTOMOT_ROOT}"
echo "[matrix] GPU_IDS=${GPU_IDS}, processes=${NPROC}"
echo "[matrix] MODEL_DIR=${MODEL_DIR}"
echo "[matrix] INDEX=${INDEX}"
echo "[matrix] DATA_ROOT=${DATA_ROOT}"
echo "[matrix] EVAL_ROOT=${EVAL_ROOT}"
echo "[matrix] 2rgb_endpoints uses source frames [0,3] (first and fourth)."

mkdir -p "${EVAL_ROOT}"

run_eval "1/10 base 4rgb production" "${EVAL_ROOT}/01_base_4rgb_production" --history-rgb-mode 4rgb
run_eval "2/10 base 4rgb audit" "${EVAL_ROOT}/02_base_4rgb_audit" --history-rgb-mode 4rgb --audit-prompt
run_eval "3/10 base 2rgb_endpoints production" "${EVAL_ROOT}/03_base_2rgb_endpoints_production" --history-rgb-mode 2rgb_endpoints
run_eval "4/10 base 2rgb_endpoints audit" "${EVAL_ROOT}/04_base_2rgb_endpoints_audit" --history-rgb-mode 2rgb_endpoints --audit-prompt

run_train "5/10 train 4rgb LoRA" 4rgb
LORA_4RGB_DIR="$(adapter_dir_for_mode 4rgb)"
run_eval "6/10 LoRA 4rgb production" "${EVAL_ROOT}/06_lora_4rgb_production" --adapter-dir "${LORA_4RGB_DIR}"
run_eval "7/10 LoRA 4rgb audit" "${EVAL_ROOT}/07_lora_4rgb_audit" --adapter-dir "${LORA_4RGB_DIR}" --audit-prompt

run_train "8/10 train 2rgb_endpoints LoRA" 2rgb_endpoints
LORA_2RGB_DIR="$(adapter_dir_for_mode 2rgb_endpoints)"
run_eval "9/10 LoRA 2rgb_endpoints production" "${EVAL_ROOT}/09_lora_2rgb_endpoints_production" --adapter-dir "${LORA_2RGB_DIR}"
run_eval "10/10 LoRA 2rgb_endpoints audit" "${EVAL_ROOT}/10_lora_2rgb_endpoints_audit" --adapter-dir "${LORA_2RGB_DIR}" --audit-prompt

echo
echo "[done] matrix results: ${EVAL_ROOT}"
