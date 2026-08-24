#!/usr/bin/env bash
# 顺序跑新 Phase2 的 4rgb / 2rgb_endpoints 完整对照：base production/audit -> train -> LoRA production/audit。
# 从 AutoMoT/ 主目录运行：
#   bash qwen3vl_local/sft_new_loop_phase2/run_rgb_mode_matrix.sh
# 默认自动选择四张空闲卡；GPU_IDS 非空时按显式列表推断进程数。

set -euo pipefail

ulimit -S -c 0 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

pick_idle_gpus() {
  local want_count="$1"
  local selected
  if command -v nvidia-smi >/dev/null 2>&1; then
    selected="$(
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); print $2, $3, $1}' \
        | sort -n -k1,1 -k2,2 \
        | head -n "${want_count}" \
        | awk '{print $3}' \
        | paste -sd, -
    )"
    if [[ -n "${selected}" ]]; then echo "${selected}"; return 0; fi
  fi
  if [[ "${want_count}" -le 1 ]]; then echo "0"; else seq -s, 0 "$((want_count - 1))"; fi
}

DDP_GPU_COUNT="${DDP_GPU_COUNT:-4}"
if [[ -z "${GPU_IDS:-}" ]]; then
  GPU_SELECTION_SOURCE="automatic"
  GPU_IDS="$(pick_idle_gpus "${DDP_GPU_COUNT}")"
else
  GPU_SELECTION_SOURCE="explicit"
fi
export GPU_IDS
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
NPROC="$(awk -F',' '{print NF}' <<< "${GPU_IDS}")"
if [[ "${GPU_SELECTION_SOURCE}" == "automatic" && "${NPROC}" -ne "${DDP_GPU_COUNT}" ]]; then
  echo "automatic selection requested ${DDP_GPU_COUNT} GPUs but found GPU_IDS=${GPU_IDS}" >&2
  exit 1
fi

find_free_master_port() {
  python -c 'import socket
s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null || echo "$((20000 + RANDOM % 20000))"
}

run_eval() {
  local title="$1"
  shift
  echo
  echo "========== ${title} =========="
  torchrun --nproc_per_node="${NPROC}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="$(find_free_master_port)" \
    qwen3vl_local/sft_new_loop_phase2/eval.py "$@"
}

run_train() {
  local title="$1"
  local history_rgb_mode="$2"
  echo
  echo "========== ${title} =========="
  MODEL_DIR="${MODEL_DIR}" INDEX="${INDEX}" DATA_ROOT="${DATA_ROOT}" \
    HISTORY_RGB_MODE="${history_rgb_mode}" DDP_GPU_COUNT="${NPROC}" \
    bash qwen3vl_local/sft_new_loop_phase2/train.sh ddp
}

adapter_dir_for_mode() {
  local history_rgb_mode="$1"
  local run_root="checkpoints/sft_new_loop_phase2_runs/run_direct_event_format_supervised_${history_rgb_mode}/latest"
  if [[ -d "${run_root}/best_generation" ]]; then
    echo "${run_root}/best_generation"
  elif [[ -d "${run_root}/best_val" ]]; then
    echo "${run_root}/best_val"
  else
    echo "${run_root}/final"
  fi
}

echo "[matrix] AutoMoT root: ${AUTOMOT_ROOT}"
echo "[matrix] GPU_IDS=${GPU_IDS}, processes=${NPROC}, source=${GPU_SELECTION_SOURCE} (DDP_GPU_COUNT only controls automatic selection)"
echo "[matrix] 2rgb_endpoints always uses source frames [0,3] (first and fourth)."

MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
INDEX="${INDEX:-checkpoints/sft_new_loop_phase2_data/frame_index.jsonl}"
DATA_ROOT="${DATA_ROOT:-lead_data}"
EVAL_COMMON_ARGS=(--model-dir "${MODEL_DIR}" --index "${INDEX}" --data-root "${DATA_ROOT}")

run_eval "1/10 base 4rgb production" "${EVAL_COMMON_ARGS[@]}" --history-rgb-mode 4rgb
run_eval "2/10 base 4rgb audit" "${EVAL_COMMON_ARGS[@]}" --history-rgb-mode 4rgb --audit-prompt
run_eval "3/10 base 2rgb_endpoints production" "${EVAL_COMMON_ARGS[@]}" --history-rgb-mode 2rgb_endpoints
run_eval "4/10 base 2rgb_endpoints audit" "${EVAL_COMMON_ARGS[@]}" --history-rgb-mode 2rgb_endpoints --audit-prompt

run_train "5/10 train 4rgb LoRA" 4rgb
LORA_4RGB_DIR="$(adapter_dir_for_mode 4rgb)"
run_eval "6/10 LoRA 4rgb production" "${EVAL_COMMON_ARGS[@]}" --adapter-dir "${LORA_4RGB_DIR}"
run_eval "7/10 LoRA 4rgb audit" "${EVAL_COMMON_ARGS[@]}" --adapter-dir "${LORA_4RGB_DIR}" --audit-prompt

run_train "8/10 train 2rgb_endpoints LoRA" 2rgb_endpoints
LORA_2RGB_DIR="$(adapter_dir_for_mode 2rgb_endpoints)"
run_eval "9/10 LoRA 2rgb_endpoints production" "${EVAL_COMMON_ARGS[@]}" --adapter-dir "${LORA_2RGB_DIR}"
run_eval "10/10 LoRA 2rgb_endpoints audit" "${EVAL_COMMON_ARGS[@]}" --adapter-dir "${LORA_2RGB_DIR}" --audit-prompt

echo
echo "[done] results are timestamped under checkpoints/sft_new_loop_phase2_eval/."
