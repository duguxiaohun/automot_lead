#!/usr/bin/env bash
# 顺序跑 Phase2 augment 的 4rgb / 2rgb_endpoints 完整对照：base production/audit -> train -> LoRA production/audit。
# 从 AutoMoT/ 主目录运行：
#   bash qwen3vl_local/sft_loop_phase2_augment/run_rgb_mode_matrix.sh
# 默认四卡 GPU 0,1,2,3；可用 GPU_IDS=0 或 GPU_IDS=0,1 覆盖。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
export GPU_IDS
NPROC="$(awk -F',' '{print NF}' <<< "${GPU_IDS}")"

run_eval() {
  local title="$1"
  shift
  echo
  echo "========== ${title} =========="
  if [[ "${NPROC}" -gt 1 ]]; then
    torchrun --nproc_per_node="${NPROC}" qwen3vl_local/sft_loop_phase2_augment/eval.py "$@"
  else
    python qwen3vl_local/sft_loop_phase2_augment/eval.py "$@"
  fi
}

run_train() {
  local title="$1"
  local history_rgb_mode="$2"
  echo
  echo "========== ${title} =========="
  if [[ "${NPROC}" -gt 1 ]]; then
    HISTORY_RGB_MODE="${history_rgb_mode}" DDP_GPU_COUNT="${NPROC}" bash qwen3vl_local/sft_loop_phase2_augment/train.sh ddp
  else
    HISTORY_RGB_MODE="${history_rgb_mode}" bash qwen3vl_local/sft_loop_phase2_augment/train.sh single
  fi
}

adapter_dir_for_mode() {
  local history_rgb_mode="$1"
  local run_root="checkpoints/sft_loop_phase2_augment_runs/run_rs_augmented_format_supervised_${history_rgb_mode}/latest"
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
echo "[matrix] 2rgb_endpoints always uses source frames [0,3] (first and fourth)."

run_eval "1/10 base 4rgb production" --history-rgb-mode 4rgb
run_eval "2/10 base 4rgb audit" --history-rgb-mode 4rgb --audit-prompt
run_eval "3/10 base 2rgb_endpoints production" --history-rgb-mode 2rgb_endpoints
run_eval "4/10 base 2rgb_endpoints audit" --history-rgb-mode 2rgb_endpoints --audit-prompt

run_train "5/10 train 4rgb LoRA" 4rgb
LORA_4RGB_DIR="$(adapter_dir_for_mode 4rgb)"
run_eval "6/10 LoRA 4rgb production" --adapter-dir "${LORA_4RGB_DIR}"
run_eval "7/10 LoRA 4rgb audit" --adapter-dir "${LORA_4RGB_DIR}" --audit-prompt

run_train "8/10 train 2rgb_endpoints LoRA" 2rgb_endpoints
LORA_2RGB_DIR="$(adapter_dir_for_mode 2rgb_endpoints)"
run_eval "9/10 LoRA 2rgb_endpoints production" --adapter-dir "${LORA_2RGB_DIR}"
run_eval "10/10 LoRA 2rgb_endpoints audit" --adapter-dir "${LORA_2RGB_DIR}" --audit-prompt

echo
echo "[done] results are timestamped under checkpoints/sft_loop_phase2_augment_eval/."
