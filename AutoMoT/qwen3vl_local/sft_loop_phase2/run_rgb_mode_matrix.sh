#!/usr/bin/env bash
# 顺序跑 Phase2 的 4rgb / 2rgb_endpoints 完整对照：base production/audit -> train -> LoRA production/audit。
# 从 AutoMoT/ 主目录运行：
#   bash qwen3vl_local/sft_loop_phase2/run_rgb_mode_matrix.sh
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
    torchrun --nproc_per_node="${NPROC}" qwen3vl_local/sft_loop_phase2/eval.py "$@"
  else
    python qwen3vl_local/sft_loop_phase2/eval.py "$@"
  fi
}

run_train() {
  local title="$1"
  local history_rgb_mode="$2"
  echo
  echo "========== ${title} =========="
  if [[ "${NPROC}" -gt 1 ]]; then
    HISTORY_RGB_MODE="${history_rgb_mode}" DDP_GPU_COUNT="${NPROC}" bash qwen3vl_local/sft_loop_phase2/train.sh ddp
  else
    HISTORY_RGB_MODE="${history_rgb_mode}" bash qwen3vl_local/sft_loop_phase2/train.sh single
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
run_eval "6/10 LoRA 4rgb production" --adapter-dir checkpoints/sft_loop_phase2_runs/run_rs_four_binary_final_4rgb/final
run_eval "7/10 LoRA 4rgb audit" --adapter-dir checkpoints/sft_loop_phase2_runs/run_rs_four_binary_final_4rgb/final --audit-prompt

run_train "8/10 train 2rgb_endpoints LoRA" 2rgb_endpoints
run_eval "9/10 LoRA 2rgb_endpoints production" --adapter-dir checkpoints/sft_loop_phase2_runs/run_rs_four_binary_final_2rgb_endpoints/final
run_eval "10/10 LoRA 2rgb_endpoints audit" --adapter-dir checkpoints/sft_loop_phase2_runs/run_rs_four_binary_final_2rgb_endpoints/final --audit-prompt

echo
echo "[done] results are timestamped under checkpoints/sft_loop_phase2_eval/."
