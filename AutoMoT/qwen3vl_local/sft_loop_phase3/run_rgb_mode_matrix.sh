#!/usr/bin/env bash
# 顺序跑 Phase3 的 4rgb / 2rgb_endpoints 完整对照：base production/audit -> train -> LoRA production/audit。
# 从 AutoMoT/ 主目录运行：
#   bash qwen3vl_local/sft_loop_phase3/run_rgb_mode_matrix.sh
# 固定四卡 DDP；可用 GPU_IDS=0,1,2,3 或 GPU_IDS=4,5,6,7 指定哪四张卡。

set -euo pipefail

ulimit -S -c 0 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
export GPU_IDS
NPROC="$(awk -F',' '{print NF}' <<< "${GPU_IDS}")"
if [[ "${NPROC}" -ne 4 ]]; then
  echo "run_rgb_mode_matrix.sh is fixed to four-card runs; got GPU_IDS=${GPU_IDS} (${NPROC} cards)." >&2
  echo "Use exactly four GPU ids, for example: GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase3/run_rgb_mode_matrix.sh" >&2
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
  torchrun --nproc_per_node=4 \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="$(find_free_master_port)" \
    qwen3vl_local/sft_loop_phase3/eval.py "$@"
}

run_train() {
  local title="$1"
  local history_rgb_mode="$2"
  echo
  echo "========== ${title} =========="
  HISTORY_RGB_MODE="${history_rgb_mode}" DDP_GPU_COUNT=4 bash qwen3vl_local/sft_loop_phase3/train.sh ddp
}

adapter_dir_for_mode() {
  local history_rgb_mode="$1"
  local run_root="checkpoints/sft_loop_phase3_runs/run_event_gate_format_supervised_${history_rgb_mode}/latest"
  if [[ -d "${run_root}/best_generation" ]]; then
    echo "${run_root}/best_generation"
  elif [[ -d "${run_root}/best_val" ]]; then
    echo "${run_root}/best_val"
  else
    echo "${run_root}/final"
  fi
}

echo "[matrix] AutoMoT root: ${AUTOMOT_ROOT}"
echo "[matrix] fixed four-card DDP: GPU_IDS=${GPU_IDS}, processes=${NPROC}"
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
echo "[done] results are timestamped under checkpoints/sft_loop_phase3_eval/."
