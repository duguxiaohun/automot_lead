#!/usr/bin/env bash
# 新 Phase3 一键评测：base production -> LoRA production -> 可选 audit prompt -> 错例审计包。
#
# 从 AutoMoT/ 目录运行：
#   ADAPTER_DIR=checkpoints/sft_new_loop_phase3_runs/latest bash qwen3vl_local/sft_new_loop_phase3/eval.sh
# 或：
#   bash qwen3vl_local/sft_new_loop_phase3/eval.sh checkpoints/sft_new_loop_phase3_runs/latest/final

set -euo pipefail

ulimit -S -c 0 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

PHASE_NAME="sft_new_loop_phase3"
EVAL_PY="qwen3vl_local/sft_new_loop_phase3/eval.py"
AUDIT_PY="qwen3vl_local/sft_new_loop_phase3/audit_eval_cases.py"
VISUAL_AUDIT_PY="qwen3vl_local/sft_new_loop_phase3/visual_audit.py"
ADAPTER_CONFIG_NAME="sft_new_loop_phase3_adapter_config.json"

MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
INDEX="${INDEX:-checkpoints/sft_new_loop_phase3_data/frame_index.jsonl}"
DATA_ROOT="${DATA_ROOT:-lead_data}"
SPLIT="${SPLIT:-test}"
CASES_PER_BIN="${CASES_PER_BIN:-64}"
ROUTE_DIVERSE_SAMPLING="${ROUTE_DIVERSE_SAMPLING:-0}"
REQUIRE_INVALID_COVERAGE="${REQUIRE_INVALID_COVERAGE:-1}"
EXCLUDE_CASES_JSONL="${EXCLUDE_CASES_JSONL:-}"
EXPECTED_EXCLUDED_CASES="${EXPECTED_EXCLUDED_CASES:-0}"
EXPECTED_TOTAL_CASES="${EXPECTED_TOTAL_CASES:-0}"
MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
AUDIT_PER_TARGET="${AUDIT_PER_TARGET:-8}"
RUN_BASE_EVAL="${RUN_BASE_EVAL:-1}"
RUN_VISUAL_AUDIT="${RUN_VISUAL_AUDIT:-1}"
SCAN_VISUAL_RISKS="${SCAN_VISUAL_RISKS:-0}"
RUN_AUDIT_PROMPT_EVAL="${RUN_AUDIT_PROMPT_EVAL:-1}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/sft_new_loop_phase3_eval_review/${TIMESTAMP}}"
ADAPTER_INPUT="${ADAPTER_DIR:-${CKPT_DIR:-${1:-}}}"

if [[ -z "${ADAPTER_INPUT}" ]]; then
  echo "Usage: ADAPTER_DIR=<lora-adapter-or-run-dir> bash qwen3vl_local/sft_new_loop_phase3/eval.sh" >&2
  exit 2
fi

resolve_adapter_dir() {
  local input="$1"
  local candidate
  for candidate in "${input}/best_generation" "${input}/final" "${input}/fallback_generation" "${input}"; do
    if [[ -f "${candidate}/${ADAPTER_CONFIG_NAME}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "Cannot resolve ${PHASE_NAME} adapter from ${input}; expected best_generation, final, fallback_generation, or an exact adapter." >&2
  return 1
}

ADAPTER_DIR="$(resolve_adapter_dir "${ADAPTER_INPUT}")"

read_adapter_history_rgb_mode() {
  python - "$1" <<'PY'
import json, pathlib, sys
adapter = pathlib.Path(sys.argv[1])
path = adapter / "sft_new_loop_phase3_adapter_config.json"
if not path.is_file():
    raise SystemExit(f"missing adapter config: {path}")
config = json.loads(path.read_text(encoding="utf-8"))
if not config.get("history_rgb_mode"):
    raise SystemExit(f"adapter config has no history_rgb_mode: {path}")
print(str(config["history_rgb_mode"]))
PY
}

BASE_HISTORY_RGB_MODE="$(read_adapter_history_rgb_mode "${ADAPTER_DIR}")"
case "${BASE_HISTORY_RGB_MODE}" in
  4rgb|2rgb_endpoints) ;;
  *)
    echo "Unknown BASE_HISTORY_RGB_MODE=${BASE_HISTORY_RGB_MODE}. Use 4rgb or 2rgb_endpoints." >&2
    exit 2
    ;;
esac

COMMON_ARGS=(
  --index "${INDEX}"
  --data-root "${DATA_ROOT}"
  --model-dir "${MODEL_DIR}"
  --split "${SPLIT}"
  --cases-per-bin "${CASES_PER_BIN}"
  --max-frames "${MAX_EVAL_FRAMES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --no-timestamp-output
  --overwrite
)
if [[ "${ROUTE_DIVERSE_SAMPLING}" == "1" ]]; then
  COMMON_ARGS+=(--route-diverse-sampling)
else
  COMMON_ARGS+=(--no-route-diverse-sampling)
fi
if [[ "${REQUIRE_INVALID_COVERAGE}" == "0" ]]; then
  COMMON_ARGS+=(--no-require-invalid-coverage)
else
  COMMON_ARGS+=(--require-invalid-coverage)
fi
if [[ -n "${EXCLUDE_CASES_JSONL}" ]]; then
  for exclusion_path in ${EXCLUDE_CASES_JSONL}; do
    COMMON_ARGS+=(--exclude-cases-jsonl "${exclusion_path}")
  done
fi
COMMON_ARGS+=(
  --expected-excluded-cases "${EXPECTED_EXCLUDED_CASES}"
  --expected-total-cases "${EXPECTED_TOTAL_CASES}"
)

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

EVAL_GPU_COUNT="${EVAL_GPU_COUNT:-4}"
if [[ -z "${GPU_IDS:-}" ]]; then
  GPU_IDS="$(pick_idle_gpus "${EVAL_GPU_COUNT}")"
fi
export GPU_IDS
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
NPROC="$(awk -F',' '{print NF}' <<< "${GPU_IDS}")"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${OUTPUT_ROOT}/eval.log") 2>&1

find_free_master_port() {
  python -c 'import socket
s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null || echo "$((20000 + RANDOM % 20000))"
}

run_eval() {
  local title="$1"
  shift
  echo
  echo "========== ${title} =========="
  if [[ "${NPROC}" -gt 1 ]]; then
    torchrun --nproc_per_node="${NPROC}" \
      --master_addr="${MASTER_ADDR:-127.0.0.1}" \
      --master_port="$(find_free_master_port)" \
      "${EVAL_PY}" "$@"
  else
    python "${EVAL_PY}" "$@"
  fi
}

echo "[phase3-eval] adapter=${ADAPTER_DIR} history_rgb_mode=${BASE_HISTORY_RGB_MODE} gpus=${GPU_IDS} output=${OUTPUT_ROOT}"

if [[ "${RUN_VISUAL_AUDIT}" == "1" ]]; then
  VISUAL_ARGS=(--output "${OUTPUT_ROOT}/visual_audit_manifest.json")
  if [[ "${SCAN_VISUAL_RISKS}" == "1" ]]; then VISUAL_ARGS+=(--scan-frame-risks); fi
  python "${VISUAL_AUDIT_PY}" "${VISUAL_ARGS[@]}"
fi

if [[ "${RUN_BASE_EVAL}" == "1" ]]; then
  run_eval "base production" "${COMMON_ARGS[@]}" \
    --history-rgb-mode "${BASE_HISTORY_RGB_MODE}" \
    --no-audit-prompt \
    --output-dir "${OUTPUT_ROOT}/base_production"
fi

run_eval "lora production" "${COMMON_ARGS[@]}" \
  --adapter-dir "${ADAPTER_DIR}" \
  --no-audit-prompt \
  --output-dir "${OUTPUT_ROOT}/lora_production"

if [[ "${RUN_AUDIT_PROMPT_EVAL}" == "1" ]]; then
  run_eval "lora audit prompt" "${COMMON_ARGS[@]}" \
    --adapter-dir "${ADAPTER_DIR}" \
    --audit-prompt \
    --output-dir "${OUTPUT_ROOT}/lora_audit"
fi

python "${AUDIT_PY}" \
  --eval-dir "${OUTPUT_ROOT}/lora_production" \
  --output-dir "${OUTPUT_ROOT}/lora_production_audit_samples" \
  --data-root "${DATA_ROOT}" \
  --per-target "${AUDIT_PER_TARGET}" \
  --overwrite

echo
echo "[phase3-eval] done: ${OUTPUT_ROOT}"
