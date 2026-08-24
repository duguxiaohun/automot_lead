#!/usr/bin/env bash
# 新 Phase2 单轮 EVENT 一键流程：build dataset -> base eval -> train -> LoRA eval -> audit samples。
#
# 从 AutoMoT/ 主目录运行：
#   bash qwen3vl_local/sft_new_loop_phase2/run_full_pipeline.sh
#
# 常用覆盖：
#   GPU_IDS=0,1,2,3 HISTORY_RGB_MODES="4rgb 2rgb_endpoints" bash qwen3vl_local/sft_new_loop_phase2/run_full_pipeline.sh
#   RUN_TRAIN=0 ADAPTER_DIR=checkpoints/sft_new_loop_phase2_runs/latest/best_generation bash qwen3vl_local/sft_new_loop_phase2/run_full_pipeline.sh

set -euo pipefail

ulimit -S -c 0 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PIPELINE_TIMESTAMP="${PIPELINE_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
PIPELINE_ROOT="${PIPELINE_ROOT:-checkpoints/sft_new_loop_phase2_pipeline/${PIPELINE_TIMESTAMP}}"
mkdir -p "${PIPELINE_ROOT}"

PIPELINE_LOG="${PIPELINE_LOG:-${PIPELINE_ROOT}/pipeline.log}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

RUN_BUILD="${RUN_BUILD:-1}"
RUN_BASE_EVAL="${RUN_BASE_EVAL:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_LORA_EVAL="${RUN_LORA_EVAL:-1}"
RUN_AUDIT_CASES="${RUN_AUDIT_CASES:-1}"
RUN_AUDIT_PROMPT_EVAL="${RUN_AUDIT_PROMPT_EVAL:-1}"

MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-checkpoints/sft_new_loop_phase2_data}"
INDEX="${INDEX:-${DATA_OUTPUT_DIR}/frame_index.jsonl}"
COLLECTION_DIR="${COLLECTION_DIR:-keyframe_filter/collection_output}"
DATA_ROOT="${DATA_ROOT:-lead_data}"
HISTORY_RGB_MODES="${HISTORY_RGB_MODES:-4rgb}"
TRAIN_FOCUS_BALANCE_COUNT="${TRAIN_FOCUS_BALANCE_COUNT:-${FOCUS_BALANCE_COUNT:-2048}}"
TRAIN_REGULAR_FOCUS_MULTIPLIER="${TRAIN_REGULAR_FOCUS_MULTIPLIER:-${REGULAR_FOCUS_MULTIPLIER:-2.0}}"
TRAIN_INVALID_FOCUS_MULTIPLIER="${TRAIN_INVALID_FOCUS_MULTIPLIER:-${INVALID_FOCUS_MULTIPLIER:-1.0}}"
TRAIN_NUM_EPOCHS="${TRAIN_NUM_EPOCHS:-${NUM_EPOCHS:-3}}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-${MAX_STEPS:-0}}"

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

count_visible_gpus() {
  local visible="$1"
  [[ -z "${visible}" ]] && { echo "0"; return; }
  awk -F',' '{print NF}' <<< "${visible}"
}

find_free_master_port() {
  python -c 'import socket
s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null || echo "$((20000 + RANDOM % 20000))"
}

DDP_GPU_COUNT="${DDP_GPU_COUNT:-${NPROC_PER_NODE:-4}}"
GPU_IDS="${GPU_IDS:-$(pick_idle_gpus "${DDP_GPU_COUNT}")}"
export GPU_IDS
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
NPROC="$(count_visible_gpus "${GPU_IDS}")"
TRAIN_MODE="${TRAIN_MODE:-$([[ "${NPROC}" -gt 1 ]] && echo ddp || echo single)}"

run_eval() {
  local title="$1"
  shift
  echo
  echo "========== ${title} =========="
  if [[ "${NPROC}" -gt 1 ]]; then
    torchrun --nproc_per_node="${NPROC}" \
      --master_addr="${MASTER_ADDR:-127.0.0.1}" \
      --master_port="$(find_free_master_port)" \
      qwen3vl_local/sft_new_loop_phase2/eval.py "$@"
  else
    python qwen3vl_local/sft_new_loop_phase2/eval.py "$@"
  fi
}

run_audit_cases() {
  local title="$1"
  local eval_dir="$2"
  local output_dir="$3"
  if [[ "${RUN_AUDIT_CASES}" != "1" ]]; then
    return
  fi
  echo
  echo "========== ${title} =========="
  python qwen3vl_local/sft_new_loop_phase2/audit_eval_cases.py \
    --eval-dir "${eval_dir}" \
    --output-dir "${output_dir}" \
    --data-root "${DATA_ROOT}" \
    --per-target "${AUDIT_PER_TARGET:-12}" \
    --overwrite
}

adapter_dir_for_train_output() {
  local train_output_dir="$1"
  if [[ -d "${train_output_dir}/best_generation" ]]; then
    echo "${train_output_dir}/best_generation"
  elif [[ -d "${train_output_dir}/best_val" ]]; then
    echo "${train_output_dir}/best_val"
  else
    echo "${train_output_dir}/final"
  fi
}

echo "[pipeline] AutoMoT root: ${AUTOMOT_ROOT}"
echo "[pipeline] PIPELINE_ROOT=${PIPELINE_ROOT}"
echo "[pipeline] PIPELINE_LOG=${PIPELINE_LOG}"
echo "[pipeline] GPU_IDS=${GPU_IDS}, processes=${NPROC}, train_mode=${TRAIN_MODE}"
echo "[pipeline] HISTORY_RGB_MODES=${HISTORY_RGB_MODES}"
echo "[pipeline] TRAIN_FOCUS_BALANCE_COUNT=${TRAIN_FOCUS_BALANCE_COUNT}"
echo "[pipeline] TRAIN_REGULAR_FOCUS_MULTIPLIER=${TRAIN_REGULAR_FOCUS_MULTIPLIER}"
echo "[pipeline] TRAIN_INVALID_FOCUS_MULTIPLIER=${TRAIN_INVALID_FOCUS_MULTIPLIER}"
echo "[pipeline] TRAIN_NUM_EPOCHS=${TRAIN_NUM_EPOCHS} TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS}"

if [[ "${RUN_BUILD}" == "1" ]]; then
  echo
  echo "========== 1/build dataset =========="
  BUILD_ARGS=(
    --collection-dir "${COLLECTION_DIR}"
    --data-root "${DATA_ROOT}"
    --output-dir "${DATA_OUTPUT_DIR}"
    --test-ratio "${TEST_RATIO:-0.10}"
    --val-ratio "${VAL_RATIO:-0.05}"
    --target-per-ue "${TARGET_PER_UE:-0}"
    --regular-multiplier "${REGULAR_MULTIPLIER:-1.0}"
    --highway-regular-fraction "${HIGHWAY_REGULAR_FRACTION:-0.25}"
    --invalid-ratio "${INVALID_RATIO:-0.20}"
    --max-routes "${BUILD_MAX_ROUTES:-0}"
    --progress-every-routes "${BUILD_PROGRESS_EVERY_ROUTES:-100}"
  )
  if [[ "${INCLUDE_VISUAL_RISK:-0}" == "1" ]]; then
    BUILD_ARGS+=(--include-visual-risk)
  fi
  python qwen3vl_local/sft_new_loop_phase2/build_dataset.py "${BUILD_ARGS[@]}"
fi

for HISTORY_RGB_MODE in ${HISTORY_RGB_MODES}; do
  case "${HISTORY_RGB_MODE}" in
    4rgb|2rgb_endpoints) ;;
    *)
      echo "Unknown HISTORY_RGB_MODE=${HISTORY_RGB_MODE}. Use 4rgb or 2rgb_endpoints." >&2
      exit 1
      ;;
  esac

  MODE_ROOT="${PIPELINE_ROOT}/${HISTORY_RGB_MODE}"
  mkdir -p "${MODE_ROOT}"
  BASE_EVAL_DIR="${MODE_ROOT}/eval_base_production"
  BASE_AUDIT_EVAL_DIR="${MODE_ROOT}/eval_base_audit_prompt"
  LORA_EVAL_DIR="${MODE_ROOT}/eval_lora_production"
  LORA_AUDIT_EVAL_DIR="${MODE_ROOT}/eval_lora_audit_prompt"
  MODE_TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-checkpoints/sft_new_loop_phase2_runs/run_direct_event_format_supervised_${HISTORY_RGB_MODE}/${PIPELINE_TIMESTAMP}}"

  if [[ "${RUN_BASE_EVAL}" == "1" ]]; then
    run_eval "base eval production ${HISTORY_RGB_MODE}" \
      --model-dir "${MODEL_DIR}" \
      --index "${INDEX}" \
      --data-root "${DATA_ROOT}" \
      --history-rgb-mode "${HISTORY_RGB_MODE}" \
      --cases-per-bin "${BASE_CASES_PER_BIN:-64}" \
      --max-frames "${BASE_MAX_EVAL_FRAMES:-0}" \
      --output-dir "${BASE_EVAL_DIR}" \
      --no-timestamp-output \
      --overwrite
    run_audit_cases "base error audit ${HISTORY_RGB_MODE}" "${BASE_EVAL_DIR}" "${MODE_ROOT}/audit_base_production"

    if [[ "${RUN_AUDIT_PROMPT_EVAL}" == "1" ]]; then
      run_eval "base eval audit-prompt ${HISTORY_RGB_MODE}" \
        --model-dir "${MODEL_DIR}" \
        --index "${INDEX}" \
        --data-root "${DATA_ROOT}" \
        --history-rgb-mode "${HISTORY_RGB_MODE}" \
        --cases-per-bin "${BASE_CASES_PER_BIN:-64}" \
        --max-frames "${BASE_MAX_EVAL_FRAMES:-0}" \
        --audit-prompt \
        --output-dir "${BASE_AUDIT_EVAL_DIR}" \
        --no-timestamp-output \
        --overwrite
    fi
  fi

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    echo
    echo "========== train LoRA ${HISTORY_RGB_MODE} =========="
    RUN_TIMESTAMP="${PIPELINE_TIMESTAMP}" \
    OUTPUT_DIR="${MODE_TRAIN_OUTPUT_DIR}" \
    MODEL_DIR="${MODEL_DIR}" \
    INDEX="${INDEX}" \
    DATA_ROOT="${DATA_ROOT}" \
    HISTORY_RGB_MODE="${HISTORY_RGB_MODE}" \
    DDP_GPU_COUNT="${NPROC}" \
    FOCUS_BALANCE_COUNT="${TRAIN_FOCUS_BALANCE_COUNT}" \
    REGULAR_FOCUS_MULTIPLIER="${TRAIN_REGULAR_FOCUS_MULTIPLIER}" \
    INVALID_FOCUS_MULTIPLIER="${TRAIN_INVALID_FOCUS_MULTIPLIER}" \
    NUM_EPOCHS="${TRAIN_NUM_EPOCHS}" \
    MAX_STEPS="${TRAIN_MAX_STEPS}" \
    bash qwen3vl_local/sft_new_loop_phase2/train.sh "${TRAIN_MODE}"
    LORA_ADAPTER_DIR="$(adapter_dir_for_train_output "${MODE_TRAIN_OUTPUT_DIR}")"
  else
    LORA_ADAPTER_DIR="${ADAPTER_DIR:-$(adapter_dir_for_train_output "${MODE_TRAIN_OUTPUT_DIR}")}"
  fi

  if [[ "${RUN_LORA_EVAL}" == "1" ]]; then
    if [[ ! -d "${LORA_ADAPTER_DIR}" ]]; then
      echo "LoRA adapter not found: ${LORA_ADAPTER_DIR}" >&2
      echo "Set ADAPTER_DIR=... or run with RUN_TRAIN=1." >&2
      exit 1
    fi
    run_eval "LoRA eval production ${HISTORY_RGB_MODE}" \
      --model-dir "${MODEL_DIR}" \
      --index "${INDEX}" \
      --data-root "${DATA_ROOT}" \
      --adapter-dir "${LORA_ADAPTER_DIR}" \
      --cases-per-bin "${LORA_CASES_PER_BIN:-64}" \
      --max-frames "${LORA_MAX_EVAL_FRAMES:-0}" \
      --output-dir "${LORA_EVAL_DIR}" \
      --no-timestamp-output \
      --overwrite
    run_audit_cases "LoRA error audit ${HISTORY_RGB_MODE}" "${LORA_EVAL_DIR}" "${MODE_ROOT}/audit_lora_production"

    if [[ "${RUN_AUDIT_PROMPT_EVAL}" == "1" ]]; then
      run_eval "LoRA eval audit-prompt ${HISTORY_RGB_MODE}" \
        --model-dir "${MODEL_DIR}" \
        --index "${INDEX}" \
        --data-root "${DATA_ROOT}" \
        --adapter-dir "${LORA_ADAPTER_DIR}" \
        --cases-per-bin "${LORA_CASES_PER_BIN:-64}" \
        --max-frames "${LORA_MAX_EVAL_FRAMES:-0}" \
        --audit-prompt \
        --output-dir "${LORA_AUDIT_EVAL_DIR}" \
        --no-timestamp-output \
        --overwrite
    fi
  fi
done

echo
echo "[done] pipeline artifacts: ${PIPELINE_ROOT}"
echo "[done] training runs: checkpoints/sft_new_loop_phase2_runs/run_direct_event_format_supervised_<mode>/${PIPELINE_TIMESTAMP}"
