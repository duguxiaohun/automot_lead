#!/usr/bin/env bash
# fused Phase1+Phase2 一键流程：视觉审计 -> 构建数据 -> base eval -> 训练 -> LoRA eval -> 错例 RGB 抽样。
#
# 从 AutoMoT/ 主目录运行：
#   bash qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh
#
# 常用覆盖：
#   GPU_IDS=0,1,2,3 HISTORY_RGB_MODES="4rgb 2rgb_endpoints" bash qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh
#   RUN_TRAIN=0 ADAPTER_DIR=checkpoints/sft_new_loop_phase1_runs/latest/best_generation bash qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh

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
PIPELINE_ROOT="${PIPELINE_ROOT:-checkpoints/sft_new_loop_phase1_pipeline/${PIPELINE_TIMESTAMP}}"
mkdir -p "${PIPELINE_ROOT}"

PIPELINE_LOG="${PIPELINE_LOG:-${PIPELINE_ROOT}/pipeline.log}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

RUN_VISUAL_AUDIT="${RUN_VISUAL_AUDIT:-1}"
RUN_BUILD="${RUN_BUILD:-1}"
RUN_BASE_EVAL="${RUN_BASE_EVAL:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_LORA_EVAL="${RUN_LORA_EVAL:-1}"
RUN_AUDIT_CASES="${RUN_AUDIT_CASES:-1}"
RUN_AUDIT_PROMPT_EVAL="${RUN_AUDIT_PROMPT_EVAL:-1}"

MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-checkpoints/sft_new_loop_phase1_data}"
INDEX="${INDEX:-${DATA_OUTPUT_DIR}/frame_index.jsonl}"
COLLECTION_DIR="${COLLECTION_DIR:-keyframe_filter/collection_output}"
ANSWER_TABLE="${ANSWER_TABLE:-keyframe_filter/collection_output/phase1_four_question_audit/phase1_four_question_answer_table.json}"
REVIEW_ROOT="${REVIEW_ROOT:-keyframe_filter/collection_output/phase1_four_question_audit/full_route_rgb_label_review_20260809}"
COVERAGE_MANIFEST="${COVERAGE_MANIFEST:-qwen3vl_local/sft_new_loop_phase1/phase2_rgb_audit_coverage.json}"
DATA_ROOT="${DATA_ROOT:-lead_data}"
HISTORY_RGB_MODES="${HISTORY_RGB_MODES:-4rgb}"
TRAIN_FOCUS_BALANCE_COUNT="${TRAIN_FOCUS_BALANCE_COUNT:-${FOCUS_BALANCE_COUNT:-9216}}"
TRAIN_MAX_FRAME_REPEAT="${TRAIN_MAX_FRAME_REPEAT:-${MAX_TRAIN_FRAME_REPEAT:-10}}"
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
      qwen3vl_local/sft_new_loop_phase1/eval.py "$@"
  else
    GPU_IDS="${GPU_IDS}" python qwen3vl_local/sft_new_loop_phase1/eval.py "$@"
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
  python qwen3vl_local/sft_new_loop_phase1/audit_eval_cases.py \
    --eval-dir "${eval_dir}" \
    --output-dir "${output_dir}" \
    --data-root "${DATA_ROOT}" \
    --per-target "${AUDIT_PER_TARGET:-12}" \
    --overwrite
}

adapter_dir_for_mode() {
  local history_rgb_mode="$1"
  local run_root="checkpoints/sft_new_loop_phase1_runs/run_${PIPELINE_TIMESTAMP}_${history_rgb_mode}_combined_phase1_phase2_${history_rgb_mode}"
  if [[ -d "${run_root}/best_generation" ]]; then
    echo "${run_root}/best_generation"
  elif [[ -d "${run_root}/best_val" ]]; then
    echo "${run_root}/best_val"
  else
    echo "${run_root}/final"
  fi
}

echo "[pipeline] AutoMoT 根目录: ${AUTOMOT_ROOT}"
echo "[pipeline] 输出目录: ${PIPELINE_ROOT}"
echo "[pipeline] 日志: ${PIPELINE_LOG}"
echo "[pipeline] GPU_IDS=${GPU_IDS}, 进程数=${NPROC}, 训练模式=${TRAIN_MODE}"
echo "[pipeline] HISTORY_RGB_MODES=${HISTORY_RGB_MODES}"
echo "[pipeline] TRAIN_FOCUS_BALANCE_COUNT=${TRAIN_FOCUS_BALANCE_COUNT}"
echo "[pipeline] TRAIN_MAX_FRAME_REPEAT=${TRAIN_MAX_FRAME_REPEAT}"
echo "[pipeline] TRAIN_NUM_EPOCHS=${TRAIN_NUM_EPOCHS} TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS}"

if [[ "${RUN_VISUAL_AUDIT}" == "1" ]]; then
  echo
  echo "========== 1/视觉审计覆盖 manifest =========="
  VISUAL_AUDIT_ARGS=(
    --collection-dir "${COLLECTION_DIR}"
    --review-root "${REVIEW_ROOT}"
    --coverage-manifest "${COVERAGE_MANIFEST}"
    --output "${DATA_OUTPUT_DIR}/visual_audit_manifest.json"
  )
  if [[ "${VISUAL_AUDIT_SCAN_FRAME_RISKS:-0}" == "1" ]]; then
    VISUAL_AUDIT_ARGS+=(--scan-frame-risks)
  fi
  python qwen3vl_local/sft_new_loop_phase1/visual_audit.py "${VISUAL_AUDIT_ARGS[@]}"
fi

if [[ "${RUN_BUILD}" == "1" ]]; then
  echo
  echo "========== 2/构建 fused 数据集 =========="
  BUILD_ARGS=(
    --collection-dir "${COLLECTION_DIR}"
    --data-root "${DATA_ROOT}"
    --answer-table "${ANSWER_TABLE}"
    --output-dir "${DATA_OUTPUT_DIR}"
    --review-root "${REVIEW_ROOT}"
    --coverage-manifest "${COVERAGE_MANIFEST}"
    --scenarios "${SCENARIOS:-all}"
    --split-seed "${SPLIT_SEED:-20260813}"
    --test-ratio "${TEST_RATIO:-0.10}"
    --val-ratio "${VAL_RATIO:-0.05}"
    --max-routes "${BUILD_MAX_ROUTES:-0}"
    --progress-every-routes "${BUILD_PROGRESS_EVERY_ROUTES:-100}"
  )
  if [[ "${INCLUDE_VISUAL_RISK:-0}" == "1" ]]; then
    BUILD_ARGS+=(--include-visual-risk)
  fi
  python qwen3vl_local/sft_new_loop_phase1/build_dataset.py "${BUILD_ARGS[@]}"
fi

for HISTORY_RGB_MODE in ${HISTORY_RGB_MODES}; do
  case "${HISTORY_RGB_MODE}" in
    4rgb|2rgb_endpoints) ;;
    *)
      echo "未知 HISTORY_RGB_MODE=${HISTORY_RGB_MODE}，只能用 4rgb 或 2rgb_endpoints。" >&2
      exit 1
      ;;
  esac

  MODE_ROOT="${PIPELINE_ROOT}/${HISTORY_RGB_MODE}"
  mkdir -p "${MODE_ROOT}"
  BASE_EVAL_DIR="${MODE_ROOT}/eval_base_production"
  BASE_AUDIT_EVAL_DIR="${MODE_ROOT}/eval_base_audit_prompt"
  LORA_EVAL_DIR="${MODE_ROOT}/eval_lora_production"
  LORA_AUDIT_EVAL_DIR="${MODE_ROOT}/eval_lora_audit_prompt"

  if [[ "${RUN_BASE_EVAL}" == "1" ]]; then
    run_eval "base production 评测 ${HISTORY_RGB_MODE}" \
      --model-dir "${MODEL_DIR}" \
      --index "${INDEX}" \
      --data-root "${DATA_ROOT}" \
      --history-rgb-mode "${HISTORY_RGB_MODE}" \
      --split "${SPLIT:-test}" \
      --cases-per-bin "${BASE_CASES_PER_BIN:-64}" \
      --max-frames "${BASE_MAX_EVAL_FRAMES:-0}" \
      --max-new-tokens "${BASE_MAX_NEW_TOKENS:-256}" \
      --output-dir "${BASE_EVAL_DIR}" \
      --no-timestamp-output \
      --overwrite
    run_audit_cases "base production 错例 RGB 抽样 ${HISTORY_RGB_MODE}" "${BASE_EVAL_DIR}" "${MODE_ROOT}/audit_base_production"

    if [[ "${RUN_AUDIT_PROMPT_EVAL}" == "1" ]]; then
      run_eval "base audit-prompt 评测 ${HISTORY_RGB_MODE}" \
        --model-dir "${MODEL_DIR}" \
        --index "${INDEX}" \
        --data-root "${DATA_ROOT}" \
        --history-rgb-mode "${HISTORY_RGB_MODE}" \
        --split "${SPLIT:-test}" \
        --cases-per-bin "${BASE_CASES_PER_BIN:-64}" \
        --max-frames "${BASE_MAX_EVAL_FRAMES:-0}" \
        --max-new-tokens "${BASE_MAX_NEW_TOKENS:-256}" \
        --audit-prompt \
        --output-dir "${BASE_AUDIT_EVAL_DIR}" \
        --no-timestamp-output \
        --overwrite
    fi
  fi

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    echo
    echo "========== 训练 LoRA ${HISTORY_RGB_MODE} =========="
    RUN_TAG="${PIPELINE_TIMESTAMP}_${HISTORY_RGB_MODE}" \
    MODEL_DIR="${MODEL_DIR}" \
    INDEX="${INDEX}" \
    DATA_ROOT="${DATA_ROOT}" \
    HISTORY_RGB_MODE="${HISTORY_RGB_MODE}" \
    DDP_GPU_COUNT="${NPROC}" \
    FOCUS_BALANCE_COUNT="${TRAIN_FOCUS_BALANCE_COUNT}" \
    MAX_TRAIN_FRAME_REPEAT="${TRAIN_MAX_FRAME_REPEAT}" \
    NUM_EPOCHS="${TRAIN_NUM_EPOCHS}" \
    MAX_STEPS="${TRAIN_MAX_STEPS}" \
    bash qwen3vl_local/sft_new_loop_phase1/train.sh "${TRAIN_MODE}"
    LORA_ADAPTER_DIR="$(adapter_dir_for_mode "${HISTORY_RGB_MODE}")"
  else
    LORA_ADAPTER_DIR="${ADAPTER_DIR:-$(adapter_dir_for_mode "${HISTORY_RGB_MODE}")}"
  fi

  if [[ "${RUN_LORA_EVAL}" == "1" ]]; then
    if [[ ! -d "${LORA_ADAPTER_DIR}" ]]; then
      echo "找不到 LoRA adapter: ${LORA_ADAPTER_DIR}" >&2
      echo "请设置 ADAPTER_DIR=...，或使用 RUN_TRAIN=1 先训练。" >&2
      exit 1
    fi
    run_eval "LoRA production 评测 ${HISTORY_RGB_MODE}" \
      --model-dir "${MODEL_DIR}" \
      --index "${INDEX}" \
      --data-root "${DATA_ROOT}" \
      --adapter-dir "${LORA_ADAPTER_DIR}" \
      --split "${SPLIT:-test}" \
      --cases-per-bin "${LORA_CASES_PER_BIN:-64}" \
      --max-frames "${LORA_MAX_EVAL_FRAMES:-0}" \
      --max-new-tokens "${LORA_MAX_NEW_TOKENS:-256}" \
      --output-dir "${LORA_EVAL_DIR}" \
      --no-timestamp-output \
      --overwrite
    run_audit_cases "LoRA production 错例 RGB 抽样 ${HISTORY_RGB_MODE}" "${LORA_EVAL_DIR}" "${MODE_ROOT}/audit_lora_production"

    if [[ "${RUN_AUDIT_PROMPT_EVAL}" == "1" ]]; then
      run_eval "LoRA audit-prompt 评测 ${HISTORY_RGB_MODE}" \
        --model-dir "${MODEL_DIR}" \
        --index "${INDEX}" \
        --data-root "${DATA_ROOT}" \
        --adapter-dir "${LORA_ADAPTER_DIR}" \
        --split "${SPLIT:-test}" \
        --cases-per-bin "${LORA_CASES_PER_BIN:-64}" \
        --max-frames "${LORA_MAX_EVAL_FRAMES:-0}" \
        --max-new-tokens "${LORA_MAX_NEW_TOKENS:-256}" \
        --audit-prompt \
        --output-dir "${LORA_AUDIT_EVAL_DIR}" \
        --no-timestamp-output \
        --overwrite
    fi
  fi
done

echo
echo "[done] pipeline 产物: ${PIPELINE_ROOT}"
echo "[done] 训练 run: checkpoints/sft_new_loop_phase1_runs/run_${PIPELINE_TIMESTAMP}_<mode>_combined_phase1_phase2_<mode>"
