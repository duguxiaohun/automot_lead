#!/usr/bin/env bash
# sft_loop_phase2_augment 训练 launcher：三类随机 RS 问法 + 可选 torch DDP。
#
# 从 AutoMoT/ 目录运行：
#   GPU_IDS=0 bash qwen3vl_local/sft_loop_phase2_augment/train.sh single
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase2_augment/train.sh ddp
# 默认使用四帧；HISTORY_RGB_MODE=2rgb_endpoints 时只喂第 1 帧和第 4 帧。

set -euo pipefail

ulimit -S -c 0 2>/dev/null || true

MODE="${1:-${MODE:-single}}"
if [[ "${MODE}" != "single" && "${MODE}" != "ddp" && "${MODE}" != "check" ]]; then
  echo "Unknown mode: ${MODE}. Use single/ddp/check." >&2
  exit 1
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
INDEX="${INDEX:-checkpoints/sft_loop_phase2_augment_data/frame_index.jsonl}"
HISTORY_RGB_MODE="${HISTORY_RGB_MODE:-4rgb}"
case "${HISTORY_RGB_MODE}" in
  4rgb|2rgb_endpoints) HISTORY_RGB_TAG="${HISTORY_RGB_MODE}" ;;
  *)
    echo "Unknown HISTORY_RGB_MODE=${HISTORY_RGB_MODE}. Use 4rgb or 2rgb_endpoints." >&2
    exit 1
    ;;
esac
OUTPUT_DIR_BASE="checkpoints/sft_loop_phase2_augment_runs"
FINAL_RUN_NAME="run_rs_augmented_format_supervised_${HISTORY_RGB_TAG}"
CHECK_RUN_NAME="check_rs_augmented_format_supervised_${HISTORY_RGB_TAG}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
FINAL_OUTPUT_DIR="${OUTPUT_DIR_BASE}/${FINAL_RUN_NAME}/${RUN_TIMESTAMP}"
CHECK_OUTPUT_DIR="${OUTPUT_DIR_BASE}/${CHECK_RUN_NAME}/${RUN_TIMESTAMP}"
if [[ "${MODE}" == "check" ]]; then
  RUN_NAME="${CHECK_RUN_NAME}"
  OUTPUT_DIR="${OUTPUT_DIR:-${CHECK_OUTPUT_DIR}}"
else
  RUN_NAME="${FINAL_RUN_NAME}"
  OUTPUT_DIR="${OUTPUT_DIR:-${FINAL_OUTPUT_DIR}}"
fi
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR_BASE}"
LATEST_TARGET="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${OUTPUT_DIR}")"
mkdir -p "${OUTPUT_DIR_BASE}/${RUN_NAME}"
ln -sfn "${LATEST_TARGET}" "${OUTPUT_DIR_BASE}/${RUN_NAME}/latest"
if [[ "${MODE}" == "check" ]]; then
  ln -sfn "${LATEST_TARGET}" "${OUTPUT_DIR_BASE}/latest_check"
else
  ln -sfn "${LATEST_TARGET}" "${OUTPUT_DIR_BASE}/latest"
fi
if [[ "${RUN_LOG:-}" != "0" ]]; then
  RUN_LOG="${RUN_LOG:-${OUTPUT_DIR}/train.log}"
  export RUN_LOG
  mkdir -p "$(dirname "${RUN_LOG}")"
  # 脚本内部 tee 一份日志，避免外层漏写 tee 时找不到训练记录。
  exec > >(tee -a "${RUN_LOG}") 2>&1
fi
export RUN_NAME RUN_TIMESTAMP

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

resolve_visible_gpus() {
  local want_count="$1"
  if [[ -n "${GPU_IDS:-}" ]]; then echo "${GPU_IDS}"; else pick_idle_gpus "${want_count}"; fi
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

EXTRA_ARGS=()
NPROC=1
case "${MODE}" in
  single)
    export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus 1)"
    NPROC=1
    ;;
  check)
    export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus 1)"
    NPROC=1
    EXTRA_ARGS+=(
      # 严格八桶均衡必须先看到每个桶；只读 index 的前 64 行会因文件顺序漏桶。
      --max-frames "${CHECK_MAX_FRAMES:-0}"
      --max-steps "${CHECK_MAX_STEPS:-2}"
      --focus-balance-count "${CHECK_FOCUS_BALANCE_COUNT:-2}"
      --eval-steps 0
      --generation-eval-steps 0
      --save-steps 0
      --no-tb
    )
    ;;
  ddp)
    DDP_GPU_COUNT="${DDP_GPU_COUNT:-${NPROC_PER_NODE:-4}}"
    export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus "${DDP_GPU_COUNT}")"
    NPROC="$(count_visible_gpus "${CUDA_VISIBLE_DEVICES}")"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-$(find_free_master_port)}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
    ;;
esac

COMMON_ARGS=(
  --model-dir "${MODEL_DIR}"
  --index "${INDEX}"
  --output-dir "${OUTPUT_DIR}"
  --history-rgb-mode "${HISTORY_RGB_MODE}"
  --num-epochs "${NUM_EPOCHS:-3}"
  --max-frames "${MAX_FRAMES:-0}"
  --max-steps "${MAX_STEPS:-0}"
  --focus-balance-count "${FOCUS_BALANCE_COUNT:-1024}"
  --eval-split "${EVAL_SPLIT:-val}"
  --eval-steps "${EVAL_STEPS:-2000}"
  --eval-balance-count "${EVAL_BALANCE_COUNT:-16}"
  --max-eval-frames "${MAX_EVAL_FRAMES:-0}"
  --format-loss-weight "${FORMAT_LOSS_WEIGHT:-0.25}"
  --generation-eval-steps "${GENERATION_EVAL_STEPS:-2000}"
  --generation-eval-balance-count "${GENERATION_EVAL_BALANCE_COUNT:-2}"
  --generation-eval-max-new-tokens "${GENERATION_EVAL_MAX_NEW_TOKENS:-64}"
  --generation-eval-min-valid-rate "${GENERATION_EVAL_MIN_VALID_RATE:-1.0}"
  --save-steps "${SAVE_STEPS:-20000}"
  --max-length "${MAX_LENGTH:-8192}"
  --learning-rate "${LEARNING_RATE:-${LR:-1e-5}}"
  --grad-accum "${GRAD_ACCUM:-1}"
  --weight-decay "${WEIGHT_DECAY:-0.0}"
  --warmup-steps "${WARMUP_STEPS:-2000}"
  --lora-rank "${LORA_RANK:-16}"
  --lora-alpha "${LORA_ALPHA:-32}"
  --lora-dropout "${LORA_DROPOUT:-0.05}"
  --lora-vision-scope "${LORA_VISION_SCOPE:-off}"
  --max-grad-norm "${MAX_GRAD_NORM:-1.0}"
  --log-steps "${LOG_STEPS:-10}"
  "${EXTRA_ARGS[@]}"
)
if [[ "${SAVE_BEST_VAL:-1}" == "0" ]]; then
  COMMON_ARGS+=(--no-save-best-val)
else
  COMMON_ARGS+=(--save-best-val)
fi
if [[ "${SAVE_BEST_GENERATION:-1}" == "0" ]]; then
  COMMON_ARGS+=(--no-save-best-generation)
else
  COMMON_ARGS+=(--save-best-generation)
fi

echo "[run] MODE=${MODE} HISTORY_RGB_MODE=${HISTORY_RGB_MODE} OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run] RUN_NAME=${RUN_NAME} RUN_TIMESTAMP=${RUN_TIMESTAMP}"
echo "[run] TB_DIR=${OUTPUT_DIR}/tb"
echo "[run] LATEST=${OUTPUT_DIR_BASE}/$([[ "${MODE}" == "check" ]] && echo latest_check || echo latest)"
if [[ "${RUN_LOG:-0}" != "0" ]]; then
  echo "[run] RUN_LOG=${RUN_LOG}"
fi
echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[gpu] NPROC=${NPROC}"

if [[ "${NPROC}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    qwen3vl_local/sft_loop_phase2_augment/train.py "${COMMON_ARGS[@]}"
else
  python qwen3vl_local/sft_loop_phase2_augment/train.py "${COMMON_ARGS[@]}"
fi

echo "[hint] eval base: GPU_IDS=0 python qwen3vl_local/sft_loop_phase2_augment/eval.py"
echo "[hint] eval lora: GPU_IDS=0 python qwen3vl_local/sft_loop_phase2_augment/eval.py --adapter-dir ${OUTPUT_DIR}"
