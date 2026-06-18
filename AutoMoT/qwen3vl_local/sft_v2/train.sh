#!/usr/bin/env bash
# SFT v2 启动器：串行 SCENE -> STATUS/SUBGOAL LoRA 训练。
#
# 在 AutoMoT/ 目录下运行：
#   GPU_IDS=0 bash qwen3vl_local/sft_v2/train.sh single
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v2/train.sh ddp

set -euo pipefail

MODE="${1:-ddp}"

MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-checkpoints/sft_v2_data/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-checkpoints/sft_v2_data/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft_v2_lora}"

OUTPUT_DIR_BASE="${OUTPUT_DIR}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/run_${RUN_TAG}"
fi

NUM_EPOCHS="${NUM_EPOCHS:-2}"
LR="${LR:-3e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_VISION="${LORA_VISION:-0}"
LABEL_WEIGHT="${LABEL_WEIGHT:-1.0}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
EVAL_STEPS="${EVAL_STEPS:-10000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_HOME:-${OUTPUT_DIR_BASE}/.hf_cache}"
mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"

if [[ "${QWEN3VL_LOG_TO_FILE:-1}" != "0" && -z "${QWEN3VL_LOG_ACTIVE:-}" ]]; then
    export QWEN3VL_LOG_ACTIVE=1
    exec > >(tee -a "${OUTPUT_DIR}/log.txt") 2>&1
    echo "[log] tee stdout/stderr to ${OUTPUT_DIR}/log.txt"
fi
if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
    ln -sfn "run_${RUN_TAG}" "${OUTPUT_DIR_BASE}/latest"
    echo "[run] OUTPUT_DIR=${OUTPUT_DIR}  (latest -> run_${RUN_TAG})"
fi

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
        if [[ -n "${selected}" ]]; then
            echo "${selected}"
            return 0
        fi
    fi
    if [[ "${want_count}" -le 1 ]]; then echo "0"; else seq -s, 0 "$((want_count - 1))"; fi
}

count_visible_gpus() {
    local visible="$1"
    [[ -z "${visible}" ]] && { echo "0"; return; }
    awk -F',' '{print NF}' <<< "${visible}"
}

resolve_visible_gpus() {
    local want_count="$1"
    if [[ -n "${GPU_IDS:-}" ]]; then echo "${GPU_IDS}"; else pick_idle_gpus "${want_count}"; fi
}

find_free_master_port() {
    python -c 'import socket
s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null || echo "$((20000 + RANDOM % 20000))"
}

EXTRA_ARGS=()
case "${MODE}" in
    single)
        echo "[mode] single"
        export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus 1)"
        NPROC=1
        ;;
    check)
        echo "[mode] check"
        export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus 1)"
        NPROC=1
        PER_DEVICE_BS=1
        GRAD_ACCUM=1
        EXTRA_ARGS+=("--check")
        ;;
    ddp)
        echo "[mode] ddp"
        DDP_GPU_COUNT="${DDP_GPU_COUNT:-8}"
        if [[ -n "${GPU_IDS:-}" ]]; then
            echo "[gpu] GPU_IDS=${GPU_IDS} takes precedence; DDP_GPU_COUNT=${DDP_GPU_COUNT} ignored"
        fi
        export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus "${DDP_GPU_COUNT}")"
        ACTUAL="$(count_visible_gpus "${CUDA_VISIBLE_DEVICES}")"
        NPROC="${ACTUAL}"
        if [[ -z "${GPU_IDS:-}" && "${ACTUAL}" -lt "${DDP_GPU_COUNT}" ]]; then
            echo "[gpu][error] requested ${DDP_GPU_COUNT}, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
            exit 1
        fi
        export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
        export MASTER_PORT="${MASTER_PORT:-$(find_free_master_port)}"
        export NCCL_P2P_LEVEL=NVL
        export NCCL_DEBUG=WARN
        ;;
    *)
        echo "Unknown mode: ${MODE}. Use single/ddp/check." >&2
        exit 1
        ;;
esac

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[gpu] NPROC=${NPROC}"

PY_ARGS=(
    --train-jsonl "${TRAIN_JSONL}"
    --val-jsonl "${VAL_JSONL}"
    --model-dir "${MODEL_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --num-epochs "${NUM_EPOCHS}"
    --per-device-batch-size "${PER_DEVICE_BS}"
    --grad-accum "${GRAD_ACCUM}"
    --learning-rate "${LR}"
    --warmup-ratio "${WARMUP_RATIO}"
    --weight-decay "${WEIGHT_DECAY}"
    --max-length "${MAX_LENGTH}"
    --lora-rank "${LORA_RANK}"
    --lora-alpha "${LORA_ALPHA}"
    --lora-dropout "${LORA_DROPOUT}"
    --label-weight "${LABEL_WEIGHT}"
    --logging-steps "${LOGGING_STEPS}"
    --save-steps "${SAVE_STEPS}"
    --eval-steps "${EVAL_STEPS}"
    --save-total-limit "${SAVE_TOTAL_LIMIT}"
    --max-eval-samples "${MAX_EVAL_SAMPLES}"
    "${EXTRA_ARGS[@]}"
)

if [[ "${LORA_VISION}" == "1" ]]; then
    PY_ARGS+=("--lora-vision")
fi

if [[ "${NPROC}" -gt 1 ]]; then
    torchrun --nproc_per_node="${NPROC}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        qwen3vl_local/sft_v2/train.py "${PY_ARGS[@]}"
else
    python qwen3vl_local/sft_v2/train.py "${PY_ARGS[@]}"
fi

echo "[done] adapter under ${OUTPUT_DIR}"
echo "[hint] GPU_IDS=0 python qwen3vl_local/sft_v2/eval.py --lora-dir ${OUTPUT_DIR}/final --save-root ${OUTPUT_DIR}"
