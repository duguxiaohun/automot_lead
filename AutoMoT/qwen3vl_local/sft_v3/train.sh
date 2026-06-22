#!/usr/bin/env bash
# SFT v3 launcher: sequence-memory OPD training.
#
# Run from AutoMoT/:
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v3/train.sh ddp
#   GPU_IDS=0 bash qwen3vl_local/sft_v3/train.sh single
#   GPU_IDS=0 bash qwen3vl_local/sft_v3/train.sh check

set -euo pipefail

# Disable core dumps so failed tool processes do not leave core.* files.
ulimit -S -c 0 2>/dev/null || true

MODE="${1:-ddp}"

# 路径默认都以远端 AutoMoT/ 为当前目录；不要在这里加 AutoMoT/ 前缀。
MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-checkpoints/sft_v3_data/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-checkpoints/sft_v3_data/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft_v3_lora}"

# 防覆盖目录约定：用户给的是 base OUTPUT_DIR，真实产物落到 run_<RUN_TAG>/。
OUTPUT_DIR_BASE="${OUTPUT_DIR}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/run_${RUN_TAG}"
fi

NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR="${LR:-3e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
OUTER_STRIDE="${OUTER_STRIDE:-1}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_VISION_SCOPE="${LORA_VISION_SCOPE:-off}"
LORA_VISION="${LORA_VISION:-0}"
VISION_LR_SCALE="${VISION_LR_SCALE:-0.1}"
MAX_VISION_LR_SCALE="${MAX_VISION_LR_SCALE:-0.25}"
VISION_CLIP_NORM="${VISION_CLIP_NORM:-0.3}"
LANGUAGE_CLIP_NORM="${LANGUAGE_CLIP_NORM:-1.0}"
STRICT_VISION_SCOPE="${STRICT_VISION_SCOPE:-1}"
VISION_GUARD_ENABLED="${VISION_GUARD_ENABLED:-1}"
VISION_GUARD_GRAD_NORM_MAX="${VISION_GUARD_GRAD_NORM_MAX:-10.0}"
VISION_GUARD_PARAM_NORM_MAX="${VISION_GUARD_PARAM_NORM_MAX:-200.0}"
VISION_GUARD_PATIENCE="${VISION_GUARD_PATIENCE:-3}"
W_A1="${W_A1:-0.2}"
W_A2="${W_A2:-0.2}"
W_A3="${W_A3:-0.2}"
W_S2="${W_S2:-1.0}"
W_S3_STATUS="${W_S3_STATUS:-1.0}"
W_S3_SUBGOAL="${W_S3_SUBGOAL:-1.0}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
# DDP 下 train.py 会拒绝 EVAL_STEPS>0；完整自由生成评估请训练后单独跑 eval.py。
EVAL_STEPS="${EVAL_STEPS:-0}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
MAX_EVAL_EPISODES="${MAX_EVAL_EPISODES:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
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
    # 默认按显存占用和 GPU util 选最空闲卡；GPU_IDS 非空时不会走这里。
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

resolve_visible_gpus() {
    # 项目统一显式 pin 写法：GPU_IDS=0 或 GPU_IDS=0,1,2,3。
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
        echo "[mode] single"
        export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus 1)"
        NPROC=1
        ;;
    check)
        # check 模式只跑极少 step，用来验证数据/模型/LoRA 链路能不能启动。
        echo "[mode] check"
        export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus 1)"
        PER_DEVICE_BS=1
        GRAD_ACCUM=1
        EXTRA_ARGS+=("--check")
        NPROC=1
        ;;
    ddp)
        # DDP_GPU_COUNT 只表示需要几张空闲卡；具体卡号默认自动挑选。
        echo "[mode] ddp"
        DDP_GPU_COUNT="${DDP_GPU_COUNT:-8}"
        if [[ -n "${GPU_IDS:-}" ]]; then
            echo "[gpu] GPU_IDS=${GPU_IDS} takes precedence; DDP_GPU_COUNT=${DDP_GPU_COUNT} ignored"
        fi
        export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus "${DDP_GPU_COUNT}")"
        NPROC="$(count_visible_gpus "${CUDA_VISIBLE_DEVICES}")"
        if [[ -z "${GPU_IDS:-}" && "${NPROC}" -lt "${DDP_GPU_COUNT}" ]]; then
            echo "[gpu][error] requested ${DDP_GPU_COUNT}, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
            exit 1
        fi
        export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
        export MASTER_PORT="${MASTER_PORT:-$(find_free_master_port)}"
        export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
        export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
        ;;
    *)
        echo "Unknown mode: ${MODE}. Use single/ddp/check." >&2
        exit 1
        ;;
esac

if [[ "${STRICT_VISION_SCOPE}" == "1" ]]; then
    EXTRA_ARGS+=("--strict-vision-scope")
else
    EXTRA_ARGS+=("--no-strict-vision-scope")
fi

if [[ "${VISION_GUARD_ENABLED}" == "1" ]]; then
    EXTRA_ARGS+=("--vision-guard-enabled")
else
    EXTRA_ARGS+=("--no-vision-guard-enabled")
fi

if [[ "${LORA_VISION}" == "1" ]]; then
    EXTRA_ARGS+=("--lora-vision")
fi

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[gpu] NPROC=${NPROC}"

PY_ARGS=(
    --train-jsonl "${TRAIN_JSONL}" \
    --val-jsonl "${VAL_JSONL}" \
    --model-dir "${MODEL_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-epochs "${NUM_EPOCHS}" \
    --per-device-batch-size "${PER_DEVICE_BS}" \
    --grad-accum "${GRAD_ACCUM}" \
    --learning-rate "${LR}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --max-length "${MAX_LENGTH}" \
    --outer-stride "${OUTER_STRIDE}" \
    --lora-rank "${LORA_RANK}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-dropout "${LORA_DROPOUT}" \
    --lora-vision-scope "${LORA_VISION_SCOPE}" \
    --vision-lr-scale "${VISION_LR_SCALE}" \
    --max-vision-lr-scale "${MAX_VISION_LR_SCALE}" \
    --language-clip-norm "${LANGUAGE_CLIP_NORM}" \
    --vision-clip-norm "${VISION_CLIP_NORM}" \
    --vision-guard-grad-norm-max "${VISION_GUARD_GRAD_NORM_MAX}" \
    --vision-guard-param-norm-max "${VISION_GUARD_PARAM_NORM_MAX}" \
    --vision-guard-patience "${VISION_GUARD_PATIENCE}" \
    --w-a1 "${W_A1}" \
    --w-a2 "${W_A2}" \
    --w-a3 "${W_A3}" \
    --w-s2 "${W_S2}" \
    --w-s3-status "${W_S3_STATUS}" \
    --w-s3-subgoal "${W_S3_SUBGOAL}" \
    --logging-steps "${LOGGING_STEPS}" \
    --save-steps "${SAVE_STEPS}" \
    --eval-steps "${EVAL_STEPS}" \
    --save-total-limit "${SAVE_TOTAL_LIMIT}" \
    --max-eval-episodes "${MAX_EVAL_EPISODES}" \
    "${EXTRA_ARGS[@]}"
)

if [[ "${NPROC}" -gt 1 ]]; then
    torchrun --nproc_per_node="${NPROC}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        qwen3vl_local/sft_v3/train.py "${PY_ARGS[@]}"
else
    python qwen3vl_local/sft_v3/train.py "${PY_ARGS[@]}"
fi

echo "[done] adapter under ${OUTPUT_DIR}"
echo "[hint] eval: GPU_IDS=0 python qwen3vl_local/sft_v3/eval.py --lora-dir ${OUTPUT_DIR}/final --save-root ${OUTPUT_DIR}"
