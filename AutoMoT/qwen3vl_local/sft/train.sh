#!/usr/bin/env bash
# SFT 训练入口（统一 LoRA 路线，不依赖 ms-swift）。
#
# 关键改动（对比历史 swift launcher）：
#   - 不再调 swift sft；直接 python qwen3vl_local/sft/train.py，
#     LoRA 由 train.py 内部用 peft.LoraConfig + get_peft_model 直接挂到 base 模型。
#   - 不再做离线 teacher 物化 / runtime cache 复用 / manifest 校验。
#     每个 train batch 在 train.py 里现场跑 frozen base 生成 ANALYSIS，不写盘。
#   - 默认数据集是 build_dataset.py 产出的 pending jsonl；train.py 接受 pending
#     直接训。
#
# 用法（**从 AutoMoT/ 目录运行**）：
#   单卡：       GPU_IDS=0 bash qwen3vl_local/sft/train.sh single
#   DDP：        DDP_GPU_COUNT=4 bash qwen3vl_local/sft/train.sh ddp
#                GPU_IDS=0,1,2,3 bash qwen3vl_local/sft/train.sh ddp
#   sanity 自检：GPU_IDS=0 bash qwen3vl_local/sft/train.sh check
#     （check 模式只跑 2 step、不保存 ckpt，用来确认 LoRA 注入和 teacher→student
#      链路是否通畅。初始 loss 应在 3-8 区间。）
#
# 常用 override：
#   MODEL_DIR=/path/to/Qwen3-VL-4B-Instruct \
#   TRAIN_JSONL=/path/to/train.jsonl \
#   VAL_JSONL=/path/to/val.jsonl \
#   OUTPUT_DIR=/path/to/sft_lora \
#   DDP_GPU_COUNT=4 \
#   bash qwen3vl_local/sft/train.sh ddp
#
# 想固定卡：在最前面再加 GPU_IDS=0,1,2,3（DDP_GPU_COUNT 被忽略，卡数从逗号数推断）。
#
# 调权重（不重启 shell，前置 export 即可生效）：
#   SFT_ANALYSIS_WEIGHT=0.5 bash qwen3vl_local/sft/train.sh ddp     # ANALYSIS 漂移
#   SFT_ANALYSIS_WEIGHT=0.1 bash qwen3vl_local/sft/train.sh ddp     # ANALYSIS 过拟合 teacher

set -euo pipefail

MODE="${1:-ddp}"

# ---------------------------------------------------------------------------
# 路径默认值
# ---------------------------------------------------------------------------
MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-checkpoints/sft_data_pending/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-checkpoints/sft_data_pending/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft_lora}"

# 防覆盖目录：OUTPUT_DIR_BASE 之下再套 run_<时间戳> 子目录；base 层维护 latest symlink。
OUTPUT_DIR_BASE="${OUTPUT_DIR}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/run_${RUN_TAG}"
fi

# ---------------------------------------------------------------------------
# 超参（可被 env override）
# ---------------------------------------------------------------------------
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LR="${LR:-3e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-3584}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
EVAL_STEPS="${EVAL_STEPS:-10000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"

# teacher 现场生成参数。
# train.sh 走单一通道：bash 变量 → 下方 CLI flag，不再 export 给 python 进程
# （以前同时 export + 传 CLI 既冗余又容易让人迷惑谁覆盖谁；CLI 是唯一权威）。
# 如果你想 bypass shell 直接 python 调，train.py 仍然支持 SFT_* env 作为 argparse default。
SFT_TEACHER_MAX_NEW_TOKENS="${SFT_TEACHER_MAX_NEW_TOKENS:-256}"
SFT_TEACHER_TEMPERATURE="${SFT_TEACHER_TEMPERATURE:-0.0}"
SFT_ANALYSIS_WEIGHT="${SFT_ANALYSIS_WEIGHT:-0.3}"
SKIP_TEACHER="${SKIP_TEACHER:-0}"

# HuggingFace 强制离线
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 缓存指向 base 层
export HF_HOME="${HF_HOME:-${OUTPUT_DIR_BASE}/.hf_cache}"
mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"

# 先开 tee 再做 symlink，让 [run] 行也进 log。
if [[ "${QWEN3VL_LOG_TO_FILE:-1}" != "0" && -z "${QWEN3VL_LOG_ACTIVE:-}" ]]; then
    export QWEN3VL_LOG_ACTIVE=1
    exec > >(tee -a "${OUTPUT_DIR}/log.txt") 2>&1
    echo "[log] tee stdout/stderr to ${OUTPUT_DIR}/log.txt"
fi
if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
    ln -sfn "run_${RUN_TAG}" "${OUTPUT_DIR_BASE}/latest"
    echo "[run] OUTPUT_DIR=${OUTPUT_DIR}  (latest -> run_${RUN_TAG})"
fi

# ---------------------------------------------------------------------------
# GPU / MASTER_PORT 选址
# ---------------------------------------------------------------------------
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
    if [[ "${want_count}" -le 1 ]]; then
        echo "0"
    else
        seq -s, 0 "$((want_count - 1))"
    fi
}

count_visible_gpus() {
    local visible="$1"
    [[ -z "${visible}" ]] && { echo "0"; return; }
    awk -F',' '{print NF}' <<< "${visible}"
}

resolve_visible_gpus() {
    local want_count="$1"
    if [[ -n "${GPU_IDS:-}" ]]; then
        echo "${GPU_IDS}"
    else
        pick_idle_gpus "${want_count}"
    fi
}

find_free_master_port() {
    python -c 'import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("", 0))
print(sock.getsockname()[1])
sock.close()
' 2>/dev/null || echo "$((20000 + RANDOM % 20000))"
}

configure_master_port() {
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-$(find_free_master_port)}"
}

# ---------------------------------------------------------------------------
# 模式分支
# ---------------------------------------------------------------------------
EXTRA_ARGS=()
case "${MODE}" in
    single)
        echo "[mode] single-GPU"
        export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus 1)"
        NPROC=1
        ;;
    check)
        echo "[mode] check (2 steps only — no checkpoint, no eval)"
        export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus 1)"
        NPROC=1
        PER_DEVICE_BS=1
        GRAD_ACCUM=1
        EXTRA_ARGS+=("--check")
        ;;
    sanity)
        echo "[mode] sanity (2 steps, skip teacher generate — student/DDP 链路自检)"
        export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus 1)"
        NPROC=1
        PER_DEVICE_BS=1
        GRAD_ACCUM=1
        EXTRA_ARGS+=("--check" "--skip-teacher")
        ;;
    ddp)
        echo "[mode] DDP"
        DDP_GPU_COUNT="${DDP_GPU_COUNT:-8}"
        if [[ -n "${GPU_IDS:-}" ]]; then
            echo "[gpu] GPU_IDS=${GPU_IDS} takes precedence; DDP_GPU_COUNT=${DDP_GPU_COUNT} ignored"
        fi
        export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus "${DDP_GPU_COUNT}")"
        ACTUAL="$(count_visible_gpus "${CUDA_VISIBLE_DEVICES}")"
        NPROC="${ACTUAL}"
        if [[ -z "${GPU_IDS:-}" && "${ACTUAL}" -lt "${DDP_GPU_COUNT}" ]]; then
            echo "[gpu][error] requested DDP_GPU_COUNT=${DDP_GPU_COUNT}, but only got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
            exit 1
        fi
        configure_master_port
        export NCCL_P2P_LEVEL=NVL
        export NCCL_DEBUG=WARN
        ;;
    *)
        echo "Unknown mode: ${MODE}. Use 'single' / 'ddp' / 'check' / 'sanity'." >&2
        exit 1
        ;;
esac

# 通用 skip-teacher 入口：任何 mode 都可以前置 SKIP_TEACHER=1 强制跳过 teacher.generate。
if [[ "${SKIP_TEACHER}" == "1" ]]; then
    echo "[skip-teacher] teacher.generate 全部走 fallback；训练出的 LoRA 不可用于生产。"
    EXTRA_ARGS+=("--skip-teacher")
fi

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[gpu] NPROC=${NPROC}"
if [[ "${MODE}" == "ddp" ]]; then
    echo "[ddp] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
fi

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
    --logging-steps "${LOGGING_STEPS}"
    --save-steps "${SAVE_STEPS}"
    --eval-steps "${EVAL_STEPS}"
    --save-total-limit "${SAVE_TOTAL_LIMIT}"
    --max-eval-samples "${MAX_EVAL_SAMPLES}"
    --analysis-weight "${SFT_ANALYSIS_WEIGHT}"
    --teacher-max-new-tokens "${SFT_TEACHER_MAX_NEW_TOKENS}"
    --teacher-temperature "${SFT_TEACHER_TEMPERATURE}"
    "${EXTRA_ARGS[@]}"
)

if [[ "${NPROC}" -gt 1 ]]; then
    torchrun --nproc_per_node="${NPROC}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        qwen3vl_local/sft/train.py "${PY_ARGS[@]}"
else
    python qwen3vl_local/sft/train.py "${PY_ARGS[@]}"
fi

echo "[done] LoRA adapter saved under ${OUTPUT_DIR}"
echo ""
echo "============================================================"
echo "[hint] 看 TensorBoard："
echo "  bash qwen3vl_local/tb_serve.sh ${OUTPUT_DIR}"
echo ""
echo "[hint] 跑 eval："
echo "  GPU_IDS=0 python qwen3vl_local/sft/eval.py --lora-dir ${OUTPUT_DIR}/final \\"
echo "      --val-jsonl ${VAL_JSONL} --save-root ${OUTPUT_DIR}"
echo ""
echo "[hint] 跑 probe："
echo "  GPU_IDS=0 python qwen3vl_local/sft/probe.py --lora-dir ${OUTPUT_DIR}/final \\"
echo "      --val-jsonl ${VAL_JSONL} --save-root ${OUTPUT_DIR} --num-per-scenario 4"
echo "============================================================"
