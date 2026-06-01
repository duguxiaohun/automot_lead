#!/usr/bin/env bash
# SFT v2 训练入口 — ms-swift LoRA on Qwen3-VL-4B-Instruct，ANALYSIS 段带蒸馏监督。
#
# 与 sft_v1_train.sh 的核心区别：
#   - --loss_scale 改用 sft_v2_analysis_supervised（v2 plugin，ANALYSIS body 权重 0.3）；
#   - 默认读 sft_v2_data_pending/，训练启动时临时物化 teacher ANALYSIS 到 OUTPUT_DIR/runtime_teacher_data/；
#   - LR 5e-5 → 3e-5（v2 监督 token 数 × 5，lr 同步下调避免过冲，详见 SFT_V2_PLAN.md §6）；
#   - MAX_LENGTH 3072 → 3584（v2 ANALYSIS 段更长）。
#
# GPU / MASTER_PORT 自动选址逻辑与 v1 完全相同，所有 SFT_RESPECT_* 环境变量同名。
#
# 用法（**从 AutoMoT/ 目录运行**）：
#   单卡：       bash tools/sft_v2_train.sh single
#   DDP：        bash tools/sft_v2_train.sh ddp
#   sanity 自检：bash tools/sft_v2_train.sh check
#     （check 模式只跑 2 step、不保存 ckpt，用来确认 loss_scale 是否生效。
#      v2 初始 loss 应在 3-8 区间，比 v1 偏高，因为多了 ANALYSIS 段约 30 个 token
#      参与 loss；判读细节见 SFT_V2_RUN.md §4。）
#
# 数据先用 tools/build_sft_dataset_v1.py --mode v2 生成 pending jsonl。
# 训练脚本默认在运行时调用冻结 teacher 生成临时 ANALYSIS 真值，不把 teacher 文本写回 pending 数据集。
#
# 常用 override：
#   MODEL_DIR=/path/to/Qwen3-VL-4B-Instruct \
#   TRAIN_JSONL=/path/to/v2_train.jsonl \
#   VAL_JSONL=/path/to/v2_val.jsonl \
#   OUTPUT_DIR=/path/to/sft_v2_lora \
#   DDP_GPU_COUNT=4 \
#   bash tools/sft_v2_train.sh ddp
#
# 调权重（不重启训练前 export 即可生效，详见 tools/sft_v2_loss_scale_plugin.py docstring）：
#   SFT_V2_ANALYSIS_WEIGHT=0.5 bash tools/sft_v2_train.sh ddp     # ANALYSIS 还在漂移时
#   SFT_V2_ANALYSIS_WEIGHT=0.1 bash tools/sft_v2_train.sh ddp     # ANALYSIS 过拟合 teacher 时

set -euo pipefail

MODE="${1:-ddp}"
DDP_GPU_COUNT_WAS_SET=0
if [[ -n "${DDP_GPU_COUNT+x}" ]]; then
    DDP_GPU_COUNT_WAS_SET=1
fi

# ---------------------------------------------------------------------------
# 路径默认值（v2 专用）
# ---------------------------------------------------------------------------
MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-checkpoints/sft_v2_data_pending/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-checkpoints/sft_v2_data_pending/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft_v2_lora}"
RUNTIME_TEACHER_DIR="${RUNTIME_TEACHER_DIR:-${OUTPUT_DIR}/runtime_teacher_data}"
RUNTIME_TEACHER_SEED="${RUNTIME_TEACHER_SEED:-20260601}"
RUNTIME_TEACHER_REFRESH="${RUNTIME_TEACHER_REFRESH:-1}"

# ---------------------------------------------------------------------------
# 超参（v2 调整版，详见 SFT_V2_PLAN.md §6）
# ---------------------------------------------------------------------------
# v2 监督信号从 v1 的 ~6 token 升到 ~30 token（多了 ANALYSIS body），
# 等效"有效梯度量"提高约 5 倍 → lr 同步下调一档（5e-5 → 3e-5）防过冲。
# MAX_LENGTH 从 3072 抬到 3584：teacher ANALYSIS body 约 80-120 token，
# 加上 system + user + 4 张图视觉 token，预留余量避免触发 truncation warning。
# 其它超参（LORA_RANK / dropout / weight_decay / warmup）沿用 v1。
NUM_EPOCHS=2
LR=3e-5
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.05
MAX_LENGTH=3584
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.1
LOGGING_STEPS=5

# v2 plugin：ANALYSIS body 权重 0.3、STATUS/SUBGOAL event_name 1.0、其它 0。
LOSS_SCALE="sft_v2_analysis_supervised"
LOSS_SCALE_PLUGIN="tools/sft_v2_loss_scale_plugin.py"

# HuggingFace 强制离线（与 v1 同）。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 缓存指向 output_dir 内，避免误读 ~/.cache。
export HF_HOME="${HF_HOME:-${OUTPUT_DIR}/.hf_cache}"
mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"

# ---------------------------------------------------------------------------
# GPU / MASTER_PORT 选址（与 v1 sft_v1_train.sh 完全相同的函数，复制保持解耦）
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
    if [[ -z "${visible}" ]]; then
        echo "0"
    else
        awk -F',' '{print NF}' <<< "${visible}"
    fi
}

is_port_free() {
    local port="$1"
    python -c 'import socket, sys
port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
' "${port}" >/dev/null 2>&1
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

    if [[ "${SFT_RESPECT_MASTER_PORT:-0}" == "1" ]]; then
        export MASTER_PORT="${MASTER_PORT:-29500}"
        return 0
    fi

    if [[ -n "${MASTER_PORT:-}" ]]; then
        if is_port_free "${MASTER_PORT}"; then
            export MASTER_PORT
            return 0
        fi
        echo "[ddp][warn] MASTER_PORT=${MASTER_PORT} is already in use; selecting a free port"
    fi

    export MASTER_PORT="$(find_free_master_port)"
}

jsonl_dataset_version() {
    local path="$1"
    python -c 'import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            print(json.loads(line).get("dataset_version", "v1"))
            break
' "${path}"
}

materialize_runtime_teacher_if_needed() {
    local train_version
    train_version="$(jsonl_dataset_version "${TRAIN_JSONL}")"
    if [[ "${train_version}" != "v2_pending" ]]; then
        echo "[teacher] TRAIN_JSONL is ${train_version}; use as-is: ${TRAIN_JSONL}"
        return 0
    fi

    local pending_dir
    pending_dir="$(dirname "${TRAIN_JSONL}")"
    local pending_val
    pending_val="${pending_dir}/val.jsonl"
    if [[ "${VAL_JSONL}" != "${pending_val}" ]]; then
        echo "[teacher][warn] pending train/val should live in one dir; override VAL_JSONL=${pending_val}"
        VAL_JSONL="${pending_val}"
    fi

    mkdir -p "${RUNTIME_TEACHER_DIR}"
    if [[ "${RUNTIME_TEACHER_REFRESH}" == "1" ]]; then
        if [[ -z "${RUNTIME_TEACHER_DIR}" || "${RUNTIME_TEACHER_DIR}" == "/" || "${RUNTIME_TEACHER_DIR}" == "." ]]; then
            echo "[teacher][err] unsafe RUNTIME_TEACHER_DIR=${RUNTIME_TEACHER_DIR}" >&2
            exit 2
        fi
        echo "[teacher] refresh runtime cache because RUNTIME_TEACHER_REFRESH=1"
        rm -f \
            "${RUNTIME_TEACHER_DIR}/train.jsonl" \
            "${RUNTIME_TEACHER_DIR}/val.jsonl" \
            "${RUNTIME_TEACHER_DIR}/stats.json" \
            "${RUNTIME_TEACHER_DIR}"/train.jsonl.rank* \
            "${RUNTIME_TEACHER_DIR}"/val.jsonl.rank*
    else
        echo "[teacher] keep existing runtime cache because RUNTIME_TEACHER_REFRESH=${RUNTIME_TEACHER_REFRESH}"
    fi
    echo "[teacher] runtime materialize teacher ANALYSIS"
    echo "[teacher] pending_dir=${pending_dir}"
    echo "[teacher] output_dir=${RUNTIME_TEACHER_DIR}"
    echo "[teacher] source pending jsonl is not modified"

    local teacher_args=(
        --pending-dir "${pending_dir}"
        --output-dir "${RUNTIME_TEACHER_DIR}"
        --model-dir "${MODEL_DIR}"
        --seed "${RUNTIME_TEACHER_SEED}"
    )
    if [[ "${MODE}" == "check" ]]; then
        teacher_args+=(--max-samples "${RUNTIME_TEACHER_MAX_SAMPLES:-32}")
    elif [[ -n "${RUNTIME_TEACHER_MAX_SAMPLES:-}" && "${RUNTIME_TEACHER_MAX_SAMPLES}" != "0" ]]; then
        teacher_args+=(--max-samples "${RUNTIME_TEACHER_MAX_SAMPLES}")
    fi

    if [[ "${MODE}" == "ddp" && "${NPROC_PER_NODE}" -gt 1 ]]; then
        torchrun --nproc_per_node="${NPROC_PER_NODE}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${MASTER_PORT}" \
            tools/build_sft_dataset_v2_teacher.py \
            "${teacher_args[@]}"
    else
        python tools/build_sft_dataset_v2_teacher.py "${teacher_args[@]}"
    fi

    TRAIN_JSONL="${RUNTIME_TEACHER_DIR}/train.jsonl"
    VAL_JSONL="${RUNTIME_TEACHER_DIR}/val.jsonl"
    echo "[teacher] runtime train=${TRAIN_JSONL}"
    echo "[teacher] runtime val=${VAL_JSONL}"
}

# ---------------------------------------------------------------------------
# 模式分支
# ---------------------------------------------------------------------------
case "${MODE}" in
    single)
        echo "[mode] single-GPU"
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(pick_idle_gpus 1)}"
        export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
        PER_DEVICE_BS=4
        GRAD_ACC=2
        SAVE_STRATEGY="epoch"
        EVAL_STRATEGY="epoch"
        EXTRA_LAUNCH=""
        ;;
    check)
        echo "[mode] check (v2 loss_scale sanity, 2 steps only — no checkpoint, no eval)"
        # v2 健康初始 loss 区间：3-8。
        # 比 v1（6-10）偏低，因为 v2 ANALYSIS body 段加 0.3 权重后单 token loss 被稀释；
        # 但比 v1 mask=0 时多了 ~30 个 token 参与 loss，整体仍在可读量级。
        # 判读细节见 SFT_V2_RUN.md §4。
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(pick_idle_gpus 1)}"
        export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
        PER_DEVICE_BS=1
        GRAD_ACC=1
        SAVE_STRATEGY="no"
        EVAL_STRATEGY="no"
        EXTRA_LAUNCH="--max_steps 2"
        ;;
    ddp)
        echo "[mode] DDP"
        PER_DEVICE_BS=2
        GRAD_ACC=2
        SAVE_STRATEGY="epoch"
        EVAL_STRATEGY="epoch"
        DDP_GPU_COUNT="${DDP_GPU_COUNT:-8}"
        if [[ "${DDP_GPU_COUNT_WAS_SET}" == "1" && "${SFT_RESPECT_CUDA_VISIBLE_DEVICES:-0}" != "1" ]]; then
            SELECTED_GPUS="$(pick_idle_gpus "${DDP_GPU_COUNT}")"
            if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "${SELECTED_GPUS}" ]]; then
                echo "[gpu][warn] override existing CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} because DDP_GPU_COUNT was set explicitly"
                echo "[gpu][warn] set SFT_RESPECT_CUDA_VISIBLE_DEVICES=1 to keep the existing visible-device mask"
            fi
            export CUDA_VISIBLE_DEVICES="${SELECTED_GPUS}"
        else
            export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(pick_idle_gpus "${DDP_GPU_COUNT}")}"
        fi
        ACTUAL_GPU_COUNT="$(count_visible_gpus "${CUDA_VISIBLE_DEVICES}")"
        if [[ "${SFT_RESPECT_NPROC_PER_NODE:-0}" == "1" ]]; then
            export NPROC_PER_NODE="${NPROC_PER_NODE:-${ACTUAL_GPU_COUNT}}"
        else
            if [[ -n "${NPROC_PER_NODE:-}" && "${NPROC_PER_NODE}" != "${ACTUAL_GPU_COUNT}" ]]; then
                echo "[gpu][warn] override existing NPROC_PER_NODE=${NPROC_PER_NODE} to match CUDA_VISIBLE_DEVICES"
                echo "[gpu][warn] set SFT_RESPECT_NPROC_PER_NODE=1 to keep the existing process count"
            fi
            export NPROC_PER_NODE="${ACTUAL_GPU_COUNT}"
        fi
        if [[ "${ACTUAL_GPU_COUNT}" -lt "${DDP_GPU_COUNT}" ]]; then
            echo "[gpu][warn] requested ${DDP_GPU_COUNT} GPUs but only selected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        fi
        configure_master_port
        export NCCL_P2P_LEVEL=NVL
        export NCCL_DEBUG=WARN
        EXTRA_LAUNCH=""
        ;;
    *)
        echo "Unknown mode: ${MODE}. Use 'single' / 'ddp' / 'check'." >&2
        exit 1
        ;;
esac

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[gpu] NPROC_PER_NODE=${NPROC_PER_NODE}"
if [[ "${MODE}" == "ddp" ]]; then
    echo "[gpu] requested DDP_GPU_COUNT=${DDP_GPU_COUNT:-8}"
    echo "[ddp] MASTER_ADDR=${MASTER_ADDR}"
    echo "[ddp] MASTER_PORT=${MASTER_PORT}"
fi

materialize_runtime_teacher_if_needed

if [[ "${MODE}" == "check" ]]; then
    VAL_ARGS=()
else
    VAL_ARGS=(--val_dataset "${VAL_JSONL}")
fi

# ANALYSIS 权重 override 提示，方便事后回溯。
echo "[plugin] SFT_V2_ANALYSIS_WEIGHT=${SFT_V2_ANALYSIS_WEIGHT:-0.3} (default 0.3)"

# ---------------------------------------------------------------------------
# best ckpt 跟踪（与 v1 一致）
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "check" ]]; then
    BEST_ARGS=()
else
    BEST_ARGS=(
        --load_best_model_at_end true
        --metric_for_best_model loss
        --greater_is_better false
    )
fi

# ---------------------------------------------------------------------------
# 启动训练
# ---------------------------------------------------------------------------
swift sft \
    --model "${MODEL_DIR}" \
    --dataset "${TRAIN_JSONL}" \
    "${VAL_ARGS[@]}" \
    --train_type lora \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --freeze_vit true \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --learning_rate "${LR}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --lr_scheduler_type cosine \
    --bf16 true \
    --gradient_checkpointing true \
    --max_length "${MAX_LENGTH}" \
    --output_dir "${OUTPUT_DIR}" \
    --logging_steps "${LOGGING_STEPS}" \
    --save_strategy "${SAVE_STRATEGY}" \
    --eval_strategy "${EVAL_STRATEGY}" \
    --save_total_limit 3 \
    --save_only_model true \
    --report_to tensorboard \
    --logging_dir "${OUTPUT_DIR}/tb" \
    --dataloader_num_workers 4 \
    --external_plugins "${LOSS_SCALE_PLUGIN}" \
    --loss_scale "${LOSS_SCALE}" \
    "${BEST_ARGS[@]}" \
    ${EXTRA_LAUNCH}

echo "[done] v2 LoRA adapter saved under ${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "[hint] 看 TensorBoard："
echo "  bash tools/tb_serve.sh ${OUTPUT_DIR}"
echo ""
echo "[hint] 在 val 集上跑 eval（注意 v2 必须显式 --val-jsonl 指向 v2 数据）："
echo "  python tools/eval_sft_v1.py --lora-dir ${OUTPUT_DIR} \\"
echo "      --val-jsonl ${VAL_JSONL} --save-root ${OUTPUT_DIR}"
echo ""
echo "[hint] 在随机场景上 dump case："
echo "  python tools/probe_sft_v1.py --lora-dir ${OUTPUT_DIR} \\"
echo "      --val-jsonl ${VAL_JSONL} --save-root ${OUTPUT_DIR} --num-per-scenario 4"
echo "============================================================"
