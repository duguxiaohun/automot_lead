#!/usr/bin/env bash
# SFT v1 训练入口 — ms-swift LoRA on Qwen3-VL-4B-Instruct.
#
# 用法（从仓库根运行）：
#   单卡：  bash AutoMoT/tools/sft_v1_train.sh single
#   8 卡 DDP：bash AutoMoT/tools/sft_v1_train.sh ddp
#
# 数据先用 AutoMoT/tools/build_sft_dataset_v1.py 生成。LoRA 只训 language model 的
# attention + MLP projections，ViT 冻结，详见 AutoMoT/tools/SFT_V1_PLAN.md。

set -euo pipefail

MODE="${1:-ddp}"

# ---------------------------------------------------------------------------
# 路径（按需 override：可以在 shell 里 export 同名变量再运行本脚本）
# ---------------------------------------------------------------------------
MODEL_DIR="${MODEL_DIR:-AutoMoT/checkpoints/Qwen3-VL-4B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-AutoMoT/checkpoints/sft_v1_data/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-AutoMoT/checkpoints/sft_v1_data/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-AutoMoT/checkpoints/sft_v1_lora}"

# ---------------------------------------------------------------------------
# 通用超参（DDP 与单卡共享）
# ---------------------------------------------------------------------------
NUM_EPOCHS=3
LR=1e-4
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.01
MAX_LENGTH=3072
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LOGGING_STEPS=5

# loss_scale 把 ANALYSIS 段 token 权重置 0（v1 不学 analysis 内容）。
# 占位字符串与 build_sft_dataset_v1.py 的 PLACEHOLDER_ANALYSIS 必须一致。
LOSS_SCALE='{"ANALYSIS:.*?(?=\nSTATUS:)": 0.0}'

# HuggingFace 强制离线（与 runner 行为一致，禁止下载）。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 防止 swift 误读 ~/.cache。所有缓存指向 output_dir 内子目录。
export HF_HOME="${HF_HOME:-${OUTPUT_DIR}/.hf_cache}"
mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"

# ---------------------------------------------------------------------------
# 模式分支
# ---------------------------------------------------------------------------
case "${MODE}" in
    single)
        echo "[mode] single-GPU"
        PER_DEVICE_BS=4
        GRAD_ACC=2
        SAVE_STEPS=200
        EVAL_STEPS=200
        EXTRA_LAUNCH=""
        ;;
    ddp)
        echo "[mode] DDP across 8 GPUs"
        PER_DEVICE_BS=2
        GRAD_ACC=2
        SAVE_STEPS=100
        EVAL_STEPS=100
        export NPROC_PER_NODE=8
        # NCCL 调优：H20 NVLink 优先。
        export NCCL_P2P_LEVEL=NVL
        export NCCL_DEBUG=WARN
        EXTRA_LAUNCH=""
        ;;
    *)
        echo "Unknown mode: ${MODE}. Use 'single' or 'ddp'." >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# 启动训练
# ---------------------------------------------------------------------------
# 说明：
# - --train_type lora 配合 --target_modules 只命中 LLM decoder 的 7 个投影。
# - --freeze_vit true 冻结视觉塔。
# - --save_only_model true 只存 LoRA adapter，省盘空间。
# - --loss_scale 把 ANALYSIS 段权重置 0；STATUS / SUBGOAL 段保持 1.0。
# - --gradient_checkpointing 在 4B 模型上影响不大但能省 ~30% 激活显存。
swift sft \
    --model "${MODEL_DIR}" \
    --dataset "${TRAIN_JSONL}" \
    --val_dataset "${VAL_JSONL}" \
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
    --save_steps "${SAVE_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --save_total_limit 3 \
    --save_only_model true \
    --report_to none \
    --dataloader_num_workers 4 \
    --loss_scale "${LOSS_SCALE}" \
    ${EXTRA_LAUNCH}

echo "[done] LoRA adapter saved under ${OUTPUT_DIR}"
