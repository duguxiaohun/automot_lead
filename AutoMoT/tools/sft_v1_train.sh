#!/usr/bin/env bash
# SFT v1 训练入口 — ms-swift LoRA on Qwen3-VL-4B-Instruct.
#
# 用法（从仓库根运行）：
#   单卡：  bash AutoMoT/tools/sft_v1_train.sh single
#   8 卡 DDP：bash AutoMoT/tools/sft_v1_train.sh ddp
#
# 数据先用 AutoMoT/tools/build_sft_dataset_v1.py 生成。LoRA 只训 language model 的
# attention + MLP projections，ViT 冻结，详见 AutoMoT/tools/SFT_V1_PLAN.md。
#
# 常用 override：
#   MODEL_DIR=/path/to/Qwen3-VL-4B-Instruct \
#   TRAIN_JSONL=/path/to/train.jsonl \
#   VAL_JSONL=/path/to/val.jsonl \
#   OUTPUT_DIR=/path/to/sft_v1_lora \
#   bash AutoMoT/tools/sft_v1_train.sh ddp
#
# 训练产物：
#   OUTPUT_DIR 下保存 LoRA adapter checkpoint。eval_sft_v1.py 的 --lora-dir
#   可以指向 OUTPUT_DIR 或某个具体 checkpoint 子目录。
#
# 重要约束：
#   1. 本脚本默认 HuggingFace 离线，只读本地 MODEL_DIR，不允许联网下载。
#   2. v1 目标是 STATUS/SUBGOAL，不学 ANALYSIS；LOSS_SCALE 会把 ANALYSIS 段 loss 置 0。
#   3. `--freeze_vit true` 冻结视觉塔，LoRA 只挂到语言 decoder 的投影层。
#   4. 如果远程发现 loss_scale 没生效，先不要继续长训，改用手动 label mask。

set -euo pipefail

# MODE 只控制 batch/step/DDP 相关配置；训练命令主体保持一致，方便单卡 smoke test
# 和 8 卡正式训练对齐。
MODE="${1:-ddp}"

# ---------------------------------------------------------------------------
# 路径（按需 override：可以在 shell 里 export 同名变量再运行本脚本）。
#
# MODEL_DIR 必须是完整本地 checkpoint 目录，例如：
#   AutoMoT/checkpoints/Qwen3-VL-4B-Instruct
# TRAIN_JSONL / VAL_JSONL 是 build_sft_dataset_v1.py 生成的 jsonl。
# OUTPUT_DIR 是 LoRA adapter 输出目录，不建议放到源码目录外的临时位置，避免后续 eval 找不到。
# ---------------------------------------------------------------------------
MODEL_DIR="${MODEL_DIR:-AutoMoT/checkpoints/Qwen3-VL-4B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-AutoMoT/checkpoints/sft_v1_data/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-AutoMoT/checkpoints/sft_v1_data/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-AutoMoT/checkpoints/sft_v1_lora}"

# ---------------------------------------------------------------------------
# 通用超参（DDP 与单卡共享）。
#
# MAX_LENGTH 需要覆盖：system prompt + user memory + 4 张图的视觉 token + assistant 三行输出。
# 如果训练日志出现 truncation warning，优先增大 MAX_LENGTH，而不是删 prompt。
#
# LORA_RANK=16 是 v1 的保守起点：参数量足够学习状态边界，又不至于大幅破坏 base 模型。
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
#
# 注意：
# - 占位字符串与 build_sft_dataset_v1.py 的 PLACEHOLDER_ANALYSIS 必须一致；
# - regex 只覆盖 "ANALYSIS:" 到 "\nSTATUS:" 之间的 token；
# - STATUS/SUBGOAL 的字面前缀和 event_name 都保留 loss，用来强化固定输出格式。
LOSS_SCALE='{"ANALYSIS:.*?(?=\nSTATUS:)": 0.0}'

# HuggingFace 强制离线（与 runner 行为一致，禁止下载）。
# 远程机器如果模型缺文件，应先由用户手动准备 checkpoint，不要让训练脚本隐式联网补齐。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 防止 swift 误读 ~/.cache。所有缓存指向 output_dir 内子目录。
# 这样一次训练的 tokenizer/model cache 和 adapter 产物在同一个树下，迁移或清理更明确。
export HF_HOME="${HF_HOME:-${OUTPUT_DIR}/.hf_cache}"
mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"

# ---------------------------------------------------------------------------
# 模式分支
# ---------------------------------------------------------------------------
case "${MODE}" in
    single)
        echo "[mode] single-GPU"
        # 单卡模式用于 smoke test 或显存足够时的小规模训练。
        # 等效 batch = PER_DEVICE_BS * GRAD_ACC = 8。
        PER_DEVICE_BS=4
        GRAD_ACC=2
        SAVE_STEPS=200
        EVAL_STEPS=200
        EXTRA_LAUNCH=""
        ;;
    ddp)
        echo "[mode] DDP across 8 GPUs"
        # 8 卡正式训练。swift 通常会读取 NPROC_PER_NODE 启动 torchrun/分布式。
        # 等效 batch = 8 GPUs * PER_DEVICE_BS * GRAD_ACC = 32。
        PER_DEVICE_BS=2
        GRAD_ACC=2
        SAVE_STEPS=100
        EVAL_STEPS=100
        export NPROC_PER_NODE=8
        # NCCL 调优：H20 NVLink 优先。若远程机器不是 NVLink 拓扑，出现 NCCL 卡住时
        # 可以临时 unset NCCL_P2P_LEVEL 或退到 single/少卡模式排查。
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
#
# 训练前建议先跑：
#   python AutoMoT/tools/eval_sft_v1.py --lora-dir "" --max-samples 32 --skip-anchor12-sanity
# 得到 base baseline；训练后再跑同样 val 子集 + LoRA，看 keep/early_advance 是否改善。
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
