#!/usr/bin/env bash
# SFT v1 训练入口 — ms-swift LoRA on Qwen3-VL-4B-Instruct.
#
# 用法（**从 AutoMoT/ 目录运行**，远程默认 cwd）：
#   单卡：       bash qwen3vl_local/sft/sft_v1_train.sh single
#   DDP：        bash qwen3vl_local/sft/sft_v1_train.sh ddp
#   sanity 自检：bash qwen3vl_local/sft/sft_v1_train.sh check
#     （check 模式只跑 2 step、不保存 ckpt，用来确认 loss_scale 是否生效。
#      正常 mask 下初始 loss 应只来自 STATUS/SUBGOAL 段，数值在 ~6-10 量级；
#      若初始 loss < 3 多半是 ANALYSIS 段也被算进去了，模型在抄占位句。
#      跑前最好先 python qwen3vl_local/sft/check_loss_mask.py 看 token 级 mask 是否对。）
#
# 数据先用 qwen3vl_local/sft/build_sft_dataset_v1.py 生成。LoRA 只训 language model 的
# attention + MLP projections，ViT 冻结，详见 qwen3vl_local/sft/SFT_PLAN.md。
#
# 常用 override：
#   MODEL_DIR=/path/to/Qwen3-VL-4B-Instruct \
#   TRAIN_JSONL=/path/to/train.jsonl \
#   VAL_JSONL=/path/to/val.jsonl \
#   OUTPUT_DIR=/path/to/sft_v1_lora \
#   DDP_GPU_COUNT=4 \
#   bash qwen3vl_local/sft/sft_v1_train.sh ddp
#
# 训练产物：
#   OUTPUT_DIR 下保存 LoRA adapter checkpoint。eval_sft_v1.py 的 --lora-dir
#   可以指向 OUTPUT_DIR 或某个具体 checkpoint 子目录。
#
# 重要约束：
#   1. 本脚本默认 HuggingFace 离线，只读本地 MODEL_DIR，不允许联网下载。
#   2. v1 目标是 STATUS/SUBGOAL，不学 ANALYSIS；自定义 loss_scale 插件会把 ANALYSIS 段 loss 置 0。
#   3. `--freeze_vit true` 冻结视觉塔，LoRA 只挂到语言 decoder 的投影层。
#   4. 如果远程发现 loss_scale 没生效，先不要继续长训，检查 qwen3vl_local/sft/sft_v1_loss_scale_plugin.py。

set -euo pipefail

# MODE 只控制 batch/step/DDP 相关配置；训练命令主体保持一致，方便单卡 smoke test
# 和 8 卡正式训练对齐。
MODE="${1:-ddp}"
DDP_GPU_COUNT_WAS_SET=0
if [[ -n "${DDP_GPU_COUNT+x}" ]]; then
    DDP_GPU_COUNT_WAS_SET=1
fi

# ---------------------------------------------------------------------------
# 路径（按需 override：可以在 shell 里 export 同名变量再运行本脚本）。
#
# MODEL_DIR 必须是完整本地 checkpoint 目录。默认相对 AutoMoT/ cwd：
#   checkpoints/Qwen3-VL-4B-Instruct   （即 AutoMoT/checkpoints/Qwen3-VL-4B-Instruct）
# TRAIN_JSONL / VAL_JSONL 是 build_sft_dataset_v1.py 生成的 jsonl。
# OUTPUT_DIR 是 LoRA adapter 输出目录，不建议放到源码目录外的临时位置，避免后续 eval 找不到。
# 想用绝对路径覆盖时直接 export 同名变量：
#   MODEL_DIR=/datashare/IOL4SGH/AutoMoT/models/Qwen3-VL-4B-Instruct bash qwen3vl_local/sft/sft_v1_train.sh ddp
# ---------------------------------------------------------------------------
MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-checkpoints/sft_v1_data/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-checkpoints/sft_v1_data/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft_v1_lora}"

# 防覆盖：每次启动自动建 run_<时间戳> 子目录，LoRA adapter / checkpoint-XXX / tb
# 产物全部写进子目录，顶层 OUTPUT_DIR_BASE/ 维护 latest symlink 指向当前 run。
# - RUN_TAG=xxx：用 run_xxx/ 做子目录名；不设用 run_$(date +%Y%m%d_%H%M%S)/；
# - NO_RUN_SUBDIR=1：回退老的"顶层覆盖"行为。
# HF_HOME 故意钉在 OUTPUT_DIR_BASE 层（见下方），让所有 run 共享 tokenizer/model
# cache，不必每个 run 重拉一份。symlink/mkdir 统一放到下方 HF_HOME 之后做。
OUTPUT_DIR_BASE="${OUTPUT_DIR}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/run_${RUN_TAG}"
fi

# ---------------------------------------------------------------------------
# 通用超参（DDP 与单卡共享）。
#
# MAX_LENGTH 需要覆盖：system prompt + user memory + 4 张图的视觉 token + assistant 三行输出。
# 如果训练日志出现 truncation warning，优先增大 MAX_LENGTH，而不是删 prompt。
#
# LORA_RANK=16 是 v1 的保守起点：参数量足够学习状态边界，又不至于大幅破坏 base 模型。
# ---------------------------------------------------------------------------
# v1 第二轮调参（前一轮 ckpt-8100 严重过训，EOS 被刷崩、出现 STATUS 行循环复读）：
# - NUM_EPOCHS 3→2：配合数据扩量到 ~14k 后，等效 batch=32 时总 step ≈ 900。
# - LR 1e-4→5e-5：13 个有效 loss token 的薄监督，1e-4 容易把 LM 头冲过头。
# - LORA_DROPOUT 0.05→0.1 / WEIGHT_DECAY 0.01→0.05：正则加倍，抑制对短目标过拟合。
# - LORA_RANK/ALPHA 不动，保留容量；MAX_LENGTH 不动，覆盖 4 图 + memory。
NUM_EPOCHS=2
LR=5e-5
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.05
MAX_LENGTH=3072
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.1
LOGGING_STEPS=5

# ---------------------------------------------------------------------------
# Step-based checkpoint 保存（用户场景：数据量上来后 epoch 周期太长）
# ---------------------------------------------------------------------------
# HF Trainer 不允许同时跑 epoch + steps 两种 save_strategy，且 load_best_model_at_end
# 要求 save_strategy == eval_strategy。所以 v1/v2 都改成纯 step 保存：每 SAVE_STEPS
# 步保存一次 + 评估一次，--save_total_limit 控制保留最近 N 个 checkpoint-XXX/。
# best 跟踪不变（按 eval/loss 选 best 装回 OUTPUT_DIR 顶层 adapter_model.*）。
# 训练结束时最后一个 step ckpt 自然约等于"最后一个 epoch 末快照"。
# 默认 SAVE_STEPS=10000 / SAVE_TOTAL_LIMIT=3 → 等效"保留最近 30k 步"。
# 想换 5k / 20k / 改保留数，export 同名变量即可，不必动脚本。
SAVE_STEPS="${SAVE_STEPS:-10000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

# loss_scale 把 ANALYSIS 段 token 权重置 0（v1 不学 analysis 内容）。
#
# 注意：
# - 占位字符串与 build_sft_dataset_v1.py 的 PLACEHOLDER_ANALYSIS 必须一致；
# - ms-swift 3.12.x 的 --loss_scale 只接受已注册策略名，不能直接传 regex JSON；
# - 插件里的 regex 只覆盖 "ANALYSIS:" 到 "\nSTATUS:" 之间的 token；
# - STATUS/SUBGOAL 的字面前缀和 event_name 都保留 loss，用来强化固定输出格式。
LOSS_SCALE="sft_v1_analysis_mask"
LOSS_SCALE_PLUGIN="qwen3vl_local/sft/sft_v1_loss_scale_plugin.py"

# HuggingFace 强制离线（与 runner 行为一致，禁止下载）。
# 远程机器如果模型缺文件，应先由用户手动准备 checkpoint，不要让训练脚本隐式联网补齐。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 防止 swift 误读 ~/.cache。HF_HOME 钉在 OUTPUT_DIR_BASE 层（不进 run 子目录），
# 让所有 run 共享同一份 tokenizer/model cache，避免每个 run 重拉占盘。
export HF_HOME="${HF_HOME:-${OUTPUT_DIR_BASE}/.hf_cache}"
mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"
# 先开 tee，再做 latest symlink + 打印 [run]，这样 "[run] OUTPUT_DIR=..." 也进 log.txt。
if [[ "${QWEN3VL_LOG_TO_FILE:-1}" != "0" && -z "${QWEN3VL_LOG_ACTIVE:-}" ]]; then
    export QWEN3VL_LOG_ACTIVE=1
    exec > >(tee -a "${OUTPUT_DIR}/log.txt") 2>&1
    echo "[log] tee stdout/stderr to ${OUTPUT_DIR}/log.txt"
fi
if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
    # ln -sfn：force + no-dereference，原子替换旧 symlink；相对目标，base 搬走仍有效。
    ln -sfn "run_${RUN_TAG}" "${OUTPUT_DIR_BASE}/latest"
    echo "[run] OUTPUT_DIR=${OUTPUT_DIR}  (latest -> run_${RUN_TAG})"
fi

# ---------------------------------------------------------------------------
# GPU 选择
# ---------------------------------------------------------------------------
# 默认用 nvidia-smi 按 memory.used、utilization.gpu 从小到大排序，自动挑最空闲 GPU，
# 并覆盖外部残留的 CUDA_VISIBLE_DEVICES。
# DDP 模式下 NPROC_PER_NODE 跟随最终 CUDA_VISIBLE_DEVICES，避免外层残留单进程配置。
# DDP_GPU_COUNT 只表示要挑多少张卡；具体卡号仍由脚本自动挑最空闲的 N 张。
# 若要严格尊重外部已有 MASTER_PORT，可设 SFT_RESPECT_MASTER_PORT=1。
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

# 用户显式 pin GPU：前置 GPU_IDS="0" / GPU_IDS="0,1,2,3" 时跳过 nvidia-smi 自动选址，
# 直接用给定卡号当 CUDA_VISIBLE_DEVICES。多卡情况下卡数从 GPU_IDS 逗号数推断，
# DDP_GPU_COUNT 在 GPU_IDS 非空时不再起作用（避免要求 N 卡却只给了 M 个 ID 这种矛盾）。
resolve_visible_gpus() {
    local want_count="$1"
    if [[ -n "${GPU_IDS:-}" ]]; then
        echo "${GPU_IDS}"
    else
        pick_idle_gpus "${want_count}"
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

# ---------------------------------------------------------------------------
# 模式分支
# ---------------------------------------------------------------------------
case "${MODE}" in
    single)
        echo "[mode] single-GPU"
        # 单卡模式用于 smoke test 或显存足够时的小规模训练。
        # 等效 batch = PER_DEVICE_BS * GRAD_ACC = 8。
        export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus 1)"
        export NPROC_PER_NODE=1
        PER_DEVICE_BS=4
        GRAD_ACC=2
        # step 触发：每 SAVE_STEPS 步保存 + eval（默认 10000）；--save_total_limit
        # 控制保留最近 N 个 checkpoint-XXX/（默认 3，等效最近 30k 步）。--load_best_model_at_end
        # 仍按 eval/loss 把 best 装回 OUTPUT_DIR 顶层（adapter_model.*）；epoch 边界
        # 不再单独 save，但训练结束时最后一个 step ckpt ≈ 最后一个 epoch 末快照。
        SAVE_STRATEGY="steps"
        EVAL_STRATEGY="steps"
        VAL_ARGS=(--val_dataset "${VAL_JSONL}")
        EXTRA_LAUNCH=""
        ;;
    check)
        echo "[mode] check (loss_scale sanity, 2 steps only — no checkpoint, no eval)"
        # ---- check 模式做什么 ----
        # 用 swift 真实训练管道跑 2 个 optimizer step 就停，目的只有一个：
        # 看初始 loss 数值合不合理，从而间接判断 --loss_scale 在多模态 chat
        # template 上是不是真的把 ANALYSIS 段权重置 0 了。
        #
        # ---- 怎么判断 ----
        # 健康初始 loss 数量级（贪心 cross-entropy 假设）：
        #   * STATUS / SUBGOAL 段约 8-15 个有效 token，每个 token 起步 loss
        #     约等于 log(vocab_size) ≈ log(152064) ≈ 11.9；
        #   * 平均下来 batch-mean loss 应落在 ~6-10 区间。
        #
        # 异常判读：
        #   * loss < 3：很可能 loss_scale 把 STATUS / SUBGOAL 也一起 mask 了，
        #     训练几乎无梯度。继续训会得到一个"什么都不学"的 LoRA。
        #   * loss > 12：ANALYSIS 段也算 loss 了；模型会优先学复读
        #     "Observations recorded."，STATUS 学习速度被稀释。
        #   * loss 在 6-10：可继续上 single / ddp 正式训。
        #
        # ---- 配置选择 ----
        # PER_DEVICE_BS=1 / GRAD_ACC=1：让 step 时间最短，2 step 总耗时 < 2 min。
        # SAVE_STRATEGY=no / EVAL_STRATEGY=no：彻底关掉 checkpoint / eval，本模式
        # 不应该写盘也不应该跑验证集。
        # EXTRA_LAUNCH="--max_steps 2"：硬截 2 step 就退出，不管 num_train_epochs。
        #
        # ---- 不进 DDP 的原因 ----
        # 8 卡 DDP 启动 NCCL handshake 通常要 10-30 秒，会把"短训观察 loss"的
        # 信号淹没在启动日志里。check 模式默认走单卡 / 单进程，让 stdout 干净。
        export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus 1)"
        export NPROC_PER_NODE=1
        PER_DEVICE_BS=1
        GRAD_ACC=1
        # check 模式不写盘也不跑 eval：SAVE_STRATEGY=no + EVAL_STRATEGY=no，
        # 同时 BEST_ARGS 会被清空（load_best 要求 save/eval 都开）。
        SAVE_STRATEGY="no"
        EVAL_STRATEGY="no"
        # check 只验证 2 个训练 step 的 loss_scale，不传 val_dataset，避免加载/评估 val 的 ~800 条样本。
        VAL_ARGS=()
        EXTRA_LAUNCH="--max_steps 2"
        ;;
    ddp)
        echo "[mode] DDP"
        # 多卡正式训练。swift 通常会读取 NPROC_PER_NODE 启动 torchrun/分布式。
        # 默认等效 batch = 8 GPUs * PER_DEVICE_BS * GRAD_ACC = 32；
        # 若用 DDP_GPU_COUNT 改卡数，等效 batch 会随卡数线性变化。
        PER_DEVICE_BS=2
        GRAD_ACC=2
        # step 触发 + best 跟踪（含义见 single 分支注释）。SAVE_STEPS/SAVE_TOTAL_LIMIT
        # 是 env 可调；用户场景"数据量大、epoch 周期太长"时这是拿到中间产物的唯一路径。
        SAVE_STRATEGY="steps"
        EVAL_STRATEGY="steps"
        DDP_GPU_COUNT="${DDP_GPU_COUNT:-8}"
        if [[ -n "${GPU_IDS:-}" ]]; then
            echo "[gpu] GPU_IDS=${GPU_IDS} takes precedence; DDP_GPU_COUNT=${DDP_GPU_COUNT} ignored"
        fi
        export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus "${DDP_GPU_COUNT}")"
        ACTUAL_GPU_COUNT="$(count_visible_gpus "${CUDA_VISIBLE_DEVICES}")"
        export NPROC_PER_NODE="${ACTUAL_GPU_COUNT}"
        if [[ "${ACTUAL_GPU_COUNT}" -lt "${DDP_GPU_COUNT}" ]]; then
            echo "[gpu][warn] requested ${DDP_GPU_COUNT} GPUs but only selected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        fi
        configure_master_port
        # NCCL 调优：H20 NVLink 优先。若远程机器不是 NVLink 拓扑，出现 NCCL 卡住时
        # 可以临时 unset NCCL_P2P_LEVEL 或退到 single/少卡模式排查。
        export NCCL_P2P_LEVEL=NVL
        export NCCL_DEBUG=WARN
        VAL_ARGS=(--val_dataset "${VAL_JSONL}")
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

# ---------------------------------------------------------------------------
# best ckpt 跟踪
# ---------------------------------------------------------------------------
# --load_best_model_at_end 行为：训练结束时把 val/loss 最小的 epoch ckpt 装载回模型，
# 并以"主权重"形式落到 OUTPUT_DIR 顶层（adapter_model.safetensors 等）；各 epoch
# 快照仍保留在 checkpoint-XXX/ 子目录，受 save_total_limit 滚动淘汰。
# 注意：HF Trainer 要求 save_strategy == eval_strategy，否则 best 跟踪会报错。
# check 模式既不 save 也不 eval，关掉 best 跟踪。
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
# 说明：
# - --train_type lora 配合 --target_modules 只命中 LLM decoder 的 7 个投影。
# - --freeze_vit true 冻结视觉塔。
# - --save_only_model true 只存 LoRA adapter，省盘空间。
# - --external_plugins 注册 SFT v1 自定义 loss_scale。
# - --loss_scale 把 ANALYSIS 段权重置 0；STATUS / SUBGOAL 段保持 1.0。
# - --gradient_checkpointing 在 4B 模型上影响不大但能省 ~30% 激活显存。
#
# 训练前建议先跑：
#   python qwen3vl_local/sft/eval_sft_v1.py --lora-dir "" --max-samples 32 --skip-anchor12-sanity
# 得到 base baseline；训练后再跑同样 val 子集 + LoRA，看 keep/early_advance 是否改善。
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
    --save_steps "${SAVE_STEPS}" \
    --eval_strategy "${EVAL_STRATEGY}" \
    --eval_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --save_only_model true \
    --report_to tensorboard \
    --logging_dir "${OUTPUT_DIR}/tb" \
    --dataloader_num_workers 4 \
    --external_plugins "${LOSS_SCALE_PLUGIN}" \
    --loss_scale "${LOSS_SCALE}" \
    "${BEST_ARGS[@]}" \
    ${EXTRA_LAUNCH}

echo "[done] LoRA adapter saved under ${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# TensorBoard / eval 路径提示
#
# 训练产物布局（与 eval_sft_v1.py / probe_sft_v1.py 同根 — 即 OUTPUT_DIR 平铺）：
#   OUTPUT_DIR/
#     ├─ adapter_model.*    训练结束 best 权重（eval/loss 最小的那个 step ckpt 装回）
#     ├─ checkpoint-*/      step LoRA adapter 快照（每 SAVE_STEPS 步一份，保留最近 SAVE_TOTAL_LIMIT 个）
#     │                     默认 SAVE_STEPS=10000、SAVE_TOTAL_LIMIT=3 → 等效最近 30k 步
#     │                     训练结束时最后一份 ≈ 最后一个 epoch 末快照
#     ├─ tb/                训练 TensorBoard events（swift 写入）
#     ├─ eval/              eval_sft_v1.py 写的 metrics.json + predictions.jsonl
#     ├─ eval_tb/           eval_sft_v1.py 写的 TB scalar/text（独立 run，TB 可同时看）
#     └─ eval_cases/        probe_sft_v1.py 随机场景 case dump（input/output/loss）
#
# 看 TensorBoard：直接把 logdir 指到 OUTPUT_DIR 根目录，左侧 run 列表会同时显示
# tb（训练）和 eval_tb（多个 ckpt 的 eval 结果）；用 qwen3vl_local/tb_serve.sh 一条命令
# 起服务，stdout 会打印本地浏览器要用的 ssh 隧道命令，本地点链接就能看。
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[hint] 看 TensorBoard："
echo "  bash qwen3vl_local/tb_serve.sh ${OUTPUT_DIR}"
echo ""
echo "[hint] 在 val 集上跑 eval（指标 + TB 标量 + 预测 jsonl）："
echo "  python qwen3vl_local/sft/eval_sft_v1.py --lora-dir ${OUTPUT_DIR} --save-root ${OUTPUT_DIR}"
echo "  # 多卡分片：torchrun --nproc_per_node=4 qwen3vl_local/sft/eval_sft_v1.py --lora-dir ${OUTPUT_DIR} --save-root ${OUTPUT_DIR}"
echo ""
echo "[hint] 在随机场景上 dump case（输入 prompt+图像，输出文本，per-token loss）："
echo "  python qwen3vl_local/sft/probe_sft_v1.py --lora-dir ${OUTPUT_DIR} --save-root ${OUTPUT_DIR} --num-per-scenario 4"
echo "============================================================"
