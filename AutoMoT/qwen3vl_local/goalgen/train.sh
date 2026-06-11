#!/usr/bin/env bash
# GoalGen v1/v2 共用训练启动脚本。请在 AutoMoT/ 目录下运行。
#
# 用法：
#   bash qwen3vl_local/goalgen/train.sh check
#   bash qwen3vl_local/goalgen/train.sh single
#   bash qwen3vl_local/goalgen/train.sh ddp
set -euo pipefail

MODE="${1:-ddp}"
DDP_GPU_COUNT_WAS_SET=0
if [[ -n "${DDP_GPU_COUNT+x}" ]]; then
    DDP_GPU_COUNT_WAS_SET=1
fi

# VERSION：v1 / v2 切换数据目录 + 输出目录 + warm-start ckpt 的"一键开关"。
# - VERSION=v1（默认）：完全沿用旧行为，从零训 DiT，数据走 goalgen_v1_data/，
#   产物落 goalgen_v1_dit/。
# - VERSION=v2：数据切到 goalgen_v2_data/（只含 middle[*]→middle[*] 两段 transition），
#   产物落 goalgen_v2_dit/，并默认从 v1 best.pt 做 warm start。
# 任何具体路径用户都可以通过显式 env 覆盖（TRAIN_JSONL / OUTPUT_DIR / INIT_FROM_CKPT）。
VERSION="${VERSION:-v1}"
if [[ "${VERSION}" == "v2" ]]; then
    TRAIN_JSONL="${TRAIN_JSONL:-checkpoints/goalgen_v2_data/train.jsonl}"
    VAL_JSONL="${VAL_JSONL:-checkpoints/goalgen_v2_data/val.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/goalgen_v2_dit}"
    # VERSION=v2 模式默认从 v1 训练产物里的 best.pt warm start：DiT 权重 + EMA shadow 都加载，
    # 不接 optimizer / scheduler / step，等同于"换数据子集 + 继承架构权重"重新训练。
    # 想从 latest.pt warm start：INIT_FROM_CKPT=checkpoints/goalgen_v1_dit/latest.pt
    # 想完全从零训 v2：INIT_FROM_CKPT=NONE（任何不存在的路径会被 train.py 报错，
    # 真要从零就显式 INIT_FROM_CKPT="" 把默认覆盖掉）。
    # 默认指向 v1 顶层 symlink：checkpoints/goalgen_v1_dit/latest/best.pt
    # （latest 是本脚本维护的 symlink，永远指向最新 v1 run_XXXXXX/）。
    # 老 schema（v1 训练时还没启用 run 子目录，best.pt 直接在 OUTPUT_DIR 顶层）：
    # 显式传 INIT_FROM_CKPT=checkpoints/goalgen_v1_dit/best.pt 即可。
    INIT_FROM_CKPT="${INIT_FROM_CKPT:-checkpoints/goalgen_v1_dit/latest/best.pt}"
    # VERSION=v2 模式默认走 **fine-tune 保守配方**（warm start 起点已是 v1 best.pt，初期 LR 过大
    # 会一步把 v1 学好的权重重新打散，得不偿失）：
    # - LR 减半（AdamW 1e-4 / Muon 1e-3）
    # - warmup 缩短到 0.02（权重已经合理，不需要长 warmup 平稳起步）
    # - NUM_EPOCHS 保持 2：v2 真实 step 数以 goalgen_v2_data/stats.json 为准；
    #   warm start + 降 LR 下先用 2 epoch 作为保守 fine-tune 默认值。
    # 想恢复 v1 from-scratch 风格（LR=2e-4 等），显式传 LR=2e-4 / MUON_LR=2e-3 即可。
    LR="${LR:-1e-4}"
    MUON_LR="${MUON_LR:-1e-3}"
    WARMUP_RATIO="${WARMUP_RATIO:-0.02}"
    NUM_EPOCHS="${NUM_EPOCHS:-2}"
elif [[ "${VERSION}" == "v1" ]]; then
    TRAIN_JSONL="${TRAIN_JSONL:-checkpoints/goalgen_v1_data/train.jsonl}"
    VAL_JSONL="${VAL_JSONL:-checkpoints/goalgen_v1_data/val.jsonl}"   # 默认与数据构建器输出一致；不存在时训练器会自动跳过验证/样例日志
    OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/goalgen_v1_dit}"
    INIT_FROM_CKPT="${INIT_FROM_CKPT:-}"
else
    echo "未知 VERSION：${VERSION}。可用：v1 / v2。" >&2
    exit 1
fi

MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
# 可选 LoRA 适配器：默认空 = 用基础 Qwen；想接 SFT v1 微调后的语言编码，传
# QWEN_ADAPTER_DIR=checkpoints/sft_v1_lora（适配器目录，不是合并后的模型目录）
QWEN_ADAPTER_DIR="${QWEN_ADAPTER_DIR:-}"

# TensorBoard / 验证 / 图像样例默认值；想关闭图像样例设 IMAGE_LOG_EVERY=0
VAL_STEPS="${VAL_STEPS:-500}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-64}"
IMAGE_LOG_EVERY="${IMAGE_LOG_EVERY:-500}"
IMAGE_LOG_SAMPLES="${IMAGE_LOG_SAMPLES:-4}"
IMAGE_LOG_EULER_STEPS="${IMAGE_LOG_EULER_STEPS:-32}"

# checkpoint 保留策略：每个 epoch 末写一份 checkpoint-XXXXXX/，更老的滚动淘汰；
# val/loss 最小的一份额外拷贝为 best.pt（顶层独立保存，不受 keep 影响）。
KEEP_RECENT_CHECKPOINTS="${KEEP_RECENT_CHECKPOINTS:-3}"

# 当前共享架构（2026-06 切换）：patch=4 / hidden=1024 / n_heads=8 -> 直接对齐 Qwen K/V (8×128)
PATCH_SIZE="${PATCH_SIZE:-4}"
HIDDEN_DIM="${HIDDEN_DIM:-1024}"
# patch/unpatch 权重来源。
# 留空（默认）= 自动调 qwen3vl_local.goalgen.dit.default_patch_unpatch_weights()
# 按 latest/ -> 无 run_subdir -> 最新 run_*/ 兜底找
# `<AutoMoT>/checkpoints/patch_unpatch_v1/.../weights/patch_unpatch_best.safetensors`，
# 找到就加载并默认冻结；找不到直接报错，避免正式训练混入随机 patch/unpatch。
# 显式给路径时仍以该路径为准（用于消融对比不同 patch_unpatch 产物）。
# 架构兼容性注意：必须用 hidden=1024 / patch=4 训出的 safetensors，
# 早期 hidden=768 / patch=2 权重不兼容。
PATCH_UNPATCH_WEIGHTS="${PATCH_UNPATCH_WEIGHTS:-}"
# 默认加载即冻结；要联合微调 patch/unpatch 设 PATCH_UNPATCH_UNFREEZE=1。
PATCH_UNPATCH_UNFREEZE="${PATCH_UNPATCH_UNFREEZE:-0}"
# warm start 继承外部 patch/unpatch 时，默认要求原 safetensors 仍存在。
# 跨机器迁移且确认 ckpt 内自带权重可用时，显式设 1 才回退到 ckpt 内权重。
PATCH_UNPATCH_CKPT_FALLBACK="${PATCH_UNPATCH_CKPT_FALLBACK:-0}"
N_HEADS="${N_HEADS:-8}"
NUM_LAYERS="${NUM_LAYERS:-12}"
COND_DIM="${COND_DIM:-256}"
MLP_RATIO="${MLP_RATIO:-4.0}"
# 历史帧数上限：仅控制 DiT 的 frame_embed 容量，**不是**控制 Qwen 喂几张图。
# Qwen 实际吃到的图数 = jsonl 里 history_rgb_paths 长度，由 build_dataset.py
# 构建时的 --num-frames 决定（默认 RGB_FRAME_COUNT=4）。
# 想真正缩短 Qwen prefill：重建数据集时调小 --num-frames，再训练；
# 这里改 MAX_HISTORY_FRAMES 不影响 Qwen wall-time。保持 8 作为上限留余量。
MAX_HISTORY_FRAMES="${MAX_HISTORY_FRAMES:-8}"
QWEN_KV_SEGMENT_MODE="${QWEN_KV_SEGMENT_MODE:-select_last}"

LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
VAE_DTYPE="${VAE_DTYPE:-float32}"
DIT_DTYPE="${DIT_DTYPE:-bfloat16}"
T_SAMPLER="${T_SAMPLER:-logit_normal}"
# 默认 Z0_PRIOR_ALPHA=0.0：纯噪声起点 z0 ~ N(0,I)，模型只学 subgoal 本身。
# 之前默认 1.0 把当前帧 latent 掺进 z0 → 低 t 区 z_t 是"当前帧+噪声"主导，
# 模型靠还原当前帧拿大部分梯度，等于在捷径学习。
# image-to-image ablation 才设回 1.0（同时也要保证推理 z_init 用同样混合方式）。
Z0_PRIOR_ALPHA="${Z0_PRIOR_ALPHA:-0.0}"
Z0_PRIOR_SIGMA="${Z0_PRIOR_SIGMA:-1.0}"
CFG_DROP_PROB="${CFG_DROP_PROB:-0.1}"
CFG_SCALE="${CFG_SCALE:-2.0}"
EMA_DECAY="${EMA_DECAY:-0.9999}"
LATENT_STATS_PATH="${LATENT_STATS_PATH:-}"
LATENT_STATS_MAX_SAMPLES="${LATENT_STATS_MAX_SAMPLES:-1000}"

# 当前共享配置：Muon optimizer（专门接管 2D 权重矩阵），与 AdamW 双轨；AdamW 接管 1D / 4D
# 参数（norm weight、embedding、Conv2d patch.proj、null_lang_k/v）。Muon LR 通常
# 比 AdamW 大 5-10×；momentum 0.95 是 Keller Jordan NanoGPT 实现默认。
MUON_LR="${MUON_LR:-2e-3}"
MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"

# 当前默认开启 gradient checkpointing（patch=4 后 token 数本就不多，启用 ckpt
# 几乎不影响速度但显存余量更宽）。GRAD_CKPT=0 关闭。
GRAD_CKPT="${GRAD_CKPT:-1}"

# 防覆盖：每次启动自动建一个 run 子目录，ckpt / TB / eval 产物全部写到子目录里。
# 顶层 OUTPUT_DIR_BASE/ 维护一个 latest symlink 指向当前 run，方便 INIT_FROM_CKPT /
# eval --dit-checkpoint 直接写 ${OUTPUT_DIR_BASE}/latest/best.pt 不用看时间戳。
# - RUN_TAG=xxx：用 run_xxx/ 作为子目录名（人类可读，便于消融对比）；
# - 不设：用 run_$(date +%Y%m%d_%H%M%S)/，保证字典序 = 时间序；
# - NO_RUN_SUBDIR=1：完全跳过子目录隔离，回退到老的"顶层覆盖"行为（仅用于排查脚本兼容性）。
OUTPUT_DIR_BASE="${OUTPUT_DIR}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/run_${RUN_TAG}"
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# HF_HOME 故意放在 OUTPUT_DIR_BASE 层（不进 run 子目录）：Qwen weights 缓存按 base
# 共享，避免每个 run 重新拉一份占 8GB。
export HF_HOME="${HF_HOME:-${OUTPUT_DIR_BASE}/.hf_cache}"
mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"

# 维护 latest symlink → run_${RUN_TAG}。
# - 用相对路径目标：OUTPUT_DIR_BASE 整个搬走时 symlink 仍然有效。
# - ln -sfn：force + no-dereference，原子地 atomic-replace 旧 symlink，并发安全
#   （DDP 多 rank 在同一节点跑同一脚本时只有 rank0 由 bash 创建，但即便撞车 ln -sfn
#   也是 idempotent）。
# - NO_RUN_SUBDIR=1 时不动 latest，OUTPUT_DIR 就是 base 本身，保持向后兼容。
if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
    ln -sfn "run_${RUN_TAG}" "${OUTPUT_DIR_BASE}/latest"
fi
if [[ "${QWEN3VL_LOG_TO_FILE:-1}" != "0" && -z "${QWEN3VL_LOG_ACTIVE:-}" ]]; then
    export QWEN3VL_LOG_ACTIVE=1
    exec > >(tee -a "${OUTPUT_DIR}/log.txt") 2>&1
    echo "[log] tee stdout/stderr to ${OUTPUT_DIR}/log.txt"
fi

# 当前默认开启 torch.compile(dit)：patch=4 后 token 数砍到 1/4，compile 的固定 overhead
# 比 v1 划算很多。COMPILE_DIT=0 关闭。注意：首次 step 编译耗时 30-90 秒，CHECK 模式
# 会显得偏慢。
COMPILE_DIT="${COMPILE_DIT:-1}"

# 可选：cuDNN benchmark（让 cuDNN 自动挑最优 conv kernel）。第一次见到每个 conv
# shape 时会同时探测多个 algorithm，瞬时显存峰值高出稳态 10-30GB。VAE conv3d
# + 大 spatial 在 H20 95GB 上实测会直接 OOM，所以默认关。显存有充裕余量
# （比如 batch=1 稳态 < 60GB）且想拿那 5-10% 速度，再置 1 启用。
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-0}"
export GOALGEN_CUDNN_BENCHMARK="${CUDNN_BENCHMARK}"

# 显存碎片缓解：分配器使用 expandable_segments，能减少 fragmentation。
# 即便 cudnn.benchmark 关掉，对 latent_stats / DiT 训练长跑也有边际收益。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
    # nvidia-smi 缺失或没返回任何行：不再盲目 fallback 到 0..N-1（ddp 下会撞忙卡），
    # 返回空串交给 require_idle_gpus 显式 exit 1，与 leadmot/train.sh 对齐。
    return 1
}

require_idle_gpus() {
    local count="$1"
    local picked
    picked="$(pick_idle_gpus "${count}" || true)"
    if [[ -z "${picked}" ]]; then
        echo "No GPU selected by nvidia-smi; refusing to guess CUDA_VISIBLE_DEVICES." >&2
        exit 1
    fi
    echo "${picked}"
}

# 用户显式 pin GPU：前置 GPU_IDS="0" / GPU_IDS="0,1,2,3" 时跳过 nvidia-smi 自动选址，
# 直接用给定卡号当 CUDA_VISIBLE_DEVICES。多卡情况下卡数从 GPU_IDS 逗号数推断，
# DDP_GPU_COUNT 在 GPU_IDS 非空时不再起作用（避免要求 N 卡却只给了 M 个 ID 这种矛盾）。
resolve_visible_gpus() {
    local want_count="$1"
    if [[ -n "${GPU_IDS:-}" ]]; then
        echo "${GPU_IDS}"
    else
        require_idle_gpus "${want_count}"
    fi
}

count_visible_gpus() {
    local visible="$1"
    if [[ -z "${visible}" ]]; then echo "0"; else awk -F',' '{print NF}' <<< "${visible}"; fi
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
    if [[ "${GOALGEN_RESPECT_MASTER_PORT:-0}" == "1" ]]; then
        export MASTER_PORT="${MASTER_PORT:-29500}"
        return 0
    fi
    if [[ -n "${MASTER_PORT:-}" ]] && is_port_free "${MASTER_PORT}"; then
        export MASTER_PORT
        return 0
    fi
    export MASTER_PORT="$(find_free_master_port)"
}

COMMON_ARGS=(
    --train-jsonl "${TRAIN_JSONL}"
    --val-jsonl "${VAL_JSONL}"
    --checkpoint-dir "${MODEL_DIR}"
    --qwen-adapter-dir "${QWEN_ADAPTER_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --patch-size "${PATCH_SIZE}"
    --hidden-dim "${HIDDEN_DIM}"
    --patch-unpatch-weights "${PATCH_UNPATCH_WEIGHTS}"
    --init-from-ckpt "${INIT_FROM_CKPT}"
    --n-heads "${N_HEADS}"
    --mlp-ratio "${MLP_RATIO}"
    --num-layers "${NUM_LAYERS}"
    --cond-dim "${COND_DIM}"
    --max-history-frames "${MAX_HISTORY_FRAMES}"
    --qwen-kv-segment-mode "${QWEN_KV_SEGMENT_MODE}"
    --learning-rate "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --muon-lr "${MUON_LR}"
    --muon-momentum "${MUON_MOMENTUM}"
    --warmup-ratio "${WARMUP_RATIO}"
    --t-sampler "${T_SAMPLER}"
    --z0-prior-alpha "${Z0_PRIOR_ALPHA}"
    --z0-prior-sigma "${Z0_PRIOR_SIGMA}"
    --cfg-drop-prob "${CFG_DROP_PROB}"
    --cfg-scale "${CFG_SCALE}"
    --ema-decay "${EMA_DECAY}"
    --latent-stats-path "${LATENT_STATS_PATH}"
    --latent-stats-max-samples "${LATENT_STATS_MAX_SAMPLES}"
    --qwen-dtype "${QWEN_DTYPE}"
    --vae-dtype "${VAE_DTYPE}"
    --dit-dtype "${DIT_DTYPE}"
    --val-steps "${VAL_STEPS}"
    --val-max-samples "${VAL_MAX_SAMPLES}"
    --image-log-every "${IMAGE_LOG_EVERY}"
    --image-log-samples "${IMAGE_LOG_SAMPLES}"
    --image-log-euler-steps "${IMAGE_LOG_EULER_STEPS}"
    --keep-recent-checkpoints "${KEEP_RECENT_CHECKPOINTS}"
)

# --patch-unpatch-unfreeze 是 store_true 旗标，只在请求时追加；store_true
# 不接受值，所以不能像普通 KV 参数那样无脑塞。
if [[ "${PATCH_UNPATCH_UNFREEZE}" == "1" ]]; then
    COMMON_ARGS+=(--patch-unpatch-unfreeze)
fi
if [[ "${PATCH_UNPATCH_CKPT_FALLBACK}" == "1" ]]; then
    COMMON_ARGS+=(--allow-patch-unpatch-ckpt-fallback)
fi

# 当前默认 torch.compile + grad-ckpt 都开；置 0 时显式追加 --no-compile / --no-grad-ckpt
# 触发 argparse 的 store_false 分支。这种 0/1 → 显式追加 flag 的写法保持 sh 端简单
# （COMPILE_DIT=0/1）同时与 argparse BooleanOptionalAction 兼容。
if [[ "${COMPILE_DIT}" == "0" ]]; then
    COMMON_ARGS+=(--no-compile)
fi
if [[ "${GRAD_CKPT}" == "0" ]]; then
    COMMON_ARGS+=(--no-grad-ckpt)
fi

echo "[version] VERSION=${VERSION}"
echo "[version] TRAIN_JSONL=${TRAIN_JSONL}"
echo "[version] OUTPUT_DIR=${OUTPUT_DIR}"
if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
    echo "[version] OUTPUT_DIR_BASE=${OUTPUT_DIR_BASE} (latest symlink -> run_${RUN_TAG})"
else
    echo "[version] NO_RUN_SUBDIR=1（已禁用 run 子目录隔离，注意可能覆盖旧 ckpt）"
fi
if [[ -n "${INIT_FROM_CKPT}" ]]; then
    echo "[version] INIT_FROM_CKPT=${INIT_FROM_CKPT}（warm start：DiT strict；EMA 非 patch/unpatch key strict）"
else
    echo "[version] INIT_FROM_CKPT=<空>（DiT 从零随机初始化）"
fi

case "${MODE}" in
    check)
        echo "[mode] check"
        export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus 1)"
        export NPROC_PER_NODE=1
        # check 模式跑 2 个优化器 step 就退出（--max-train-steps 2）；epoch 末 save 分支
        # 不会被触发，循环外的 fallback 会写一份 ckpt 兜底。
        python qwen3vl_local/goalgen/train.py \
            "${COMMON_ARGS[@]}" \
            --num-epochs 1 \
            --grad-accum-steps 1 \
            --logging-steps 1 \
            --max-train-steps 2
        ;;
    single)
        echo "[mode] single"
        export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus 1)"
        export NPROC_PER_NODE=1
        # NUM_EPOCHS 默认 2：831k 样本 / GRAD_ACC=4 ≈ 207k optimizer step / epoch
        # （单卡），DiT 从零训通常 100-200k step 才看到收敛趋势，1 epoch 偏少；
        # 2 epoch 配合 cosine decay 收尾。想再多就显式 NUM_EPOCHS=3 之类。
        python qwen3vl_local/goalgen/train.py \
            "${COMMON_ARGS[@]}" \
            --num-epochs "${NUM_EPOCHS:-2}" \
            --grad-accum-steps "${GRAD_ACC:-4}" \
            --logging-steps "${LOGGING_STEPS:-10}"
        ;;
    ddp)
        echo "[mode] ddp"
        DDP_GPU_COUNT="${DDP_GPU_COUNT:-8}"
        if [[ -n "${GPU_IDS:-}" ]]; then
            echo "[gpu] GPU_IDS=${GPU_IDS} takes precedence; DDP_GPU_COUNT=${DDP_GPU_COUNT} ignored"
        fi
        export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus "${DDP_GPU_COUNT}")"
        ACTUAL_GPU_COUNT="$(count_visible_gpus "${CUDA_VISIBLE_DEVICES}")"
        export NPROC_PER_NODE="${ACTUAL_GPU_COUNT}"
        configure_master_port
        export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
        export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
        echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        echo "[gpu] NPROC_PER_NODE=${NPROC_PER_NODE}"
        echo "[ddp] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
        # NUM_EPOCHS 默认 2：831k / 4 GPU / GRAD_ACC=4 ≈ 52k optimizer step / epoch；
        # DiT 从零训通常 100-200k step 才稳定收敛，1 epoch 偏少；2 epoch 给 cosine
        # decay 留尾段精修。需要更多就显式 NUM_EPOCHS=3 之类。
        torchrun --nproc_per_node="${NPROC_PER_NODE}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${MASTER_PORT}" \
            qwen3vl_local/goalgen/train.py \
            "${COMMON_ARGS[@]}" \
            --num-epochs "${NUM_EPOCHS:-2}" \
            --grad-accum-steps "${GRAD_ACC:-4}" \
            --logging-steps "${LOGGING_STEPS:-10}"
        ;;
    *)
        echo "未知模式：${MODE}。可用模式：check / single / ddp。" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# TensorBoard / eval / probe 入口提示
#
# 训练产物 OUTPUT_DIR_BASE/ 多 run 布局：
#   OUTPUT_DIR_BASE/                        e.g. checkpoints/goalgen_v2_dit/
#     ├─ latest -> run_YYYYmmdd_HHMMSS      symlink，永远指向最新 run（本脚本维护）
#     ├─ run_20260605_1430/                 一次完整训练的所有产物
#     │   ├─ best.pt + best.json
#     │   ├─ latest.pt
#     │   ├─ checkpoint-XXXXXX/
#     │   ├─ step-checkpoint-XXXXXX/
#     │   ├─ tb/
#     │   ├─ eval/ + eval_tb/
#     │   └─ eval_cases/
#     ├─ run_20260606_0915/                 下一次训练，完全独立不会覆盖上一份
#     └─ .hf_cache/                         Qwen HF 缓存按 base 共享
#
# 下游命令既可写 ${OUTPUT_DIR_BASE}/latest/best.pt（自动跟最新 run），也可写
# ${OUTPUT_DIR_BASE}/run_XXXX/best.pt 指定历史 run 做对比。
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[hint] 看 TensorBoard（多次 run 自动对比，base 目录下所有 run 都会展开）："
echo "  bash qwen3vl_local/tb_serve.sh ${OUTPUT_DIR_BASE}"
echo ""
echo "[hint] eval 最新 run（latest symlink）："
echo "  python qwen3vl_local/goalgen/eval.py \\"
echo "    --dit-checkpoint ${OUTPUT_DIR_BASE}/latest/best.pt \\"
echo "    --qwen-adapter-dir \"${QWEN_ADAPTER_DIR}\" \\"
echo "    --save-root ${OUTPUT_DIR_BASE}/latest"
echo ""
echo "[hint] eval 当前 run（绑定本次 RUN_TAG，不受后续新 run 影响）："
echo "  python qwen3vl_local/goalgen/eval.py \\"
echo "    --dit-checkpoint ${OUTPUT_DIR}/best.pt \\"
echo "    --qwen-adapter-dir \"${QWEN_ADAPTER_DIR}\" \\"
echo "    --save-root ${OUTPUT_DIR}"
echo ""
echo "[hint] 多卡 eval 分片："
echo "  torchrun --standalone --nproc_per_node=4 qwen3vl_local/goalgen/eval.py \\"
echo "    --dit-checkpoint ${OUTPUT_DIR_BASE}/latest/best.pt --save-root ${OUTPUT_DIR_BASE}/latest"
echo ""
echo "[hint] 随机场景 case dump（输入历史/预测/真值 PNG + memory + per-step v_cos）："
echo "  python qwen3vl_local/goalgen/probe.py \\"
echo "    --dit-checkpoint ${OUTPUT_DIR_BASE}/latest/best.pt \\"
echo "    --qwen-adapter-dir \"${QWEN_ADAPTER_DIR}\" \\"
echo "    --save-root ${OUTPUT_DIR_BASE}/latest --num-per-scenario 4"
echo "============================================================"
