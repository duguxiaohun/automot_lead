#!/usr/bin/env bash
# GoalGen v1 训练启动脚本。请在 AutoMoT/ 目录下运行。
#
# 用法：
#   bash qwen3vl_local/goalgen/train_v1.sh check
#   bash qwen3vl_local/goalgen/train_v1.sh single
#   bash qwen3vl_local/goalgen/train_v1.sh ddp
set -euo pipefail

MODE="${1:-ddp}"
DDP_GPU_COUNT_WAS_SET=0
if [[ -n "${DDP_GPU_COUNT+x}" ]]; then
    DDP_GPU_COUNT_WAS_SET=1
fi

TRAIN_JSONL="${TRAIN_JSONL:-checkpoints/goalgen_v1_data/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-checkpoints/goalgen_v1_data/val.jsonl}"   # 默认与数据构建器输出一致；不存在时训练器会自动跳过验证/样例日志
MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
# 可选 LoRA 适配器：默认空 = 用基础 Qwen；想接 SFT v1 微调后的语言编码，传
# QWEN_ADAPTER_DIR=checkpoints/sft_v1_lora（适配器目录，不是合并后的模型目录）
QWEN_ADAPTER_DIR="${QWEN_ADAPTER_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/goalgen_v1_dit}"

# TensorBoard / 验证 / 图像样例默认值；想关闭图像样例设 IMAGE_LOG_EVERY=0
VAL_STEPS="${VAL_STEPS:-500}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-64}"
IMAGE_LOG_EVERY="${IMAGE_LOG_EVERY:-500}"
IMAGE_LOG_SAMPLES="${IMAGE_LOG_SAMPLES:-4}"
IMAGE_LOG_EULER_STEPS="${IMAGE_LOG_EULER_STEPS:-32}"

# checkpoint 保留策略：每个 epoch 末写一份 checkpoint-XXXXXX/，更老的滚动淘汰；
# val/loss 最小的一份额外拷贝为 best.pt（顶层独立保存，不受 keep 影响）。
KEEP_RECENT_CHECKPOINTS="${KEEP_RECENT_CHECKPOINTS:-3}"

PATCH_SIZE="${PATCH_SIZE:-2}"
HIDDEN_DIM="${HIDDEN_DIM:-768}"
# 可选：把 AutoMoT/vae_standalone/train_patch_unpatch.py 训出来的权重塞回 DiT。
# 留空 = 维持原行为（patch/unpatch 随机初始化跟 DiT 一起训练）。
PATCH_UNPATCH_WEIGHTS="${PATCH_UNPATCH_WEIGHTS:-}"
# 默认加载即冻结；要联合微调 patch/unpatch 设 PATCH_UNPATCH_UNFREEZE=1。
PATCH_UNPATCH_UNFREEZE="${PATCH_UNPATCH_UNFREEZE:-0}"
N_HEADS="${N_HEADS:-12}"
NUM_LAYERS="${NUM_LAYERS:-12}"
COND_DIM="${COND_DIM:-256}"
MLP_RATIO="${MLP_RATIO:-4.0}"
LANGUAGE_KV_INPUT_DIM="${LANGUAGE_KV_INPUT_DIM:-auto}"   # train_v1.py 会用首条样本的分段 KV 推维度；显式给整数（如 1024）可跳过探测
MAX_HISTORY_FRAMES="${MAX_HISTORY_FRAMES:-8}"
QWEN_KV_SEGMENT_MODE="${QWEN_KV_SEGMENT_MODE:-select_last}"

LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
VAE_DTYPE="${VAE_DTYPE:-float32}"
DIT_DTYPE="${DIT_DTYPE:-bfloat16}"
T_SAMPLER="${T_SAMPLER:-logit_normal}"
Z0_PRIOR_ALPHA="${Z0_PRIOR_ALPHA:-1.0}"
Z0_PRIOR_SIGMA="${Z0_PRIOR_SIGMA:-1.0}"
CFG_DROP_PROB="${CFG_DROP_PROB:-0.1}"
CFG_SCALE="${CFG_SCALE:-2.0}"
EMA_DECAY="${EMA_DECAY:-0.9999}"
LATENT_STATS_PATH="${LATENT_STATS_PATH:-}"
LATENT_STATS_MAX_SAMPLES="${LATENT_STATS_MAX_SAMPLES:-1000}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_HOME:-${OUTPUT_DIR}/.hf_cache}"
mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"

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
    --n-heads "${N_HEADS}"
    --mlp-ratio "${MLP_RATIO}"
    --num-layers "${NUM_LAYERS}"
    --cond-dim "${COND_DIM}"
    --max-history-frames "${MAX_HISTORY_FRAMES}"
    --qwen-kv-segment-mode "${QWEN_KV_SEGMENT_MODE}"
    --language-kv-input-dim "${LANGUAGE_KV_INPUT_DIM}"
    --learning-rate "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
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

case "${MODE}" in
    check)
        echo "[mode] check"
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(pick_idle_gpus 1)}"
        export NPROC_PER_NODE=1
        # check 模式跑 2 个优化器 step 就退出（--max-train-steps 2）；epoch 末 save 分支
        # 不会被触发，循环外的 fallback 会写一份 ckpt 兜底。
        python qwen3vl_local/goalgen/train_v1.py \
            "${COMMON_ARGS[@]}" \
            --num-epochs 1 \
            --grad-accum-steps 1 \
            --logging-steps 1 \
            --max-train-steps 2
        ;;
    single)
        echo "[mode] single"
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(pick_idle_gpus 1)}"
        export NPROC_PER_NODE=1
        python qwen3vl_local/goalgen/train_v1.py \
            "${COMMON_ARGS[@]}" \
            --num-epochs "${NUM_EPOCHS:-1}" \
            --grad-accum-steps "${GRAD_ACC:-4}" \
            --logging-steps "${LOGGING_STEPS:-10}"
        ;;
    ddp)
        echo "[mode] ddp"
        DDP_GPU_COUNT="${DDP_GPU_COUNT:-8}"
        if [[ "${DDP_GPU_COUNT_WAS_SET}" == "1" && "${GOALGEN_RESPECT_CUDA_VISIBLE_DEVICES:-0}" != "1" ]]; then
            export CUDA_VISIBLE_DEVICES="$(pick_idle_gpus "${DDP_GPU_COUNT}")"
        else
            export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(pick_idle_gpus "${DDP_GPU_COUNT}")}"
        fi
        ACTUAL_GPU_COUNT="$(count_visible_gpus "${CUDA_VISIBLE_DEVICES}")"
        if [[ "${GOALGEN_RESPECT_NPROC_PER_NODE:-0}" == "1" ]]; then
            export NPROC_PER_NODE="${NPROC_PER_NODE:-${ACTUAL_GPU_COUNT}}"
        else
            export NPROC_PER_NODE="${ACTUAL_GPU_COUNT}"
        fi
        configure_master_port
        export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
        export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
        echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        echo "[gpu] NPROC_PER_NODE=${NPROC_PER_NODE}"
        echo "[ddp] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
        torchrun --nproc_per_node="${NPROC_PER_NODE}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${MASTER_PORT}" \
            qwen3vl_local/goalgen/train_v1.py \
            "${COMMON_ARGS[@]}" \
            --num-epochs "${NUM_EPOCHS:-1}" \
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
# 训练产物 OUTPUT_DIR 平铺布局（与 eval_v1.py / probe_v1.py 同根）：
#   OUTPUT_DIR/
#     ├─ best.pt + best.json          val/loss 历史最小的轻量权重（eval 默认指向它）
#     ├─ latest.pt                    最近一次保存的轻量权重（无 val 时 eval 回退到它）
#     ├─ checkpoint-*/                各 epoch DiT 全量 ckpt（含 optimizer/scheduler，保留最近 N=KEEP_RECENT_CHECKPOINTS）
#     ├─ tb/                          训练 TensorBoard events（含 train/* val/* epoch_end/* image_samples）
#     ├─ eval/                        eval_v1.py 写的 metrics + perline + samples PNG
#     ├─ eval_tb/                     eval_v1.py 写的 TB scalar / image（独立 run）
#     └─ eval_cases/                  probe_v1.py 随机场景 case dump
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[hint] 看 TensorBoard（训练曲线 + 多次 eval 同时显示）："
echo "  bash tools/tb_serve.sh ${OUTPUT_DIR}"
echo ""
echo "[hint] eval（指标 + TB scalar/image + perline jsonl）："
echo "  python qwen3vl_local/goalgen/eval_v1.py \\"
echo "    --dit-checkpoint ${OUTPUT_DIR}/best.pt \\"
echo "    --qwen-adapter-dir \"${QWEN_ADAPTER_DIR}\" \\"
echo "    --save-root ${OUTPUT_DIR}"
echo ""
echo "[hint] 多卡 eval 分片："
echo "  torchrun --standalone --nproc_per_node=4 qwen3vl_local/goalgen/eval_v1.py \\"
echo "    --dit-checkpoint ${OUTPUT_DIR}/best.pt --save-root ${OUTPUT_DIR}"
echo ""
echo "[hint] 随机场景 case dump（输入历史/预测/真值 PNG + memory + per-step v_cos）："
echo "  python qwen3vl_local/goalgen/probe_v1.py \\"
echo "    --dit-checkpoint ${OUTPUT_DIR}/best.pt \\"
echo "    --qwen-adapter-dir \"${QWEN_ADAPTER_DIR}\" \\"
echo "    --save-root ${OUTPUT_DIR} --num-per-scenario 4"
echo "============================================================"
