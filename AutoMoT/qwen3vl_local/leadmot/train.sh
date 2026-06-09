#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-ddp}"  # check | single | ddp

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# 显存碎片缓解：LeadMoT batched 训练会反复构造变长 Qwen K/V padding batch，
# expandable_segments 能降低长跑时 allocator fragmentation 导致的偶发 OOM。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_JSONL="${TRAIN_JSONL:-checkpoints/leadmot_v1_data/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-checkpoints/leadmot_v1_data/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/leadmot_v1_decoder}"

# 防覆盖：每次启动自动建 run_<时间戳> 子目录，best.pt / latest.pt / tb / eval 产物
# 全部写进子目录，顶层 OUTPUT_DIR_BASE/ 维护一个 latest symlink 指向当前 run。
# - RUN_TAG=xxx：用 run_xxx/ 做子目录名（人类可读，便于消融对比）；
# - 不设：用 run_$(date +%Y%m%d_%H%M%S)/，字典序 = 时间序；
# - NO_RUN_SUBDIR=1：回退老的"顶层覆盖"行为（仅排查兼容性时用）。
# bash 段只在主进程执行一次，torchrun 各 worker 共享同一个 OUTPUT_DIR，无竞态。
OUTPUT_DIR_BASE="${OUTPUT_DIR}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR_BASE}/run_${RUN_TAG}"
  mkdir -p "${OUTPUT_DIR}"
  # ln -sfn：force + no-dereference，原子替换旧 symlink；相对目标，base 整个搬走仍有效。
  ln -sfn "run_${RUN_TAG}" "${OUTPUT_DIR_BASE}/latest"
  echo "[run] OUTPUT_DIR=${OUTPUT_DIR}  (latest -> run_${RUN_TAG})"
fi

MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
LEAD_BEV_CKPT="${LEAD_BEV_CKPT:-checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth}"
RESUME="${RESUME:-}"
INIT_FROM_CKPT="${INIT_FROM_CKPT:-}"

LR="${LR:-4.9e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
# 2026-06 三轮调优后 8 卡等效 global batch=384（vs 旧 batch=1 路径的 64，6x）；
# cosine schedule 总步数同比缩到 1/6（约 26k → 4k step），原 ratio=0.05 只剩 ~200
# warmup step，配合 LR=4.9e-4 容易在 warmup 末期 overshoot；ratio 升到 0.20 给足
# 预热（~800 warmup step）。
WARMUP_RATIO="${WARMUP_RATIO:-0.20}"
# NUM_EPOCHS 从 3 升到 10：等效 batch 翻 6x 后单 epoch optimizer step 砍到 1/6，
# 想保持总 step 数与旧 batch=1 + 3 epoch 量级一致需要 18 epoch；折中取 10 epoch
# （≈40k optimizer step）保证 decoder 从零训有足够梯度更新次数。
NUM_EPOCHS="${NUM_EPOCHS:-10}"
GRAD_ACC="${GRAD_ACC:-2}"
# 2026-06 H20 batched 训练：decoder 单步 forward 处理的 sample 数。
# =1 时走 runtime.forward_sample fast path（与历史 ckpt 字节级等价）；
# >1 时启用 runtime.forward_batch + _pad_segmented_kv_batch + prefix_key_padding_mask。
# 等效 global batch = world_size * BATCH_SIZE * GRAD_ACC（默认 8卡 * 24 * 2 = 384）。
# 相比旧 batch=1 路径等效 batch 约 6x，LR 按 sqrt(6) 从 2e-4 上调到 4.9e-4。
# LeadMoT 的 frozen Qwen prefill 仍是 no-grad 串行，显存不会像 SFT 那样线性吃满；
# 默认 BATCH_SIZE=24 是 H20 上更接近 80% 显存的吞吐点。
# OOM 时先 BATCH_SIZE÷2 / GRAD_ACC×2 保持等效不变；坏样本会污染整批，保守回退用 16。
BATCH_SIZE="${BATCH_SIZE:-24}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-500}"
KEEP_RECENT_CHECKPOINTS="${KEEP_RECENT_CHECKPOINTS:-3}"
STEP_SAVE_EVERY="${STEP_SAVE_EVERY:-10000}"
KEEP_RECENT_STEP_CHECKPOINTS="${KEEP_RECENT_STEP_CHECKPOINTS:-3}"
VAL_STEPS="${VAL_STEPS:-500}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-64}"
VAL_SAMPLE_SEED="${VAL_SAMPLE_SEED:-202607}"
ROUTE_LOSS_WEIGHT="${ROUTE_LOSS_WEIGHT:-0.5}"
WAYPOINT_LOSS_WEIGHT="${WAYPOINT_LOSS_WEIGHT:-1.0}"
LOSS_TYPE="${LOSS_TYPE:-l1}"
LEADMOT_ROPE_TYPE="${LEADMOT_ROPE_TYPE:-mrope}"
DECODER_DROPOUT="${DECODER_DROPOUT:-0.1}"
DECODER_DTYPE="${DECODER_DTYPE:-bfloat16}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
QWEN_LOAD_STAGGER_S="${QWEN_LOAD_STAGGER_S:-2.0}"
# EMA defaults on. Set EMA=0 to save raw-only checkpoints.
# EMA decay：2026-06 三轮调优后 step 数从 ~26k 缩到 ~4k/epoch，等效 batch ×6 后单
# step 推进权重的步幅更大；旧默认 0.999 → 等效平均窗仅 ~1000 step（之前 ~6000
# step），EMA shadow 平滑不足。升到 0.9999 让平均窗回到 ~10k 量级，与 NUM_EPOCHS=10
# 配合更稳。短 sanity / 调试可显式 EMA_DECAY=0.999。
EMA="${EMA:-1}"
EMA_DECAY="${EMA_DECAY:-0.9999}"
# TensorBoard planning overlays; set IMAGE_LOG_EVERY=0 to disable.
IMAGE_LOG_EVERY="${IMAGE_LOG_EVERY:-1000}"
IMAGE_LOG_SAMPLES="${IMAGE_LOG_SAMPLES:-4}"
IMAGE_LOG_SEED="${IMAGE_LOG_SEED:-20260101}"
RGB_FRAME_COUNT="${RGB_FRAME_COUNT:-4}"
RGB_FRAME_STEP="${RGB_FRAME_STEP:-1}"
BEV_FRAME_COUNT="${BEV_FRAME_COUNT:-1}"
BEV_FRAME_STEP="${BEV_FRAME_STEP:-1}"
# USE_BEV=1（默认）：decoder 在 gen 序列里拼 120 个 BEV token，对应 v1 全套行为。
# USE_BEV=0：消融配置，decoder 完全靠 frozen Qwen prefix K/V + ego 状态做 planning，
# BEV encoder 也跳过 forward 不算图，节省一份 LEAD TransfuserBackbone forward 时间/显存。
# 注意切换 USE_BEV 会导致 state_dict 不兼容（decoder.bev_projector 子模块存在性变化），
# 不能跨 USE_BEV 加载 --init-from-ckpt。
USE_BEV="${USE_BEV:-1}"
FRAME_INTERVAL_S="${FRAME_INTERVAL_S:-0.25}"
TP_LOOKAHEAD_S="${TP_LOOKAHEAD_S:-1.0}"
NTP_LOOKAHEAD_S="${NTP_LOOKAHEAD_S:-2.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

pick_idle_gpus() {
  local count="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t, -k2,2n \
    | head -n "${count}" \
    | awk -F, '{gsub(/ /, "", $1); print $1}' \
    | paste -sd, -
}

require_idle_gpus() {
  local count="$1"
  local picked
  picked="$(pick_idle_gpus "${count}" || true)"
  if [[ -z "${picked}" ]]; then
    echo "No GPU selected by nvidia-smi; refusing to reuse external CUDA_VISIBLE_DEVICES." >&2
    exit 1
  fi
  echo "${picked}"
}

count_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' '
  else
    echo 0
  fi
}

is_port_free() {
  local port="$1"
  python - "$port" <<'PY'
import socket, sys
port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.bind(("", port))
    except OSError:
        raise SystemExit(1)
PY
}

find_free_master_port() {
  python - <<'PY' 2>/dev/null || echo "$((20000 + RANDOM % 20000))"
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
}

configure_master_port() {
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  if [[ "${LEADMOT_RESPECT_MASTER_PORT:-0}" == "1" && -n "${MASTER_PORT:-}" ]]; then
    if is_port_free "${MASTER_PORT}"; then
      export MASTER_PORT
      return 0
    fi
    echo "[port][err] MASTER_PORT=${MASTER_PORT} is already in use and LEADMOT_RESPECT_MASTER_PORT=1" >&2
    exit 1
  fi
  if [[ -n "${MASTER_PORT:-}" ]]; then
    if is_port_free "${MASTER_PORT}"; then
      export MASTER_PORT
      return 0
    fi
    echo "[port][warn] MASTER_PORT=${MASTER_PORT} is already in use; selecting a free port"
  fi
  export MASTER_PORT="$(find_free_master_port)"
}

export_torchrun_master_env() {
  export PET_MASTER_ADDR="${MASTER_ADDR}"
  export PET_MASTER_PORT="${MASTER_PORT}"
}

common_args=(
  --train-jsonl "${TRAIN_JSONL}"
  --val-jsonl "${VAL_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --model-dir "${MODEL_DIR}"
  --lead-bev-ckpt "${LEAD_BEV_CKPT}"
  --learning-rate "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --warmup-ratio "${WARMUP_RATIO}"
  --num-epochs "${NUM_EPOCHS}"
  --grad-accum-steps "${GRAD_ACC}"
  --batch-size "${BATCH_SIZE}"
  --logging-steps "${LOGGING_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --keep-recent-checkpoints "${KEEP_RECENT_CHECKPOINTS}"
  --step-save-every "${STEP_SAVE_EVERY}"
  --keep-recent-step-checkpoints "${KEEP_RECENT_STEP_CHECKPOINTS}"
  --val-steps "${VAL_STEPS}"
  --val-max-samples "${VAL_MAX_SAMPLES}"
  --val-sample-seed "${VAL_SAMPLE_SEED}"
  --route-loss-weight "${ROUTE_LOSS_WEIGHT}"
  --waypoint-loss-weight "${WAYPOINT_LOSS_WEIGHT}"
  --loss-type "${LOSS_TYPE}"
  --leadmot-rope-type "${LEADMOT_ROPE_TYPE}"
  --decoder-dropout "${DECODER_DROPOUT}"
  --decoder-dtype "${DECODER_DTYPE}"
  --qwen-dtype "${QWEN_DTYPE}"
  --qwen-load-stagger-s "${QWEN_LOAD_STAGGER_S}"
  --ema-decay "${EMA_DECAY}"
  --image-log-every "${IMAGE_LOG_EVERY}"
  --image-log-samples "${IMAGE_LOG_SAMPLES}"
  --image-log-seed "${IMAGE_LOG_SEED}"
  --rgb-frame-count "${RGB_FRAME_COUNT}"
  --rgb-frame-step "${RGB_FRAME_STEP}"
  --bev-frame-count "${BEV_FRAME_COUNT}"
  --bev-frame-step "${BEV_FRAME_STEP}"
  --frame-interval-s "${FRAME_INTERVAL_S}"
  --target-point-lookahead-s "${TP_LOOKAHEAD_S}"
  --next-target-point-lookahead-s "${NTP_LOOKAHEAD_S}"
)

if [[ -n "${RESUME}" ]]; then
  common_args+=(--resume "${RESUME}")
fi
if [[ -n "${INIT_FROM_CKPT}" ]]; then
  common_args+=(--init-from-ckpt "${INIT_FROM_CKPT}")
fi
# EMA=0 disables EMA; eval/probe default to --use-ema and fall back to raw if missing.
# Raw-only checkpoints remain compatible with eval/probe.
if [[ "${EMA}" == "0" ]]; then
  common_args+=(--no-ema)
else
  common_args+=(--ema)
fi
# USE_BEV=0 时把 BooleanOptionalAction 的 --no-use-bev 透传过去，关闭 decoder BEV 通路。
if [[ "${USE_BEV}" == "0" ]]; then
  common_args+=(--no-use-bev)
else
  common_args+=(--use-bev)
fi

case "${MODE}" in
  check)
    export CUDA_VISIBLE_DEVICES="$(require_idle_gpus 1)"
    python qwen3vl_local/leadmot/train.py \
      "${common_args[@]}" \
      --limit-train-samples "${LIMIT_TRAIN_SAMPLES:-2}" \
      --limit-val-samples "${LIMIT_VAL_SAMPLES:-1}" \
      --max-train-steps "${MAX_TRAIN_STEPS:-2}" \
      --save-steps 0 \
      --val-steps 0 \
      --image-log-every 0 \
      --no-tb \
      ${EXTRA_ARGS}
    ;;
  single)
    export CUDA_VISIBLE_DEVICES="$(require_idle_gpus 1)"
    python qwen3vl_local/leadmot/train.py "${common_args[@]}" ${EXTRA_ARGS}
    ;;
  ddp)
    DDP_GPU_COUNT="${DDP_GPU_COUNT:-8}"
    export CUDA_VISIBLE_DEVICES="$(require_idle_gpus "${DDP_GPU_COUNT}")"
    configure_master_port
    export_torchrun_master_env
    NPROC_PER_NODE="$(count_visible_gpus)"
    if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
      echo "No GPU found for ddp mode." >&2
      exit 1
    fi
    echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "[gpu] NPROC_PER_NODE=${NPROC_PER_NODE}"
    echo "[port] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} PET_MASTER_PORT=${PET_MASTER_PORT}"
    torchrun \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      qwen3vl_local/leadmot/train.py "${common_args[@]}" ${EXTRA_ARGS}
    ;;
  *)
    echo "Usage: $0 [check|single|ddp]" >&2
    exit 2
    ;;
esac

# ---------------------------------------------------------------------------
# 产物布局（OUTPUT_DIR 平铺，与 eval.py / probe.py 同根）：
#   OUTPUT_DIR/
#     ├─ best.pt + best.json        val/loss 历史最小（eval/probe 默认指向它）
#     ├─ latest.pt                  最近一次保存（无 val 时回退到它）
#     ├─ checkpoint-epochNN.pt      各 epoch 全量 ckpt（保留最近 KEEP_RECENT_CHECKPOINTS）
#     ├─ step-checkpoint-NNNNNN.pt  每 STEP_SAVE_EVERY 步快照（保留最近 KEEP_RECENT_STEP_CHECKPOINTS）
#     ├─ tb/                        训练 TensorBoard events
#     └─ invocations/               每次启动的 argv / env / git commit
# ---------------------------------------------------------------------------
if [[ "${MODE}" != "check" ]]; then
  echo ""
  echo "============================================================"
  echo "[hint] TensorBoard:"
  echo "  bash qwen3vl_local/tb_serve.sh ${OUTPUT_DIR}"
  echo "[hint] offline eval:"
  echo "  torchrun --standalone --nproc_per_node=4 qwen3vl_local/leadmot/eval.py --save-root ${OUTPUT_DIR}"
  echo "[hint] case probe:"
  echo "  python qwen3vl_local/leadmot/probe.py --save-root ${OUTPUT_DIR}"
  echo "============================================================"
fi

