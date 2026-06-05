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

TRAIN_JSONL="${TRAIN_JSONL:-checkpoints/leadmot_v1_data/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-checkpoints/leadmot_v1_data/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/leadmot_v1_decoder}"
MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
LEAD_BEV_CKPT="${LEAD_BEV_CKPT:-checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth}"
RESUME="${RESUME:-}"
INIT_FROM_CKPT="${INIT_FROM_CKPT:-}"

LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
GRAD_ACC="${GRAD_ACC:-8}"
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
# Longer schedules can try EMA_DECAY=0.9999; short runs should keep 0.999.
EMA="${EMA:-1}"
EMA_DECAY="${EMA_DECAY:-0.999}"
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
TP_LOOKAHEAD_S="${TP_LOOKAHEAD_S:-1.5}"
NTP_LOOKAHEAD_S="${NTP_LOOKAHEAD_S:-3.0}"
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
  local start="${1:-29500}"
  local port
  for port in $(seq "${start}" "$((start + 200))"); do
    if is_port_free "${port}"; then
      echo "${port}"
      return 0
    fi
  done
  echo "No free port found near ${start}" >&2
  return 1
}

configure_master_port() {
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  if [[ -z "${MASTER_PORT:-}" ]]; then
    export MASTER_PORT="$(find_free_master_port 29500)"
  elif [[ "${LEADMOT_RESPECT_MASTER_PORT:-1}" != "0" ]]; then
    if ! is_port_free "${MASTER_PORT}"; then
      echo "MASTER_PORT=${MASTER_PORT} is busy. Set MASTER_PORT or LEADMOT_RESPECT_MASTER_PORT=0." >&2
      exit 1
    fi
  fi
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
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      picked="$(pick_idle_gpus 1 || true)"
      if [[ -n "${picked}" ]]; then
        export CUDA_VISIBLE_DEVICES="${picked}"
      fi
    fi
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
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      picked="$(pick_idle_gpus 1 || true)"
      if [[ -n "${picked}" ]]; then
        export CUDA_VISIBLE_DEVICES="${picked}"
      fi
    fi
    python qwen3vl_local/leadmot/train.py "${common_args[@]}" ${EXTRA_ARGS}
    ;;
  ddp)
    if [[ -n "${DDP_GPU_COUNT:-}" ]]; then
      picked="$(pick_idle_gpus "${DDP_GPU_COUNT}" || true)"
      if [[ -n "${picked}" ]]; then
        export CUDA_VISIBLE_DEVICES="${picked}"
      fi
    elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      picked="$(pick_idle_gpus 8 || true)"
      if [[ -n "${picked}" ]]; then
        export CUDA_VISIBLE_DEVICES="${picked}"
      fi
    fi
    configure_master_port
    NPROC_PER_NODE="${NPROC_PER_NODE:-$(count_visible_gpus)}"
    if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
      echo "No GPU found for ddp mode." >&2
      exit 1
    fi
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
  echo "  bash tools/tb_serve.sh ${OUTPUT_DIR}"
  echo "[hint] offline eval:"
  echo "  torchrun --standalone --nproc_per_node=4 qwen3vl_local/leadmot/eval.py --save-root ${OUTPUT_DIR}"
  echo "[hint] case probe:"
  echo "  python qwen3vl_local/leadmot/probe.py --save-root ${OUTPUT_DIR}"
  echo "============================================================"
fi

