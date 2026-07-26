#!/usr/bin/env bash
# SFT base 训练 launcher：RS/EVENT 两问直接 token SFT + true torch DDP。
#
# 从 AutoMoT/ 目录运行：
#   GPU_IDS=0 bash qwen3vl_local/sft_base/train.sh single
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_base/train.sh ddp
#   GPU_IDS=0 bash qwen3vl_local/sft_base/train.sh check

set -euo pipefail

# 禁用 core dump，避免失败的训练/工具进程在仓库里留下 core.* 大文件。
ulimit -S -c 0 2>/dev/null || true

MODE="${1:-${MODE:-ddp}}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

TRAIN_INDEX="${TRAIN_INDEX:-checkpoints/sft_base_data/train_sequence_index.jsonl}"
VAL_INDEX="${VAL_INDEX:-checkpoints/sft_base_data/val_sequence_index.jsonl}"
MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft_base_runs}"
OUTPUT_DIR_BASE="${OUTPUT_DIR}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
  # 防覆盖约定：用户给的是 base OUTPUT_DIR，真实训练产物落到 run_<RUN_TAG>/。
  # base 层的 latest 软链始终指向最近一次 run，方便 eval/probe 文档写稳定路径。
  OUTPUT_DIR="${OUTPUT_DIR_BASE}/run_${RUN_TAG}"
fi

HF_HOME="${HF_HOME:-${OUTPUT_DIR_BASE}/.hf_cache}"
export HF_HOME
mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"

if [[ "${QWEN3VL_LOG_TO_FILE:-1}" != "0" && -z "${QWEN3VL_LOG_ACTIVE:-}" ]]; then
  # 所有 rank 的 stdout/stderr 默认落到 log.txt；QWEN3VL_LOG_ACTIVE 防止递归 tee。
  export QWEN3VL_LOG_ACTIVE=1
  exec > >(tee -a "${OUTPUT_DIR}/log.txt") 2>&1
  echo "[log] tee stdout/stderr to ${OUTPUT_DIR}/log.txt"
fi

if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
  ln -sfn "run_${RUN_TAG}" "${OUTPUT_DIR_BASE}/latest"
  echo "[run] OUTPUT_DIR=${OUTPUT_DIR}  (latest -> run_${RUN_TAG})"
fi

pick_idle_gpus() {
  local want_count="$1"
  local selected
  if command -v nvidia-smi >/dev/null 2>&1; then
    # 自动选卡规则与 v3/v4 保持一致：优先显存占用低，再看 GPU util。
    # 显式 GPU_IDS 非空时不会走这里，便于用户固定复现实验。
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
    echo "[mode] check"
    export CUDA_VISIBLE_DEVICES="$(resolve_visible_gpus 1)"
    EXTRA_ARGS+=("--check")
    NPROC=1
    ;;
  ddp)
    echo "[mode] ddp"
    DDP_GPU_COUNT="${DDP_GPU_COUNT:-${NPROC_PER_NODE:-4}}"
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
    # sft_base 使用普通 DDP + teacher-forced CE；这里仅负责进程数和 master 端口。
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    # NCCL RAS 在共享服务器上可能因默认端口冲突打印 bind failed；训练不依赖 RAS。
    export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use single/ddp/check." >&2
    exit 1
    ;;
esac

if [[ "${STRICT_VISION_SCOPE:-1}" == "1" ]]; then
  EXTRA_ARGS+=("--strict-vision-scope")
else
  EXTRA_ARGS+=("--no-strict-vision-scope")
fi

if [[ "${NO_GRAD_CHECKPOINT:-0}" == "1" ]]; then
  EXTRA_ARGS+=("--no-grad-checkpoint")
fi

if [[ "${VISION_GUARD_ENABLED:-1}" == "1" ]]; then
  EXTRA_ARGS+=("--vision-guard-enabled")
else
  EXTRA_ARGS+=("--no-vision-guard-enabled")
fi

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[gpu] NPROC=${NPROC}"

COMMON_ARGS=(
  # 下面所有参数都可以通过同名大写环境变量覆盖；这里保持和 v3/v4 launcher
  # 类似的写法，方便远端批量实验只改 shell 环境，不手动编辑脚本。
  --train-index "${TRAIN_INDEX}"
  --val-index "${VAL_INDEX}"
  --model-dir "${MODEL_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE:-${PER_DEVICE_BS:-1}}"
  --grad-accum "${GRAD_ACCUM:-1}"
  --num-epochs "${NUM_EPOCHS:-1}"
  --learning-rate "${LEARNING_RATE:-${LR:-3e-5}}"
  --weight-decay "${WEIGHT_DECAY:-0.05}"
  --warmup-ratio "${WARMUP_RATIO:-0.03}"
  --lora-rank "${LORA_RANK:-16}"
  --lora-alpha "${LORA_ALPHA:-32}"
  --lora-dropout "${LORA_DROPOUT:-0.1}"
  --lora-vision-scope "${LORA_VISION_SCOPE:-merger}"
  # sft_base 默认微调视觉桥接层；仍给较小 LR/clip，避免 RS/EVENT 小任务冲坏视觉层。
  --vision-lr-scale "${VISION_LR_SCALE:-0.1}"
  --max-vision-lr-scale "${MAX_VISION_LR_SCALE:-0.25}"
  --language-clip-norm "${LANGUAGE_CLIP_NORM:-1.0}"
  --vision-clip-norm "${VISION_CLIP_NORM:-0.3}"
  --vision-guard-grad-norm-max "${VISION_GUARD_GRAD_NORM_MAX:-10.0}"
  --vision-guard-param-norm-max "${VISION_GUARD_PARAM_NORM_MAX:-200.0}"
  --vision-guard-patience "${VISION_GUARD_PATIENCE:-3}"
  --max-length "${MAX_LENGTH:-8192}"
  --max-routes "${MAX_ROUTES:-0}"
  --max-frames-per-route "${MAX_FRAMES_PER_ROUTE:-0}"
  --num-workers "${NUM_WORKERS:-0}"
  --logging-steps "${LOGGING_STEPS:-5}"
  --save-steps "${SAVE_STEPS:-200}"
  --eval-steps "${EVAL_STEPS:-200}"
  --max-eval-samples "${MAX_EVAL_SAMPLES:-256}"
  --max-steps "${MAX_STEPS:-0}"
  --frames-per-sync "${FRAMES_PER_SYNC:-64}"
  --ue-event-loss-weight "${UE_EVENT_LOSS_WEIGHT:-3.0}"
  --re-event-loss-weight "${RE_EVENT_LOSS_WEIGHT:-1.0}"
  --ue-frame-repeat "${UE_FRAME_REPEAT:-2}"
  --seed "${SEED:-20260711}"
  "${EXTRA_ARGS[@]}"
)

if [[ "${NPROC}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    qwen3vl_local/sft_base/train.py "${COMMON_ARGS[@]}"
else
  python qwen3vl_local/sft_base/train.py "${COMMON_ARGS[@]}"
fi

echo "[done] adapter under ${OUTPUT_DIR}"
echo "[hint] eval: GPU_IDS=0 python qwen3vl_local/sft_base/eval.py --index ${VAL_INDEX} --model-dir ${MODEL_DIR} --adapter-dir ${OUTPUT_DIR}/final --output-json ${OUTPUT_DIR}/eval_metrics.json"
