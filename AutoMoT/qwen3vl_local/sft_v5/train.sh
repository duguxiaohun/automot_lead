#!/usr/bin/env bash
# SFT v5 训练 launcher：RS/EVENT 两问 OPSD + torchrun 多进程训练。
#
# 从 AutoMoT/ 目录运行：
#   GPU_IDS=0 bash qwen3vl_local/sft_v5/train.sh single
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp
#   GPU_IDS=0 bash qwen3vl_local/sft_v5/train.sh check
#
# ddp 模式默认按四张 H20 的当前“max_util”口径启动：每卡 8 条 route sequence，
# 同一 timestep 最多 8 个 frame 做 batched Qwen rollout，优先追求 GPU 利用率；
# optimizer 默认按 512 个 global frame 组成流式窗口，最迟 32 个 timestep 更新一次，
# 不再等待整个超长 route batch 结束。
# single/check 模式仍保守用 1，避免单卡调试时意外把显存吃爆。
# 用户可用 BATCH_PROFILE=debug/balanced/max_util 在 4/6/8 路间切换；
# 显式传 PER_DEVICE_BATCH_SIZE / QWEN_BATCH_SIZE 时永远优先使用
# 用户配置。多卡默认启用 length_balanced sampler，按 route frame 数均衡各 rank；
# 如需复现旧分片行为，可设置 SAMPLER_MODE=distributed。

set -euo pipefail

# 禁用 core dump，避免失败的训练/工具进程在仓库里留下 core.* 大文件。
ulimit -S -c 0 2>/dev/null || true

MODE="${1:-${MODE:-ddp}}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# v5 的 Qwen3-VL OPSD 会频繁创建/释放 KV 与 logits 张量；启用 expandable
# segments 可降低 CUDA allocator 碎片化导致的小额分配 OOM。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_INDEX="${TRAIN_INDEX:-checkpoints/sft_v5_data/train_sequence_index.jsonl}"
VAL_INDEX="${VAL_INDEX:-checkpoints/sft_v5_data/val_sequence_index.jsonl}"
MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft_v5_runs}"
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
    # v5 使用 torchrun 多进程；各 rank 的 batch 会先 local padding，再由 train.py
    # all-reduce 出 global max_T，并手动 all-reduce LoRA 梯度。这里仅负责进程数和 master 端口。
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    # NCCL 2.26 RAS 会额外尝试绑定本地监听端口；同机多实验时容易出现
    # "NCCL WARN Call to bind failed: Address already in use"。训练不依赖 RAS，
    # 默认关闭，用户需要 NCCL RAS 诊断时可显式 NCCL_RAS_ENABLE=1。
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
if [[ "${PARALLEL_KL:-1}" == "0" ]]; then
  EXTRA_ARGS+=("--no-parallel-kl")
else
  EXTRA_ARGS+=("--parallel-kl")
fi
if [[ "${CHECKPOINT_PROBE:-1}" == "0" ]]; then
  EXTRA_ARGS+=("--no-checkpoint-probe")
else
  EXTRA_ARGS+=("--checkpoint-probe")
fi
if [[ "${CHECKPOINT_PROBE_BASE:-1}" == "0" ]]; then
  EXTRA_ARGS+=("--no-checkpoint-probe-base")
else
  EXTRA_ARGS+=("--checkpoint-probe-base")
fi
if [[ "${CHECKPOINT_PROBE_WITH_TEACHER:-1}" == "0" ]]; then
  EXTRA_ARGS+=("--no-checkpoint-probe-with-teacher")
else
  EXTRA_ARGS+=("--checkpoint-probe-with-teacher")
fi

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[gpu] NPROC=${NPROC}"

# launcher 统一在这里决定“默认 batch 口径”。这样文档里的
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp
# 就会真实变成 H20 max_util 8 路，而不是只启动 4 个 rank、每个 rank 内部仍单样本。
BATCH_PROFILE="${BATCH_PROFILE:-${SFT_V5_BATCH_PROFILE:-max_util}}"
if [[ "${MODE}" == "ddp" ]]; then
  case "${BATCH_PROFILE}" in
    debug|conservative|4)
      DEFAULT_PER_DEVICE_BATCH_SIZE=4
      DEFAULT_QWEN_BATCH_SIZE=4
      ;;
    balanced|6)
      DEFAULT_PER_DEVICE_BATCH_SIZE=6
      DEFAULT_QWEN_BATCH_SIZE=6
      ;;
    aggressive|max_util|8)
      DEFAULT_PER_DEVICE_BATCH_SIZE=8
      DEFAULT_QWEN_BATCH_SIZE=8
      ;;
    *)
      echo "[batch][error] unknown BATCH_PROFILE=${BATCH_PROFILE}; use debug/balanced/max_util or set PER_DEVICE_BATCH_SIZE/QWEN_BATCH_SIZE" >&2
      exit 1
      ;;
  esac
  DEFAULT_PROGRESS_FRAMES=20
else
  BATCH_PROFILE="single"
  DEFAULT_PER_DEVICE_BATCH_SIZE=1
  DEFAULT_QWEN_BATCH_SIZE=1
  DEFAULT_PROGRESS_FRAMES=5
fi

EFFECTIVE_PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-${PER_DEVICE_BS:-${DEFAULT_PER_DEVICE_BATCH_SIZE}}}"
# 如果用户只调 PER_DEVICE_BATCH_SIZE/PER_DEVICE_BS 而没调 QWEN_BATCH_SIZE，
# Qwen rollout batch 默认跟随每卡 route 数；这样 max_util 默认是 8/8，
# 降到 balanced/debug/2 路时也会自然变成 6/6、4/4、2/2。
EFFECTIVE_QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-${EFFECTIVE_PER_DEVICE_BATCH_SIZE:-${DEFAULT_QWEN_BATCH_SIZE}}}"
EFFECTIVE_PROGRESS_FRAMES="${PROGRESS_FRAMES:-${DEFAULT_PROGRESS_FRAMES}}"
# 正式训练默认按全局有效 frame 组成短 optimizer 窗口：四卡 8 路时一个
# timestep 最多贡献 32 frame，通常约 16 个 timestep 达到 512。若后期 route
# 大量结束导致每个 timestep 的有效 frame 变少，32 timestep 上限负责及时更新。
# GRAD_ACCUM 会在 train.py 内同时放大下面两个阈值；这里保留原始配置用于清晰打印。
EFFECTIVE_UPDATE_MODE="${UPDATE_MODE:-streaming_frames}"
EFFECTIVE_TARGET_GLOBAL_FRAMES="${TARGET_GLOBAL_FRAMES_PER_STEP:-512}"
EFFECTIVE_MAX_TIMESTEPS="${MAX_TIMESTEPS_PER_STEP:-32}"
# 8 路 rollout 没有 autograd graph，可以保持吞吐；Q2 KL 约 3k token 时有梯度的
# scoring 才是显存峰值。正式 H20 配置默认拆成 2+2+2+2 并逐微批 backward，
# 不降低 rollout 并行度，也不缩短 Q1/Q2 的 1024 token 安全上限。
EFFECTIVE_PARALLEL_KL_MICROBATCH_SIZE="${PARALLEL_KL_MICROBATCH_SIZE:-2}"

echo "[batch] PER_DEVICE_BATCH_SIZE=${EFFECTIVE_PER_DEVICE_BATCH_SIZE}"
echo "[batch] QWEN_BATCH_SIZE=${EFFECTIVE_QWEN_BATCH_SIZE}"
echo "[batch] GRAD_ACCUM=${GRAD_ACCUM:-1}"
echo "[batch] BATCH_PROFILE=${BATCH_PROFILE}"
echo "[sampler] SAMPLER_MODE=${SAMPLER_MODE:-length_balanced}"
echo "[parallel] PARALLEL_KL=${PARALLEL_KL:-1}"
echo "[parallel] PARALLEL_KL_MICROBATCH_SIZE=${EFFECTIVE_PARALLEL_KL_MICROBATCH_SIZE}"
echo "[update] UPDATE_MODE=${EFFECTIVE_UPDATE_MODE}"
echo "[update] TARGET_GLOBAL_FRAMES_PER_STEP=${EFFECTIVE_TARGET_GLOBAL_FRAMES}"
echo "[update] MAX_TIMESTEPS_PER_STEP=${EFFECTIVE_MAX_TIMESTEPS}"
echo "[save] SAVE_STEPS=${SAVE_STEPS:-40}"
echo "[probe] CHECKPOINT_PROBE=${CHECKPOINT_PROBE:-1} BASE=${CHECKPOINT_PROBE_BASE:-1} TEACHER=${CHECKPOINT_PROBE_WITH_TEACHER:-1} CASES=${CHECKPOINT_PROBE_NUM_CASES:-24} ROUTES=${CHECKPOINT_PROBE_NUM_ROUTES:-1} MODE=${CHECKPOINT_PROBE_SAMPLE_MODE:-random} ARTIFACTS=${CHECKPOINT_PROBE_ARTIFACT_LEVEL:-review} RADIUS=${CHECKPOINT_PROBE_CONTEXT_RADIUS:-8}"
if [[ "${MODE}" == "ddp" && "${EFFECTIVE_PER_DEVICE_BATCH_SIZE}" -lt "${EFFECTIVE_QWEN_BATCH_SIZE}" ]]; then
  echo "[batch][warn] QWEN_BATCH_SIZE=${EFFECTIVE_QWEN_BATCH_SIZE} > PER_DEVICE_BATCH_SIZE=${EFFECTIVE_PER_DEVICE_BATCH_SIZE}; extra Qwen slots will be unused"
fi

COMMON_ARGS=(
  # 下面所有参数都可以通过同名大写环境变量覆盖；这里保持和 v3/v4 launcher
  # 类似的写法，方便远端批量实验只改 shell 环境，不手动编辑脚本。
  --train-index "${TRAIN_INDEX}"
  --val-index "${VAL_INDEX}"
  --model-dir "${MODEL_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --per-device-batch-size "${EFFECTIVE_PER_DEVICE_BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM:-1}"
  --num-epochs "${NUM_EPOCHS:-1}"
  --learning-rate "${LEARNING_RATE:-${LR:-1e-5}}"
  --weight-decay "${WEIGHT_DECAY:-0.05}"
  --warmup-ratio "${WARMUP_RATIO:-0.03}"
  --lora-rank "${LORA_RANK:-16}"
  --lora-alpha "${LORA_ALPHA:-32}"
  --lora-dropout "${LORA_DROPOUT:-0.1}"
  --lora-vision-scope "${LORA_VISION_SCOPE:-off}"
  # 视觉 LoRA 默认 off；开启时仍给较小 LR/clip，避免 RS/EVENT 小任务冲坏视觉层。
  --vision-lr-scale "${VISION_LR_SCALE:-0.1}"
  --language-clip-norm "${LANGUAGE_CLIP_NORM:-1.0}"
  --vision-clip-norm "${VISION_CLIP_NORM:-0.3}"
  --max-new-tokens-q1 "${MAX_NEW_TOKENS_Q1:-1024}"
  --max-new-tokens-q2 "${MAX_NEW_TOKENS_Q2:-1024}"
  --temperature "${TEMPERATURE:-1.0}"
  # RS_SLOW 以 4 帧为中心；下一行默认把每次稳定复核间隔随机化为 3/4/5 帧。
  --rs-slow-interval "${RS_SLOW_INTERVAL:-4}"
  # 正式默认不锁死 4 帧；每次稳定 RS query 后按 route/seed 可复现地从
  # 3/4/5 帧选下一次复核间隔。设 RS_SLOW_INTERVAL_JITTER=0 可做固定周期消融。
  --rs-slow-interval-jitter "${RS_SLOW_INTERVAL_JITTER:-1}"
  # 错误记忆课程：稳定 RS 低频，错误期间逐帧运行 Q1；EVENT_FAST 每个 RS 有效帧运行。
  # EVENT 只有在 RS gate 正确、实际进入 Q2 后才累计自己的错误 streak。
  --rs-error-patience "${RS_ERROR_PATIENCE:-4}"
  --event-error-patience "${EVENT_ERROR_PATIENCE:-3}"
  --rs-repair-interval "${RS_REPAIR_INTERVAL:-2}"
  --event-repair-interval "${EVENT_REPAIR_INTERVAL:-1}"
  # 正式默认是延迟硬修复：RS/EVENT 先连续出错用完 patience，到 review slot
  # 才写回 GT。这保留连续纠偏数据，同时避免纯 UNKNOWN 软修复在极端复制模型上
  # 永久卡住。只做消融时可显式设 RS_REPAIR_MODE/EVENT_REPAIR_MODE=unknown。
  --rs-repair-mode "${RS_REPAIR_MODE:-ground_truth}"
  --event-repair-mode "${EVENT_REPAIR_MODE:-ground_truth}"
  # relation curriculum：RS 的 5% contradiction + 7% omission 会因额外触发慢问，
  # 在理想当帧纠偏下约映射为 Q1 60/24/16（aligned/omission/contradiction）。
  # EVENT 的 eligible 比例为 55/25/20；受 RS pre-gate 影响，最终 Q2 理想值
  # 约 60/22/17。closed-loop 实测比例以 TensorBoard 为准。
  --rs-memory-corrupt-prob "${RS_MEMORY_CORRUPT_PROB:-0.05}"
  --rs-memory-unknown-prob "${RS_MEMORY_UNKNOWN_PROB:-0.07}"
  --event-memory-corrupt-prob "${EVENT_MEMORY_CORRUPT_PROB:-0.20}"
  --event-memory-unknown-prob "${EVENT_MEMORY_UNKNOWN_PROB:-0.25}"
  --rs-initial-gt-prob "${RS_INITIAL_GT_PROB:-0.5}"
  --event-initial-gt-prob "${EVENT_INITIAL_GT_PROB:-0.5}"
  --max-routes "${MAX_ROUTES:-0}"
  --max-frames-per-route "${MAX_FRAMES_PER_ROUTE:-0}"
  --num-workers "${NUM_WORKERS:-0}"
  --qwen-batch-size "${EFFECTIVE_QWEN_BATCH_SIZE}"
  --sampler-mode "${SAMPLER_MODE:-length_balanced}"
  --parallel-kl-microbatch-size "${EFFECTIVE_PARALLEL_KL_MICROBATCH_SIZE}"
  --update-mode "${EFFECTIVE_UPDATE_MODE}"
  --target-global-frames-per-step "${EFFECTIVE_TARGET_GLOBAL_FRAMES}"
  --max-timesteps-per-step "${EFFECTIVE_MAX_TIMESTEPS}"
  --logging-steps "${LOGGING_STEPS:-1}"
  --progress-frames "${EFFECTIVE_PROGRESS_FRAMES}"
  --heartbeat-seconds "${HEARTBEAT_SECONDS:-120}"
  # 用户实测约 80 optimizer steps/day；默认 40 step 约半天保存一次，既降低长跑
  # 中断风险，也避免 checkpoint/probe 过于频繁地占用磁盘和暂停训练。
  --save-steps "${SAVE_STEPS:-40}"
  --max-steps "${MAX_STEPS:-0}"
  --checkpoint-probe-num-cases "${CHECKPOINT_PROBE_NUM_CASES:-24}"
  --checkpoint-probe-num-routes "${CHECKPOINT_PROBE_NUM_ROUTES:-1}"
  --checkpoint-probe-sample-mode "${CHECKPOINT_PROBE_SAMPLE_MODE:-random}"
  --checkpoint-probe-context-radius "${CHECKPOINT_PROBE_CONTEXT_RADIUS:-8}"
  --checkpoint-probe-sequence-length "${CHECKPOINT_PROBE_SEQUENCE_LENGTH:-24}"
  --checkpoint-probe-artifact-level "${CHECKPOINT_PROBE_ARTIFACT_LEVEL:-review}"
  --checkpoint-probe-max-new-tokens-q1 "${CHECKPOINT_PROBE_MAX_NEW_TOKENS_Q1:-256}"
  --checkpoint-probe-max-new-tokens-q2 "${CHECKPOINT_PROBE_MAX_NEW_TOKENS_Q2:-192}"
  --seed "${SEED:-20260711}"
  "${EXTRA_ARGS[@]}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  # 只检查 launcher 解析后的 GPU、batch profile 和 train.py 参数，不加载 Qwen 权重。
  # 用法：DRY_RUN=1 GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp
  echo "[dry-run] NPROC=${NPROC}"
  printf "[dry-run] command:"
  if [[ "${NPROC}" -gt 1 ]]; then
    printf " torchrun --nproc_per_node=%q --master_addr=%q --master_port=%q qwen3vl_local/sft_v5/train.py" \
      "${NPROC}" "${MASTER_ADDR}" "${MASTER_PORT}"
  else
    printf " python qwen3vl_local/sft_v5/train.py"
  fi
  printf " %q" "${COMMON_ARGS[@]}"
  printf "\n"
  exit 0
fi

if [[ "${NPROC}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    qwen3vl_local/sft_v5/train.py "${COMMON_ARGS[@]}"
else
  python qwen3vl_local/sft_v5/train.py "${COMMON_ARGS[@]}"
fi

echo "[done] adapter under ${OUTPUT_DIR}"
echo "[hint] eval: GPU_IDS=0 python qwen3vl_local/sft_v5/eval.py --index ${VAL_INDEX} --model-dir ${MODEL_DIR} --adapter-dir ${OUTPUT_DIR}/final --output-json ${OUTPUT_DIR}/eval_metrics.json"
