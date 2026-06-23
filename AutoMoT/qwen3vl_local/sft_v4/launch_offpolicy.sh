#!/usr/bin/env bash
# SFT v4 off-policy launcher: 2 learner DDP ranks + async collectors.
# 关键约定：learner 才进 DDP/NCCL；collector 只读 LoRA snapshot、写 replay。
#
# Run from AutoMoT/:
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v4/launch_offpolicy.sh
#
# Exclusive_Process 机器上每张 GPU 只能承载一个 CUDA 进程，因此本脚本默认使用
# 2 张卡跑 learner DDP、2 张卡各跑 1 个 collector。若需要覆盖忙卡检查，可显式
# 设置 ALLOW_BUSY_GPUS=1。

set -euo pipefail

ulimit -S -c 0 2>/dev/null || true

MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-checkpoints/sft_v4_data/train.jsonl}"
OUTPUT_DIR_BASE="${OUTPUT_DIR:-checkpoints/sft_v4_lora}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/run_${RUN_TAG}"
else
    OUTPUT_DIR="${OUTPUT_DIR_BASE}"
fi
REPLAY_DIR="${REPLAY_DIR:-${OUTPUT_DIR}/replay}"
LATEST_LORA_DIR="${LATEST_LORA_DIR:-${OUTPUT_DIR}/latest_lora}"

MAX_STEPS="${MAX_STEPS:-10000}"
LR="${LR:-3e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SNAPSHOT_EVERY_STEPS="${SNAPSHOT_EVERY_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
REPLAY_STARTUP_TIMEOUT_SEC="${REPLAY_STARTUP_TIMEOUT_SEC:-600}"

REPLAY_CAPACITY="${REPLAY_CAPACITY:-256}"
COLLECTORS_PER_GPU="${COLLECTORS_PER_GPU:-1}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-0}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}"
P_INIT_CORRECT="${P_INIT_CORRECT:-0.5}"
PHASE_B_NOISE_PROB="${PHASE_B_NOISE_PROB:-0.15}"
OUTER_STRIDE="${OUTER_STRIDE:-1}"

LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_VISION_SCOPE="${LORA_VISION_SCOPE:-off}"
VISION_LR_SCALE="${VISION_LR_SCALE:-0.1}"
MAX_VISION_LR_SCALE="${MAX_VISION_LR_SCALE:-0.25}"
LANGUAGE_CLIP_NORM="${LANGUAGE_CLIP_NORM:-1.0}"
VISION_CLIP_NORM="${VISION_CLIP_NORM:-0.3}"
VISION_GUARD_ENABLED="${VISION_GUARD_ENABLED:-1}"
VISION_GUARD_GRAD_NORM_MAX="${VISION_GUARD_GRAD_NORM_MAX:-10.0}"
VISION_GUARD_PARAM_NORM_MAX="${VISION_GUARD_PARAM_NORM_MAX:-200.0}"
VISION_GUARD_PATIENCE="${VISION_GUARD_PATIENCE:-3}"

# 固定离线模式；HF_HOME 挂在 base 层，避免每个 run 子目录重复缓存。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_HOME:-${OUTPUT_DIR_BASE}/.hf_cache}"
mkdir -p "${OUTPUT_DIR}" "${REPLAY_DIR}" "${HF_HOME}"
if [[ "${RESUME_FROM_CHECKPOINT}" == "latest" ]]; then
    prev_latest="${OUTPUT_DIR_BASE}/latest"
    latest_ckpt=""
    if [[ -d "${prev_latest}" || -L "${prev_latest}" ]]; then
        latest_ckpt="$(find "${prev_latest}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
    fi
    if [[ -z "${latest_ckpt}" ]]; then
        echo "[error] RESUME_FROM_CHECKPOINT=latest but no checkpoint-* found under ${prev_latest}" >&2
        exit 1
    fi
    RESUME_FROM_CHECKPOINT="${latest_ckpt}"
fi
if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
    # base/latest 永远指向最近一次 run，方便 eval/probe 和 RESUME_FROM_CHECKPOINT=latest。
    ln -sfn "run_${RUN_TAG}" "${OUTPUT_DIR_BASE}/latest"
fi

if [[ "${QWEN3VL_LOG_TO_FILE:-1}" != "0" && -z "${QWEN3VL_LOG_ACTIVE:-}" ]]; then
    export QWEN3VL_LOG_ACTIVE=1
    exec > >(tee -a "${OUTPUT_DIR}/offpolicy.log") 2>&1
    echo "[log] tee stdout/stderr to ${OUTPUT_DIR}/offpolicy.log"
fi

pick_idle_gpus() {
    local want_count="$1"
    local selected
    if command -v nvidia-smi >/dev/null 2>&1; then
        selected="$(
            nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
                | awk -F',' -v max_mem="${GPU_MAX_USED_MB}" '{gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); if ($2 <= max_mem) print $2, $3, $1}' \
                | sort -n -k1,1 -k2,2 \
                | head -n "${want_count}" \
                | awk '{print $3}' \
                | paste -sd, -
        )"
        if [[ -n "${selected}" ]]; then
            echo "${selected}"
            return 0
        fi
        return 1
    fi
    seq -s, 0 "$((want_count - 1))"
}

gpu_compute_mode() {
    local gpu="$1"
    nvidia-smi -i "${gpu}" --query-gpu=compute_mode --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true
}

gpu_memory_used_mb() {
    local gpu="$1"
    nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}' || echo 999999
}

assert_gpu_plan_safe() {
    local csv="$1"
    local collectors_per_gpu="$2"
    local seen=","
    local has_exclusive=0
    local gpu used mode

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[gpu][warn] nvidia-smi not found; skip GPU safety checks" >&2
        return 0
    fi

    IFS=',' read -r -a _gpu_plan_array <<< "${csv}"
    for gpu in "${_gpu_plan_array[@]}"; do
        gpu="$(tr -d ' ' <<< "${gpu}")"
        if [[ -z "${gpu}" ]]; then
            continue
        fi
        if [[ "${seen}" == *",${gpu},"* ]]; then
            echo "[error] duplicate GPU ${gpu} in GPU_IDS=${csv}; Exclusive_Process needs one CUDA process per physical GPU" >&2
            exit 1
        fi
        seen="${seen}${gpu},"

        mode="$(gpu_compute_mode "${gpu}")"
        used="$(gpu_memory_used_mb "${gpu}")"
        echo "[gpu] id=${gpu} compute_mode=${mode:-unknown} memory_used=${used}MiB"
        if [[ "${mode}" == "Exclusive_Process" || "${mode}" == "E.Process" ]]; then
            has_exclusive=1
        fi
        if [[ "${ALLOW_BUSY_GPUS}" != "1" && "${used}" -gt "${GPU_MAX_USED_MB}" ]]; then
            echo "[error] GPU ${gpu} is busy (${used}MiB > ${GPU_MAX_USED_MB}MiB). Pick idle GPUs or set ALLOW_BUSY_GPUS=1 intentionally." >&2
            exit 1
        fi
    done

    if [[ "${has_exclusive}" == "1" && "${collectors_per_gpu}" -gt 1 ]]; then
        echo "[error] detected Exclusive_Process GPU mode, but COLLECTORS_PER_GPU=${collectors_per_gpu}. Use COLLECTORS_PER_GPU=1." >&2
        exit 1
    fi
}

split_csv() {
    local csv="$1"
    local idx="$2"
    awk -v n="$idx" -F',' '{gsub(/ /, "", $n); print $n}' <<< "${csv}"
}

find_free_master_port() {
    python -c 'import socket
s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null || echo "$((20000 + RANDOM % 20000))"
}

if [[ -n "${GPU_IDS:-}" ]]; then
    visible="${GPU_IDS}"
else
    if ! visible="$(pick_idle_gpus 4)"; then
        visible=""
    fi
fi
gpu_count="$(awk -F',' '{print NF}' <<< "${visible}")"
if [[ "${gpu_count}" -lt 4 ]]; then
    echo "[error] need 4 idle GPUs via GPU_IDS or auto selection; got ${visible}" >&2
    echo "[hint] current filter requires memory.used <= ${GPU_MAX_USED_MB}MiB; set GPU_IDS=0,1,2,3 or relax GPU_MAX_USED_MB if appropriate" >&2
    exit 1
fi
assert_gpu_plan_safe "${visible}" "${COLLECTORS_PER_GPU}"
learner_gpus="$(split_csv "${visible}" 1),$(split_csv "${visible}" 2)"
collector_gpus="$(split_csv "${visible}" 3),$(split_csv "${visible}" 4)"

IFS=',' read -r -a collector_gpu_array <<< "${collector_gpus}"
collector_processes=$(( ${#collector_gpu_array[@]} * COLLECTORS_PER_GPU ))
master_port="${MASTER_PORT:-$(find_free_master_port)}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${master_port}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"

echo "[run] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[gpu] learner=${learner_gpus} collector=${collector_gpus} collectors_per_gpu=${COLLECTORS_PER_GPU}"
echo "[cfg] max_steps=${MAX_STEPS} replay_capacity=${REPLAY_CAPACITY} startup_replay_timeout=${REPLAY_STARTUP_TIMEOUT_SEC}s p_init=${P_INIT_CORRECT} phase_b_noise=${PHASE_B_NOISE_PROB}"

resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    resume_args=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
vision_guard_args=()
if [[ "${VISION_GUARD_ENABLED}" == "1" ]]; then
    vision_guard_args=(--vision-guard-enabled)
else
    vision_guard_args=(--no-vision-guard-enabled)
fi

rm -f "${OUTPUT_DIR}/STOP"

pids=()
cleanup() {
    echo "[launcher] stopping children"
    echo "stop" > "${OUTPUT_DIR}/STOP" || true
    for pid in "${pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup INT TERM

# learner 独占前两张卡并进入 DDP；视觉 LoRA guard 参数只影响 learner。
CUDA_VISIBLE_DEVICES="${learner_gpus}" torchrun --nproc_per_node=2 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    qwen3vl_local/sft_v4/learn.py \
    --model-dir "${MODEL_DIR}" \
    --replay-dir "${REPLAY_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-steps "${MAX_STEPS}" \
    --learning-rate "${LR}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --startup-replay-timeout-sec "${REPLAY_STARTUP_TIMEOUT_SEC}" \
    --snapshot-every-steps "${SNAPSHOT_EVERY_STEPS}" \
    --save-steps "${SAVE_STEPS}" \
    "${resume_args[@]}" \
    --lora-rank "${LORA_RANK}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-dropout "${LORA_DROPOUT}" \
    --lora-vision-scope "${LORA_VISION_SCOPE}" \
    --vision-lr-scale "${VISION_LR_SCALE}" \
    --max-vision-lr-scale "${MAX_VISION_LR_SCALE}" \
    --language-clip-norm "${LANGUAGE_CLIP_NORM}" \
    --vision-clip-norm "${VISION_CLIP_NORM}" \
    "${vision_guard_args[@]}" \
    --vision-guard-grad-norm-max "${VISION_GUARD_GRAD_NORM_MAX}" \
    --vision-guard-param-norm-max "${VISION_GUARD_PARAM_NORM_MAX}" \
    --vision-guard-patience "${VISION_GUARD_PATIENCE}" \
    --learner-world-size 2 \
    --collector-processes "${collector_processes}" &
pids+=("$!")
learner_pid="${pids[0]}"

sleep "${LEARNER_STARTUP_GRACE_SEC:-5}"
if ! jobs -pr | grep -qx "${learner_pid}"; then
    set +e
    wait "${learner_pid}"
    learner_status="$?"
    set -e
    echo "stop" > "${OUTPUT_DIR}/STOP"
    echo "[launcher][error] learner exited before collectors started with ${learner_status}" >&2
    exit "${learner_status}"
fi

collector_idx=0
for gpu in "${collector_gpu_array[@]}"; do
    for local_i in $(seq 1 "${COLLECTORS_PER_GPU}"); do
        collector_id="coll${collector_idx}_gpu${gpu}"
        # collector 不进 DDP，只在对应 GPU 上 rollout 并写 replay/ready。
        CUDA_VISIBLE_DEVICES="${gpu}" python qwen3vl_local/sft_v4/collect.py \
            --train-jsonl "${TRAIN_JSONL}" \
            --model-dir "${MODEL_DIR}" \
            --replay-dir "${REPLAY_DIR}" \
            --latest-lora-dir "${LATEST_LORA_DIR}" \
            --collector-id "${collector_id}" \
            --replay-capacity "${REPLAY_CAPACITY}" \
            --p-init-correct "${P_INIT_CORRECT}" \
            --phase-b-noise-prob "${PHASE_B_NOISE_PROB}" \
            --outer-stride "${OUTER_STRIDE}" \
            --lora-rank "${LORA_RANK}" \
            --lora-alpha "${LORA_ALPHA}" \
            --lora-dropout "${LORA_DROPOUT}" \
            --lora-vision-scope "${LORA_VISION_SCOPE}" \
            --stop-file "${OUTPUT_DIR}/STOP" &
        pids+=("$!")
        collector_idx=$((collector_idx + 1))
    done
done

set +e
wait "${learner_pid}"
learner_status="$?"
# learner 正常或异常退出后都写 STOP，让 collectors 在 episode 边界收尾退出。
echo "stop" > "${OUTPUT_DIR}/STOP"
for pid in "${pids[@]:1}"; do
    wait "${pid}"
done
set -e

if [[ "${learner_status}" -ne 0 ]]; then
    echo "[launcher][error] learner exited with ${learner_status}" >&2
    exit "${learner_status}"
fi

echo "[launcher] completed: ${OUTPUT_DIR}"
