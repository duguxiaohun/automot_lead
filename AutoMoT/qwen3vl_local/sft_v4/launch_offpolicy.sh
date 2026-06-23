#!/usr/bin/env bash
# SFT v4 off-policy launcher: 2 learner DDP ranks + 6 async collectors.
#
# Run from AutoMoT/:
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v4/launch_offpolicy.sh

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
REPLAY_CAPACITY="${REPLAY_CAPACITY:-256}"
COLLECTORS_PER_GPU="${COLLECTORS_PER_GPU:-3}"
P_INIT_CORRECT="${P_INIT_CORRECT:-0.5}"
PHASE_B_NOISE_PROB="${PHASE_B_NOISE_PROB:-0.15}"
OUTER_STRIDE="${OUTER_STRIDE:-1}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_VISION_SCOPE="${LORA_VISION_SCOPE:-off}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_HOME:-${OUTPUT_DIR_BASE}/.hf_cache}"
mkdir -p "${OUTPUT_DIR}" "${REPLAY_DIR}" "${HF_HOME}"
if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
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
    seq -s, 0 "$((want_count - 1))"
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

if [[ -n "${LEARNER_GPU_IDS:-}" && -n "${COLLECTOR_GPU_IDS:-}" ]]; then
    learner_gpus="${LEARNER_GPU_IDS}"
    collector_gpus="${COLLECTOR_GPU_IDS}"
else
    visible="${GPU_IDS:-$(pick_idle_gpus 4)}"
    gpu_count="$(awk -F',' '{print NF}' <<< "${visible}")"
    if [[ "${gpu_count}" -lt 4 ]]; then
        echo "[error] need 4 GPUs or set LEARNER_GPU_IDS and COLLECTOR_GPU_IDS explicitly; got ${visible}" >&2
        exit 1
    fi
    learner_gpus="$(split_csv "${visible}" 1),$(split_csv "${visible}" 2)"
    collector_gpus="$(split_csv "${visible}" 3),$(split_csv "${visible}" 4)"
fi

IFS=',' read -r -a collector_gpu_array <<< "${collector_gpus}"
collector_processes=$(( ${#collector_gpu_array[@]} * COLLECTORS_PER_GPU ))
master_port="${MASTER_PORT:-$(find_free_master_port)}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${master_port}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"

echo "[run] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[gpu] learner=${learner_gpus} collector=${collector_gpus} collectors_per_gpu=${COLLECTORS_PER_GPU}"
echo "[cfg] max_steps=${MAX_STEPS} replay_capacity=${REPLAY_CAPACITY} p_init=${P_INIT_CORRECT} phase_b_noise=${PHASE_B_NOISE_PROB}"

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
    --snapshot-every-steps "${SNAPSHOT_EVERY_STEPS}" \
    --save-steps "${SAVE_STEPS}" \
    --lora-rank "${LORA_RANK}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-dropout "${LORA_DROPOUT}" \
    --lora-vision-scope "${LORA_VISION_SCOPE}" \
    --learner-world-size 2 \
    --collector-processes "${collector_processes}" &
pids+=("$!")
learner_pid="${pids[0]}"

sleep 5

collector_idx=0
for gpu in "${collector_gpu_array[@]}"; do
    for local_i in $(seq 1 "${COLLECTORS_PER_GPU}"); do
        collector_id="coll${collector_idx}_gpu${gpu}"
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
