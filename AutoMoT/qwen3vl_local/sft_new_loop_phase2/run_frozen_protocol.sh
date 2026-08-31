#!/usr/bin/env bash
# 冻结 v3 实验协议：多 seed 只看 validation 选优，再对排除旧 384 条后的 456 条做一次 unseen 验收。
#
# 从 AutoMoT/ 目录运行：
#   bash qwen3vl_local/sft_new_loop_phase2/run_frozen_protocol.sh train
#   bash qwen3vl_local/sft_new_loop_phase2/run_frozen_protocol.sh unseen
#   bash qwen3vl_local/sft_new_loop_phase2/run_frozen_protocol.sh all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

ACTION="${1:-train}"
case "${ACTION}" in
  train|unseen|all) ;;
  *) echo "Usage: $0 [train|unseen|all]" >&2; exit 2 ;;
esac

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

EXPECTED_PROMPT_NAME="sft_new_loop_phase2_direct_event_visual_v3"
EXPECTED_PROMPT_HASH="cd564634257fe0f072de70947200a820d6dd2b43375981b60120a1fe2296dd7f"
HISTORY_RGB_MODE="${HISTORY_RGB_MODE:-2rgb_endpoints}"
if [[ "${HISTORY_RGB_MODE}" != "2rgb_endpoints" ]]; then
  echo "Frozen protocol requires HISTORY_RGB_MODE=2rgb_endpoints, got ${HISTORY_RGB_MODE}" >&2
  exit 2
fi

python - "${EXPECTED_PROMPT_NAME}" "${EXPECTED_PROMPT_HASH}" <<'PY'
import sys
from qwen3vl_local.sft_new_loop_phase2.history_rgb import HISTORY_RGB_MODE_END2
from qwen3vl_local.sft_new_loop_phase2.prompts import PROMPT_NAME, event_prompt_sha256
expected_name, expected_hash = sys.argv[1:]
actual_hash = event_prompt_sha256(history_rgb_mode=HISTORY_RGB_MODE_END2)
if PROMPT_NAME != expected_name or actual_hash != expected_hash:
    raise SystemExit(
        f"frozen prompt mismatch: name={PROMPT_NAME!r} hash={actual_hash}; "
        f"expected name={expected_name!r} hash={expected_hash}"
    )
print(f"[freeze] prompt={PROMPT_NAME} hash={actual_hash}")
PY

MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
INDEX="${INDEX:-checkpoints/sft_new_loop_phase2_data/frame_index.jsonl}"
DATA_ROOT="${DATA_ROOT:-lead_data}"
EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M%S)_v3_frozen}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-checkpoints/sft_new_loop_phase2_frozen_protocol/${EXPERIMENT_ID}}"
TRAIN_ROOT="${TRAIN_ROOT:-${EXPERIMENT_ROOT}/train_runs}"
SELECTION_JSON="${SELECTION_JSON:-${EXPERIMENT_ROOT}/seed_selection.json}"
SEEDS="${SEEDS:-20260810 20260811 20260812}"
REQUIRED_SEEDS="${REQUIRED_SEEDS:-3}"
TRAIN_MODE="${TRAIN_MODE:-ddp}"

if [[ ! -f "${INDEX}" ]]; then
  echo "Missing dataset index: ${INDEX}" >&2
  echo "Build it once with the build_dataset.py command in SFT_NEW_LOOP_PHASE2_RUN.md, then rerun." >&2
  exit 2
fi

run_seed_training() {
  local seed="$1"
  local output_dir="${TRAIN_ROOT}/seed_${seed}"
  if [[ -f "${output_dir}/generation_selection_status.json" ]]; then
    echo "[seed ${seed}] completed run found; reusing ${output_dir}"
    return
  fi
  echo "[seed ${seed}] training -> ${output_dir}"
  SEED="${seed}" \
  RUN_TIMESTAMP="${EXPERIMENT_ID}_seed${seed}" \
  OUTPUT_DIR="${output_dir}" \
  MODEL_DIR="${MODEL_DIR}" \
  INDEX="${INDEX}" \
  DATA_ROOT="${DATA_ROOT}" \
  HISTORY_RGB_MODE="${HISTORY_RGB_MODE}" \
  DDP_GPU_COUNT="${DDP_GPU_COUNT:-4}" \
  FOCUS_BALANCE_COUNT="${FOCUS_BALANCE_COUNT:-2048}" \
  REGULAR_FOCUS_MULTIPLIER="${REGULAR_FOCUS_MULTIPLIER:-2.0}" \
  INVALID_FOCUS_MULTIPLIER="${INVALID_FOCUS_MULTIPLIER:-1.0}" \
  GENERATION_EVAL_BALANCE_COUNT="${GENERATION_EVAL_BALANCE_COUNT:-32}" \
  GENERATION_EVAL_MIN_UE3_TARGET_RECALL="${GENERATION_EVAL_MIN_UE3_TARGET_RECALL:-0.625}" \
  GENERATION_EVAL_MIN_UE6_TARGET_RECALL="${GENERATION_EVAL_MIN_UE6_TARGET_RECALL:-0.80}" \
  GENERATION_EVAL_MIN_INVALID_EXACT="${GENERATION_EVAL_MIN_INVALID_EXACT:-0.80}" \
  GENERATION_EVAL_MIN_APPLICABLE_REGULAR_EXACT="${GENERATION_EVAL_MIN_APPLICABLE_REGULAR_EXACT:-0.50}" \
  bash qwen3vl_local/sft_new_loop_phase2/train.sh "${TRAIN_MODE}"
}

select_seed() {
  local args=()
  local seed
  for seed in ${SEEDS}; do
    args+=(--run-root "${TRAIN_ROOT}/seed_${seed}")
  done
  python qwen3vl_local/sft_new_loop_phase2/select_seed_checkpoint.py \
    "${args[@]}" \
    --required-seeds "${REQUIRED_SEEDS}" \
    --output "${SELECTION_JSON}"
}

if [[ "${ACTION}" == "train" || "${ACTION}" == "all" ]]; then
  mkdir -p "${TRAIN_ROOT}"
  for seed in ${SEEDS}; do
    run_seed_training "${seed}"
  done
  select_seed
  echo "[train-complete] validation-only selection: ${SELECTION_JSON}"
fi

if [[ "${ACTION}" == "unseen" || "${ACTION}" == "all" ]]; then
  DEV_CASES_JSONL="${DEV_CASES_JSONL:-qwen3vl_local/sft_new_loop_phase2/frozen_dev_cases_v3_384.jsonl}"
  UNSEEN_ROOT="${UNSEEN_ROOT:-${EXPERIMENT_ROOT}/unseen_456}"
  ACCEPTANCE_JSON="${ACCEPTANCE_JSON:-${UNSEEN_ROOT}/unseen_acceptance.json}"
  if [[ -e "${ACCEPTANCE_JSON}" && "${ALLOW_UNSEEN_RERUN:-0}" != "1" ]]; then
    echo "Refusing to rerun frozen unseen evaluation: ${ACCEPTANCE_JSON} already exists." >&2
    echo "Set ALLOW_UNSEEN_RERUN=1 only for infrastructure recovery, never for prompt tuning." >&2
    exit 2
  fi
  if [[ ! -e "${DEV_CASES_JSONL}" ]]; then
    echo "Missing frozen 384-case exclusion source: ${DEV_CASES_JSONL}" >&2
    exit 2
  fi
  if [[ -n "${SELECTED_ADAPTER:-}" ]]; then
    ADAPTER_TO_EVAL="${SELECTED_ADAPTER}"
  else
    if [[ ! -f "${SELECTION_JSON}" ]]; then
      echo "Missing validation-only seed selection: ${SELECTION_JSON}" >&2
      exit 2
    fi
    ADAPTER_TO_EVAL="$(python - "${SELECTION_JSON}" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["selected_adapter_dir"])
PY
)"
  fi
  echo "[unseen] adapter=${ADAPTER_TO_EVAL}"
  echo "[unseen] exclude=${DEV_CASES_JSONL}; expected 840-384=456"
  ADAPTER_DIR="${ADAPTER_TO_EVAL}" \
  MODEL_DIR="${MODEL_DIR}" \
  INDEX="${INDEX}" \
  DATA_ROOT="${DATA_ROOT}" \
  SPLIT=test \
  CASES_PER_BIN=0 \
  EXCLUDE_CASES_JSONL="${DEV_CASES_JSONL}" \
  EXPECTED_EXCLUDED_CASES=384 \
  EXPECTED_TOTAL_CASES=456 \
  OUTPUT_ROOT="${UNSEEN_ROOT}" \
  TIMESTAMP="${EXPERIMENT_ID}_unseen456" \
  RUN_AUDIT_PROMPT_EVAL="${RUN_AUDIT_PROMPT_EVAL:-0}" \
  RUN_VISUAL_AUDIT="${RUN_VISUAL_AUDIT:-1}" \
  EVAL_GPU_COUNT="${EVAL_GPU_COUNT:-4}" \
  bash qwen3vl_local/sft_new_loop_phase2/eval.sh

  python qwen3vl_local/sft_new_loop_phase2/check_acceptance.py \
    --metrics "${UNSEEN_ROOT}/lora_production/metrics.json" \
    --output "${ACCEPTANCE_JSON}" \
    --min-overall-exact "${MIN_OVERALL_EXACT:-0.80}" \
    --min-format-valid-rate "${MIN_FORMAT_VALID_RATE:-1.0}" \
    --min-ue3-recall "${MIN_UE3_RECALL:-0.80}" \
    --min-ue6-recall "${MIN_UE6_RECALL:-0.80}" \
    --min-invalid-recall "${MIN_INVALID_RECALL:-0.80}" \
    --min-applicable-regular-exact "${MIN_APPLICABLE_REGULAR_EXACT:-0.50}"
  echo "[unseen-complete] acceptance=${ACCEPTANCE_JSON}"
fi
