#!/usr/bin/env bash
# v23 choice：一个主要动作或 KEEP；STOP > 首次跨线 > 纵向，需新索引和新训练。
# 默认 RGB-stage 候选：隔离建库 -> 完整训练 -> test评测；无需历史审计包。
# PROMPT_VARIANT=baseline 可显式运行冻结 v23；标签/阈值不随候选更改。
#
# 从 AutoMoT/ 目录运行，默认 v23 四图 + choice（选择题）：
#   bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
#
# 两图 + choice：
#   HISTORY_RGB_MODE=2rgb_endpoints bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
#   GPU_IDS=0,1,2,3 HISTORY_RGB_MODE=2rgb_endpoints bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
# 四图 + binary（逐动作 YES/NO）：
#   ACTION_OUTPUT_MODE=binary bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
#   GPU_IDS=0,1,2,3 ACTION_OUTPUT_MODE=binary bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
# 两图 + binary：
#   HISTORY_RGB_MODE=2rgb_endpoints ACTION_OUTPUT_MODE=binary bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
#   GPU_IDS=0,1,2,3 HISTORY_RGB_MODE=2rgb_endpoints ACTION_OUTPUT_MODE=binary bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
# 复用已构建索引/训练产物（跳过训练时需指定原 RUN_ROOT）：
#   SKIP_BUILD=1 SKIP_TRAIN=1 bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh

set -euo pipefail

ulimit -S -c 0 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PROMPT_VARIANT="${PROMPT_VARIANT:-v23_rgb_stage_candidate_20261006}"
if [[ "${PROMPT_VARIANT}" == "baseline" ]]; then
  DEFAULT_DATA_DIR="checkpoints/sft_new_loop_phase3_data_v23"
else
  DEFAULT_DATA_DIR="checkpoints/sft_new_loop_phase3_data_rgb_stage_20261006_isolated"
fi
DATA_DIR="${DATA_DIR:-${DEFAULT_DATA_DIR}}"
INDEX="${INDEX:-${DATA_DIR}/frame_index.jsonl}"
DATA_ROOT="${DATA_ROOT:-lead_data}"
COLLECTION_DIR="${COLLECTION_DIR:-keyframe_filter/collection_output}"
MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3.5-4B}"
HISTORY_RGB_MODE="${HISTORY_RGB_MODE:-4rgb}"
ACTION_OUTPUT_MODE="${ACTION_OUTPUT_MODE:-choice}"
TRAIN_MODE="${TRAIN_MODE:-ddp}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
RUN_REGRESSION="${RUN_REGRESSION:-0}"
RUN_AUDITS="${RUN_AUDITS:-0}"
AUDIT_ROOT="${AUDIT_ROOT:-checkpoints}"
REPLAY_AUTOMOT_ROOT="${REPLAY_AUTOMOT_ROOT:-${AUTOMOT_ROOT}}"
if [[ "${PIPELINE_CHECK_ONLY:-0}" == "1" ]]; then SKIP_TRAIN=1; SKIP_EVAL=1; fi
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S_%N)}"
PIPELINE_ROOT="${PIPELINE_ROOT:-checkpoints/sft_new_loop_phase3_pipeline/${TIMESTAMP}_${HISTORY_RGB_MODE}_${ACTION_OUTPUT_MODE}}"

RUN_ROOT="${RUN_ROOT:-${OUTPUT_DIR:-${PIPELINE_ROOT}/train}}"

# Never append a second group to an existing output, even with a fixed TIMESTAMP.
mkdir -p "$(dirname "${PIPELINE_ROOT}")"
mkdir "${PIPELINE_ROOT}"
exec > >(tee "${PIPELINE_ROOT}/pipeline.log") 2>&1
for flag in SKIP_BUILD SKIP_TRAIN SKIP_EVAL RUN_REGRESSION RUN_AUDITS; do
  if [[ "${!flag}" != 0 && "${!flag}" != 1 ]]; then echo "Invalid ${flag}=${!flag}" >&2; exit 2; fi
done
case "${TRAIN_MODE}" in single|ddp) ;; *) echo "Use TRAIN_MODE=single/ddp; PIPELINE_CHECK_ONLY=1 for CPU checks" >&2; exit 2 ;; esac
# Full pipeline means an untruncated scheduled run and the entire test index.
for knob in MAX_STEPS MAX_FRAMES MAX_EVAL_FRAMES CASES_PER_BIN; do
  if [[ "${!knob:-0}" != 0 ]]; then echo "Full pipeline requires ${knob}=0" >&2; exit 2; fi
done
if [[ "${EVAL_SPLIT:-val}" != val || "${SPLIT:-test}" != test || -n "${EXCLUDE_CASES_JSONL:-}" ]]; then
  echo "Full pipeline requires validation on val, testing on test, without exclusions" >&2; exit 2
fi
export PROMPT_VARIANT INDEX DATA_ROOT MODEL_DIR HISTORY_RGB_MODE ACTION_OUTPUT_MODE
export TRAIN_MODE PIPELINE_ROOT RUN_ROOT SKIP_TRAIN SKIP_EVAL RUN_REGRESSION RUN_AUDITS AUDIT_ROOT REPLAY_AUTOMOT_ROOT
if [[ "${SKIP_BUILD}" == 1 && ! -f "${INDEX}" ]]; then echo "SKIP_BUILD=1 requires existing ${INDEX}" >&2; exit 2; fi
python -m qwen3vl_local.sft_new_loop_phase3.pipeline_support start

case "${ACTION_OUTPUT_MODE}" in
  binary|choice) ;;
  *) echo "Unknown ACTION_OUTPUT_MODE=${ACTION_OUTPUT_MODE}. Use binary or choice." >&2; exit 2 ;;
esac
echo "[phase3-pipeline] root=${PIPELINE_ROOT} index=${INDEX} history_rgb_mode=${HISTORY_RGB_MODE} action_output_mode=${ACTION_OUTPUT_MODE} prompt=${PROMPT_VARIANT}"

echo
if [[ "${RUN_AUDITS}" == 1 ]]; then
  echo "========== optional RGB audit coverage =========="
  python qwen3vl_local/sft_new_loop_phase3/visual_audit.py \
    --output "${PIPELINE_ROOT}/visual_audit_manifest.json"
fi

echo
echo "========== 1/3 build action index =========="
if [[ "${SKIP_BUILD}" == "1" ]]; then
  echo "[skip] SKIP_BUILD=1; reusing ${INDEX}"
elif [[ -f "${INDEX}" ]]; then
  echo "[reuse] Existing index; index contracts and split isolation will be rechecked. Use a fresh DATA_DIR to rebuild."
else
  if [[ "${INDEX}" != "${DATA_DIR}/frame_index.jsonl" || -e "${DATA_DIR}" ]]; then
    echo "Refusing partial/custom build destination; use a fresh DATA_DIR (INDEX=DATA_DIR/frame_index.jsonl)" >&2; exit 2
  fi
  BUILD_ARGS=(
    --workers "${BUILD_WORKERS:-0}"
    --collection-dir "${COLLECTION_DIR}"
    --data-root "${DATA_ROOT}"
    --output-dir "${DATA_DIR}"
    --scenarios "${SCENARIOS:-all}"
    --split-seed "${SPLIT_SEED:-20260920}"
    --test-ratio "${TEST_RATIO:-0.10}"
    --val-ratio "${VAL_RATIO:-0.05}"
    --target-per-context "${TARGET_PER_CONTEXT:-0}"
    --invalid-ratio "${INVALID_RATIO:-0.20}"
    --max-routes "${MAX_ROUTES:-0}"
    --max-routes-per-scenario "${MAX_ROUTES_PER_SCENARIO:-0}"
    --progress-every-routes "${PROGRESS_EVERY_ROUTES:-200}"
  )
  if [[ "${REQUIRE_INVALID_TRUE_RS_COVERAGE:-1}" == "0" ]]; then
    BUILD_ARGS+=(--no-require-invalid-true-rs-coverage)
  else
    BUILD_ARGS+=(--require-invalid-true-rs-coverage)
  fi
  if [[ "${PROMPT_VARIANT}" != baseline ]]; then
    BUILD_ARGS+=(--development-groups-extra "${SCRIPT_DIR}/development_route_groups_rgb_stage_20261006.json")
  fi
  python qwen3vl_local/sft_new_loop_phase3/build_dataset.py "${BUILD_ARGS[@]}"
fi

python -m qwen3vl_local.sft_new_loop_phase3.pipeline_support bind

if [[ "${RUN_AUDITS}" == 1 ]]; then
  # 标签完整精度复算、RGB路径与split核验结果随pipeline保存，失败时不启动训练。
  python qwen3vl_local/sft_new_loop_phase3/audit_rebuilt_index.py \
    --index "${INDEX}" --data-root "${DATA_ROOT}" \
    --action-output-mode "${ACTION_OUTPUT_MODE}" \
    --output "${PIPELINE_ROOT}/index_audit.json"

  echo
  # 再从原始 meta 复算，不能只用索引自己保存的未来速度自证。
  python qwen3vl_local/sft_new_loop_phase3/audit_raw_index.py \
    --index "${INDEX}" --data-root "${DATA_ROOT}" \
    --action-output-mode "${ACTION_OUTPUT_MODE}" \
    --workers "${AUDIT_WORKERS:-8}" --output "${PIPELINE_ROOT}/raw_index_audit.json"

  python qwen3vl_local/sft_new_loop_phase3/audit_temporal_slices.py \
    --index "${INDEX}" --output "${PIPELINE_ROOT}/temporal_slices.json"

  # 包括被隔离而未进入索引的源帧，保留修复前后字段便于继续逐帧回查。
  python qwen3vl_local/sft_new_loop_phase3/audit_annotation_repairs.py \
    --data-root "${DATA_ROOT}" --collection-dir "${COLLECTION_DIR}" --scenarios "${SCENARIOS:-all}" \
    --output-dir "${PIPELINE_ROOT}/annotation_repair_audit"

fi

# Exercise the actual sampler with the same requested process count before loading weights.
SAMPLING_WORLD="${DDP_GPU_COUNT:-${NPROC_PER_NODE:-4}}"
if [[ -n "${GPU_IDS:-}" ]]; then IFS=',' read -ra PIPELINE_GPUS <<< "${GPU_IDS}"; SAMPLING_WORLD="${#PIPELINE_GPUS[@]}"; fi
if [[ "${TRAIN_MODE}" == single ]]; then SAMPLING_WORLD=1; fi
NPROC_PER_NODE="${SAMPLING_WORLD}" bash qwen3vl_local/sft_new_loop_phase3/train.sh sampling
python -m qwen3vl_local.sft_new_loop_phase3.pipeline_support verify

echo "========== 2/3 train LoRA =========="
if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo "[skip] SKIP_TRAIN=1"
else
  INDEX="${INDEX}" DATA_ROOT="${DATA_ROOT}" MODEL_DIR="${MODEL_DIR}" \
  HISTORY_RGB_MODE="${HISTORY_RGB_MODE}" ACTION_OUTPUT_MODE="${ACTION_OUTPUT_MODE}" OUTPUT_DIR="${RUN_ROOT}" \
    bash qwen3vl_local/sft_new_loop_phase3/train.sh "${TRAIN_MODE}"
fi

echo
echo "========== 3/3 standalone test evaluation =========="
if [[ "${SKIP_EVAL}" == "1" ]]; then
  echo "[skip] SKIP_EVAL=1"
elif [[ ! -e "${RUN_ROOT}" ]]; then
  echo "[error] no trained run at ${RUN_ROOT}; set RUN_ROOT when SKIP_TRAIN=1" >&2
  exit 1
else
  ADAPTER_PATH="$(python -m qwen3vl_local.sft_new_loop_phase3.pipeline_support adapter)"
  BUNDLE_NAME="${BUNDLE_BASENAME:-sft_new_loop_phase3_${TIMESTAMP}_${HISTORY_RGB_MODE}_${ACTION_OUTPUT_MODE}_audit_bundle}"
  ADAPTER_DIR="${ADAPTER_PATH}" SPLIT=test CASES_PER_BIN=0 MAX_EVAL_FRAMES=0 INDEX="${INDEX}" DATA_ROOT="${DATA_ROOT}" MODEL_DIR="${MODEL_DIR}" \
  TIMESTAMP="${TIMESTAMP}" BUNDLE_BASENAME="${BUNDLE_NAME}" BUNDLE_MAX_MB="${BUNDLE_MAX_MB:-30}" \
  OUTPUT_ROOT="${PIPELINE_ROOT}/eval" \
    bash qwen3vl_local/sft_new_loop_phase3/eval.sh
  if [[ "${RUN_AUDITS}" == 1 ]]; then
    echo "[phase3-pipeline] audit bundle: ${PIPELINE_ROOT}/eval/${BUNDLE_NAME}.tar.gz"
  fi
fi

echo
python -m qwen3vl_local.sft_new_loop_phase3.pipeline_support finish
echo "[phase3-pipeline] done: ${PIPELINE_ROOT}; see pipeline_manifest.json"
