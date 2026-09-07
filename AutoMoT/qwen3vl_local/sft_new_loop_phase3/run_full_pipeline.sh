#!/usr/bin/env bash
# 新 Phase3 全流程：RGB 审计覆盖检查 -> 构建动作索引 -> 训练 -> 独立评测 + 错例审计包。
#
# 从 AutoMoT/ 目录运行：
#   bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
#
# 常用覆盖：
#   SKIP_BUILD=1 SKIP_TRAIN=1 bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
#   HISTORY_RGB_MODE=2rgb_endpoints bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh

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

DATA_DIR="${DATA_DIR:-checkpoints/sft_new_loop_phase3_data}"
INDEX="${INDEX:-${DATA_DIR}/frame_index.jsonl}"
DATA_ROOT="${DATA_ROOT:-lead_data}"
MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
HISTORY_RGB_MODE="${HISTORY_RGB_MODE:-4rgb}"
TRAIN_MODE="${TRAIN_MODE:-ddp}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
PIPELINE_ROOT="${PIPELINE_ROOT:-checkpoints/sft_new_loop_phase3_pipeline/${TIMESTAMP}}"

mkdir -p "${PIPELINE_ROOT}"
exec > >(tee -a "${PIPELINE_ROOT}/pipeline.log") 2>&1

echo "[phase3-pipeline] root=${PIPELINE_ROOT} index=${INDEX} history_rgb_mode=${HISTORY_RGB_MODE}"

echo
echo "========== 1/4 RGB audit coverage =========="
python qwen3vl_local/sft_new_loop_phase3/visual_audit.py \
  --output "${PIPELINE_ROOT}/visual_audit_manifest.json"

echo
echo "========== 2/4 build action index =========="
if [[ "${SKIP_BUILD}" == "1" ]]; then
  echo "[skip] SKIP_BUILD=1; reusing ${INDEX}"
else
  BUILD_ARGS=(
    --data-root "${DATA_ROOT}"
    --output-dir "${DATA_DIR}"
    --scenarios "${SCENARIOS:-all}"
    --split-seed "${SPLIT_SEED:-20260819}"
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
  python qwen3vl_local/sft_new_loop_phase3/build_dataset.py "${BUILD_ARGS[@]}"
fi

echo
echo "========== 3/4 train LoRA =========="
if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo "[skip] SKIP_TRAIN=1"
else
  INDEX="${INDEX}" DATA_ROOT="${DATA_ROOT}" MODEL_DIR="${MODEL_DIR}" \
  HISTORY_RGB_MODE="${HISTORY_RGB_MODE}" \
    bash qwen3vl_local/sft_new_loop_phase3/train.sh "${TRAIN_MODE}"
fi

RUN_ROOT="checkpoints/sft_new_loop_phase3_runs/latest"

adapter_history_rgb_mode() {
  local adapter_input="$1"
  python - "${adapter_input}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for candidate in (root / "best_generation", root / "final", root / "fallback_generation", root):
    cfg_path = candidate / "sft_new_loop_phase3_adapter_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        mode = str(cfg.get("history_rgb_mode", ""))
        if mode not in {"4rgb", "2rgb_endpoints"}:
            raise SystemExit(f"invalid history_rgb_mode={mode!r} in {cfg_path}")
        print(mode)
        raise SystemExit(0)
raise SystemExit(f"missing sft_new_loop_phase3_adapter_config.json under {root}")
PY
}

echo
echo "========== 4/4 standalone eval + error audit =========="
if [[ "${SKIP_EVAL}" == "1" ]]; then
  echo "[skip] SKIP_EVAL=1"
elif [[ ! -e "${RUN_ROOT}" ]]; then
  echo "[skip] no trained run at ${RUN_ROOT}"
else
  ADAPTER_RGB_MODE="$(adapter_history_rgb_mode "${RUN_ROOT}")"
  BUNDLE_NAME="${BUNDLE_BASENAME:-sft_new_loop_phase3_${TIMESTAMP}_${ADAPTER_RGB_MODE}_audit_bundle}"
  ADAPTER_DIR="${RUN_ROOT}" INDEX="${INDEX}" DATA_ROOT="${DATA_ROOT}" MODEL_DIR="${MODEL_DIR}" \
  TIMESTAMP="${TIMESTAMP}" BUNDLE_BASENAME="${BUNDLE_NAME}" BUNDLE_MAX_MB="${BUNDLE_MAX_MB:-30}" \
  OUTPUT_ROOT="${PIPELINE_ROOT}/eval" \
    bash qwen3vl_local/sft_new_loop_phase3/eval.sh
  echo "[phase3-pipeline] audit bundle: ${PIPELINE_ROOT}/eval/${BUNDLE_NAME}.tar.gz"
fi

echo
echo "[phase3-pipeline] done: ${PIPELINE_ROOT}"
