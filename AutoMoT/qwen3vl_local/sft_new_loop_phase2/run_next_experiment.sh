#!/usr/bin/env bash
# New Phase2 下一轮正式实验一键入口。
#
# 默认执行：
#   1. 缺少 frame_index 时构建冻结数据索引；
#   2. 在训练前验证 test=840、历史 dev=384、unseen=456；
#   3. 顺序训练 3 个 seed，只按 validation 门槛选 checkpoint；
#   4. 对剩余 456 条 unseen 只评测一次并写验收结论。
#
# 默认当前目录已经是 AutoMoT/：
#   bash qwen3vl_local/sft_new_loop_phase2/run_next_experiment.sh

set -euo pipefail

ulimit -S -c 0 2>/dev/null || true

# 默认执行命令时已经位于 AutoMoT/，相对路径全部以当前目录为准。
AUTOMOT_ROOT="${AUTOMOT_ROOT:-$(pwd)}"
if [[ ! -d "${AUTOMOT_ROOT}/qwen3vl_local" || ! -d "${AUTOMOT_ROOT}/keyframe_filter" ]]; then
  echo "Current directory is not AutoMoT/: ${AUTOMOT_ROOT}" >&2
  echo "cd AutoMoT first, or set AUTOMOT_ROOT explicitly." >&2
  exit 2
fi
AUTOMOT_ROOT="$(cd "${AUTOMOT_ROOT}" && pwd)"
cd "${AUTOMOT_ROOT}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# 这是冻结 v3 的唯一正式实验 ID。默认固定，脚本中断后再次运行会复用已完成的 seed。
EXPERIMENT_ID="${EXPERIMENT_ID:-v3_frozen_3seed_unseen456_20260831}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-checkpoints/sft_new_loop_phase2_frozen_protocol/${EXPERIMENT_ID}}"
DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-checkpoints/sft_new_loop_phase2_data}"
INDEX="${INDEX:-${DATA_OUTPUT_DIR}/frame_index.jsonl}"
MANIFEST="${MANIFEST:-${DATA_OUTPUT_DIR}/manifest.json}"
MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
DATA_ROOT="${DATA_ROOT:-lead_data}"
COLLECTION_DIR="${COLLECTION_DIR:-keyframe_filter/collection_output}"
DEV_CASES_JSONL="${DEV_CASES_JSONL:-qwen3vl_local/sft_new_loop_phase2/frozen_dev_cases_v3_384.jsonl}"
ACCEPTANCE_JSON="${ACCEPTANCE_JSON:-${EXPERIMENT_ROOT}/unseen_456/unseen_acceptance.json}"
REBUILD_DATA="${REBUILD_DATA:-0}"

mkdir -p "${EXPERIMENT_ROOT}"
LOG_PATH="${EXPERIMENT_ROOT}/run_next_experiment.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[one-click] AutoMoT root=${AUTOMOT_ROOT}"
echo "[one-click] experiment_id=${EXPERIMENT_ID}"
echo "[one-click] experiment_root=${EXPERIMENT_ROOT}"
echo "[one-click] log=${LOG_PATH}"
echo "[one-click] model=${MODEL_DIR}"
echo "[one-click] index=${INDEX}"
echo "[one-click] old_dev_cases=${DEV_CASES_JSONL}"

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 2
  fi
}

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Missing local Qwen model: ${MODEL_DIR}" >&2
  echo "This experiment is offline-only. Mount/copy Qwen3-VL-4B-Instruct there, or set MODEL_DIR to its local path." >&2
  exit 2
fi
require_path "${DATA_ROOT}" "LEAD data root"
require_path "${COLLECTION_DIR}" "collection_output"
require_path "${DEV_CASES_JSONL}" "frozen 384-case audit source"

if [[ "${REBUILD_DATA}" == "1" || ! -f "${INDEX}" || ! -f "${MANIFEST}" ]]; then
  echo
  echo "========== 1/4 build frozen dataset index =========="
  python qwen3vl_local/sft_new_loop_phase2/build_dataset.py \
    --collection-dir "${COLLECTION_DIR}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${DATA_OUTPUT_DIR}" \
    --test-ratio 0.10 \
    --val-ratio 0.05 \
    --regular-multiplier 1.0 \
    --highway-regular-fraction 0.25 \
    --invalid-ratio 0.20
else
  echo "[one-click] reuse existing dataset index and manifest"
fi

echo
echo "========== 2/4 preflight frozen split identities =========="
python - "${INDEX}" "${DEV_CASES_JSONL}" <<'PY'
import json
import pathlib
import sys

index_path = pathlib.Path(sys.argv[1])
dev_input = pathlib.Path(sys.argv[2])

def identity(obj):
    return (
        str(obj.get("scenario", "")),
        str(obj.get("route_id", "")),
        int(obj.get("frame_id", -1)),
        str(obj.get("question_domain", "")),
        str(obj.get("event", "")),
        str(obj.get("invalid_source") or ""),
    )

test_keys = set()
test_rows = 0
with index_path.open("r", encoding="utf-8") as handle:
    for line_no, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("split")) != "test":
            continue
        test_rows += 1
        test_keys.add(identity(row))

dev_files = sorted(dev_input.glob("cases*.jsonl")) if dev_input.is_dir() else [dev_input]
if not dev_files:
    raise SystemExit(f"no cases*.jsonl under frozen dev source: {dev_input}")
dev_keys = set()
for path in dev_files:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                dev_keys.add(identity(json.loads(line)))

matched = test_keys & dev_keys
remaining = test_keys - dev_keys
report = {
    "test_rows": test_rows,
    "test_unique": len(test_keys),
    "dev_files": len(dev_files),
    "dev_unique": len(dev_keys),
    "matched_dev": len(matched),
    "unseen_remaining": len(remaining),
}
print(json.dumps(report, ensure_ascii=False, indent=2))
expected = {
    "test_rows": 840,
    "test_unique": 840,
    "dev_unique": 384,
    "matched_dev": 384,
    "unseen_remaining": 456,
}
bad = {key: (report[key], value) for key, value in expected.items() if report[key] != value}
if bad:
    raise SystemExit(f"frozen split preflight failed: {bad}")
PY

if [[ -f "${ACCEPTANCE_JSON}" ]]; then
  echo
  echo "[one-click] unseen acceptance already exists; refusing to evaluate it again."
  python - "${ACCEPTANCE_JSON}" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps(payload, ensure_ascii=False, indent=2))
raise SystemExit(0 if payload.get("accepted") else 2)
PY
  exit 0
fi

echo
echo "========== 3/4 train 3 seeds and select by validation =========="
echo "========== 4/4 run one-time unseen-456 acceptance =========="
set +e
EXPERIMENT_ID="${EXPERIMENT_ID}" \
EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" \
MODEL_DIR="${MODEL_DIR}" \
INDEX="${INDEX}" \
DATA_ROOT="${DATA_ROOT}" \
DEV_CASES_JSONL="${DEV_CASES_JSONL}" \
ACCEPTANCE_JSON="${ACCEPTANCE_JSON}" \
HISTORY_RGB_MODE=2rgb_endpoints \
SEEDS="${SEEDS:-20260810 20260811 20260812}" \
REQUIRED_SEEDS="${REQUIRED_SEEDS:-3}" \
TRAIN_MODE="${TRAIN_MODE:-ddp}" \
DDP_GPU_COUNT="${DDP_GPU_COUNT:-4}" \
EVAL_GPU_COUNT="${EVAL_GPU_COUNT:-4}" \
bash qwen3vl_local/sft_new_loop_phase2/run_frozen_protocol.sh all
protocol_status=$?
set -e

if [[ -f "${ACCEPTANCE_JSON}" ]]; then
  echo
  echo "========== final unseen acceptance =========="
  python - "${ACCEPTANCE_JSON}" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps(payload, ensure_ascii=False, indent=2))
print("FINAL_DECISION=" + ("ACCEPT" if payload.get("accepted") else "REJECT"))
PY
else
  echo "[one-click] no unseen acceptance was produced." >&2
  echo "If training finished, inspect generation_selection_status.json and fallback_generation.json." >&2
fi

exit "${protocol_status}"
