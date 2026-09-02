#!/usr/bin/env bash
# 用已有三个 final adapter 做 route-diverse validation 复评；通过四项 guard 后才运行 unseen-456。
#
# 从 AutoMoT/ 目录运行：
#   bash qwen3vl_local/sft_new_loop_phase2/run_route_diverse_validation_rescore.sh
#
# 默认复用 v3_frozen_3seed_unseen456_20260831 的训练权重，不重新训练。
# RUN_UNSEEN=0 可只做 validation 复评并停下。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

EXPERIMENT_ID="${EXPERIMENT_ID:-v3_frozen_3seed_unseen456_20260831}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-checkpoints/sft_new_loop_phase2_frozen_protocol/${EXPERIMENT_ID}}"
TRAIN_ROOT="${TRAIN_ROOT:-${EXPERIMENT_ROOT}/train_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_ROOT}/route_diverse_validation_rescore}"
MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
INDEX="${INDEX:-checkpoints/sft_new_loop_phase2_data/frame_index.jsonl}"
DATA_ROOT="${DATA_ROOT:-lead_data}"
SEEDS="${SEEDS:-20260810 20260811 20260812}"
CASES_PER_BIN="${CASES_PER_BIN:-32}"
SAMPLING_SEED="${SAMPLING_SEED:-20260831}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
RUN_UNSEEN="${RUN_UNSEEN:-1}"
SELECTION_JSON="${SELECTION_JSON:-${OUTPUT_ROOT}/selection.json}"

EXPECTED_PROMPT_NAME="sft_new_loop_phase2_direct_event_visual_v3"
EXPECTED_PROMPT_HASH="cd564634257fe0f072de70947200a820d6dd2b43375981b60120a1fe2296dd7f"
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

if [[ ! -f "${INDEX}" ]]; then
  echo "Missing dataset index: ${INDEX}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
echo "[rescore] experiment_root=${EXPERIMENT_ROOT}"
echo "[rescore] output_root=${OUTPUT_ROOT}"
echo "[rescore] cases_per_bin=${CASES_PER_BIN} sampling_seed=${SAMPLING_SEED}"

for seed in ${SEEDS}; do
  adapter="${TRAIN_ROOT}/seed_${seed}/final"
  output="${OUTPUT_ROOT}/seed_${seed}"
  if [[ ! -f "${adapter}/sft_new_loop_phase2_adapter_config.json" ]]; then
    echo "Missing final adapter for seed ${seed}: ${adapter}" >&2
    exit 2
  fi
  echo "[rescore seed=${seed}] adapter=${adapter}"
  python qwen3vl_local/sft_new_loop_phase2/eval.py \
    --model-dir "${MODEL_DIR}" \
    --index "${INDEX}" \
    --data-root "${DATA_ROOT}" \
    --adapter-dir "${adapter}" \
    --output-dir "${output}" \
    --split val \
    --cases-per-bin "${CASES_PER_BIN}" \
    --seed "${SAMPLING_SEED}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --route-diverse-sampling \
    --no-timestamp-output \
    --overwrite \
    --no-save-prompts \
    --no-save-error-rgb \
    --no-save-all-rgb
done

selection_status=0
python - "${OUTPUT_ROOT}" "${SELECTION_JSON}" "${TRAIN_ROOT}" ${SEEDS} <<'PY' || selection_status=$?
import json
import pathlib
import sys

output_root = pathlib.Path(sys.argv[1])
selection_path = pathlib.Path(sys.argv[2])
train_root = pathlib.Path(sys.argv[3])
seeds = sys.argv[4:]
floors = {
    "format_valid_rate": 1.0,
    "ue3_target_recall": 0.625,
    "ue6_target_recall": 0.80,
    "invalid_exact": 0.80,
    "applicable_regular_exact": 0.50,
}
records = []
for seed in seeds:
    metrics_path = output_root / f"seed_{seed}" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    variant = (metrics.get("variant_reports") or {}).get("all_random_order") or {}
    questions = metrics.get("per_question") or {}
    slices = metrics.get("slice_reports") or {}
    values = {
        "format_valid_rate": float(variant.get("format_valid_rate", 0.0)),
        "ue3_target_recall": float((questions.get("UE3") or {}).get("recall", 0.0)),
        "ue6_target_recall": float((questions.get("UE6") or {}).get("recall", 0.0)),
        "invalid_exact": float((slices.get("invalid") or {}).get("exact_match_accuracy", 0.0)),
        "applicable_regular_exact": float(
            (slices.get("applicable_regular") or {}).get("exact_match_accuracy", 0.0)
        ),
    }
    passed = {key: values[key] >= floors[key] for key in floors}
    records.append({
        "seed": int(seed),
        "adapter_dir": str(train_root / f"seed_{seed}" / "final"),
        "metrics_path": str(metrics_path),
        "overall_exact": float(metrics.get("exact_match_accuracy", 0.0)),
        "values": values,
        "floors": floors,
        "passed": passed,
        "all_guards_ok": all(passed.values()),
        "route_diversity": ((metrics.get("sampling_verification") or {}).get("route_diversity") or {}),
    })

eligible = [record for record in records if record["all_guards_ok"]]
selected = max(eligible, key=lambda record: (record["overall_exact"], -record["seed"])) if eligible else None
payload = {
    "format": "sft_new_loop_phase2_route_diverse_validation_selection_v1",
    "contract": (
        "Validation-only rescore of existing final adapters. Same route-diverse case set and fixed sampling seed "
        "for all training seeds; original frozen metrics remain unchanged."
    ),
    "records": records,
    "selected": selected,
    "selected_adapter_dir": selected["adapter_dir"] if selected else None,
    "ready_for_unseen": selected is not None,
}
selection_path.parent.mkdir(parents=True, exist_ok=True)
selection_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
if selected is None:
    raise SystemExit(2)
PY

if [[ "${selection_status}" -ne 0 ]]; then
  if [[ "${selection_status}" -eq 2 ]]; then
    audit_output="${EXPERIMENT_ROOT}/ue3_route_diverse_full_rgb_audit"
    echo "[rescore] no seed passed all guards; unseen remains untouched"
    echo "[rescore] building all-positive UE3 RGB audit (TP controls + FN)"
    python qwen3vl_local/sft_new_loop_phase2/build_ue3_validation_rgb_audit.py \
      --experiment-root "${EXPERIMENT_ROOT}" \
      --index "${INDEX}" \
      --data-root "${DATA_ROOT}" \
      --source-mode eval \
      --eval-root "${OUTPUT_ROOT}" \
      --include-correct \
      --output-dir "${audit_output}" \
      --overwrite
    echo "[rescore] UE3 full RGB audit=${audit_output}"
    echo "[rescore] archive=${audit_output}.tar.gz"
  fi
  exit "${selection_status}"
fi

SELECTED_ADAPTER="$(python - "${SELECTION_JSON}" <<'PY'
import json
import pathlib
import sys

print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["selected_adapter_dir"])
PY
)"
echo "[rescore-complete] selected_adapter=${SELECTED_ADAPTER}"
echo "[rescore-complete] selection=${SELECTION_JSON}"

if [[ "${RUN_UNSEEN}" == "1" ]]; then
  echo "[rescore] all validation guards passed; starting one-time unseen-456 acceptance"
  EXPERIMENT_ID="${EXPERIMENT_ID}" \
  EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" \
  TRAIN_ROOT="${TRAIN_ROOT}" \
  MODEL_DIR="${MODEL_DIR}" \
  INDEX="${INDEX}" \
  DATA_ROOT="${DATA_ROOT}" \
  SELECTED_ADAPTER="${SELECTED_ADAPTER}" \
  bash qwen3vl_local/sft_new_loop_phase2/run_frozen_protocol.sh unseen
else
  echo "[rescore] RUN_UNSEEN=0; unseen-456 remains untouched"
fi
