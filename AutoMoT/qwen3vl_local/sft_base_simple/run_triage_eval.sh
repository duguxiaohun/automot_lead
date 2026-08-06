#!/usr/bin/env bash
set -euo pipefail

# Lightweight triage launcher for sft_base_simple. Run from AutoMoT/.

CKPT="${CKPT:-checkpoints/sft_base_simple_runs/latest/final}"
MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
INDEX="${INDEX:-checkpoints/sft_base_simple_data/val_sequence_index.jsonl}"
OUT_ROOT="${OUT_ROOT:-$(dirname "${CKPT}")/eval_results/triage_$(date +%Y%m%d_%H%M%S)}"
BALANCED_CASES_PER_BIN="${BALANCED_CASES_PER_BIN:-64}"
TRIAGE_PROFILE="${TRIAGE_PROFILE:-fast}"

mkdir -p "${OUT_ROOT}"

COMMON=(
  python qwen3vl_local/sft_base_simple/eval.py
  --index "${INDEX}"
  --model-dir "${MODEL_DIR}"
  --adapter-dir "${CKPT}"
  --task full
)

if [[ "${TRIAGE_PROFILE}" == "fast" ]]; then
  COMMON+=(--no-write-frames --no-write-tb)
fi

run_case() {
  local name="$1"
  shift
  echo "[triage] ${name}"
  "${COMMON[@]}" \
    --output-dir "${OUT_ROOT}/${name}" \
    "$@"
}

run_case "joint_fourbin" \
  --full-balance-mode joint \
  --full-balance-cases-per-bin "${BALANCED_CASES_PER_BIN}"

run_case "full_route_closed_loop" \
  --full-balance-mode none \
  --sample-routes "${FULL_ROUTE_SAMPLE_ROUTES:-64}"

run_case "joint_fourbin_score" \
  --full-balance-mode joint \
  --full-balance-cases-per-bin "${BALANCED_CASES_PER_BIN}" \
  --prediction-mode score

python - "${OUT_ROOT}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
keys = [
    "name",
    "task",
    "eval_mode",
    "full_balance_mode",
    "frames",
    "rollout_frames",
    "change_pairs",
    "road_acc",
    "highway_f1",
    "event_acc",
    "ue_f1",
    "joint_acc",
    "road_change_f1",
    "event_change_f1",
    "prediction_mode",
    "full_balance_cases_per_bin",
    "road_logit_bias",
    "event_logit_bias",
    "image_ablation",
    "goal_ablation",
]
rows = []
for metrics_path in sorted(root.glob("*/metrics.json")):
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    row = {"name": metrics_path.parent.name}
    for key in keys[1:]:
        row[key] = metrics.get(key)
    rows.append(row)

csv_path = root / "triage_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
print(f"[triage] summary -> {csv_path}")
PY

echo "[triage] done -> ${OUT_ROOT}"
