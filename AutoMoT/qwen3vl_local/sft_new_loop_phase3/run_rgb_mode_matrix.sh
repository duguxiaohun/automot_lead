#!/usr/bin/env bash
# 新 Phase3 的 4RGB / 2RGB_endpoints 输入合同对比矩阵。
#
# 从 AutoMoT/ 目录运行：
#   bash qwen3vl_local/sft_new_loop_phase3/run_rgb_mode_matrix.sh
#
# 每个模式独立训练一个 adapter，再各自评测，避免用一个 adapter 混跑两种输入合同。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

MODES="${MODES:-4rgb 2rgb_endpoints}"
INDEX="${INDEX:-checkpoints/sft_new_loop_phase3_data/frame_index.jsonl}"
DATA_ROOT="${DATA_ROOT:-lead_data}"
MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
TRAIN_MODE="${TRAIN_MODE:-ddp}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
MATRIX_ROOT="${MATRIX_ROOT:-checkpoints/sft_new_loop_phase3_rgb_matrix/${TIMESTAMP}}"

mkdir -p "${MATRIX_ROOT}"
exec > >(tee -a "${MATRIX_ROOT}/matrix.log") 2>&1

for mode in ${MODES}; do
  echo
  echo "########## history_rgb_mode=${mode} ##########"
  RUN_ROOT="checkpoints/sft_new_loop_phase3_runs/run_high_level_action_${mode}/latest"
  if [[ "${SKIP_TRAIN}" != "1" ]]; then
    INDEX="${INDEX}" DATA_ROOT="${DATA_ROOT}" MODEL_DIR="${MODEL_DIR}" HISTORY_RGB_MODE="${mode}" \
      bash qwen3vl_local/sft_new_loop_phase3/train.sh "${TRAIN_MODE}"
  fi
  if [[ ! -e "${RUN_ROOT}" ]]; then
    echo "[skip] no trained run for ${mode} at ${RUN_ROOT}"
    continue
  fi
  ADAPTER_DIR="${RUN_ROOT}" INDEX="${INDEX}" DATA_ROOT="${DATA_ROOT}" MODEL_DIR="${MODEL_DIR}" \
  OUTPUT_ROOT="${MATRIX_ROOT}/${mode}" RUN_BASE_EVAL="${RUN_BASE_EVAL:-1}" \
    bash qwen3vl_local/sft_new_loop_phase3/eval.sh
done

python - "${MATRIX_ROOT}" <<'PY'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
rows = []
for metrics_path in sorted(root.glob("*/*/metrics.json")):
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows.append(
        {
            "mode": metrics_path.parent.parent.name,
            "run": metrics_path.parent.name,
            "cases": payload.get("total_cases"),
            "exact": payload.get("exact_match_accuracy"),
            "invalid_joint_ok": (payload.get("invalid_contract") or {}).get("invalid_joint_ok_rate"),
        }
    )
(root / "matrix_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
lines = ["# sft_new_loop_phase3 RGB mode matrix", "", "| mode | run | cases | exact | invalid_joint_ok |", "|---|---|---:|---:|---:|"]
for row in rows:
    lines.append(
        f"| {row['mode']} | {row['run']} | {row['cases']} | "
        f"{float(row['exact'] or 0.0):.4f} | {float(row['invalid_joint_ok'] or 0.0):.4f} |"
    )
(root / "matrix_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
