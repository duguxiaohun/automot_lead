#!/usr/bin/env bash
# 不重跑模型：把 route-diverse rescore 的 32 个 UE3 正例全部导出为四帧 RGB 审计包。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

EXPERIMENT_ID="${EXPERIMENT_ID:-v3_frozen_3seed_unseen456_20260831}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-checkpoints/sft_new_loop_phase2_frozen_protocol/${EXPERIMENT_ID}}"
EVAL_ROOT="${EVAL_ROOT:-${EXPERIMENT_ROOT}/route_diverse_validation_rescore}"
INDEX="${INDEX:-checkpoints/sft_new_loop_phase2_data/frame_index.jsonl}"
DATA_ROOT="${DATA_ROOT:-lead_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_ROOT}/ue3_route_diverse_full_rgb_audit}"

if [[ ! -f "${INDEX}" ]]; then
  echo "Missing dataset index: ${INDEX}" >&2
  exit 2
fi
if ! compgen -G "${EVAL_ROOT}/seed_*/cases*.jsonl" >/dev/null; then
  echo "Missing rescore case records: ${EVAL_ROOT}/seed_*/cases*.jsonl" >&2
  echo "Run RUN_UNSEEN=0 bash qwen3vl_local/sft_new_loop_phase2/run_route_diverse_validation_rescore.sh first." >&2
  exit 2
fi

python qwen3vl_local/sft_new_loop_phase2/build_ue3_validation_rgb_audit.py \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --index "${INDEX}" \
  --data-root "${DATA_ROOT}" \
  --source-mode eval \
  --eval-root "${EVAL_ROOT}" \
  --include-correct \
  --output-dir "${OUTPUT_DIR}" \
  --overwrite

echo "[done] UE3 full RGB audit=${OUTPUT_DIR}"
echo "[done] archive=${OUTPUT_DIR}.tar.gz"
