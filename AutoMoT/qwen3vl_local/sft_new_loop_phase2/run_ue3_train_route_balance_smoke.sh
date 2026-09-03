#!/usr/bin/env bash
# CPU-only：验证 UE3 route-balanced 训练曝光，不加载模型、不改 index、不触碰 unseen。
# 从 AutoMoT/ 目录运行：
#   bash qwen3vl_local/sft_new_loop_phase2/run_ue3_train_route_balance_smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

INDEX="${INDEX:-checkpoints/sft_new_loop_phase2_data/frame_index.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/ue3_train_route_balance_smoke}"
FOCUS_BALANCE_COUNT="${FOCUS_BALANCE_COUNT:-2048}"
SEED="${SEED:-20260810}"
MAX_TRAIN_UE3_FRAME_REPEAT="${MAX_TRAIN_UE3_FRAME_REPEAT:-10}"

if [[ ! -f "${INDEX}" ]]; then
  echo "missing index: ${INDEX}" >&2
  exit 2
fi

python qwen3vl_local/sft_new_loop_phase2/audit_ue3_train_route_balance.py \
  --index "${INDEX}" \
  --output-dir "${OUTPUT_DIR}" \
  --target "${FOCUS_BALANCE_COUNT}" \
  --seed "${SEED}" \
  --max-frame-repeat "${MAX_TRAIN_UE3_FRAME_REPEAT}"

echo "[done] ${OUTPUT_DIR}/summary.json"
echo "[done] ${OUTPUT_DIR}/summary.md"
echo "[stop] CPU sampling smoke complete; no model, training, or unseen evaluation was started"
