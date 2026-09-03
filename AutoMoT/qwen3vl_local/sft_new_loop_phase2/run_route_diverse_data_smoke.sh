#!/usr/bin/env bash
# CPU-only：重建 route-diverse train index，并硬校验 val/test frozen 身份完全不变。
# 从 AutoMoT/ 目录运行：
#   bash qwen3vl_local/sft_new_loop_phase2/run_route_diverse_data_smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

OLD_INDEX="${OLD_INDEX:-checkpoints/sft_new_loop_phase2_data/frame_index.jsonl}"
NEW_DATA_DIR="${NEW_DATA_DIR:-checkpoints/sft_new_loop_phase2_data_route_diverse_smoke}"
COLLECTION_DIR="${COLLECTION_DIR:-keyframe_filter/collection_output}"
DATA_ROOT="${DATA_ROOT:-lead_data}"

if [[ ! -f "${OLD_INDEX}" ]]; then
  echo "missing old index: ${OLD_INDEX}" >&2
  exit 2
fi

echo "[route-data-smoke] old_index=${OLD_INDEX}"
echo "[route-data-smoke] new_data_dir=${NEW_DATA_DIR}"

python qwen3vl_local/sft_new_loop_phase2/build_dataset.py \
  --collection-dir "${COLLECTION_DIR}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${NEW_DATA_DIR}" \
  --test-ratio 0.10 \
  --val-ratio 0.05 \
  --regular-multiplier 1.0 \
  --highway-regular-fraction 0.25 \
  --invalid-ratio 0.20

python qwen3vl_local/sft_new_loop_phase2/compare_dataset_route_diversity.py \
  --old-index "${OLD_INDEX}" \
  --new-index "${NEW_DATA_DIR}/frame_index.jsonl" \
  --output-dir "${NEW_DATA_DIR}/route_diversity_comparison"

echo "[done] ${NEW_DATA_DIR}/route_diversity_comparison/comparison.md"
echo "[stop] CPU data smoke complete; no training or unseen evaluation was started"
