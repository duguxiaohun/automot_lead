#!/usr/bin/env bash
# 将已完成的逐帧 RGB decisions 与源 EVENT 标注和现有 Phase2 index 联表；不加载模型。
# 从 AutoMoT/ 目录运行：
#   bash qwen3vl_local/sft_new_loop_phase2/run_ue3_label_alignment_audit.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

AUDIT_ROOT="${AUDIT_ROOT:-checkpoints/ue3_route_diverse_full_rgb_audit}"
COLLECTION_DIR="${COLLECTION_DIR:-keyframe_filter/collection_output}"
INDEX="${INDEX:-checkpoints/sft_new_loop_phase2_data/frame_index.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${AUDIT_ROOT}/label_alignment}"

echo "[ue3-alignment] audit_root=${AUDIT_ROOT}"
echo "[ue3-alignment] collection_dir=${COLLECTION_DIR}"
echo "[ue3-alignment] index=${INDEX}"
echo "[ue3-alignment] output=${OUTPUT_DIR}"

python qwen3vl_local/sft_new_loop_phase2/rescore_ue3_rgb_decisions.py \
  --audit-root "${AUDIT_ROOT}"

python qwen3vl_local/sft_new_loop_phase2/audit_ue3_label_alignment.py \
  --audit-root "${AUDIT_ROOT}" \
  --collection-dir "${COLLECTION_DIR}" \
  --index "${INDEX}" \
  --output-dir "${OUTPUT_DIR}"

echo "[done] inspect ${OUTPUT_DIR}/summary.md"
