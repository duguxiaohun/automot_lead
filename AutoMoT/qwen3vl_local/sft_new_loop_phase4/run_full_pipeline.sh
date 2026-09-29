#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P4_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${P4_DIR}/../.."
export DATASET="${DATASET:-checkpoints/sft_new_loop_phase4_data_$(date +%Y%m%d_%H%M%S_%N)}"
export OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft_new_loop_phase4_runs/$(date +%Y%m%d_%H%M%S_%N)}"
"${PYTHON:-python}" -m qwen3vl_local.sft_new_loop_phase4.dataset \
  --annotations "${ANNOTATIONS:-qwen3vl_local/sft_new_loop_phase4/reviewed_state_pairs_v3.json}" \
  --data-root "${DATA_ROOT:-lead_data}" --output-dir "$DATASET" --rgb-mode "${RGB_MODE:-4}"
bash "$P4_DIR/train.sh" "${MODE:-ddp}" "$@"
# Sampling-only checks deliberately stop before adapter-dependent stages.
if [[ "${MODE:-ddp}" == check ]]; then exit 0; fi
P4_BEST="$("${PYTHON:-python}" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print((p/json.loads((p/"best.json").read_text())["adapter"]).resolve())' "$OUTPUT_DIR")"
P4_EVAL_GPU="$("${PYTHON:-python}" -c 'from qwen3vl_local.sft_new_loop_phase4.devices import select_gpus; print(select_gpus(1))')"
CUDA_VISIBLE_DEVICES="$P4_EVAL_GPU" "${PYTHON:-python}" -m qwen3vl_local.sft_new_loop_phase4.evaluate \
  --dataset "$DATASET" --adapter "$P4_BEST" --model-dir "${MODEL_DIR:-checkpoints/Qwen3.5-4B}" \
  --data-root "${DATA_ROOT:-lead_data}" --split test --output-dir "$OUTPUT_DIR/test" \
  --device "${EVAL_DEVICE:-cuda:0}"
"${PYTHON:-python}" -m qwen3vl_local.sft_new_loop_phase4.audit_bundle \
  --run "$OUTPUT_DIR" --output "$OUTPUT_DIR/audit.zip"
