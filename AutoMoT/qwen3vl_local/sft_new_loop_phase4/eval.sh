#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P4_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${P4_DIR}/../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [[ "${1:-}" != --help ]]; then
  export CUDA_VISIBLE_DEVICES="$("${PYTHON:-python}" -c 'from qwen3vl_local.sft_new_loop_phase4.devices import select_gpus; print(select_gpus(1))')"
fi
exec "${PYTHON:-python}" -m qwen3vl_local.sft_new_loop_phase4.evaluate "$@"
