#!/usr/bin/env bash
# CPU-only export of existing results. No model loading, training, or evaluation.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"
exec python -m qwen3vl_local.sft_new_loop_phase3.pack_results "$@"
