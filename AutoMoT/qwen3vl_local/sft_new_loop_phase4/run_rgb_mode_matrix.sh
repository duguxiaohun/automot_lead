#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P4_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for P4_RGB_MODE in 2 4; do
  RGB_MODE="$P4_RGB_MODE" DATASET="checkpoints/sft_new_loop_phase4_${P4_RGB_MODE}rgb_$(date +%Y%m%d_%H%M%S_%N)" \
    bash "$P4_DIR/run_full_pipeline.sh" "$@"
done
