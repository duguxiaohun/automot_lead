#!/usr/bin/env bash
# CPU-only bounded labeling audit. Run from AutoMoT, normally inside tmux.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
ulimit -H -c 0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec ionice -c 3 "${PYTHON_BIN:-python}" -u -m qwen3vl_local.audit_joint.local_replay "$@"
