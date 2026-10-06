#!/usr/bin/env bash
# Four sequential, complete candidate training/evaluation runs. No automatic promotion.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "${SCRIPT_DIR}/four_runs.py" run "$@"
