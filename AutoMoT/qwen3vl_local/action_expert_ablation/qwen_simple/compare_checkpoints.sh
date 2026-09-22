#!/usr/bin/env bash
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 统一编辑 action_prior/compare_checkpoints.sh 的 CKPT_DIRS，或直接传多个目录。
exec bash "$SCRIPT_DIR/../../action_prior/compare_checkpoints.sh" "$@"
