#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P3_DIAG_APP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$P3_DIAG_APP"
export PYTHONUNBUFFERED=1
exec "${PYTHON:-python}" -m qwen3vl_local.audit_joint.phase3_matched_image_launch "$@"
