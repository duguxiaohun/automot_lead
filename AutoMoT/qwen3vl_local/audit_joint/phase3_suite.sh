#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P3_SUITE_APP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$P3_SUITE_APP"
P3_SUITE_CHECK=()
if [[ "${SKIP_TRAIN:-0}" == 1 || "${PIPELINE_CHECK_ONLY:-0}" == 1 ]]; then
  P3_SUITE_CHECK=(--prepare-only)
fi
exec "${PYTHON:-python}" -m qwen3vl_local.audit_joint.phase3_suite "$@" "${P3_SUITE_CHECK[@]}"
