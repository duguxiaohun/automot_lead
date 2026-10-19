#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P3_TRIAL_APP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$P3_TRIAL_APP"
exec "${PYTHON:-python}" -m qwen3vl_local.audit_joint.phase3_base_trial "$@"
