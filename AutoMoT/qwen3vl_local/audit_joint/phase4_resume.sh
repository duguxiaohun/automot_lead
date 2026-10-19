#!/usr/bin/env bash
# CPU continuation only; use explicit assets on the machine the user chooses.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P4_RESUME_APP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$P4_RESUME_APP"
exec "${PYTHON:-python}" -m qwen3vl_local.audit_joint.phase4_resume "$@"
