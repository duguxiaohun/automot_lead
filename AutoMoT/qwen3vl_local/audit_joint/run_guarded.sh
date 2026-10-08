#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
ulimit -H -c 0
[[ "$(ulimit -Sc)" == 0 && "$(ulimit -Hc)" == 0 ]] || { echo 'Cannot disable core limits; no job launched.' >&2; exit 2; }
exec "${PYTHON:-python}" -m qwen3vl_local.audit_joint.guard "$@"
