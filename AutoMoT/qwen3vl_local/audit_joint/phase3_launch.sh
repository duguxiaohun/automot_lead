#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P3_LAUNCH_APP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$P3_LAUNCH_APP"
P3_LAUNCH_CHECK=()
for P3_LAUNCH_FLAG in SKIP_TRAIN PIPELINE_CHECK_ONLY; do
  case "${!P3_LAUNCH_FLAG:-0}" in
    0) ;;
    1) P3_LAUNCH_CHECK=(--prepare-only) ;;
    *) printf 'Ambiguous %s; use 0 or 1\n' "$P3_LAUNCH_FLAG" >&2; exit 2 ;;
  esac
done
exec "${PYTHON:-python}" -m qwen3vl_local.audit_joint.phase3_launch "$@" "${P3_LAUNCH_CHECK[@]}"
