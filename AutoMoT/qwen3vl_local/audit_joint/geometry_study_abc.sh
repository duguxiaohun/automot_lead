#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$APP/qwen3vl_local/runtime_env.sh" "$APP"
cd "$APP"
if [[ "${QWEN_EXECUTION_VERSION:-}/app" != "$APP" && "${1:---help}" != "--help" && "${1:-}" != "status" ]]; then
  exec "$PYTHON" -B "$APP/qwen3vl_local/runtime_version.py" --app "$APP" -- qwen3vl_local/audit_joint/geometry_study_abc.sh "$@"
fi
cd "$APP"
export PYTHONPATH="$APP${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$PYTHON" -m qwen3vl_local.audit_joint.geometry_study_abc "$@"
