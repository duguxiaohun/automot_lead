#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P3_COMPARE_APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$P3_COMPARE_APP"
source "$P3_COMPARE_APP/qwen3vl_local/runtime_env.sh" "$P3_COMPARE_APP"
cd "$P3_COMPARE_APP"
if [[ "${QWEN_EXECUTION_VERSION:-}/app" != "$P3_COMPARE_APP" && "${1:---help}" != "--help" && "${1:-}" != "status" ]]; then
  exec "$PYTHON" -B "$P3_COMPARE_APP/qwen3vl_local/runtime_version.py" --app "$P3_COMPARE_APP" -- qwen3vl_local/audit_joint/phase3_compare.sh "$@"
fi
export PYTHON PYTHONPATH="$P3_COMPARE_APP${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$PYTHON" -m qwen3vl_local.audit_joint.phase3_compare "$@"
