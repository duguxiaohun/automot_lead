#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
ABC_APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
source "$ABC_APP/qwen3vl_local/runtime_env.sh" "$ABC_APP"
cd "$ABC_APP"
# Start the whole pipeline in one immutable local source version.
if [[ "${QWEN_EXECUTION_VERSION:-}/app" != "$ABC_APP" ]]; then
  exec "$PYTHON" -B "$ABC_APP/qwen3vl_local/runtime_version.py" --app "$ABC_APP" -- qwen3vl_local/audit_joint/phase3_abc_refine/run.sh "$@"
fi
cd "$ABC_APP"
"$PYTHON" -B "$ABC_APP/qwen3vl_local/runtime_dependencies.py" abc_refine "${1:---help}"
export PYTHONPATH="$ABC_APP${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "${PYTHON:-python}" -B -m qwen3vl_local.audit_joint.phase3_abc_refine "$@"
