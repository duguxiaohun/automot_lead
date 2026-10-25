#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P3_SINGLE_DEFAULT_APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$P3_SINGLE_DEFAULT_APP"
source "$P3_SINGLE_DEFAULT_APP/qwen3vl_local/runtime_env.sh" "$P3_SINGLE_DEFAULT_APP"
export PYTHONPATH="$P3_SINGLE_DEFAULT_APP${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$PYTHON" -B -m qwen3vl_local.audit_joint.phase3_single_default "$@"
