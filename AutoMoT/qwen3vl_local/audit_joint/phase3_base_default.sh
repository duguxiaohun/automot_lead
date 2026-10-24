#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P3_DEFAULT_APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$P3_DEFAULT_APP"
if [[ -z ${PYTHON:-} ]]; then
  if [[ -x "$P3_DEFAULT_APP/.venv/bin/python" ]]; then PYTHON="$P3_DEFAULT_APP/.venv/bin/python"; else PYTHON=$(command -v python); fi
fi
export PYTHON PYTHONPATH="$P3_DEFAULT_APP${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$PYTHON" -B -m qwen3vl_local.audit_joint.phase3_base_default "$@"
