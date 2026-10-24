#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -z ${PYTHON:-} ]]; then
  if [[ -x "$APP/.venv/bin/python" ]]; then PYTHON="$APP/.venv/bin/python"; else PYTHON=$(command -v python); fi
fi
export PYTHONPATH="$APP" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$APP"
exec "$PYTHON" -B -m qwen3vl_local.audit_joint.phase3_single_refine.pipeline "$@"
