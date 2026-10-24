#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -z ${PYTHON:-} ]]; then
  if [[ -n ${VIRTUAL_ENV:-} && -x "$VIRTUAL_ENV/bin/python" ]]; then PYTHON="$VIRTUAL_ENV/bin/python"
  elif [[ -n ${CONDA_PREFIX:-} && -x "$CONDA_PREFIX/bin/python" ]]; then PYTHON="$CONDA_PREFIX/bin/python"
  elif [[ -x "$APP/.venv/bin/python" ]]; then PYTHON="$APP/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then PYTHON=$(command -v python)
  else PYTHON=$(command -v python3); fi
fi
PYTHON=$(command -v "$PYTHON")
export PYTHONPATH="$APP" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$APP"
exec "$PYTHON" -B "$APP/qwen3vl_local/audit_joint/phase3_single_refine/bootstrap.py" "$@"
