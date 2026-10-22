#!/usr/bin/env bash
# Standalone CPU evidence; no model weights, GPU selection, SSH or source changes.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -z ${PYTHON:-} ]]; then
  if [[ -x "$APP/.venv/bin/python" ]]; then PYTHON="$APP/.venv/bin/python"; else PYTHON=$(command -v python); fi
fi
export PYTHONPATH="$APP${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$PYTHON" -m qwen3vl_local.audit_joint.geometry_study_cpu_audit "$@"
