#!/usr/bin/env bash
# Preserve the chosen interpreter while separating torchrun/application options.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
: "${AUTOMOT_REAL_PYTHON:?Set the real Python interpreter; use the audit workflow entry}"
exec "$AUTOMOT_REAL_PYTHON" -m qwen3vl_local.audit_joint.python_dispatch "$@"
