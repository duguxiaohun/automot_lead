#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P3_BOUNDED_APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$P3_BOUNDED_APP"
if [[ -z ${PYTHON:-} ]]; then
  if [[ -x "$P3_BOUNDED_APP/.venv/bin/python" ]]; then
    PYTHON="$P3_BOUNDED_APP/.venv/bin/python"
  else
    PYTHON=$(command -v python)
  fi
fi
export PYTHON PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec bash qwen3vl_local/audit_joint/phase3_launch.sh --method base --base-budget bounded_v3 \
  --output "$P3_BOUNDED_APP/checkpoints/phase3_base_bounded_runs_v3" "$@"
