#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P3_COVERAGE_APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$P3_COVERAGE_APP"
if [[ -z ${PYTHON:-} ]]; then
  if [[ -x "$P3_COVERAGE_APP/.venv/bin/python" ]]; then
    PYTHON="$P3_COVERAGE_APP/.venv/bin/python"
  else
    PYTHON=$(command -v python)
  fi
fi
export PYTHON PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec bash qwen3vl_local/audit_joint/phase3_launch.sh --method base --base-budget coverage_v2 \
  --output "$P3_COVERAGE_APP/checkpoints/phase3_base_coverage_v2" "$@"
