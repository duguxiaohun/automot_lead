#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
ABC_APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$ABC_APP${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "${PYTHON:-python}" -B -m qwen3vl_local.audit_joint.phase3_abc_refine "$@"
