#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -S -c 0 2>/dev/null || true
trap 'echo "[FAILED] Phase1 paired prompt audit: see error above" >&2' ERR
P1_REFINE_APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$P1_REFINE_APP"
P1_REFINE_PYTHON=${PYTHON:-python}
command -v "$P1_REFINE_PYTHON" >/dev/null
export PYTHONPATH="$P1_REFINE_APP${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$P1_REFINE_PYTHON" -B -m qwen3vl_local.audit_joint.phase1_visual_refine.remote "$@"
