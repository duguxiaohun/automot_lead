#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -S -c 0 2>/dev/null || true
trap 'code=$?; echo "[STOPPED] Phase1 v2 exit=$code; see the error or REVIEW_REQUIRED details above" >&2' ERR
P1_V2_APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$P1_V2_APP"
P1_V2_PYTHON=${PYTHON:-python}
command -v "$P1_V2_PYTHON" >/dev/null
export PYTHONPATH="$P1_V2_APP${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$P1_V2_PYTHON" -B -m qwen3vl_local.audit_joint.phase1_visual_refine_v2.remote "$@"
