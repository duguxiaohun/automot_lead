#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -S -c 0 2>/dev/null || true
trap 'code=$?; echo "[FAILED] fresh training/evaluation exit=${code}; see error above" >&2' ERR
APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
source "$APP/qwen3vl_local/runtime_env.sh" "$APP"
export PATH="$(dirname "$PYTHON"):$PATH"
cd "$APP"
"$PYTHON" -B -m qwen3vl_local.audit_joint.phase12_retrain "$@"
