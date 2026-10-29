#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -S -c 0 2>/dev/null || true
trap 'code=$?; echo "[FAILED] Phase1+2 shared LoRA exit=$code; see error above" >&2' ERR
APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$APP/qwen3vl_local/runtime_env.sh" "$APP"
cd "$APP"
export PYTHONPATH="$APP" PATH="$(dirname "$PYTHON"):$PATH"
"$PYTHON" -B -m qwen3vl_local.sft_new_loop_phase12 "$@"
