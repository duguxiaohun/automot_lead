#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
QWEN_VERSION_APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$QWEN_VERSION_APP/qwen3vl_local/runtime_env.sh" "$QWEN_VERSION_APP"
exec "$PYTHON" -B "$QWEN_VERSION_APP/qwen3vl_local/runtime_version.py" --app "$QWEN_VERSION_APP" "$@"
