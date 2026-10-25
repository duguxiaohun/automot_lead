#!/usr/bin/env bash
# Launch a frozen project script with the explicitly selected/active interpreter.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
QWEN_APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$QWEN_APP/qwen3vl_local/runtime_env.sh" "$QWEN_APP"
if [[ $# -lt 1 ]]; then echo 'Usage: bash qwen3vl_local/run_with_env.sh qwen3vl_local/path/to/script.sh [arguments]' >&2; exit 2; fi
QWEN_TARGET=$1
shift
cd "$QWEN_APP"
if [[ "${QWEN_EXECUTION_VERSION:-}/app" == "$QWEN_APP" ]]; then exec bash "$QWEN_TARGET" "$@"; fi
exec "$PYTHON" -B "$QWEN_APP/qwen3vl_local/runtime_version.py" --app "$QWEN_APP" -- "$QWEN_TARGET" "$@"
