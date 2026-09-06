#!/usr/bin/env bash
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# --bench2drive 显式进入正式闭环；其余参数维持离线评测兼容。
if [[ "${1:-}" == "--bench2drive" ]]; then
 shift
 exec bash "$HERE/bench2drive.sh" "$@"
fi
exec python "$HERE/launch.py" eval "$@"
