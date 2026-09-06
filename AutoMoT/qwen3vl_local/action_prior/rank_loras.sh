#!/usr/bin/env bash
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 只读评测日志与权重指纹，不申请 GPU，不启动模型或训练。
exec python "$HERE/rank_loras.py" "$@"
