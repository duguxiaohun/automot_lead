#!/usr/bin/env bash
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 对比已有指标，默认复制并压缩推荐 LoRA；不申请 GPU、不改源权重。
# --no-export-bundle 仅审计；默认打印 tar.gz 路径、SHA256 和固定组合训练命令。
exec python "$HERE/rank_loras.py" "$@"
