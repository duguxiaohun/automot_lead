#!/usr/bin/env bash
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# CPU 合同/代数/反向传播测试，不加载真实 Qwen 或 CARLA。
python -m pytest "$HERE/tests" -q "$@"
