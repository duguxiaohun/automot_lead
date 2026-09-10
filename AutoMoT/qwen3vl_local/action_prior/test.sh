#!/usr/bin/env bash
# CPU 均衡采样回归（不读取真实数据/权重、不训练）：
#   bash qwen3vl_local/action_prior/test.sh -k 'event_balance or repeat or diversity'
# 完整回归：bash qwen3vl_local/action_prior/test.sh；训练开关示例见 run.md。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# CPU 合同/代数/反向传播测试，不加载真实 Qwen 或 CARLA。
python -m pytest "$HERE/tests" -q "$@"
