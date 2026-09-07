#!/usr/bin/env bash
# 在 AutoMoT/ 下直接复制执行：
#   bash qwen3vl_local/action_prior/probe.sh --checkpoint checkpoints/action_prior/latest/best.pt --split test
#   bash qwen3vl_local/action_prior/probe.sh --checkpoint checkpoints/action_prior/latest/best.pt --split test --no-dataset-priors
# 与 eval.sh 一样默认跟随 checkpoint 记录的先验来源。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$HERE/launch.py" probe "$@"
