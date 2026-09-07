#!/usr/bin/env bash
# 在 AutoMoT/ 下直接复制执行：
#   bash qwen3vl_local/action_prior/resume.sh checkpoints/action_prior/latest/latest.pt
# 先验来源从原 run 的 config.json 原样恢复，续训不能换先验来源。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$HERE/resume.py" "$@"
