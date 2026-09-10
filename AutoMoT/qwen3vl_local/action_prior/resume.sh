#!/usr/bin/env bash
# 在 AutoMoT/ 下直接复制执行：
#   bash qwen3vl_local/action_prior/resume.sh checkpoints/action_prior/latest/latest.pt
# 先验来源从原 run 的 config.json 原样恢复，续训不能换先验来源。
# 均衡开关/配额/重复上限从 checkpoint 的 config.json 恢复，不在续训中切换课程。
# 索引搬迁（内容相同）示例；旧算法的 checkpoint 必须使用其原代码恢复，新算法另开新 run：
#   bash qwen3vl_local/action_prior/resume.sh checkpoints/action_prior/latest/latest.pt --event-balance-index checkpoints/action_prior_event_balance_v2/full_event_mapping.jsonl
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/resume.sh checkpoints/action_prior/latest/latest.pt --event-balance-index checkpoints/action_prior_event_balance_v2/full_event_mapping.jsonl
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$HERE/resume.py" "$@"
