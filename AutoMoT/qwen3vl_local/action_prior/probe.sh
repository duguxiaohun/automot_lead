#!/usr/bin/env bash
# 在 AutoMoT/ 下直接复制执行：
#   bash qwen3vl_local/action_prior/probe.sh --checkpoint checkpoints/action_prior/latest/best.pt --split test
#   bash qwen3vl_local/action_prior/probe.sh --checkpoint checkpoints/action_prior/latest/best.pt --split test --no-dataset-priors
# 与 eval.sh 一样默认跟随 checkpoint 记录的先验来源。
# 均衡训练 checkpoint 的离线评测：采样配置自动恢复，val/test 保持自然分布。
# 索引搬迁时只覆盖路径（内容必须相同），无需再次传 EVENT_BALANCED：
#   bash qwen3vl_local/action_prior/probe.sh --checkpoint checkpoints/action_prior/latest/best.pt --event-balance-index checkpoints/action_prior_event_balance_v2/full_event_mapping.jsonl
#   GPU_IDS=0 bash qwen3vl_local/action_prior/probe.sh --checkpoint checkpoints/action_prior/latest/best.pt --event-balance-index checkpoints/action_prior_event_balance_v2/full_event_mapping.jsonl
# 新训练开关与构建步骤见 run.md「UE/特殊 RE 均衡课程」。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$HERE/launch.py" probe "$@"
