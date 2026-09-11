#!/usr/bin/env bash
# 在 AutoMoT/ 下直接复制执行（latest 是最新 run 的软链接）：
#   bash qwen3vl_local/action_prior/eval.sh --checkpoint checkpoints/action_prior/latest/best.pt --split test
#   bash qwen3vl_local/action_prior/eval.sh --checkpoint checkpoints/action_prior/latest/best.pt --split test --no-dataset-priors
#   bash qwen3vl_local/action_prior/eval.sh --checkpoint checkpoints/action_prior/latest/best.pt --split test --prior-noise 0
# 默认跟随 checkpoint 记录的先验来源与注入噪声；--prior-noise 0 是干净先验对照。
# 显式改成另一种属于条件迁移，metrics.json 记 prior_source_override=true，不能当同条件复现。
# 均衡训练 checkpoint 的离线评测：采样配置自动恢复，val/test 保持自然分布。
# 索引搬迁时只覆盖路径（内容必须相同），无需再次传 EVENT_BALANCED：
#   bash qwen3vl_local/action_prior/eval.sh --checkpoint checkpoints/action_prior/latest/best.pt --event-balance-index checkpoints/action_prior_event_balance_v2/full_event_mapping.jsonl
#   GPU_IDS=0 bash qwen3vl_local/action_prior/eval.sh --checkpoint checkpoints/action_prior/latest/best.pt --event-balance-index checkpoints/action_prior_event_balance_v2/full_event_mapping.jsonl
# 新训练开关与构建步骤见 run.md。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# --bench2drive 显式进入正式闭环；其余参数维持离线评测兼容。
if [[ "${1:-}" == "--bench2drive" ]]; then
 shift
 exec bash "$HERE/bench2drive.sh" "$@"
fi
# 默认按 checkpoint 记录的先验来源评测；--dataset-priors/--no-dataset-priors 可显式覆盖。
args=("$@")
has_flag() { local flag="$1"; shift; [[ " $* " == *" $flag "* || " $* " == *" $flag="* ]]; }
if has_flag --dataset-priors "$@" && ! has_flag --prior-labels "$@"; then
 args+=(--prior-labels "${PRIOR_LABELS:-checkpoints/action_prior_labels/prior_labels.jsonl}")
fi
exec python "$HERE/launch.py" eval "${args[@]}"
