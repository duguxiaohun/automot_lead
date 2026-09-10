#!/usr/bin/env bash
# 在已配 CARLA 0.9.15 的 AutoMoT/ 下直接复制执行：
#   bash qwen3vl_local/action_prior/eval.sh --bench2drive --checkpoint checkpoints/action_prior/latest/best.pt
#   ACTION_DATASET_PRIORS=0 bash qwen3vl_local/action_prior/eval.sh --bench2drive --checkpoint checkpoints/action_prior/latest/best.pt
# 闭环实时帧没有数据集标签：数据集先验 checkpoint 必须用第二条显式换回 LoRA 先验，
# 并在报告里声明条件迁移。
# 纯 EVENT_BALANCED=1 采样模型在闭环无需 full map，也不需要重新打开训练采样开关。
# EVENT_BALANCED_SCENE_PRIORS=1 的离线模型会被拒绝；不要用 GT transition 伪造在线输入。
#   GPU_IDS=0 ACTION_DATASET_PRIORS=0 bash qwen3vl_local/action_prior/bench2drive.sh --checkpoint checkpoints/action_prior/latest/best.pt
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$HERE/bench2drive.py" "$@"
