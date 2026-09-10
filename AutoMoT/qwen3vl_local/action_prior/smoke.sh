#!/usr/bin/env bash
# 在 AutoMoT/ 下直接复制执行：
#   bash qwen3vl_local/action_prior/smoke.sh
#   DATASET_PRIORS=1 bash qwen3vl_local/action_prior/smoke.sh
#   DATASET_PRIORS=1 PRIOR_NOISE=0.1 bash qwen3vl_local/action_prior/smoke.sh
# UE/特殊 RE 均衡采样（先按 run.md 构建 action index 和 v2 full map）：
#   DATA_DIR=checkpoints/action_prior_data_event_v1 EVENT_BALANCED=1 EVENT_BALANCE_INDEX=checkpoints/action_prior_event_balance_v2/full_event_mapping.jsonl bash qwen3vl_local/action_prior/smoke.sh --dataset-priors
#   GPU_IDS=0 DATA_DIR=checkpoints/action_prior_data_event_v1 EVENT_BALANCED=1 EVENT_BALANCE_INDEX=checkpoints/action_prior_event_balance_v2/full_event_mapping.jsonl bash qwen3vl_local/action_prior/smoke.sh --dataset-priors
# 关闭采样开关：上面的命令追加 --sampling-mode uniform；恢复自然分布，保留 dataset-priors 选择。
# 重复上限 EVENT_BALANCE_MAX_FRAME_REPEATS=8；自动 epoch 预算 EVENT_BALANCED_EPOCH_SAMPLES=0。
# 可选 BEST_SELECTION_METRIC=event_balanced_ade 要求 val 全桶覆盖；默认 natural_ade。
# 可选 EVENT_BALANCED_SCENE_PRIORS=1 只用于 dataset-priors + PRIOR_NOISE=0 的离线条件实验，闭环禁用。
# smoke 仍使用真实模型；四次更新后的 best 验证会遍历完整 val，val-max-samples 仅限制周期诊断。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 用真实模型/真实索引跑四个更新；显式区别于 61 epoch 正式训练。
DDP_GPU_COUNT="${DDP_GPU_COUNT:-1}" GRAD_ACCUM=1 NUM_WORKERS=0 VAL_STEPS=2 \
 bash "$HERE/train.sh" --max-train-steps 4 --val-max-samples 4 --logging-steps 1 "$@"
