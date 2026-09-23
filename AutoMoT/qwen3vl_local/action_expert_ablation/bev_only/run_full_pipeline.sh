#!/usr/bin/env bash
# 在 AutoMoT/ 下执行；自动准备数据、选卡。默认 event 均衡、7轮、每轮116256次。
# 默认 event 均衡：
#   bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh
# 显式单图参数（BEV 始终为当前单帧 RGB + LiDAR；图数参数用于入口对齐）：
#   bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --rgb-frame-count 1
# action 均衡 + token + 显式单图参数：
#   bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --action-balanced --high-level-action-token --rgb-frame-count 1
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --action-balanced --high-level-action-token --rgb-frame-count 1
# 续训（恢复原配置，使用匹配源码）：
#   bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --resume checkpoints/action_expert_ablation/bev_only/latest/latest.pt
# 两种采样默认每轮116256次、单帧上限11；action内小事件权重最多2倍。
# 容量不足报错；--event-balanced-epoch-samples 0 可恢复自动预算，显式参数/续训原配置优先。
# BEV 本身使用当前单帧 RGB + LiDAR，无 Qwen 图数区别。
# 其它参数及实验说明见同目录 run.md。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
source qwen3vl_local/action_expert_ablation/pipeline_common.sh
action_ablation_run_full_pipeline bev_only checkpoints/action_expert_ablation/bev_only "$@"
