#!/usr/bin/env bash
# 开启动作 token 默认加弱分离（weight=0.01, cosine margin=0.5）；新 run 对照加 --action-token-separation-weight 0。
# 默认7轮：首轮5% optimizer更新warmup（占用首周期），1/2/4轮cosine，累计第1/3/7轮末到谷底。
# 优化细节统一默认；每个epoch训练/验证后自动更新run目录的 training_audit.zip，无需审计开关。
# 优化器/LR 共用 action_prior Python 配置：默认 muon_adamw + cosine_restarts。
# 可追加 --optimizer adamw --lr-scheduler cosine 作基线；环境变量 OPTIMIZER/LR_SCHEDULER 同样生效，CLI 优先。
# 续训恢复原配置并严格校验；完整参数和三组对照见 action_prior/OPTIMIZATION.md。
# 运行示例（在 AutoMoT/ 下执行）：
# 构建/复用索引 → 训练 → 用本次 best.pt 做 test；默认自动选四张空闲 GPU。
#   bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh
# 事件均衡（full pipeline 自动准备；train.sh 需显式 EVENT_BALANCE_INDEX 或 --event-balance-index）：
#   bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced
# 兼容 EVENT_BALANCED=1；--no-event-balanced 关闭。课程/比例直接引用 action_prior，详见 ../run.md。
# 单当前图 demo（默认仍为四图；更换图数需新开 run）：
#   bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced --rgb-frame-count 1
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced --rgb-frame-count 1
# 单当前图＋high-level 动作 token（KEEP 不细分）：
#   bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced --rgb-frame-count 1 --high-level-action-token
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced --rgb-frame-count 1 --high-level-action-token
# 环境变量等价写法；显式 CLI 优先：
#   RGB_FRAME_COUNT=1 HIGH_LEVEL_ACTION_TOKEN=1 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced
# 四图对照：将 --rgb-frame-count 1 换成 --rgb-frame-count 4；关闭 token 用 --no-high-level-action-token。
# bev_only 不使用 Qwen，BEV 原本就是当前单帧 RGB＋LiDAR；此选项不改变其有效模型输入。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
source qwen3vl_local/action_expert_ablation/pipeline_common.sh
action_ablation_run_full_pipeline bev_only checkpoints/action_expert_ablation/bev_only "$@"
