#!/usr/bin/env bash
# 优化细节统一默认；每个epoch训练/验证后自动更新run目录的 training_audit.zip，无需审计开关。
# 优化器/LR 共用 action_prior Python 配置：默认 muon_adamw + cosine_restarts。
# 可追加 --optimizer adamw --lr-scheduler cosine 作基线；环境变量 OPTIMIZER/LR_SCHEDULER 同样生效，CLI 优先。
# 续训恢复原配置并严格校验；完整参数和三组对照见 action_prior/OPTIMIZATION.md。
# 运行示例（在 AutoMoT/ 下执行；需已有共享数据索引）：
#   bash qwen3vl_local/action_expert_ablation/bev_only/train.sh
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/train.sh
# 续训（使用与 checkpoint 匹配的代码版本）：
#   bash qwen3vl_local/action_expert_ablation/bev_only/train.sh --resume checkpoints/action_expert_ablation/bev_only/latest/latest.pt
# 事件均衡（full pipeline 自动准备；train.sh 需显式 EVENT_BALANCE_INDEX 或 --event-balance-index）：
#   bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced
# 兼容 EVENT_BALANCED=1；--no-event-balanced 关闭。课程/比例直接引用 action_prior，详见 ../run.md。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
export PYTHONUNBUFFERED=1
source qwen3vl_local/action_prior/event_balance_common.sh
action_event_balance_options "$@"
set -- "${ACTION_EVENT_BALANCE_ARGS[@]}"
# 消融拒绝特权 scene prior；纯采样开关不会引入先验文本。
if [[ "${EVENT_BALANCED_SCENE_PRIORS:-0}" == 1 ]]; then
  set -- --event-balanced-scene-priors "$@"
fi

resume_requested=0
if [[ -n "${RESUME:-}" ]]; then
  resume_requested=1
fi
for item in "$@"; do
  case "$item" in
    --resume|--resume=*)
      resume_requested=1
      ;;
  esac
done

args=()
add_default_arg() {
  local env_name="$1"
  local option="$2"
  local default_value="$3"
  if [[ "$resume_requested" == 0 || -v "$env_name" ]]; then
    args+=("$option" "${!env_name:-$default_value}")
  fi
}

add_default_arg DATA_ROOT --data-root lead_data
add_default_arg DATA_DIR --data-dir checkpoints/action_prior_data
add_default_arg LEAD_BEV_CKPT --lead-bev-ckpt checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth
add_default_arg NUM_EPOCHS --num-epochs 61
add_default_arg LR --learning-rate 0.0002
add_default_arg GRAD_ACCUM --grad-accum-steps 16
add_default_arg VAL_STEPS --val-steps 250
add_default_arg SAVE_STEPS --save-steps 1000
add_default_arg LOGGING_STEPS --logging-steps 10
add_default_arg NUM_WORKERS --num-workers 8
add_default_arg FLOW_SAMPLE_STEPS --flow-sample-steps 10
add_default_arg FLOW_ROUTE_COORDINATE_SCALE_M --flow-route-coordinate-scale-m 30
add_default_arg FLOW_WAYPOINT_COORDINATE_SCALE_M --flow-waypoint-coordinate-scale-m 20
add_default_arg FLOW_TIME_EMBED_DIM --flow-time-embed-dim 64
add_default_arg FLOW_TRAJECTORY_LAYERS --flow-trajectory-layers 2
add_default_arg FLOW_TRAJECTORY_HEADS --flow-trajectory-heads 8
if [[ -v TRAIN_SAMPLED_METRICS ]]; then
  if [[ "${TRAIN_SAMPLED_METRICS:-0}" == 1 ]]; then
    args+=(--train-sampled-metrics)
  else
    args+=(--no-train-sampled-metrics)
  fi
elif [[ "$resume_requested" == 0 && "${TRAIN_SAMPLED_METRICS:-0}" == 1 ]]; then
  args+=(--train-sampled-metrics)
fi


exec python qwen3vl_local/action_expert_ablation/launch.py train --variant bev_only "${args[@]}" "$@"
