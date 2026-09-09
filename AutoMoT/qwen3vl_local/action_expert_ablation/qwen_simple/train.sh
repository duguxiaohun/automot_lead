#!/usr/bin/env bash
# 运行示例（在 AutoMoT/ 下执行；需已有共享数据索引）：
#   bash qwen3vl_local/action_expert_ablation/qwen_simple/train.sh
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/qwen_simple/train.sh
# 续训（使用与 checkpoint 匹配的代码版本）：
#   bash qwen3vl_local/action_expert_ablation/qwen_simple/train.sh --resume checkpoints/action_expert_ablation/qwen_simple/latest/latest.pt
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
export PYTHONUNBUFFERED=1

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
add_default_arg MODEL_DIR --model-dir checkpoints/Qwen3-VL-4B-Instruct
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


exec python qwen3vl_local/action_expert_ablation/launch.py train --variant qwen_simple "${args[@]}" "$@"
