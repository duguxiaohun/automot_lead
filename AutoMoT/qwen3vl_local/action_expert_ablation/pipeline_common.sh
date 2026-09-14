#!/usr/bin/env bash
# 供两个 run_full_pipeline.sh source 的共享函数；完整运行请使用下列入口。
# 运行示例（在 AutoMoT/ 下执行；默认自动选四张空闲 GPU）：
#   bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh
# 将 bev_only 换成 qwen_simple 即运行另一组消融。
ulimit -S -c 0 2>/dev/null || true
source qwen3vl_local/action_prior/event_balance_common.sh
action_ablation_index_ready() {
  action_shared_index_ready "$1"
}

action_ablation_build_index_if_needed() {
  action_build_shared_index_if_needed "$1" "$2"
}

action_ablation_config_value() {
  local config_path="$1"
  local key="$2"
  local fallback="$3"
  command python - "$config_path" "$key" "$fallback" <<'PY'
import json
import sys
path, key, fallback = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
value = data.get(key, fallback)
print(fallback if value is None else value)
PY
}

action_ablation_restore_config_value() {
  local var_name="$1"
  local config_path="$2"
  local key="$3"
  local explicit="$4"
  if [[ "$explicit" == 0 ]]; then
    local value
    value="$(action_ablation_config_value "$config_path" "$key" "${!var_name}")"
    printf -v "$var_name" "%s" "$value"
  fi
}

action_ablation_run_full_pipeline() {
  local variant="$1"
  local default_output_dir="$2"
  shift 2

  action_event_balance_options "$@"
  set -- "${ACTION_EVENT_BALANCE_ARGS[@]}"
  local sampling_mode="$ACTION_EVENT_SAMPLING_MODE"
  local event_balance_index="$ACTION_EVENT_BALANCE_INDEX"
  # 在构建数据之前拒绝先验输入，避免把标签静默加入消融。
  local scene_priors="${EVENT_BALANCED_SCENE_PRIORS:-0}"
  local item
  for item in "$@"; do
    case "$item" in
      --event-balanced-scene-priors) scene_priors=1 ;;
      --no-event-balanced-scene-priors) scene_priors=0 ;;
    esac
  done
  if [[ "$scene_priors" == 1 ]]; then
    echo "ablations do not accept --event-balanced-scene-priors; use --event-balanced for sampling only" >&2
    return 2
  fi

  local data_root="${DATA_ROOT:-lead_data}"
  local data_dir="${DATA_DIR:-checkpoints/action_prior_data}"
  local train_output_dir="${OUTPUT_DIR:-$default_output_dir}"
  local train_resume="${RESUME:-}"
  local model_dir="${MODEL_DIR:-}"
  local lead_bev_ckpt="${LEAD_BEV_CKPT:-}"
  local data_root_explicit=0
  local data_dir_explicit=0
  local model_dir_explicit=0
  local lead_bev_ckpt_explicit=0
  [[ ${DATA_ROOT+x} ]] && data_root_explicit=1
  [[ ${DATA_DIR+x} ]] && data_dir_explicit=1
  [[ ${MODEL_DIR+x} ]] && model_dir_explicit=1
  [[ ${LEAD_BEV_CKPT+x} ]] && lead_bev_ckpt_explicit=1
  local args=("$@")
  local passthrough=()

  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
      --data-root)
        if (( i + 1 >= ${#args[@]} )); then
          echo "--data-root requires a value" >&2
          return 2
        fi
        data_root="${args[$((i + 1))]}"
        data_root_explicit=1
        ((i += 1))
        ;;
      --data-root=*)
        data_root="${args[$i]#--data-root=}"
        data_root_explicit=1
        ;;
      --data-dir)
        if (( i + 1 >= ${#args[@]} )); then
          echo "--data-dir requires a value" >&2
          return 2
        fi
        data_dir="${args[$((i + 1))]}"
        data_dir_explicit=1
        ((i += 1))
        ;;
      --data-dir=*)
        data_dir="${args[$i]#--data-dir=}"
        data_dir_explicit=1
        ;;
      --output-dir)
        if (( i + 1 >= ${#args[@]} )); then
          echo "--output-dir requires a value" >&2
          return 2
        fi
        train_output_dir="${args[$((i + 1))]}"
        ((i += 1))
        ;;
      --output-dir=*)
        train_output_dir="${args[$i]#--output-dir=}"
        ;;
      --resume)
        if (( i + 1 >= ${#args[@]} )); then
          echo "--resume requires a value" >&2
          return 2
        fi
        train_resume="${args[$((i + 1))]}"
        ((i += 1))
        ;;
      --resume=*)
        train_resume="${args[$i]#--resume=}"
        ;;
      --model-dir)
        if (( i + 1 >= ${#args[@]} )); then
          echo "--model-dir requires a value" >&2
          return 2
        fi
        model_dir="${args[$((i + 1))]}"
        model_dir_explicit=1
        ((i += 1))
        ;;
      --model-dir=*)
        model_dir="${args[$i]#--model-dir=}"
        model_dir_explicit=1
        ;;
      --lead-bev-ckpt)
        if (( i + 1 >= ${#args[@]} )); then
          echo "--lead-bev-ckpt requires a value" >&2
          return 2
        fi
        lead_bev_ckpt="${args[$((i + 1))]}"
        lead_bev_ckpt_explicit=1
        ((i += 1))
        ;;
      --lead-bev-ckpt=*)
        lead_bev_ckpt="${args[$i]#--lead-bev-ckpt=}"
        lead_bev_ckpt_explicit=1
        ;;
      *)
        passthrough+=("${args[$i]}")
        ;;
    esac
  done

  local run_dir
  if [[ -n "$train_resume" ]]; then
    local requested_resume="$train_resume"
    if ! train_resume="$(realpath "$requested_resume")"; then
      echo "cannot resolve --resume checkpoint: $requested_resume" >&2
      return 2
    fi
    run_dir="$(dirname "$train_resume")"
    local resume_config="$run_dir/config.json"
    if [[ ! -s "$resume_config" ]]; then
      echo "missing resume config: $resume_config" >&2
      return 2
    fi
    action_ablation_restore_config_value data_root "$resume_config" data_root "$data_root_explicit"
    action_ablation_restore_config_value data_dir "$resume_config" data_dir "$data_dir_explicit"
    action_ablation_restore_config_value lead_bev_ckpt "$resume_config" lead_bev_ckpt "$lead_bev_ckpt_explicit"
    if [[ "$variant" == "qwen_simple" ]]; then
      action_ablation_restore_config_value model_dir "$resume_config" model_dir "$model_dir_explicit"
    fi
  elif [[ "${NO_RUN_SUBDIR:-}" == "1" ]]; then
    run_dir="$train_output_dir"
  else
    export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
    run_dir="$train_output_dir/run_$RUN_TAG"
  fi

  action_ablation_build_index_if_needed "$data_root" "$data_dir"

  # 续训的采样参数/来源由 config.json 恢复，不自动重选或重建 full map。
  if [[ -z "$train_resume" && "$sampling_mode" == event_balanced ]]; then
    if [[ -z "$event_balance_index" ]]; then
      event_balance_index="$(action_prepare_event_balance_index "$data_root" "$data_dir")"
      passthrough+=(--event-balance-index "$event_balance_index")
    fi
    echo "[event balance index] $event_balance_index"
  fi

  local train_args=(
    --data-root "$data_root"
    --data-dir "$data_dir"
    --output-dir "$train_output_dir"
  )
  if [[ -n "$train_resume" ]]; then
    train_args+=(--resume "$train_resume")
  fi
  if [[ "$variant" == "qwen_simple" && -n "$model_dir" ]]; then
    train_args+=(--model-dir "$model_dir")
  fi
  if [[ -n "$lead_bev_ckpt" ]]; then
    train_args+=(--lead-bev-ckpt "$lead_bev_ckpt")
  fi
  train_args+=("${passthrough[@]}")

  bash "qwen3vl_local/action_expert_ablation/$variant/train.sh" "${train_args[@]}"

  local ckpt="$run_dir/best.pt"
  if [[ ! -s "$ckpt" ]]; then
    echo "missing expected checkpoint: $ckpt" >&2
    return 1
  fi
  local eval_args=(
    --checkpoint "$ckpt"
    --split test
    --data-root "$data_root"
    --data-dir "$data_dir"
  )
  if [[ "$variant" == "qwen_simple" && -n "$model_dir" ]]; then
    eval_args+=(--model-dir "$model_dir")
  fi
  if [[ -n "$lead_bev_ckpt" ]]; then
    eval_args+=(--lead-bev-ckpt "$lead_bev_ckpt")
  fi
  if [[ -n "$event_balance_index" ]]; then
    eval_args+=(--event-balance-index "$event_balance_index")
  fi
  bash "qwen3vl_local/action_expert_ablation/$variant/eval.sh" "${eval_args[@]}"
}
