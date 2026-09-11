#!/usr/bin/env bash
# 推荐自动准备数据并训练：
#   bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced
# 本脚本是已准备好索引时的底层训练入口，不负责自动构建。日常操作见 run.md。
#   bash qwen3vl_local/tb_serve.sh checkpoints/action_prior/latest/tb
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
export PYTHONUNBUFFERED=1
# 参数用数组传递，路径包含空格时也不会被拆开。
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
has_flag() { local flag="$1"; shift; [[ " $* " == *" $flag "* || " $* " == *" $flag="* ]]; }
has_value() { local flag="$1" value="$2"; shift 2; [[ " $* " == *" $flag $value "* || " $* " == *" $flag=$value "* ]]; }
args=(--data-root "${DATA_ROOT:-lead_data}" --data-dir "${DATA_DIR:-checkpoints/action_prior_data}"
 --checkpoint-root "${CHECKPOINT_ROOT:-checkpoints}" --selection-policy "${SELECTION_POLICY:-available}"
 --model-dir "${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
 --lead-bev-ckpt "${LEAD_BEV_CKPT:-checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth}"
 --num-epochs "${NUM_EPOCHS:-61}" --learning-rate "${LR:-0.0002}"
 --grad-accum-steps "${GRAD_ACCUM:-16}" --val-steps "${VAL_STEPS:-250}"
 --save-steps "${SAVE_STEPS:-1000}" --num-workers "${NUM_WORKERS:-8}")
args+=(--logging-steps "${LOGGING_STEPS:-10}")
# EVENT_BALANCED=1 或 --sampling-mode event_balanced：按 action_prior 全帧映射的
# UE1-7/RE2/RE3/RE5 十桶各一份、确认常规背景两份重建每个 epoch。只影响训练抽样；
# EVENT_BALANCED_SCENE_PRIORS=1 是另一个 dataset-only 的离线自然文本条件，闭环禁用。
if [[ "${EVENT_BALANCED:-0}" == 1 ]] && ! has_flag --sampling-mode "$@"; then
 args+=(--sampling-mode event_balanced)
fi
# 采样课程与离线 scene prior 都读取同一份 full map。uniform + scene-priors
# 同样必须传索引，不能只因未启用重采样而遗漏 EVENT_BALANCE_INDEX。
scene_priors_requested=0
if [[ "${EVENT_BALANCED_SCENE_PRIORS:-0}" == 1 ]] || has_flag --event-balanced-scene-priors "$@"; then
 scene_priors_requested=1
fi
if has_flag --no-event-balanced-scene-priors "$@"; then scene_priors_requested=0; fi
if has_value --sampling-mode event_balanced "$@" || { [[ "${EVENT_BALANCED:-0}" == 1 ]] && ! has_flag --sampling-mode "$@"; } || [[ "$scene_priors_requested" == 1 ]]; then
 if ! has_flag --event-balance-index "$@"; then
  : "${EVENT_BALANCE_INDEX:?set EVENT_BALANCE_INDEX to action_prior full_event_mapping.jsonl}"
  args+=(--event-balance-index "$EVENT_BALANCE_INDEX")
 fi
fi
if [[ "$scene_priors_requested" == 1 ]] && ! has_flag --event-balanced-scene-priors "$@"; then
 args+=(--event-balanced-scene-priors)
fi
if ! has_flag --event-balanced-epoch-samples "$@"; then
 args+=(--event-balanced-epoch-samples "${EVENT_BALANCED_EPOCH_SAMPLES:-0}")
fi
if ! has_flag --event-balance-max-frame-repeats "$@"; then
 args+=(--event-balance-max-frame-repeats "${EVENT_BALANCE_MAX_FRAME_REPEATS:-8}")
fi
if ! has_flag --best-selection-metric "$@"; then
 args+=(--best-selection-metric "${BEST_SELECTION_METRIC:-natural_ade}")
fi
# v5 条件 Flow Matching：10 步 Euler 是默认起点；坐标缩放/时间编码均写入 checkpoint 合同。
args+=(--flow-sample-steps "${FLOW_SAMPLE_STEPS:-10}"
 --flow-route-coordinate-scale-m "${FLOW_ROUTE_COORDINATE_SCALE_M:-30}"
 --flow-waypoint-coordinate-scale-m "${FLOW_WAYPOINT_COORDINATE_SCALE_M:-20}"
 --flow-time-embed-dim "${FLOW_TIME_EMBED_DIM:-64}"
 --flow-trajectory-layers "${FLOW_TRAJECTORY_LAYERS:-2}"
 --flow-trajectory-heads "${FLOW_TRAJECTORY_HEADS:-8}")
if [[ "${TRAIN_SAMPLED_METRICS:-0}" == 1 ]]; then
 args+=(--train-sampled-metrics)
fi
# 环境变量与命令行开关都要参与判断，否则直接调用 train.sh --dataset-priors 会退回 LoRA 默认。
dataset="${DATASET_PRIORS:-0}"
if has_flag --dataset-priors "$@"; then dataset=1; fi
if has_flag --no-dataset-priors "$@"; then dataset=0; fi
if [[ "$dataset" == 1 ]]; then
 has_flag --dataset-priors "$@" || args+=(--dataset-priors)
 has_flag --prior-labels "$@" ||
  args+=(--prior-labels "${PRIOR_LABELS:-checkpoints/action_prior_labels/prior_labels.jsonl}")
 has_flag --prior-noise "$@" || args+=(--prior-noise "${PRIOR_NOISE:-0}")
 has_flag --prior-noise-invalid-share "$@" ||
  args+=(--prior-noise-invalid-share "${PRIOR_NOISE_INVALID_SHARE:-0.25}")
fi
# 标定真值不需要再让 base 复核自己是否与先验矛盾；默认省掉第二次生成。
if ! has_flag --analysis-review "$@" && ! has_flag --no-analysis-review "$@"; then
 review="${ANALYSIS_REVIEW:-$([[ "$dataset" == 1 ]] && echo 0 || echo 1)}"
 [[ "$review" == 1 ]] && args+=(--analysis-review) || args+=(--no-analysis-review)
fi
[[ -z "${PHASE1_ADAPTER:-}" ]] || args+=(--phase1-adapter "$PHASE1_ADAPTER")
[[ -z "${PHASE2_ADAPTER:-}" ]] || args+=(--phase2-adapter "$PHASE2_ADAPTER")
exec python "$HERE/launch.py" "${ACTION_MODE:-train}" "${args[@]}" "$@"
