#!/usr/bin/env bash
# 在 AutoMoT/ 下执行；需已有训练索引，自动准备请用 run_full_pipeline.sh。
# 默认 event 均衡：
#   bash qwen3vl_local/action_prior/train.sh --dataset-priors
# 单当前图（默认 event 均衡，token 默认关闭；省略图数参数时默认四图）：
#   bash qwen3vl_local/action_prior/train.sh --dataset-priors --rgb-frame-count 1
# 单当前图 + action 均衡（token 默认关闭）：
#   bash qwen3vl_local/action_prior/train.sh --dataset-priors --action-balanced --rgb-frame-count 1
# action 均衡 + token + 单当前图：
#   bash qwen3vl_local/action_prior/train.sh --dataset-priors --action-balanced --high-level-action-token --rgb-frame-count 1
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/train.sh --dataset-priors --action-balanced --high-level-action-token --rgb-frame-count 1
# 续训：
#   bash qwen3vl_local/action_prior/train.sh --resume checkpoints/action_prior/latest/latest.pt
# 两种采样的完整 demo 见 run_full_pipeline.sh；参数说明见 run.md。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
export PYTHONUNBUFFERED=1
# 参数用数组传递，路径包含空格时也不会被拆开。
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "$HERE/scene_policy.py" "$@" >/dev/null
source "$HERE/event_balance_common.sh"
action_event_balance_options "$@"
set -- "${ACTION_EVENT_BALANCE_ARGS[@]}"
has_flag() { local flag="$1"; shift; [[ " $* " == *" $flag "* || " $* " == *" $flag="* ]]; }
# 在拼接任何新训练默认值之前分流；保留数组边界，兼容带空格的路径与 CLI > RESUME。
resume_checkpoint="${RESUME:-}"
train_cli=()
while (( $# )); do
 case "$1" in
  --resume)
   [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "--resume needs a value" >&2; exit 2; }
   resume_checkpoint="$2"; shift ;;
  --resume=*)
   resume_checkpoint="${1#*=}"
   [[ -n "$resume_checkpoint" ]] || { echo "--resume needs a value" >&2; exit 2; } ;;
  *) train_cli+=("$1") ;;
 esac
 shift
done
set -- "${train_cli[@]}"
if [[ -n "$resume_checkpoint" ]]; then
 resume_checkpoint="$(realpath -e -- "$resume_checkpoint")"
 [[ -f "$resume_checkpoint" ]] || { echo "resume checkpoint is not a file: $resume_checkpoint" >&2; exit 2; }
 # 只传用户显式设置的环境值；未设置时全部由 resume.py 从原 run 恢复。
 # CLI 放最后，继续覆盖环境变量；摘要开关环境值由共用 resume.py 处理。
 resume_overrides=()
 for mapping in DATA_ROOT:data-root DATA_DIR:data-dir MODEL_DIR:model-dir LEAD_BEV_CKPT:lead-bev-ckpt \
  CHECKPOINT_ROOT:checkpoint-root SELECTION_POLICY:selection-policy NUM_EPOCHS:num-epochs \
  LR:learning-rate GRAD_ACCUM:grad-accum-steps VAL_STEPS:val-steps SAVE_STEPS:save-steps \
  NUM_WORKERS:num-workers LOGGING_STEPS:logging-steps PRIOR_LABELS:prior-labels PRIOR_NOISE:prior-noise \
  PRIOR_NOISE_INVALID_SHARE:prior-noise-invalid-share PHASE1_ADAPTER:phase1-adapter PHASE2_ADAPTER:phase2-adapter \
  EVENT_BALANCE_INDEX:event-balance-index FLOW_SAMPLE_STEPS:flow-sample-steps \
  FLOW_ROUTE_COORDINATE_SCALE_M:flow-route-coordinate-scale-m \
  FLOW_WAYPOINT_COORDINATE_SCALE_M:flow-waypoint-coordinate-scale-m FLOW_TIME_EMBED_DIM:flow-time-embed-dim \
  FLOW_TRAJECTORY_LAYERS:flow-trajectory-layers FLOW_TRAJECTORY_HEADS:flow-trajectory-heads; do
  name="${mapping%%:*}"; option="${mapping#*:}"
  if [[ -v "$name" ]]; then resume_overrides+=("--$option" "${!name}"); fi
 done
 for mapping in DATASET_PRIORS:dataset-priors ANALYSIS_REVIEW:analysis-review \
  TRAIN_SAMPLED_METRICS:train-sampled-metrics; do
  name="${mapping%%:*}"; option="${mapping#*:}"
  if [[ -v "$name" ]]; then
   case "${!name}" in
    1) resume_overrides+=("--$option") ;;
    0) resume_overrides+=("--no-$option") ;;
    *) echo "$name must be 0 or 1" >&2; exit 2 ;;
   esac
  fi
 done
 exec bash "$HERE/resume.sh" "$resume_checkpoint" "${resume_overrides[@]}" "$@"
fi
args=(--data-root "${DATA_ROOT:-lead_data}" --data-dir "${DATA_DIR:-checkpoints/action_prior_data}"
 --checkpoint-root "${CHECKPOINT_ROOT:-checkpoints}" --selection-policy "${SELECTION_POLICY:-available}"
 --model-dir "${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
 --lead-bev-ckpt "${LEAD_BEV_CKPT:-checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth}"
 --num-epochs "${NUM_EPOCHS:-7}" --learning-rate "${LR:-0.0002}"
 --grad-accum-steps "${GRAD_ACCUM:-16}" --val-steps "${VAL_STEPS:-250}"
 --save-steps "${SAVE_STEPS:-1000}" --num-workers "${NUM_WORKERS:-8}")
args+=(--logging-steps "${LOGGING_STEPS:-10}")
# 具体动作默认自动复用/生成 Phase3 标注；默认关闭，只追加所选动作的逐场景因果句。
if ! has_flag --high-level-action-prior "$@" && ! has_flag --no-high-level-action-prior "$@"; then
 case "${HIGH_LEVEL_ACTION_PRIOR:-0}" in
  1) args+=(--high-level-action-prior) ;;
  0) args+=(--no-high-level-action-prior) ;;
  *) echo "HIGH_LEVEL_ACTION_PRIOR must be 0 or 1" >&2; exit 2 ;;
 esac
fi
if [[ -n "${HIGH_LEVEL_ACTION_INDEX:-}" ]] && ! has_flag --high-level-action-index "$@"; then
 args+=(--high-level-action-index "$HIGH_LEVEL_ACTION_INDEX")
fi
# 默认一次图文 prefill；CLI 明确开关优先于环境变量，摘要复核不能隐式开启生成。
if ! has_flag --generate-analysis "$@" && ! has_flag --no-generate-analysis "$@"; then
 case "${GENERATE_ANALYSIS:-0}" in
  1) args+=(--generate-analysis) ;;
  0) args+=(--no-generate-analysis) ;;
  *) echo "GENERATE_ANALYSIS must be 0 or 1" >&2; exit 2 ;;
 esac
fi
# 均衡采样只改抽样；特殊 RE 场景由共用策略自动决定。
if ! has_flag --event-balance-index "$@" && [[ -n "${EVENT_BALANCE_INDEX:-}" ]]; then
 args+=(--event-balance-index "$EVENT_BALANCE_INDEX")
fi
# 两种均衡模式都允许 Python 在模型预检前准备 full map；续训已在上方分流。
# 采样默认值统一由 config.DEFAULTS 提供，显式环境变量已由共享 helper 转成 CLI。
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
# 仅 generate-analysis 开启时该复核设置才生效；标定真值默认省掉复核生成。
if ! has_flag --analysis-review "$@" && ! has_flag --no-analysis-review "$@"; then
 review="${ANALYSIS_REVIEW:-$([[ "$dataset" == 1 ]] && echo 0 || echo 1)}"
 [[ "$review" == 1 ]] && args+=(--analysis-review) || args+=(--no-analysis-review)
fi
[[ -z "${PHASE1_ADAPTER:-}" ]] || args+=(--phase1-adapter "$PHASE1_ADAPTER")
[[ -z "${PHASE2_ADAPTER:-}" ]] || args+=(--phase2-adapter "$PHASE2_ADAPTER")
exec python "$HERE/launch.py" "${ACTION_MODE:-train}" "${args[@]}" "$@"
