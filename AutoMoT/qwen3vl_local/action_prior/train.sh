#!/usr/bin/env bash
# 在 AutoMoT/ 下直接复制执行：
#   bash qwen3vl_local/action_prior/train.sh
#   python qwen3vl_local/action_prior/build_prior_labels.py
#   DATASET_PRIORS=1 bash qwen3vl_local/action_prior/train.sh
#   DATASET_PRIORS=1 PRIOR_NOISE=0.1 bash qwen3vl_local/action_prior/train.sh
#   DATASET_PRIORS=1 PRIOR_LABELS=/自定义/prior_labels.jsonl bash qwen3vl_local/action_prior/train.sh
# DATASET_PRIORS=1 时不加载 LoRA，用 $PRIOR_LABELS 的标定真值，且默认关闭独立复核，
# base 每帧只生成一次；ANALYSIS_REVIEW=1 可要求保留复核（每帧两次生成）。
# PRIOR_NOISE=0.1：10% 的帧按审计错误方向把 RS 或 EVENT 先验改成错误值/invalid。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
export PYTHONUNBUFFERED=1
# 参数用数组传递，路径包含空格时也不会被拆开。
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
args=(--data-root "${DATA_ROOT:-lead_data}" --data-dir "${DATA_DIR:-checkpoints/action_prior_data}"
 --checkpoint-root "${CHECKPOINT_ROOT:-checkpoints}" --selection-policy "${SELECTION_POLICY:-available}"
 --model-dir "${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
 --lead-bev-ckpt "${LEAD_BEV_CKPT:-checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth}"
 --num-epochs "${NUM_EPOCHS:-61}" --learning-rate "${LR:-0.0002}"
 --grad-accum-steps "${GRAD_ACCUM:-16}" --val-steps "${VAL_STEPS:-250}"
 --save-steps "${SAVE_STEPS:-1000}" --num-workers "${NUM_WORKERS:-8}")
args+=(--logging-steps "${LOGGING_STEPS:-10}")
has_flag() { local flag="$1"; shift; [[ " $* " == *" $flag "* || " $* " == *" $flag="* ]]; }
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
