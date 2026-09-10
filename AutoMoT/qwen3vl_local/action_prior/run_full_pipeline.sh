#!/usr/bin/env bash
# 在 AutoMoT/ 下直接复制执行：
#   bash qwen3vl_local/action_prior/run_full_pipeline.sh
#   bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors
#   PRIOR_NOISE=0.1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors
#   GPU_IDS=0,1,2,3 PRIOR_NOISE=0.1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors
#   bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --prior-labels /自定义/prior_labels.jsonl
# 不传开关：Phase1/Phase2 LoRA 逐帧问答作先验（冷启动 11 次生成）。
# --dataset-priors：不加载 LoRA，用数据集标定真值，base 每帧只生成一次（默认不再复核）；
# 只在使用默认路径且文件缺失时自动生成 checkpoints/action_prior_labels/prior_labels.jsonl。
# PRIOR_NOISE=0.1：10% 的帧按审计错误方向把 RS 或 EVENT 先验改成错误值/invalid。
# UE/特殊 RE 均衡采样（先按 run.md 构建 action index 和 v2 full map）：
#   DATA_DIR=checkpoints/action_prior_data_event_v1 EVENT_BALANCED=1 EVENT_BALANCE_INDEX=checkpoints/action_prior_event_balance_v2/full_event_mapping.jsonl bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors
#   GPU_IDS=0,1,2,3 DATA_DIR=checkpoints/action_prior_data_event_v1 EVENT_BALANCED=1 EVENT_BALANCE_INDEX=checkpoints/action_prior_event_balance_v2/full_event_mapping.jsonl bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors
# 关闭采样开关：上面的命令追加 --sampling-mode uniform；恢复自然分布，保留 dataset-priors 选择。
# 重复上限 EVENT_BALANCE_MAX_FRAME_REPEATS=8；自动 epoch 预算 EVENT_BALANCED_EPOCH_SAMPLES=0。
# 可选 BEST_SELECTION_METRIC=event_balanced_ade 要求 val 全桶覆盖；默认 natural_ade。
# 可选 EVENT_BALANCED_SCENE_PRIORS=1 只用于 dataset-priors + PRIOR_NOISE=0 的离线条件实验，闭环禁用。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 开关会写进 checkpoint 的 args 与合同身份；后续 eval/probe 默认按模型自己的记录跑。
DATASET_PRIORS_ENV_SET=0
[[ -z "${DATASET_PRIORS+x}" ]] || DATASET_PRIORS_ENV_SET=1
DATASET_PRIORS="${DATASET_PRIORS:-0}"
DEFAULT_PRIOR_LABELS=checkpoints/action_prior_labels/prior_labels.jsonl
explicit_labels=0
[[ -z "${PRIOR_LABELS+x}" ]] || explicit_labels=1
PRIOR_LABELS="${PRIOR_LABELS:-$DEFAULT_PRIOR_LABELS}"
explicit_event_balance_index=0
[[ -z "${EVENT_BALANCE_INDEX+x}" ]] || explicit_event_balance_index=1
EVENT_BALANCE_INDEX="${EVENT_BALANCE_INDEX:-}"
ARGS=()
want_labels=0
want_event_balance_index=0
event_balance_index_in_args=0
explicit_prior_source="$DATASET_PRIORS_ENV_SET"
for item in "$@"; do
 if [[ "$want_labels" == 1 ]]; then
  PRIOR_LABELS="$item"
  explicit_labels=1
  want_labels=0
  continue
 fi
 if [[ "$want_event_balance_index" == 1 ]]; then
  EVENT_BALANCE_INDEX="$item"
  explicit_event_balance_index=1
  ARGS+=(--event-balance-index "$item")
  event_balance_index_in_args=1
  want_event_balance_index=0
  continue
 fi
 case "$item" in
  --dataset-priors) DATASET_PRIORS=1; explicit_prior_source=1 ;;
  --no-dataset-priors) DATASET_PRIORS=0; explicit_prior_source=1 ;;
  # 自定义标签索引由本脚本统一转交 train.sh，避免出现两个 --prior-labels。
  --prior-labels) want_labels=1 ;;
  --prior-labels=*) PRIOR_LABELS="${item#*=}"; explicit_labels=1 ;;
  --event-balance-index) want_event_balance_index=1 ;;
  --event-balance-index=*) EVENT_BALANCE_INDEX="${item#*=}"; explicit_event_balance_index=1; event_balance_index_in_args=1; ARGS+=("$item") ;;
  *) ARGS+=("$item") ;;
 esac
done
[[ "$want_labels" == 0 ]] || { echo "--prior-labels needs a path" >&2; exit 2; }
[[ "$want_event_balance_index" == 0 ]] || { echo "--event-balance-index needs a path" >&2; exit 2; }
if [[ -n "${RESUME:-}" && "$explicit_prior_source" == 0 ]]; then
 RESUME_CONFIG="$(dirname -- "$RESUME")/config.json"
 if [[ -f "$RESUME_CONFIG" ]]; then
  DATASET_PRIORS="$(python - "$RESUME_CONFIG" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print("1" if json.load(handle).get("dataset_priors", False) else "0")
PY
)"
 fi
fi
export DATASET_PRIORS PRIOR_LABELS
# run tag 只计算一次，数据、训练、测试都绑定本次 run。
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/action_prior}"
export DATA_DIR="${DATA_DIR:-checkpoints/action_prior_data/run_${RUN_TAG}}"
export DATA_ROOT="${DATA_ROOT:-lead_data}"
# 预检在 run 创建前执行，完整日志先放根目录 logs；launcher 在 run 内建立入口链接。
export PYTHONUNBUFFERED=1
PIPELINE_LOG="${PIPELINE_LOG:-$OUTPUT_DIR/logs/pipeline_$RUN_TAG.log}"
mkdir -p -- "$(dirname -- "$PIPELINE_LOG")"
export ACTION_PIPELINE_LOG="$PIPELINE_LOG"
exec > >(tee -a "$PIPELINE_LOG") 2>&1
echo "[pipeline log] $PIPELINE_LOG"
if [[ "$DATASET_PRIORS" == 1 ]]; then
 echo "[prior source] dataset labels: $PRIOR_LABELS (no Phase1/Phase2 LoRA is loaded, injected noise rate ${PRIOR_NOISE:-0})"
else
 echo "[prior source] phase1/phase2 LoRA inference"
fi
if [[ -n "${RESUME:-}" ]]; then
 # 标签索引搬家后旧 config.json 的路径已失效；显式路径必须继续传给续训入口。
 RESUME_ARGS=("${ARGS[@]+"${ARGS[@]}"}")
 if [[ "$DATASET_PRIORS" == 1 && "$explicit_labels" != 0 ]]; then
  RESUME_ARGS+=(--prior-labels "$PRIOR_LABELS")
 fi
 if [[ "$explicit_event_balance_index" != 0 && "$event_balance_index_in_args" == 0 ]]; then
  RESUME_ARGS+=(--event-balance-index "$EVENT_BALANCE_INDEX")
 fi
 bash "$HERE/resume.sh" "$RESUME" "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}"
elif [[ "$DATASET_PRIORS" == 1 ]]; then
 # 标定真值逐帧命中，不选择也不固定任何 LoRA；预检只核验 base、BEV 与标签索引。
 if [[ ! -f "$PRIOR_LABELS" ]]; then
  if [[ "$PRIOR_LABELS" != "$DEFAULT_PRIOR_LABELS" ]]; then
   echo "[prior labels missing] $PRIOR_LABELS: build it explicitly with build_prior_labels.py --output-dir $(dirname -- "$PRIOR_LABELS")" >&2
   exit 2
  fi
  python "$HERE/build_prior_labels.py" --output-dir "$(dirname -- "$PRIOR_LABELS")"
 fi
 ACTION_MODE=preflight bash "$HERE/train.sh" --models-only "${ARGS[@]+"${ARGS[@]}"}"
 if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
  python "$HERE/build_dataset.py" --data-root "$DATA_ROOT" --output-dir "$DATA_DIR"
 fi
 bash "$HERE/train.sh" "${ARGS[@]+"${ARGS[@]}"}"
else
 # 先核验权重和 prompt 合同，缺权重时不先构建全量索引。
 SELECTION_FILE="$OUTPUT_DIR/selection_${RUN_TAG}.json"
 ACTION_MODE=preflight bash "$HERE/train.sh" --models-only --selection-output "$SELECTION_FILE" "${ARGS[@]+"${ARGS[@]}"}"
 if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
  python "$HERE/build_dataset.py" --data-root "$DATA_ROOT" --output-dir "$DATA_DIR"
 fi
 # 数据构建期间即使上游产生新 best，也必须继续使用本次预检已展示的权重。
 bash "$HERE/train.sh" "${ARGS[@]+"${ARGS[@]}"}" --selection-manifest "$SELECTION_FILE"
fi
RUN_DIR="$OUTPUT_DIR/run_$RUN_TAG"
[[ "${NO_RUN_SUBDIR:-0}" != 1 ]] || RUN_DIR="$OUTPUT_DIR"
[[ -z "${RESUME:-}" ]] || RUN_DIR="$(dirname -- "$RESUME")"
# 轨迹 head 的最优点按验证采样 ADE（可显式改为事件均衡 ADE），和上游 LoRA 的 best_generation 区分。
test -f "$RUN_DIR/best.pt"
# eval/probe 默认沿用 checkpoint 自己记录的先验来源；显式标签搬迁必须贯穿旧 best.pt。
EVAL_ARGS=()
if [[ "$DATASET_PRIORS" == 1 && "$explicit_labels" != 0 ]]; then
 EVAL_ARGS+=(--prior-labels "$PRIOR_LABELS")
fi
if [[ "$explicit_event_balance_index" != 0 ]]; then
 EVAL_ARGS+=(--event-balance-index "$EVENT_BALANCE_INDEX")
fi
bash "$HERE/eval.sh" --checkpoint "$RUN_DIR/best.pt" --split test --output-dir "$RUN_DIR/test" "${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}"
bash "$HERE/probe.sh" --checkpoint "$RUN_DIR/best.pt" --split test --output-dir "$RUN_DIR/probe" "${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}"

# 训练/验证历史 + 最终离线 test/probe 的轻量包；权重/文本缓存/视频不入包。
python "$HERE/audit_bundle.py" --root "$RUN_DIR"
# 正式 220 路线只用于最终报告；CARLA 环境上显式开启，不能替代训练期 val。
if [[ "${BENCH2DRIVE:-0}" == 1 ]]; then
 if [[ "$DATASET_PRIORS" == 1 && "${ACTION_DATASET_PRIORS:-}" != 0 ]]; then
  # 闭环帧没有数据集标签；要跑必须显式 ACTION_DATASET_PRIORS=0 并承认条件迁移。
  echo "[bench2drive skipped] dataset-prior checkpoint has no closed-loop labels; rerun with ACTION_DATASET_PRIORS=0" >&2
 else
  B2D_ARGS=()
  if [[ -f "$RUN_DIR/bench2drive/run_manifest.json" ]]; then
   B2D_ARGS+=(--resume)
  fi
  bash "$HERE/eval.sh" --bench2drive --checkpoint "$RUN_DIR/best.pt" \
   --output-dir "$RUN_DIR/bench2drive" --num-gpus "${EVAL_GPU_COUNT:-1}" "${B2D_ARGS[@]+"${B2D_ARGS[@]}"}"
 fi
fi
