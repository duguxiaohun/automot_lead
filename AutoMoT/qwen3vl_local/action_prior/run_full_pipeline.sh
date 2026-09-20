#!/usr/bin/env bash
# 优化细节统一默认；每个epoch训练/验证后自动更新run目录的 training_audit.zip，无需审计开关。
# 优化器/LR 共用 action_prior Python 配置：默认 muon_adamw + cosine_restarts。
# 可追加 --optimizer adamw --lr-scheduler cosine 作基线；环境变量 OPTIMIZER/LR_SCHEDULER 同样生效，CLI 优先。
# 续训恢复原配置并严格校验；完整参数和三组对照见 action_prior/OPTIMIZATION.md。
# v13 Phase3 对齐：主要动作或 NONE；自动 v4 动作索引，NONE 保留场景事实 但不追加具体动作。
# 在 AutoMoT/ 下运行；自动准备索引、自动选卡，默认直接图文 KV，不生成摘要。
# 1. 数据集先验 + 均衡采样（去掉 --event-balanced 即自然采样）：
#   bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced
#
# 2. 再加 一句 Phase3 所选动作及场景原因（离线动作真值）（自动标注，无需动作索引）：
#   bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --high-level-action-prior
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --high-level-action-prior
#   去掉 --high-level-action-prior 保留原自然 RS/EVENT；动作开关默认关闭。
#
# 3. 续训（自动恢复原配置和开关）：
#   bash qwen3vl_local/action_prior/run_full_pipeline.sh --resume checkpoints/action_prior/latest/latest.pt
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --resume checkpoints/action_prior/latest/latest.pt
#
# 可选追加：--generate-analysis 生成摘要；--prior-noise 0.1 注入先验噪声。
# 更多开关/环境变量见 run.md，审计见 AUDIT.md；动作真值模式暂不支持闭环。
# 干净的 dataset-priors + high-level-action-prior 自动提供已确认特殊 RE；无需额外场景开关。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
scene_priors="$(python "$HERE/scene_policy.py" "$@")"
source "$HERE/event_balance_common.sh"
action_event_balance_options "$@"
set -- "${ACTION_EVENT_BALANCE_ARGS[@]}"
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
# 只记录用户显式给出的路径，续训不能把稍后生成的脚本默认值当作覆盖。
declare -A pipeline_paths=()
for name in DATA_ROOT DATA_DIR MODEL_DIR LEAD_BEV_CKPT HIGH_LEVEL_ACTION_INDEX; do
 if [[ -v "$name" ]]; then pipeline_paths["$name"]="${!name}"; fi
done
ARGS=()
sampling_mode=uniform
if [[ "${EVENT_BALANCED:-0}" == 1 ]]; then sampling_mode=event_balanced; fi
action_priors="${HIGH_LEVEL_ACTION_PRIOR:-0}"
sampling_explicit=0
explicit_prior_source="$DATASET_PRIORS_ENV_SET"
# 消费 pipeline 负责的选项，其余原样传给训练；CLI 优先于环境变量。
while (( $# )); do
 case "$1" in
  --dataset-priors) DATASET_PRIORS=1; explicit_prior_source=1 ;;
  --no-dataset-priors) DATASET_PRIORS=0; explicit_prior_source=1 ;;
  --event-balanced) sampling_mode=event_balanced; sampling_explicit=1 ;;
  --no-event-balanced) sampling_mode=uniform; sampling_explicit=1 ;;
  --resume|--sampling-mode|--data-dir|--data-root|--model-dir|--lead-bev-ckpt|--prior-labels|--event-balance-index|--high-level-action-index)
   flag="$1"
   [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "$flag needs a value" >&2; exit 2; }
   value="$2"; shift
   case "$flag" in
    --resume) RESUME="$value" ;;
    --sampling-mode) sampling_mode="$value"; sampling_explicit=1 ;;
    --data-dir) DATA_DIR="$value"; pipeline_paths[DATA_DIR]="$value" ;;
    --data-root) DATA_ROOT="$value"; pipeline_paths[DATA_ROOT]="$value" ;;
    --model-dir) pipeline_paths[MODEL_DIR]="$value" ;;
    --lead-bev-ckpt) pipeline_paths[LEAD_BEV_CKPT]="$value" ;;
    --high-level-action-index) pipeline_paths[HIGH_LEVEL_ACTION_INDEX]="$value" ;;
    --prior-labels) PRIOR_LABELS="$value"; explicit_labels=1 ;;
    --event-balance-index) EVENT_BALANCE_INDEX="$value"; explicit_event_balance_index=1 ;;
   esac ;;
  --resume=*)
   RESUME="${1#*=}"
   [[ -n "$RESUME" ]] || { echo "--resume needs a value" >&2; exit 2; } ;;
  --sampling-mode=*) sampling_mode="${1#*=}"; sampling_explicit=1 ;;
  --data-dir=*) DATA_DIR="${1#*=}"; pipeline_paths[DATA_DIR]="$DATA_DIR" ;;
  --data-root=*) DATA_ROOT="${1#*=}"; pipeline_paths[DATA_ROOT]="$DATA_ROOT" ;;
  --model-dir=*) pipeline_paths[MODEL_DIR]="${1#*=}" ;;
  --lead-bev-ckpt=*) pipeline_paths[LEAD_BEV_CKPT]="${1#*=}" ;;
  --high-level-action-index=*) pipeline_paths[HIGH_LEVEL_ACTION_INDEX]="${1#*=}" ;;
  --prior-labels=*) PRIOR_LABELS="${1#*=}"; explicit_labels=1 ;;
  --event-balance-index=*) EVENT_BALANCE_INDEX="${1#*=}"; explicit_event_balance_index=1 ;;
  --high-level-action-prior) action_priors=1; ARGS+=("$1") ;;
  --no-high-level-action-prior) action_priors=0; ARGS+=("$1") ;;
  *) ARGS+=("$1") ;;
 esac
 shift
done
PATH_ARGS=()
for name in DATA_ROOT DATA_DIR MODEL_DIR LEAD_BEV_CKPT HIGH_LEVEL_ACTION_INDEX; do
 if [[ -v "pipeline_paths[$name]" ]]; then
  option="${name,,}"
  PATH_ARGS+=("--${option//_/-}" "${pipeline_paths[$name]}")
 fi
done
ARGS+=("${PATH_ARGS[@]}")
# CLI 与环境变量共用同一续训分支；提前固定真实路径，防止 latest 在训练期间被改指。
if [[ -n "${RESUME:-}" ]]; then
 RESUME="$(realpath -e -- "$RESUME")"
 [[ -f "$RESUME" ]] || { echo "resume checkpoint is not a file: $RESUME" >&2; exit 2; }
 export RESUME
fi
[[ "$sampling_mode" == uniform || "$sampling_mode" == event_balanced ]] || { echo "invalid sampling mode: $sampling_mode" >&2; exit 2; }
[[ "$sampling_explicit" == 0 ]] || ARGS+=(--sampling-mode "$sampling_mode")
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
prepare_event_inputs() {
 # 所有自动生成均在模型预检前完成；显式索引保留原内容并由训练预检校验。
 if [[ "$sampling_mode" == event_balanced || "$scene_priors" == 1 || "$action_priors" == 1 ]]; then
  action_build_shared_index_if_needed "$DATA_ROOT" "$DATA_DIR"
  if [[ -z "$EVENT_BALANCE_INDEX" ]]; then
   EVENT_BALANCE_INDEX="$(action_prepare_event_balance_index "$DATA_ROOT" "$DATA_DIR")"
  fi
  export EVENT_BALANCE_INDEX
  echo "[event balance index] $EVENT_BALANCE_INDEX"
 fi
}
if [[ -n "${RESUME:-}" ]]; then
 # 标签索引搬家后旧 config.json 的路径已失效；显式路径必须继续传给续训入口。
 RESUME_ARGS=("${ARGS[@]+"${ARGS[@]}"}")
 if [[ "$explicit_prior_source" != 0 ]]; then
  if [[ "$DATASET_PRIORS" == 1 ]]; then
   RESUME_ARGS+=(--dataset-priors)
  else
   RESUME_ARGS+=(--no-dataset-priors)
  fi
 fi
 if [[ "$DATASET_PRIORS" == 1 && "$explicit_labels" != 0 ]]; then
  RESUME_ARGS+=(--prior-labels "$PRIOR_LABELS")
 fi
 if [[ "$explicit_event_balance_index" != 0 ]]; then
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
  if [[ ! -f checkpoints/sft_new_loop_phase1_data/frame_index.jsonl ]]; then
   python "$HERE/../sft_new_loop_phase1/build_dataset.py" --data-root "$DATA_ROOT" --output-dir checkpoints/sft_new_loop_phase1_data
  fi
  python "$HERE/build_prior_labels.py" --output-dir "$(dirname -- "$PRIOR_LABELS")"
 fi
 prepare_event_inputs
 ACTION_MODE=preflight bash "$HERE/train.sh" --models-only "${ARGS[@]+"${ARGS[@]}"}"
 action_build_shared_index_if_needed "$DATA_ROOT" "$DATA_DIR"
 bash "$HERE/train.sh" "${ARGS[@]+"${ARGS[@]}"}"
else
 # 先核验权重和 prompt 合同，缺权重时不先构建全量索引。
 SELECTION_FILE="$OUTPUT_DIR/selection_${RUN_TAG}.json"
 prepare_event_inputs
 ACTION_MODE=preflight bash "$HERE/train.sh" --models-only --selection-output "$SELECTION_FILE" "${ARGS[@]+"${ARGS[@]}"}"
 action_build_shared_index_if_needed "$DATA_ROOT" "$DATA_DIR"
 # 数据构建期间即使上游产生新 best，也必须继续使用本次预检已展示的权重。
 bash "$HERE/train.sh" "${ARGS[@]+"${ARGS[@]}"}" --selection-manifest "$SELECTION_FILE"
fi
RUN_DIR="$OUTPUT_DIR/run_$RUN_TAG"
[[ "${NO_RUN_SUBDIR:-0}" != 1 ]] || RUN_DIR="$OUTPUT_DIR"
[[ -z "${RESUME:-}" ]] || RUN_DIR="$(dirname -- "$RESUME")"
# 轨迹 head 的最优点按验证采样 ADE（可显式改为事件均衡 ADE），和上游 LoRA 的 best_generation 区分。
test -f "$RUN_DIR/best.pt"
# eval/probe 默认沿用 checkpoint 自己记录的先验来源；显式标签搬迁必须贯穿旧 best.pt。
EVAL_ARGS=("${PATH_ARGS[@]}")
if [[ "$DATASET_PRIORS" == 1 && "$explicit_labels" != 0 ]]; then
 EVAL_ARGS+=(--prior-labels "$PRIOR_LABELS")
fi
if [[ -n "$EVENT_BALANCE_INDEX" ]]; then
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
