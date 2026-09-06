#!/usr/bin/env bash
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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
if [[ -n "${RESUME:-}" ]]; then
 bash "$HERE/resume.sh" "$RESUME" "$@"
else
 # 先核验权重和 prompt 合同，缺权重时不先构建全量索引。
 SELECTION_FILE="$OUTPUT_DIR/selection_${RUN_TAG}.json"
 ACTION_MODE=preflight bash "$HERE/train.sh" --models-only --selection-output "$SELECTION_FILE" "$@"
 if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
  python "$HERE/build_dataset.py" --data-root "$DATA_ROOT" --output-dir "$DATA_DIR"
 fi
 # 数据构建期间即使上游产生新 best，也必须继续使用本次预检已展示的权重。
 bash "$HERE/train.sh" "$@" --selection-manifest "$SELECTION_FILE"
fi
RUN_DIR="$OUTPUT_DIR/run_$RUN_TAG"
[[ "${NO_RUN_SUBDIR:-0}" != 1 ]] || RUN_DIR="$OUTPUT_DIR"
[[ -z "${RESUME:-}" ]] || RUN_DIR="$(dirname -- "$RESUME")"
# 轨迹 head 的最优点按验证 loss，和上游 LoRA 的 best_generation 区分。
test -f "$RUN_DIR/best.pt"
bash "$HERE/eval.sh" --checkpoint "$RUN_DIR/best.pt" --split test --output-dir "$RUN_DIR/test"
bash "$HERE/probe.sh" --checkpoint "$RUN_DIR/best.pt" --split test --output-dir "$RUN_DIR/probe"

# 训练/验证历史 + 最终离线 test/probe 的轻量包；权重/文本缓存/视频不入包。
python "$HERE/audit_bundle.py" --root "$RUN_DIR"
# 正式 220 路线只用于最终报告；CARLA 环境上显式开启，不能替代训练期 val。
if [[ "${BENCH2DRIVE:-0}" == 1 ]]; then
 B2D_ARGS=()
 if [[ -f "$RUN_DIR/bench2drive/run_manifest.json" ]]; then
  B2D_ARGS+=(--resume)
 fi
 bash "$HERE/eval.sh" --bench2drive --checkpoint "$RUN_DIR/best.pt" \
  --output-dir "$RUN_DIR/bench2drive" --num-gpus "${EVAL_GPU_COUNT:-1}" "${B2D_ARGS[@]}"
fi
