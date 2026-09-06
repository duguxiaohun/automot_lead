#!/usr/bin/env bash
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# run tag 只计算一次，数据、训练、测试都绑定本次 run。
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/action_prior}"
export DATA_DIR="${DATA_DIR:-checkpoints/action_prior_data/run_${RUN_TAG}}"
export DATA_ROOT="${DATA_ROOT:-lead_data}"
if [[ -n "${RESUME:-}" ]]; then
 bash "$HERE/resume.sh" "$RESUME" "$@"
else
 # 先核验权重和 prompt 合同，缺权重时不先构建全量索引。
 ACTION_MODE=preflight bash "$HERE/train.sh" --models-only "$@"
 if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
  python "$HERE/build_dataset.py" --data-root "$DATA_ROOT" --output-dir "$DATA_DIR"
 fi
 bash "$HERE/train.sh" "$@"
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
