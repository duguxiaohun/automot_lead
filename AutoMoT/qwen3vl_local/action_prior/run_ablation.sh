#!/usr/bin/env bash
# 在 AutoMoT/ 下直接复制执行：
#   DATA_DIR=checkpoints/action_prior_data bash qwen3vl_local/action_prior/run_ablation.sh
#   DATASET_PRIORS=1 DATA_DIR=checkpoints/action_prior_data bash qwen3vl_local/action_prior/run_ablation.sh
# 开关只对 prior 臂生效，base 臂本来就没有先验。
# EVENT_BALANCED 是 train.sh / run_full_pipeline.sh 的轨迹训练开关；本入口不启用均衡训练。
# 构建、开启/关闭以及独立场景先验 demo 见 run.md。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${RESUME:-}" || "${NO_RUN_SUBDIR:-0}" == 1 ]]; then
 echo "ablation requires two fresh run subdirectories" >&2
 exit 2
fi
: "${DATA_DIR:?请先设置同一个完整 action 索引 DATA_DIR}"
# 两次独立初始化；所有共有 CLI、seed、数据、预算完全相同，只改变 condition_mode。
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ABLATION_DIR="${OUTPUT_DIR:-checkpoints/action_prior_ablation}"
for mode in base prior; do
 # base 消融本来就不产生先验；数据集标定真值只对 prior 臂有意义。
 arm_dataset_priors=0
 [[ "$mode" != prior ]] || arm_dataset_priors="${DATASET_PRIORS:-0}"
 OUTPUT_DIR="$ABLATION_DIR/$mode" DATASET_PRIORS="$arm_dataset_priors" \
  bash "$HERE/train.sh" "$@" --condition-mode "$mode"
 bash "$HERE/eval.sh" --checkpoint "$ABLATION_DIR/$mode/run_$RUN_TAG/best.pt" --split test
 bash "$HERE/eval.sh" --checkpoint "$ABLATION_DIR/$mode/run_$RUN_TAG/best.pt" --split test \
  --max-samples "${ABLATION_EVAL_SAMPLES:-256}" --dump-cases --output-dir "$ABLATION_DIR/$mode/run_$RUN_TAG/paired_test"
done
python "$HERE/compare_ablation.py" --base "$ABLATION_DIR/base/run_$RUN_TAG/paired_test" \
 --prior "$ABLATION_DIR/prior/run_$RUN_TAG/paired_test" --output "$ABLATION_DIR/comparison_$RUN_TAG.json"
