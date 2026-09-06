#!/usr/bin/env bash
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
 OUTPUT_DIR="$ABLATION_DIR/$mode" bash "$HERE/train.sh" "$@" --condition-mode "$mode"
 bash "$HERE/eval.sh" --checkpoint "$ABLATION_DIR/$mode/run_$RUN_TAG/best.pt" --split test
 bash "$HERE/eval.sh" --checkpoint "$ABLATION_DIR/$mode/run_$RUN_TAG/best.pt" --split test \
  --max-samples "${ABLATION_EVAL_SAMPLES:-256}" --dump-cases --output-dir "$ABLATION_DIR/$mode/run_$RUN_TAG/paired_test"
done
python "$HERE/compare_ablation.py" --base "$ABLATION_DIR/base/run_$RUN_TAG/paired_test" \
 --prior "$ABLATION_DIR/prior/run_$RUN_TAG/paired_test" --output "$ABLATION_DIR/comparison_$RUN_TAG.json"
