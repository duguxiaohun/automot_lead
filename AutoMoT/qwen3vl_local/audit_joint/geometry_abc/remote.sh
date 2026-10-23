#!/usr/bin/env bash
# Run from this machine's existing AutoMoT installation. No SSH, sync or downloads.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$APP${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -z ${PYTHON:-} ]]; then
  if [[ -x "$APP/.venv/bin/python" ]]; then PYTHON="$APP/.venv/bin/python"; else PYTHON=$(command -v python); fi
fi
export PYTHON
ENTRY="$APP/qwen3vl_local/audit_joint/geometry_abc/run.sh"
STUDY_ROOT="$APP/checkpoints/geometry_abc_v1"
STUDY_PER_EVENT=${STUDY_PER_EVENT:-1280}
[[ $STUDY_PER_EVENT == 1280 || $STUDY_PER_EVENT == 1024 ]] || { echo 'Use 1280 or the matched 1024-step-budget control' >&2; exit 2; }
cd "$APP"
case ${1:-} in
  cpu)
    [[ $# == 1 ]] || exit 2
    "$PYTHON" -u -m qwen3vl_local.audit_joint.geometry_abc.cpu --per-event "$STUDY_PER_EVENT"
    ;;
  train)
    [[ $# == 2 && $2 =~ ^(a|b|c)$ ]] || { echo 'Usage: remote.sh train a|b|c' >&2; exit 2; }
    [[ -f "$STUDY_ROOT/ready_$STUDY_PER_EVENT.json" ]] || { echo 'Run the study CPU preparation first' >&2; exit 2; }
    name=${STUDY_RUN_NAME:-${2}_${STUDY_PER_EVENT}}
    [[ $name =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid semantic run name' >&2; exit 2; }
    RUN="$STUDY_ROOT/runs/$name"
    [[ ! -e "$RUN" ]] || { echo "Existing run: $RUN; no overwrite or implicit resume" >&2; exit 2; }
    export STUDY_ROOT STUDY_PER_EVENT
    DATASET=$("$PYTHON" -B - <<'PYREADY'
import json,os
from pathlib import Path
from qwen3vl_local.audit_joint.geometry_abc.prepare import load_ready
root=Path(os.environ['STUDY_ROOT']);quota=int(os.environ['STUDY_PER_EVENT'])
ready=load_ready(root/f'ready_{quota}.json',per_event=quota,world_size=4)
print(ready['dataset'])
PYREADY
)
    export NPROC_PER_NODE=4
    if [[ -n ${GPU_IDS:-} ]]; then
      IFS=, read -ra devices <<< "$GPU_IDS"
      [[ ${#devices[@]} == 4 ]] || { echo 'Matched standard budget requires four GPUs' >&2; exit 2; }
    fi
    bash "$ENTRY" train --dataset "$DATASET" --variant "$2" \
      --model-dir "$APP/checkpoints/Qwen3.5-4B" --output "$RUN" --per-event "$STUDY_PER_EVENT"
    export RUN
    selected=$("$PYTHON" -c 'import json,os; from pathlib import Path; p=Path(os.environ["RUN"]); print(p/json.loads((p/"training_complete.json").read_text())["selected_checkpoint"])')
    eval_gpu=${GPU_IDS:-}
    eval_gpu=${eval_gpu%%,*}
    GPU_IDS="$eval_gpu" bash "$ENTRY" evaluate --dataset "$DATASET" --variant "$2" \
      --model-dir "$APP/checkpoints/Qwen3.5-4B" --checkpoint "$selected" --split test --output "$RUN/test"
    ;;
  *) echo 'Usage: remote.sh cpu | train a|b|c' >&2; exit 2 ;;
esac
