#!/usr/bin/env bash
# Project-local assets only. Run in foreground or under nohup; no SSH or sync.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ENTRY="$APP/qwen3vl_local/audit_joint/geometry_v1/run.sh"
export GEOMETRY_ROOT="$APP/checkpoints/geometry_v1"
export DATA_ROOT="$APP/lead_data"
export MODEL_DIR="$APP/checkpoints/Qwen3.5-4B"
export COLLECTION_DIR="$APP/keyframe_filter/collection_output"
export CARLA_MAPS="$APP/CARLA_0915/CarlaUE4/Content/Carla/Maps"
export MAPS_DIR="$GEOMETRY_ROOT/maps"
export PYTHONPATH="$APP${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -z ${PYTHON:-} ]]; then
  if [[ -x "$APP/.venv/bin/python" ]]; then
    PYTHON="$APP/.venv/bin/python"
  else
    PYTHON=$(command -v python)
  fi
fi
export PYTHON
case ${1:-} in
  cpu|train) ;;
  *) echo 'Usage: bash remote.sh cpu|train' >&2; exit 2 ;;
esac
[[ $# == 1 ]] || { echo 'Unexpected arguments' >&2; exit 2; }
cd "$APP"
printf 'APP=%s\nPYTHON=%s\nDATA=%s\nMODEL=%s\nCARLA_MAPS=%s\nOUTPUT=%s\n' \
  "$APP" "$PYTHON" "$DATA_ROOT" "$MODEL_DIR" "$CARLA_MAPS" "$GEOMETRY_ROOT"

if [[ $1 == cpu ]]; then
  # Validate before reserving a fresh output. Never rebuild or overwrite a run.
  "$PYTHON" -B - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import carla, numpy, torch, transformers, peft
from PIL import Image
from qwen3vl_local.sft_new_loop_phase3.preflight import check_model

for name in ('DATA_ROOT', 'COLLECTION_DIR', 'CARLA_MAPS'):
    if not Path(os.environ[name]).is_dir():
        raise SystemExit(f'Missing project asset: {name}={os.environ[name]}')
check_model(os.environ['MODEL_DIR'])
root = Path(os.environ['CARLA_MAPS'])
towns = [f'Town{i:02d}' for i in range(1, 8)] + ['Town10HD', 'Town11', 'Town12', 'Town13', 'Town15']
selected = {}
for town in towns:
    candidates = [root/'OpenDrive'/f'{town}.xodr', root/town/'OpenDrive'/f'{town}.xodr']
    found = [p for p in candidates if p.is_file()]
    if not found:
        raise SystemExit(f'Missing {town}.xodr in project CARLA Maps')
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in found]
    if len(set(hashes)) != 1:
        raise SystemExit(f'Conflicting project maps: {town}')
    selected[town] = dict(source=str(found[0].resolve(strict=True)), sha256=hashes[0])
out = Path(os.environ['GEOMETRY_ROOT'])
out.parent.mkdir(parents=True, exist_ok=True)
out.mkdir(exist_ok=False)
view = Path(os.environ['MAPS_DIR'])
view.mkdir()
for town, item in selected.items():
    (view/f'{town}.xodr').symlink_to(item['source'])
(out/'assets.json').write_text(json.dumps(dict(
    python=os.sys.executable, data_root=os.environ['DATA_ROOT'],
    model_dir=os.environ['MODEL_DIR'], carla_maps=str(root), maps=selected,
), indent=2) + '\n')
print(f'Prepared {len(selected)} project map links: {view}', flush=True)
PY
  bash "$ENTRY" build --collection "$COLLECTION_DIR" --data-root "$DATA_ROOT" \
    --maps "$MAPS_DIR" --output "$GEOMETRY_ROOT/data"
  bash "$ENTRY" plan --dataset "$GEOMETRY_ROOT/data/dataset.json" \
    --output "$GEOMETRY_ROOT/plan.json" --world-size 4 --per-event 1024 --epochs 3
  printf '%s\n' ready > "$GEOMETRY_ROOT/cpu.done"
else
  [[ -f "$GEOMETRY_ROOT/cpu.done" ]] || { echo 'CPU build and plan not complete' >&2; exit 2; }
  [[ ! -e "$GEOMETRY_ROOT/run" && ! -e "$GEOMETRY_ROOT/test" ]] || {
    echo 'Existing run/test output; refusing to overwrite or resume' >&2; exit 2;
  }
  export NPROC_PER_NODE=4
  if [[ -n ${GPU_IDS:-} ]]; then
    IFS=, read -ra devices <<< "$GPU_IDS"
    [[ ${#devices[@]} == 4 ]] || { echo 'This plan requires four GPUs' >&2; exit 2; }
  fi
  bash "$ENTRY" train --dataset "$GEOMETRY_ROOT/data/dataset.json" \
    --model-dir "$MODEL_DIR" --output "$GEOMETRY_ROOT/run" \
    --per-event 1024 --epochs 3 --max-startup-hours 1 --max-epoch-hours 10
  eval_gpu=${GEOMETRY_EVAL_GPU:-}
  if [[ -z $eval_gpu && -n ${GPU_IDS:-} ]]; then eval_gpu=${GPU_IDS%%,*}; fi
  GPU_IDS="$eval_gpu" bash "$ENTRY" evaluate \
    --dataset "$GEOMETRY_ROOT/data/dataset.json" --model-dir "$MODEL_DIR" \
    --checkpoint "$GEOMETRY_ROOT/run/epoch_3" --split test --output "$GEOMETRY_ROOT/test"
  printf '%s\n' complete > "$GEOMETRY_ROOT/train_test.done"
fi
