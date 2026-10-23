#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
APP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="${APP}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON=${PYTHON:-python}
MODULE=qwen3vl_local.audit_joint.geometry_abc
if [[ ${1:-} == train || ${1:-} == evaluate ]]; then
  if [[ -n ${GPU_IDS:-} ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  else
    count=${NPROC_PER_NODE:-4}
    [[ ${1:-} != train ]] && count=1
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -n "$count" | cut -d, -f1 | paste -sd, -)
  fi
  [[ -n $CUDA_VISIBLE_DEVICES ]] || { echo 'No GPUs selected' >&2; exit 2; }
  "$PYTHON" -c 'import os; d=os.environ["CUDA_VISIBLE_DEVICES"].split(","); assert all(x.isdigit() for x in d) and len(d)==len(set(d)), "GPU_IDS must contain unique numeric IDs"'
  export STUDY_GPU_SELECTED=1
  if [[ ${1:-} == train ]]; then
    IFS=, read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
    exec "$PYTHON" -m torch.distributed.run --standalone --max_restarts=0 --nproc_per_node="${#devices[@]}" --module -- "$MODULE" "$@"
  fi
fi
exec "$PYTHON" -m "$MODULE" "$@"
