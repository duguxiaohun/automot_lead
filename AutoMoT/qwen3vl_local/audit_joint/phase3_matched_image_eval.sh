#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P3_DIAG_APP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$P3_DIAG_APP"
P3_DIAG_CPU=0
for P3_DIAG_ARG in "$@"; do
  case "$P3_DIAG_ARG" in --preflight-only|--help|-h) P3_DIAG_CPU=1 ;; esac
done
if [[ "$P3_DIAG_CPU" == 0 ]]; then
  if [[ -n "${GPU_IDS:-}" ]]; then
    [[ "$GPU_IDS" =~ ^[0-9]+$ ]] || { echo 'GPU_IDS must specify exactly one numeric GPU ID' >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  else
    P3_DIAG_GPU="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | "${PYTHON:-python}" -c 'import csv,sys; rows=[tuple(int(v.strip()) for v in r) for r in csv.reader(sys.stdin)]; print(min(rows,key=lambda r:(r[1],r[2],r[0]))[0])')"
    export CUDA_VISIBLE_DEVICES="$P3_DIAG_GPU"
  fi
  echo "Diagnostic inference GPU: $CUDA_VISIBLE_DEVICES"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
exec "${PYTHON:-python}" -m qwen3vl_local.audit_joint.phase3_matched_image_eval \
  --plan qwen3vl_local/audit_joint/phase3_behavior_diagnostic_v1/matched_images_seed_11.json \
         qwen3vl_local/audit_joint/phase3_behavior_diagnostic_v1/matched_images_seed_29.json \
         qwen3vl_local/audit_joint/phase3_behavior_diagnostic_v1/matched_images_seed_47.json \
  "$@"
