#!/usr/bin/env bash
# Four-rank independent generation; no gradient synchronization is needed.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P4_EVAL_PYTHON="${PYTHON:-python}"
P4_EVAL_KIND="${1:?Use evaluate.sh evaluate|reload_check [arguments]}"
shift
[[ "$P4_EVAL_KIND" == evaluate || "$P4_EVAL_KIND" == reload_check ]] || { echo 'Unknown evaluation mode' >&2; exit 2; }
P4_EVAL_GPUS="$("$P4_EVAL_PYTHON" -c 'from qwen3vl_local.sft_new_loop_phase4.devices import select_gpus; print(select_gpus(4))')"
IFS=',' read -ra P4_EVAL_CARDS <<< "$P4_EVAL_GPUS"
[[ ${#P4_EVAL_CARDS[@]} == 4 ]] || { echo 'Evaluation requires four visible idle GPUs or four explicit GPU_IDS' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$P4_EVAL_GPUS"
exec "$P4_EVAL_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m "qwen3vl_local.sft_new_loop_phase4.rgb_short.$P4_EVAL_KIND" "$@"
