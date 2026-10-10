#!/usr/bin/env bash
# Full paired weak-supervision build -> exact event-equal training -> test -> audit.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P4_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${P4_DIR}/../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHON="${PYTHON:-python}"
fail() { echo "[Phase4 pipeline] $*" >&2; exit 2; }
[[ "${ACTION_OUTPUT_MODE:-binary}" == binary ]] || fail 'Phase4 predicts binary YES/NO only; ACTION_OUTPUT_MODE=choice is not supported.'
case "${HISTORY_RGB_MODE:-}" in
  '') P4_RGB="${RGB_MODE:-4}" ;;
  4rgb) P4_RGB=4 ;;
  2rgb_endpoints) P4_RGB=2 ;;
  *) fail 'HISTORY_RGB_MODE must be 4rgb or 2rgb_endpoints.' ;;
esac
[[ "$P4_RGB" == 2 || "$P4_RGB" == 4 ]] || fail 'RGB_MODE must be 2 or 4.'
[[ -z "${RGB_MODE:-}" || "$RGB_MODE" == "$P4_RGB" ]] || fail 'RGB_MODE conflicts with HISTORY_RGB_MODE.'
export RGB_MODE="$P4_RGB"
P4_MODE="${TRAIN_MODE:-${MODE:-ddp}}"
[[ -z "${MODE:-}" || "$MODE" == "$P4_MODE" ]] || fail 'MODE conflicts with TRAIN_MODE.'
case "$P4_MODE" in ddp|single|check|preflight|host-preflight) ;; *) fail 'TRAIN_MODE must be ddp/single/check/preflight/host-preflight.' ;; esac
for P4_FLAG in SKIP_BUILD SKIP_TRAIN SKIP_EVAL; do
  P4_VALUE="${!P4_FLAG:-0}"
  [[ "$P4_VALUE" == 0 || "$P4_VALUE" == 1 ]] || fail "$P4_FLAG must be 0 or 1."
done
[[ -z "${CONDITION_STREAM:-}" ]] || fail 'Use the dataset CLI for condition streams; this entry builds the full teacher pool.'
[[ "${MAX_TEACHER_QUESTIONS_PER_ROUTE:-0}" == 0 ]] || fail 'Full training requires MAX_TEACHER_QUESTIONS_PER_ROUTE=0.'
if [[ -n "${PRODUCTION_INDEX:-}" ]]; then
  [[ -n "${TEACHER_REGISTRY:-}" ]] || fail 'PRODUCTION_INDEX requires a source-bound teacher registry (TEACHER_REGISTRY).'
  [[ -n "${CANDIDATE_POOL:-}" ]] || fail 'PRODUCTION_INDEX requires CANDIDATE_POOL.'
else
  [[ -z "${TEACHER_REGISTRY:-}" ]] || fail 'TEACHER_REGISTRY requires PRODUCTION_INDEX.'
fi
for P4_ARG in "$@"; do
  P4_OPTION="${P4_ARG%%=*}"
  for P4_FIXED in --dataset --model-dir --data-root --output-dir --paired-with --sampling-policy; do
    if [[ "$P4_OPTION" == --?* && "$P4_FIXED" == "$P4_OPTION"* ]]; then
      fail 'Set paths through environment variables; phase3_balanced and the RGB pair are fixed by this pipeline.'
    fi
  done
done
export DATA_ROOT="${DATA_ROOT:-lead_data}"
export MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3.5-4B}"
if [[ "$P4_MODE" == ddp || "$P4_MODE" == single || "$P4_MODE" == host-preflight ]]; then
  [[ -f "$MODEL_DIR/config.json" ]] && compgen -G "$MODEL_DIR/*.safetensors" >/dev/null || fail "MODEL_DIR=$MODEL_DIR: local base model missing; set MODEL_DIR to the installed complete Qwen3.5-4B before building/training."
fi
DATA_DIR="${DATA_DIR:-checkpoints/phase4_v47_full}"
# Canonical paths prevent a caller from building one pair and training another.
mkdir -p -- "$DATA_DIR"
DATA_DIR="$(cd -- "$DATA_DIR" && pwd)"
[[ -z "${DATASET:-}" || "$(realpath -m -- "$DATASET")" == "$DATA_DIR/data$P4_RGB" ]] || fail 'DATASET must be DATA_DIR/data2 or DATA_DIR/data4 for the chosen mode; set DATA_DIR.'
[[ -z "${PAIRED_WITH:-}" || "$(realpath -m -- "$PAIRED_WITH")" == "$DATA_DIR/data$((6-P4_RGB))" ]] || fail 'PAIRED_WITH must be the other mode under DATA_DIR.'
export DATASET="$DATA_DIR/data$P4_RGB" PAIRED_WITH="$DATA_DIR/data$((6-P4_RGB))"
if [[ "${SKIP_TRAIN:-0}" == 1 ]]; then
  [[ -n "${RUN_ROOT:-${OUTPUT_DIR:-}}" ]] || fail 'SKIP_TRAIN=1 requires RUN_ROOT or OUTPUT_DIR of the completed training run.'
  [[ "$P4_MODE" == ddp || "$P4_MODE" == single ]] || fail 'Check modes cannot be combined with SKIP_TRAIN=1.'
fi
[[ -z "${RUN_ROOT:-}" || -z "${OUTPUT_DIR:-}" || "$(realpath -m -- "$RUN_ROOT")" == "$(realpath -m -- "$OUTPUT_DIR")" ]] || fail 'RUN_ROOT and OUTPUT_DIR conflict.'
P4_STAMP="$(date +%Y%m%d_%H%M%S_%N)"
PIPELINE_ROOT="${PIPELINE_ROOT:-checkpoints/sft_new_loop_phase4_pipeline/${P4_STAMP}_rgb${P4_RGB}}"
export OUTPUT_DIR="${RUN_ROOT:-${OUTPUT_DIR:-${PIPELINE_ROOT}/train}}"
mkdir -p -- "$PIPELINE_ROOT"
exec > >(tee -a "$PIPELINE_ROOT/pipeline.log") 2>&1
printf '[Phase4 pipeline] RGB=%s, mode=%s, all-route pool, Phase3-style 1024/event epochs\nDATA_DIR=%s\nOUTPUT_DIR=%s\n' "$P4_RGB" "$P4_MODE" "$DATA_DIR" "$OUTPUT_DIR"
P4_BUILD=(--data-dir "$DATA_DIR" --data-root "$DATA_ROOT" --workers "${BUILD_WORKERS:-16}" --annotations "${ANNOTATIONS:-$P4_DIR/reviewed_state_pairs_v9.json}")
[[ "${SKIP_BUILD:-0}" != 1 ]] || P4_BUILD+=(--skip-build)
[[ -z "${CANDIDATE_POOL:-}" ]] || P4_BUILD+=(--candidate-pool "$CANDIDATE_POOL")
[[ -z "${PRODUCTION_INDEX:-}" ]] || P4_BUILD+=(--production-index "$PRODUCTION_INDEX" --teacher-registry "$TEACHER_REGISTRY")
"$PYTHON" -m qwen3vl_local.sft_new_loop_phase4.full_pipeline "${P4_BUILD[@]}"
if [[ "${SKIP_TRAIN:-0}" != 1 ]]; then
  bash "$P4_DIR/train.sh" "$P4_MODE" --sampling-policy phase3_balanced --paired-with "$PAIRED_WITH" --epochs "${EPOCHS:-7}" --accumulation "${ACCUMULATION:-8}" "$@"
fi
# Checks must not touch any existing adapter or evaluate a previous run.
case "$P4_MODE" in check|preflight|host-preflight) exit 0 ;; esac
if [[ "${SKIP_EVAL:-0}" != 1 ]]; then
  P4_BEST="$("$PYTHON" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print((p/json.loads((p/"best.json").read_text())["adapter"]).resolve())' "$OUTPUT_DIR")"
  if [[ "${EVAL_DEVICE:-cuda:0}" == cpu ]]; then
    P4_EVAL_GPU="${CUDA_VISIBLE_DEVICES:-}"
  else
    P4_EVAL_GPU="$("$PYTHON" -c 'from qwen3vl_local.sft_new_loop_phase4.devices import select_gpus; print(select_gpus(1))')"
  fi
  CUDA_VISIBLE_DEVICES="$P4_EVAL_GPU" "$PYTHON" -m qwen3vl_local.sft_new_loop_phase4.evaluate \
    --dataset "$DATASET" --adapter "$P4_BEST" --model-dir "$MODEL_DIR" \
    --data-root "$DATA_ROOT" --split test --output-dir "$OUTPUT_DIR/test" --device "${EVAL_DEVICE:-cuda:0}"
fi
"$PYTHON" -m qwen3vl_local.sft_new_loop_phase4.audit_bundle --run "$OUTPUT_DIR" --output "$OUTPUT_DIR/audit.zip"
echo "[Phase4 pipeline] complete: $OUTPUT_DIR"
