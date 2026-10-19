#!/usr/bin/env bash
# Default restored four-image choice base. Legacy profiles require explicit opt-in.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P3_DEFAULT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 0 && "$1" == --* ]]; then
  exec bash "$P3_DEFAULT_DIR/phase3_base_trial.sh" "$@"
fi
[[ $# -le 1 && "${1:-ddp}" == ddp ]] || {
  echo 'Default base uses four ranks. Use CLI flags, or LEGACY_PHASE3_SEQUENCE=1 for a historical profile.' >&2; exit 2;
}
for P3_DEFAULT_PAIR in 'HISTORY_RGB_MODE:4rgb' 'ACTION_OUTPUT_MODE:choice' 'BATCH_SIZE:1' 'GRAD_ACCUM:1' 'NUM_EPOCHS:3'; do
  P3_DEFAULT_KEY="${P3_DEFAULT_PAIR%%:*}"; P3_DEFAULT_VALUE="${P3_DEFAULT_PAIR#*:}"
  [[ -z "${!P3_DEFAULT_KEY:-}" || "${!P3_DEFAULT_KEY}" == "$P3_DEFAULT_VALUE" ]] || {
    echo "Default base requires $P3_DEFAULT_KEY=$P3_DEFAULT_VALUE; legacy settings cannot silently carry over." >&2; exit 2;
  }
done
: "${INDEX:?Set INDEX to the frozen historical four-image choice index}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a fresh experiment directory}"
P3_DEFAULT_ARGS=(--index "$INDEX" --output "$OUTPUT_DIR" --stage "${P3_STAGE:-quick}")
[[ -z "${DATA_ROOT:-}" ]] || P3_DEFAULT_ARGS+=(--data-root "$DATA_ROOT")
[[ -z "${MODEL_DIR:-}" ]] || P3_DEFAULT_ARGS+=(--model-dir "$MODEL_DIR")
[[ -z "${RESUME_FROM:-}" ]] || P3_DEFAULT_ARGS+=(--resume "$RESUME_FROM")
exec bash "$P3_DEFAULT_DIR/phase3_base_trial.sh" "${P3_DEFAULT_ARGS[@]}"
