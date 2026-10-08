#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P4_FULL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
P4_RGB="${1:?usage: train_full.sh 2|4 [ddp|single|check|preflight|host-preflight] [training flags]}"
shift
[[ "$P4_RGB" == 2 || "$P4_RGB" == 4 ]] || { echo 'RGB mode must be 2 or 4' >&2; exit 2; }
P4_MODE="${1:-ddp}"
if [[ $# -gt 0 ]]; then shift; fi
for P4_ARG in "$@"; do
  case "$P4_ARG" in
    --sampling-policy|--sampling-policy=*) echo 'train_full.sh fixes full_event_equal; use train.sh for other policies' >&2; exit 2 ;;
  esac
done
export RGB_MODE="$P4_RGB"
export DATASET="${DATASET:-${P4_FULL_DIR}/../../checkpoints/phase4_v40_full/data${P4_RGB}}"
P4_PEER="${PAIRED_WITH:-$(dirname -- "$DATASET")/data$((6-P4_RGB))}"
echo '[Phase4 full] Every admitted question each epoch; exact event 1:1 by oversampling. Repetition is intentionally uncapped.'
exec bash "$P4_FULL_DIR/train.sh" "$P4_MODE" --sampling-policy full_event_equal --paired-with "$P4_PEER" "$@"
