#!/usr/bin/env bash
set -Eeuo pipefail
# Fresh Phase2 task composition; training and inference share the new input layout.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMMAND=${1:-all}
if (($#)); then shift; fi
export AUTOMOT_TEXT_PREFIX=${AUTOMOT_TEXT_PREFIX:-1}
export AUTOMOT_TEXT_PREFIX_CACHE=${AUTOMOT_TEXT_PREFIX_CACHE:-1}
exec bash "$HERE/run.sh" "$COMMAND" 2 --revision taskmix_20261031 --budget-profile phase2_10k "$@"
