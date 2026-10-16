#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
: "${PILOT_ROOT:?Set PILOT_ROOT to a pilot directory built with the current v54 producer contract (old versions are incompatible)}"
FINISH_PYTHON="${PYTHON:-python}"
FINISH_OUT="${OUTPUT_DIR:-checkpoints/v54_automatic_$(date +%Y%m%d_%H%M%S_%N)}"
mkdir "$FINISH_OUT"
export FINISH_OUT PILOT_ROOT
"$FINISH_PYTHON" - <<'PY'
import json,os
from pathlib import Path
from qwen3vl_local.audit_joint.server_readiness import source_check
r=source_check(Path('.'),Path('qwen3vl_local/audit_joint/release_sources_v70_selected_b2_20261016.json'))
(Path(os.environ['FINISH_OUT'])/'source_check.json').write_text(json.dumps(r,indent=2)+'\n')
if r['status']!='matched':raise SystemExit('Source mismatch; inspect source_check.json and synchronize latest files')
PY
FINISH_RESERVE=$("$FINISH_PYTHON" - <<'PY'
import os,shutil
from pathlib import Path
out=Path(os.environ['FINISH_OUT']);pilot=Path(os.environ['PILOT_ROOT'])
if pilot.stat().st_dev != out.stat().st_dev:raise SystemExit('Replay and output must share the guarded filesystem')
print(max(20.,shutil.disk_usage(out).free/2**30-12.))
PY
)
set +e
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 PYTHON="$FINISH_PYTHON" bash qwen3vl_local/audit_joint/run_guarded.sh \
 --space-path "$FINISH_OUT" --min-free-gib "$FINISH_RESERVE" -- \
 nice -n 10 "$FINISH_PYTHON" -m qwen3vl_local.audit_joint.server_finish build \
 --pilot "$PILOT_ROOT" --output "$FINISH_OUT" --data-root "${DATA_ROOT:-lead_data}" \
 --model-dir "${MODEL_DIR:-checkpoints/Qwen3.5-4B}" --workers "${REPLAY_WORKERS:-4}" \
 > "$FINISH_OUT/build.log" 2>&1
FINISH_RC=$?
set -e
"$FINISH_PYTHON" -m qwen3vl_local.audit_joint.server_finish package --output "$FINISH_OUT" --exit-code "$FINISH_RC"
printf 'CPU result: %s/finish_receipt.json\nReturn: %s/handoff/fullflow_report.zip and %s/handoff/fullflow_package.json\n' "$FINISH_OUT" "$FINISH_OUT" "$FINISH_OUT"
exit "$FINISH_RC"
