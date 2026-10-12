#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
PILOT_PYTHON="${PYTHON:-python}"
PILOT_OUT="${OUTPUT_DIR:-checkpoints/v52_pilot_$(date +%Y%m%d_%H%M%S_%N)}"
mkdir "$PILOT_OUT"
export PILOT_OUT
# Code synchronization only: no repeated pytest/model readiness or legacy .git checks.
"$PILOT_PYTHON" - <<'PY'
import json,os
from pathlib import Path
from qwen3vl_local.audit_joint.server_readiness import source_check
p=Path(os.environ['PILOT_OUT'])
r=source_check(Path('.'),Path('qwen3vl_local/audit_joint/release_sources_v52_20261012.json'))
(p/'source_check.json').write_text(json.dumps(r,indent=2)+'\n')
if r['status']!='matched':raise SystemExit('Source mismatch; synchronize release files before building. See source_check.json')
PY
# Stop at 20GiB free or after about 2GiB net disk consumption, whichever comes first.
# Polling is not a hard quota; unrelated writes also consume this allowance.
PILOT_RESERVE=$("$PILOT_PYTHON" - <<'PY'
import os,shutil
print(max(20.,shutil.disk_usage(os.environ['PILOT_OUT']).free/2**30-2.))
PY
)
set +e
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHON="$PILOT_PYTHON" bash qwen3vl_local/audit_joint/run_guarded.sh \
  --space-path "$PILOT_OUT" --min-free-gib "$PILOT_RESERVE" -- \
  nice -n 10 "$PILOT_PYTHON" -m qwen3vl_local.audit_joint.server_pilot run \
  --data-root "${DATA_ROOT:-lead_data}" --output "$PILOT_OUT" > "$PILOT_OUT/pilot.log" 2>&1
PILOT_RC=$?
set -e
"$PILOT_PYTHON" -m qwen3vl_local.audit_joint.server_pilot report --output "$PILOT_OUT" --exit-code "$PILOT_RC"
printf 'Return these files: %s/pilot_report.zip and %s/pilot_package.json\n' "$PILOT_OUT" "$PILOT_OUT"
exit "$PILOT_RC"
