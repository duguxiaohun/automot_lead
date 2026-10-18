#!/usr/bin/env bash
# CPU relabel/build from existing lead_data only. No CARLA, new capture or GPU training.
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P4_EXISTING_PYTHON="${PYTHON:-python}"
P4_EXISTING_OUT="${OUTPUT_DIR:-checkpoints/phase4_existing_$(date +%Y%m%d_%H%M%S_%N)}"
[[ $# == 0 ]] || { echo 'Use environment variables for this fixed entry.' >&2; exit 2; }
mkdir "$P4_EXISTING_OUT"
export P4_EXISTING_OUT
set +e
"$P4_EXISTING_PYTHON" - <<'PY'
import json,os
from pathlib import Path
from qwen3vl_local.audit_joint.server_readiness import source_check
r=source_check(Path('.'),Path('qwen3vl_local/audit_joint/release_sources_v84_label_sampling_20261018.json'))
(Path(os.environ['P4_EXISTING_OUT'])/'source_check.json').write_text(json.dumps(r,indent=2)+'\n')
if r['status']!='matched':raise SystemExit('Source mismatch; synchronize this release before building')
PY
P4_EXISTING_RC=$?
if [[ "$P4_EXISTING_RC" == 0 ]]; then
  P4_EXISTING_RESERVE=$("$P4_EXISTING_PYTHON" - <<'PY'
import os,shutil
print(max(20.,shutil.disk_usage(os.environ['P4_EXISTING_OUT']).free/2**30-12.))
PY
  )
  P4_EXISTING_RC=$?
  if [[ "$P4_EXISTING_RC" == 0 ]]; then
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHON="$P4_EXISTING_PYTHON" bash qwen3vl_local/audit_joint/run_guarded.sh \
      --space-path "$P4_EXISTING_OUT" --min-free-gib "$P4_EXISTING_RESERVE" -- \
      nice -n 10 "$P4_EXISTING_PYTHON" -m qwen3vl_local.audit_joint.server_finish prepare \
      --output "$P4_EXISTING_OUT" --data-root "${DATA_ROOT:-lead_data}" \
      --model-dir "${MODEL_DIR:-checkpoints/Qwen3.5-4B}" --workers "${REPLAY_WORKERS:-2}" \
      --coverage-policy existing_data_development > "$P4_EXISTING_OUT/build.log" 2>&1
    P4_EXISTING_RC=$?
  fi
fi
set -e
"$P4_EXISTING_PYTHON" -m qwen3vl_local.audit_joint.server_finish package \
  --output "$P4_EXISTING_OUT" --exit-code "$P4_EXISTING_RC"
printf 'Return: %s/handoff/fullflow_report.zip and %s/handoff/fullflow_package.json\n' "$P4_EXISTING_OUT" "$P4_EXISTING_OUT"
exit "$P4_EXISTING_RC"
