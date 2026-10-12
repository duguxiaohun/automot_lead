#!/usr/bin/env bash
# Run from AutoMoT/. Only CPU tests, local preflight, metadata inventory and ZIP.
set -euo pipefail
ulimit -S -c 0
ulimit -H -c 0
P50_PYTHON="${PYTHON:-python}"
P50_CHECKPOINTS="${CHECKPOINTS_ROOT:-checkpoints}"
P50_MODEL="${MODEL_DIR:-checkpoints/Qwen3.5-4B}"
P50_OUTPUT="${READINESS_OUTPUT:-checkpoints/server_readiness_v50_$(date +%Y%m%d_%H%M%S_%N)}"
test ! -e "$P50_OUTPUT" || { echo 'Use a fresh READINESS_OUTPUT.' >&2; exit 2; }
mkdir -p "$P50_OUTPUT"
P50_TEST_EXIT=0
P50_SOURCE_PREFLIGHT=0
"$P50_PYTHON" - <<'PY' > "$P50_OUTPUT/tests.log" 2>&1 || P50_SOURCE_PREFLIGHT=$?
import json
from pathlib import Path
from qwen3vl_local.audit_joint.server_readiness import source_check
report = source_check(Path('.'), Path('qwen3vl_local/audit_joint/release_sources_v50_20261011.json'))
print(json.dumps(dict(stage='source_before_pytest', **report), ensure_ascii=False, indent=2))
raise SystemExit(0 if report['status'] == 'matched' else 2)
PY
if [[ "$P50_SOURCE_PREFLIGHT" == 0 ]]; then
PYTHON="$P50_PYTHON" bash qwen3vl_local/audit_joint/run_guarded.sh \
  --space-path "$P50_OUTPUT" --min-free-gib 20 -- \
  "$P50_PYTHON" -m qwen3vl_local.audit_joint.server_readiness --run-tests \
  >> "$P50_OUTPUT/tests.log" 2>&1 || P50_TEST_EXIT=$?
else
  P50_TEST_EXIT=2
  echo 'Published source hashes failed; pytest not run. Synchronize the release files; local extras are preserved.' >> "$P50_OUTPUT/tests.log"
fi
P50_PREFLIGHT_EXIT=0
"$P50_PYTHON" -m qwen3vl_local.qwen35.preflight --model-dir "$P50_MODEL" \
  > "$P50_OUTPUT/preflight.log" 2>&1 || P50_PREFLIGHT_EXIT=$?
"$P50_PYTHON" - "$P50_OUTPUT/commands.json" "$P50_TEST_EXIT" "$P50_PREFLIGHT_EXIT" "$P50_MODEL" "$P50_SOURCE_PREFLIGHT" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(
    pytest_exit_code=int(sys.argv[2]), preflight_exit_code=int(sys.argv[3]),
    source_preflight_exit_code=int(sys.argv[5]), pytest_started=int(sys.argv[5]) == 0,
    model_dir=sys.argv[4], gpu_training_run=False, dataset_build_run=False), indent=2)+'\n')
PY
"$P50_PYTHON" -m qwen3vl_local.audit_joint.server_readiness \
  --checkpoints "$P50_CHECKPOINTS" --output "$P50_OUTPUT/report" \
  --log "$P50_OUTPUT/tests.log" --log "$P50_OUTPUT/preflight.log" --log "$P50_OUTPUT/commands.json" \
  > "$P50_OUTPUT/package.json"
cat "$P50_OUTPUT/package.json"
echo "Send $P50_OUTPUT/report.zip and $P50_OUTPUT/package.json"
P50_SOURCE_EXIT=0
"$P50_PYTHON" - "$P50_OUTPUT/package.json" <<'PY' || P50_SOURCE_EXIT=$?
import json, sys
from pathlib import Path
raise SystemExit(0 if json.loads(Path(sys.argv[1]).read_text())['source_status']=='matched' else 1)
PY
if [[ "$P50_TEST_EXIT" != 0 || "$P50_PREFLIGHT_EXIT" != 0 || "$P50_SOURCE_EXIT" != 0 ]]; then
  echo 'CPU checks failed or were blocked; diagnostics were packaged. Do not build/train yet.' >&2
  exit 2
fi
