#!/usr/bin/env bash
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P3_PYTHON="${PYTHON:-python}"
P3_OUTPUT="${OUTPUT_DIR:-checkpoints/phase3_prepare_$(date +%Y%m%d_%H%M%S_%N)}"
mkdir "$P3_OUTPUT"
P3_RESERVE=$("$P3_PYTHON" - "$P3_OUTPUT" <<'PY'
import shutil,sys
print(max(20., shutil.disk_usage(sys.argv[1]).free / 2**30 - 8.))
PY
)
set +e
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 PYTHON="$P3_PYTHON" bash qwen3vl_local/audit_joint/run_guarded.sh \
 --space-path "$P3_OUTPUT" --min-free-gib "$P3_RESERVE" -- \
 nice -n 10 "$P3_PYTHON" -m qwen3vl_local.audit_joint.server_phase3_prepare build \
 --output "$P3_OUTPUT" --data-root "${DATA_ROOT:-lead_data}" \
 --collection-dir "${COLLECTION_DIR:-keyframe_filter/collection_output}" \
 --workers "${BUILD_WORKERS:-2}" > "$P3_OUTPUT/prepare.log" 2>&1
P3_RC=$?
"$P3_PYTHON" -m qwen3vl_local.audit_joint.server_phase3_prepare package \
 --output "$P3_OUTPUT" --exit-code "$P3_RC"
P3_PACKAGE_RC=$?
set -e
printf 'Return: %s/phase3_prepare_report.zip and %s/phase3_prepare_package.json\n' "$P3_OUTPUT" "$P3_OUTPUT"
if [[ "$P3_RC" -ne 0 ]]; then exit "$P3_RC"; fi
exit "$P3_PACKAGE_RC"
