#!/usr/bin/env bash
# 将已完成的逐帧 RGB decisions 与源 EVENT 标注和现有 Phase2 index 联表；不加载模型。
# 从 AutoMoT/ 目录运行：
#   bash qwen3vl_local/sft_new_loop_phase2/run_ue3_label_alignment_audit.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

COLLECTION_DIR="${COLLECTION_DIR:-keyframe_filter/collection_output}"
INDEX="${INDEX:-checkpoints/sft_new_loop_phase2_data/frame_index.jsonl}"

# 旧机器可能把审计包复制到 checkpoints 顶层；正式生成脚本则写在 frozen experiment
# 目录。未显式传 AUDIT_ROOT 时，按 32-case decisions 身份自动定位，避免选错实验。
if [[ -z "${AUDIT_ROOT:-}" ]]; then
  AUDIT_ROOT="$(python - "${SCRIPT_DIR}/ue3_route_diverse_rgb_decisions_v1.jsonl" <<'PY'
import json
import pathlib
import sys

decisions_path = pathlib.Path(sys.argv[1])


def identities(path: pathlib.Path):
    out = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out.add(
                (
                    str(row.get("scenario", "")),
                    str(row.get("route_id", "")),
                    int(row.get("frame_id", -1)),
                    str(row.get("question_domain", "")),
                )
            )
    return out


expected = identities(decisions_path)
preferred = pathlib.Path(
    "checkpoints/sft_new_loop_phase2_frozen_protocol/"
    "v3_frozen_3seed_unseen456_20260831/ue3_route_diverse_full_rgb_audit"
)
candidates = [pathlib.Path("checkpoints/ue3_route_diverse_full_rgb_audit"), preferred]
candidates.extend(
    pathlib.Path("checkpoints/sft_new_loop_phase2_frozen_protocol").glob(
        "*/ue3_route_diverse_full_rgb_audit"
    )
)
matches = []
seen = set()
for root in candidates:
    key = str(root)
    if key in seen:
        continue
    seen.add(key)
    manifest = root / "manifest.jsonl"
    if manifest.is_file() and identities(manifest) == expected:
        matches.append(root)

if preferred in matches:
    print(preferred)
elif len(matches) == 1:
    print(matches[0])
elif not matches:
    raise SystemExit(
        "cannot find a UE3 RGB audit whose manifest matches the 32 reviewed decisions; "
        "set AUDIT_ROOT=<directory containing manifest.jsonl> or rerun "
        "run_ue3_full_validation_rgb_audit.sh"
    )
else:
    rendered = ", ".join(str(path) for path in matches)
    raise SystemExit(f"multiple matching UE3 RGB audits found; set AUDIT_ROOT explicitly: {rendered}")
PY
)"
fi

if [[ ! -f "${AUDIT_ROOT}/manifest.jsonl" ]]; then
  echo "missing UE3 audit manifest: ${AUDIT_ROOT}/manifest.jsonl" >&2
  echo "Set AUDIT_ROOT to the directory containing manifest.jsonl." >&2
  exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-${AUDIT_ROOT}/label_alignment}"

echo "[ue3-alignment] audit_root=${AUDIT_ROOT}"
echo "[ue3-alignment] collection_dir=${COLLECTION_DIR}"
echo "[ue3-alignment] index=${INDEX}"
echo "[ue3-alignment] output=${OUTPUT_DIR}"

python qwen3vl_local/sft_new_loop_phase2/rescore_ue3_rgb_decisions.py \
  --audit-root "${AUDIT_ROOT}"

python qwen3vl_local/sft_new_loop_phase2/audit_ue3_label_alignment.py \
  --audit-root "${AUDIT_ROOT}" \
  --collection-dir "${COLLECTION_DIR}" \
  --index "${INDEX}" \
  --output-dir "${OUTPUT_DIR}"

echo "[done] inspect ${OUTPUT_DIR}/summary.md"
