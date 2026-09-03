#!/usr/bin/env bash
set -euo pipefail

# 禁用 core dump，避免模型加载异常时生成巨大 core.* 文件。
ulimit -S -c 0 2>/dev/null || true

MODE="${1:-preflight}"  # preflight | smoke | train | eval
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-checkpoints/sft_new_loop_phase2_frozen_protocol/v3_frozen_3seed_unseen456_20260831/train_runs/seed_20260810/fallback_generation}"
DATA_ROOT="${DATA_ROOT:-lead_data}"
LEADMOT_DATA_DIR="${LEADMOT_DATA_DIR:-checkpoints/leadmot_v1_data}"
TRAIN_JSONL="${TRAIN_JSONL:-${LEADMOT_DATA_DIR}/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${LEADMOT_DATA_DIR}/val.jsonl}"
AUTO_BUILD_LEADMOT_DATASET="${AUTO_BUILD_LEADMOT_DATASET:-1}"
DATASET_SEED="${DATASET_SEED:-2026}"
DATASET_STRIDE="${DATASET_STRIDE:-5}"
AB_ID="${AB_ID:-v3_seed20260810_fallback4000_$(date +%Y%m%d_%H%M%S)}"
AB_ROOT="${AB_ROOT:-checkpoints/leadmot_qwen_adapter_ab/${AB_ID}}"
TRAIN_LAUNCH_MODE="${TRAIN_LAUNCH_MODE:-ddp}"
SEED="${SEED:-2026}"
USE_BEV="${USE_BEV:-1}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
MIN_ROUTES="${MIN_ROUTES:-10}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

echo "[leadmot-ab] mode=${MODE}"
echo "[leadmot-ab] AutoMoT root=${AUTOMOT_ROOT}"
echo "[leadmot-ab] model=${MODEL_DIR}"
echo "[leadmot-ab] adapter=${ADAPTER_DIR}"
echo "[leadmot-ab] data_root=${DATA_ROOT}"
echo "[leadmot-ab] train_jsonl=${TRAIN_JSONL}"
echo "[leadmot-ab] val_jsonl=${VAL_JSONL}"
echo "[leadmot-ab] output=${AB_ROOT}"

prepare_leadmot_dataset() {
  if [[ -f "${TRAIN_JSONL}" && -f "${VAL_JSONL}" ]]; then
    echo "[leadmot-ab] reuse existing LeadMoT train/val JSONL"
    return 0
  fi
  if [[ -f "${TRAIN_JSONL}" || -f "${VAL_JSONL}" ]]; then
    echo "[leadmot-ab][error] partial LeadMoT dataset: train/val must either both exist or both be absent." >&2
    echo "  train=${TRAIN_JSONL}" >&2
    echo "  val=${VAL_JSONL}" >&2
    return 1
  fi
  if [[ "${MODE}" == "eval" ]]; then
    echo "[leadmot-ab][error] eval mode requires the original train/val JSONL; refusing to rebuild a new split." >&2
    return 1
  fi
  if [[ "${AUTO_BUILD_LEADMOT_DATASET}" != "1" ]]; then
    echo "[leadmot-ab][error] missing LeadMoT train/val JSONL and AUTO_BUILD_LEADMOT_DATASET=${AUTO_BUILD_LEADMOT_DATASET}." >&2
    return 1
  fi
  if [[ "$(dirname "${TRAIN_JSONL}")" != "$(dirname "${VAL_JSONL}")" \
        || "$(basename "${TRAIN_JSONL}")" != "train.jsonl" \
        || "$(basename "${VAL_JSONL}")" != "val.jsonl" ]]; then
    echo "[leadmot-ab][error] automatic build requires sibling train.jsonl/val.jsonl paths." >&2
    return 1
  fi
  if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[leadmot-ab][error] missing LEAD data root: ${DATA_ROOT}" >&2
    echo "Set DATA_ROOT to the LEAD route root (normally AutoMoT/lead_data)." >&2
    return 1
  fi

  local output_dir
  output_dir="$(dirname "${TRAIN_JSONL}")"
  echo "[leadmot-ab] LeadMoT JSONL missing; building once from ${DATA_ROOT} -> ${output_dir}"
  # 当前 A/B 必须可进入 CARLA，因此固定 USE_SUBGOAL=0；索引无需依赖 keyframes，
  # 但仍由 build_dataset.py 统一执行异常时长 route 剔除和 route-level train/val split。
  python qwen3vl_local/leadmot/build_dataset.py \
    --data-root "${DATA_ROOT}" \
    --output-dir "${output_dir}" \
    --no-with-subgoal-fields \
    --samples-per-scenario 0 \
    --stride "${DATASET_STRIDE}" \
    --seed "${DATASET_SEED}"
}

if [[ "${USE_SUBGOAL:-0}" != "0" ]]; then
  echo "[leadmot-ab][error] this closing A/B requires USE_SUBGOAL=0 so its winner can enter CARLA." >&2
  exit 1
fi
prepare_leadmot_dataset

# 这一步只检查文件与配置，不加载 GPU 模型。Phase2 v3 / 2RGB / seed10 是当前
# RGB 审计后的唯一研究候选；不允许脚本静默换成另一个 adapter。
python - "${MODEL_DIR}" "${ADAPTER_DIR}" "${TRAIN_JSONL}" "${VAL_JSONL}" <<'PY'
import json
import sys
from pathlib import Path

from qwen3vl_local.leadmot.config import build_qwen_backbone_contract

model_dir, adapter_dir, train_jsonl, val_jsonl = map(Path, sys.argv[1:])

def inspect_jsonl(path):
    identities = set()
    routes = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            route_dir = str(row.get("route_dir") or "")
            if not route_dir or "anchor" not in row:
                raise SystemExit(f"invalid LeadMoT row without route_dir/anchor: {path}:{line_no}")
            identity = (route_dir, int(row["anchor"]))
            if identity in identities:
                raise SystemExit(f"duplicate LeadMoT case identity: {path}:{line_no}: {identity}")
            identities.add(identity)
            routes.add(route_dir)
    if not identities:
        raise SystemExit(f"empty LeadMoT JSONL: {path}")
    return identities, routes

train_ids, train_routes = inspect_jsonl(train_jsonl)
val_ids, val_routes = inspect_jsonl(val_jsonl)
route_overlap = sorted(train_routes & val_routes)
if route_overlap:
    raise SystemExit(f"LeadMoT train/val route leakage: {route_overlap[:5]}")
contract = build_qwen_backbone_contract(model_dir, adapter_dir)
meta = contract.get("adapter_metadata") or {}
expected = {
    "config_file": "sft_new_loop_phase2_adapter_config.json",
    "schema": "sft_new_loop_phase2_adapter_config",
    "route": "sft_new_loop_phase2_direct_event",
    "dataset_name": "sft_new_loop_phase2_direct_event",
    "prompt_name": "sft_new_loop_phase2_direct_event_visual_v3",
    "production_prompt_sha256": "cd564634257fe0f072de70947200a820d6dd2b43375981b60120a1fe2296dd7f",
    "history_rgb_mode": "2rgb_endpoints",
    "history_rgb_count": 2,
    "history_rgb_selected_indices": [0, 3],
    "global_step": 4000,
    "seed": 20260810,
}
mismatch = {
    key: {"expected": value, "actual": meta.get(key)}
    for key, value in expected.items()
    if meta.get(key) != value
}
if mismatch:
    raise SystemExit("Phase2 candidate contract mismatch: " + json.dumps(mismatch, ensure_ascii=False))
print(json.dumps({
    "preflight": "ok",
    "train_cases": len(train_ids),
    "train_routes": len(train_routes),
    "val_cases": len(val_ids),
    "val_routes": len(val_routes),
    "base_config_sha256": contract["base_config_sha256"],
    "adapter_sha256": contract["adapter_sha256"],
    "adapter_metadata": meta,
}, ensure_ascii=False, indent=2))
PY

if [[ "${MODE}" == "preflight" ]]; then
  echo "[leadmot-ab] preflight passed; no GPU model loaded and no training started."
  exit 0
fi

mkdir -p "${AB_ROOT}"
python - "${AB_ROOT}" "${MODEL_DIR}" "${ADAPTER_DIR}" "${TRAIN_JSONL}" "${VAL_JSONL}" "${SEED}" <<'PY'
import json, sys, time
from pathlib import Path
root, model, adapter, train, val, seed = sys.argv[1:]
path = Path(root) / "ab_manifest.json"
path.write_text(json.dumps({
    "format": "leadmot_qwen_adapter_ab_manifest_v1",
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "model_dir": model,
    "adapter_dir": adapter,
    "train_jsonl": train,
    "val_jsonl": val,
    "seed": int(seed),
    "arms": {"base": {"qwen_adapter": ""}, "lora": {"qwen_adapter": adapter}},
}, ensure_ascii=False, indent=2), encoding="utf-8")
PY

run_train_arm() {
  local arm="$1"
  local adapter="$2"
  local launch_mode="$3"
  local arm_root="${AB_ROOT}/${arm}"
  echo "[leadmot-ab] training arm=${arm} adapter=${adapter:-<base>} mode=${launch_mode}"
  NO_RUN_SUBDIR=1 \
  OUTPUT_DIR="${arm_root}" \
  TRAIN_JSONL="${TRAIN_JSONL}" \
  VAL_JSONL="${VAL_JSONL}" \
  MODEL_DIR="${MODEL_DIR}" \
  QWEN_ADAPTER_DIR="${adapter}" \
  SEED="${SEED}" \
  USE_BEV="${USE_BEV}" \
  USE_SUBGOAL=0 \
  bash qwen3vl_local/leadmot/train.sh "${launch_mode}"
}

resolve_checkpoint() {
  local arm_root="$1"
  if [[ -f "${arm_root}/best.pt" ]]; then
    echo "${arm_root}/best.pt"
  elif [[ -f "${arm_root}/latest.pt" ]]; then
    echo "${arm_root}/latest.pt"
  else
    echo "missing LeadMoT checkpoint under ${arm_root}" >&2
    return 1
  fi
}

run_eval_arm() {
  local arm="$1"
  local max_samples="$2"
  local arm_root="${AB_ROOT}/${arm}"
  local checkpoint
  checkpoint="$(resolve_checkpoint "${arm_root}")"
  local args=(
    --jsonl "${VAL_JSONL}"
    --checkpoint "${checkpoint}"
    --output-dir "${arm_root}/eval"
    --model-dir "${MODEL_DIR}"
    --qwen-adapter-dir auto
    --seed "${SEED}"
  )
  if [[ "${max_samples}" -gt 0 ]]; then
    args+=(--max-samples "${max_samples}")
  fi
  echo "[leadmot-ab] evaluating arm=${arm} checkpoint=${checkpoint} max_samples=${max_samples}"
  if [[ -n "${GPU_IDS:-}" && "${GPU_IDS}" == *,* ]]; then
    local nproc
    nproc="$(awk -F, '{print NF}' <<< "${GPU_IDS}")"
    GPU_IDS="${GPU_IDS}" torchrun --standalone --nproc_per_node="${nproc}" \
      qwen3vl_local/leadmot/eval.py "${args[@]}"
  elif [[ -n "${GPU_IDS:-}" ]]; then
    GPU_IDS="${GPU_IDS}" python qwen3vl_local/leadmot/eval.py "${args[@]}"
  else
    python qwen3vl_local/leadmot/eval.py "${args[@]}"
  fi
}

compare_arms() {
  python qwen3vl_local/sft_new_loop_phase2/compare_leadmot_qwen_ab.py \
    --base "${AB_ROOT}/base/eval/eval_v1_perline.jsonl" \
    --lora "${AB_ROOT}/lora/eval/eval_v1_perline.jsonl" \
    --output-json "${AB_ROOT}/comparison.json" \
    --output-md "${AB_ROOT}/comparison.md" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
    --min-routes "${MIN_ROUTES}"
}

case "${MODE}" in
  smoke)
    # 两臂各跑 2 个 optimizer step，再对同 8 个 val case 做链路等价性检查。
    # smoke 只确认加载/保存/自动恢复/配对统计，不用于判断 LoRA 是否更好。
    LIMIT_TRAIN_SAMPLES=2 MAX_TRAIN_STEPS=2 NUM_WORKERS=0 \
      run_train_arm base "" check
    LIMIT_TRAIN_SAMPLES=2 MAX_TRAIN_STEPS=2 NUM_WORKERS=0 \
      run_train_arm lora "${ADAPTER_DIR}" check
    run_eval_arm base 8
    run_eval_arm lora 8
    compare_arms
    ;;
  train)
    run_train_arm base "" "${TRAIN_LAUNCH_MODE}"
    run_train_arm lora "${ADAPTER_DIR}" "${TRAIN_LAUNCH_MODE}"
    run_eval_arm base 0
    run_eval_arm lora 0
    compare_arms
    ;;
  eval)
    run_eval_arm base 0
    run_eval_arm lora 0
    compare_arms
    ;;
  *)
    echo "Usage: $0 [preflight|smoke|train|eval]" >&2
    exit 2
    ;;
esac

echo "[leadmot-ab] done: ${AB_ROOT}"
