#!/usr/bin/env bash
set -euo pipefail

# 禁用 core dump，避免模型加载异常时生成巨大 core.* 文件。
ulimit -S -c 0 2>/dev/null || true

MODE="${1:-preflight}"  # preflight | smoke | train | eval
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-checkpoints/sft_new_loop_phase2_frozen_protocol/v3_frozen_3seed_unseen456_20260831/train_runs/seed_20260810/final}"
TRAIN_JSONL="${TRAIN_JSONL:-checkpoints/leadmot_v1_data/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-checkpoints/leadmot_v1_data/val.jsonl}"
AB_ID="${AB_ID:-v3_seed20260810_$(date +%Y%m%d_%H%M%S)}"
AB_ROOT="${AB_ROOT:-checkpoints/leadmot_qwen_adapter_ab/${AB_ID}}"
TRAIN_LAUNCH_MODE="${TRAIN_LAUNCH_MODE:-ddp}"
SEED="${SEED:-2026}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
MIN_ROUTES="${MIN_ROUTES:-10}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

echo "[leadmot-ab] mode=${MODE}"
echo "[leadmot-ab] AutoMoT root=${AUTOMOT_ROOT}"
echo "[leadmot-ab] model=${MODEL_DIR}"
echo "[leadmot-ab] adapter=${ADAPTER_DIR}"
echo "[leadmot-ab] output=${AB_ROOT}"

# 这一步只检查文件与配置，不加载 GPU 模型。Phase2 v3 / 2RGB / seed10 是当前
# RGB 审计后的唯一研究候选；不允许脚本静默换成另一个 adapter。
python - "${MODEL_DIR}" "${ADAPTER_DIR}" "${TRAIN_JSONL}" "${VAL_JSONL}" <<'PY'
import json
import sys
from pathlib import Path

from qwen3vl_local.leadmot.config import build_qwen_backbone_contract

model_dir, adapter_dir, train_jsonl, val_jsonl = map(Path, sys.argv[1:])
for path, kind in ((train_jsonl, "train jsonl"), (val_jsonl, "val jsonl")):
    if not path.is_file():
        raise SystemExit(f"missing {kind}: {path}")
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
