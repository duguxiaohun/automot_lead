#!/usr/bin/env bash
# SFT baseline 诊断评估脚本：集中跑 full/road/event、黑图消融和 ROAD/EVENT bias sweep。
#
# 从远端 AutoMoT/ 目录运行：
#   GPU_IDS=0 bash qwen3vl_local/sft_baseline/run_triage_eval.sh
#   GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_baseline/run_triage_eval.sh
# 可覆盖：
#   CKPT=checkpoints/sft_baseline_runs/run_xxx/final
#   OUT_ROOT=checkpoints/sft_baseline_runs/run_xxx/triage_eval
#   SAMPLE_ROUTES=60
#   TRANSITION_CASES=128
#   SEED=20260804

set -euo pipefail

# 禁用 core dump，避免失败的评估进程在仓库里留下 core.* 大文件。
ulimit -S -c 0 2>/dev/null || true

CKPT="${CKPT:-checkpoints/sft_baseline_runs/run_v3_event_cooldown_probe3/final}"
RUN_DIR="$(dirname "${CKPT}")"
STAMP="${TRIAGE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${RUN_DIR}/triage_eval_${STAMP}}"
SAMPLE_ROUTES="${SAMPLE_ROUTES:-60}"
TRANSITION_CASES="${TRANSITION_CASES:-128}"
SEED="${SEED:-20260804}"
GPU_IDS="${GPU_IDS:-0}"
EVENT_BIASES="${EVENT_BIASES:-0 0.5 1 1.5 2 2.5 3}"
ROAD_BIASES="${ROAD_BIASES:--4 -3 -2 -1 0 1 2}"

count_gpus() {
  local visible="$1"
  if [[ -z "${visible}" ]]; then
    echo "0"
    return
  fi
  awk -F',' '{print NF}' <<< "${visible}"
}

find_free_master_port() {
  python - <<'PY' 2>/dev/null || echo "$((20000 + RANDOM % 20000))"
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
}

GPU_COUNT="$(count_gpus "${GPU_IDS}")"

mkdir -p "${OUT_ROOT}"
exec > >(tee -a "${OUT_ROOT}/run.log") 2>&1

echo "[triage] ckpt=${CKPT}"
echo "[triage] out=${OUT_ROOT}"
echo "[triage] sample_routes=${SAMPLE_ROUTES} transition_cases=${TRANSITION_CASES} seed=${SEED}"
echo "[triage] gpu_ids=${GPU_IDS} gpu_count=${GPU_COUNT}"

run_eval() {
  local name="$1"
  shift
  local out_dir="${OUT_ROOT}/${name}"
  echo "[triage] >>> ${name}"
  local common_args=(
    qwen3vl_local/sft_baseline/eval.py
    --adapter-dir "${CKPT}"
    --seed "${SEED}"
    --output-dir "${out_dir}"
    "$@"
  )
  if [[ "${GPU_COUNT}" -gt 1 ]]; then
    local master_port
    master_port="$(find_free_master_port)"
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" GPU_IDS="${GPU_IDS}" MASTER_PORT="${master_port}" \
      torchrun --nproc_per_node "${GPU_COUNT}" "${common_args[@]}"
  else
    GPU_IDS="${GPU_IDS}" python "${common_args[@]}"
  fi
}

run_eval "full_60" \
  --task full \
  --sample-routes "${SAMPLE_ROUTES}"

run_eval "road_transition_128" \
  --task road \
  --max-transition-cases "${TRANSITION_CASES}"

run_eval "event_transition_128" \
  --task event \
  --max-transition-cases "${TRANSITION_CASES}"

run_eval "full_60_black_nogoal" \
  --task full \
  --sample-routes "${SAMPLE_ROUTES}" \
  --image-ablation black \
  --ablate-goal

for bias in ${EVENT_BIASES}; do
  safe_bias="${bias//-/_neg_}"
  safe_bias="${safe_bias//./p}"
  run_eval "event_score_bias_${safe_bias}" \
    --task event \
    --prediction-mode score \
    --event-logit-bias "${bias}" \
    --max-transition-cases "${TRANSITION_CASES}" \
    --no-write-frames
done

for bias in ${ROAD_BIASES}; do
  safe_bias="${bias//-/_neg_}"
  safe_bias="${safe_bias//./p}"
  run_eval "road_score_bias_${safe_bias}" \
    --task road \
    --prediction-mode score \
    --road-logit-bias "${bias}" \
    --max-transition-cases "${TRANSITION_CASES}" \
    --no-write-frames
done

python - "${OUT_ROOT}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
keys = [
    "name",
    "task",
    "eval_mode",
    "frames",
    "road_acc",
    "highway_precision",
    "highway_recall",
    "highway_f1",
    "event_acc",
    "ue_precision",
    "ue_recall",
    "ue_f1",
    "joint_acc",
    "road_change_f1",
    "event_change_f1",
    "transition_hit_rate",
    "transition_post_acc",
    "prediction_mode",
    "road_logit_bias",
    "event_logit_bias",
    "image_ablation",
    "goal_ablation",
]
for metrics_path in sorted(root.glob("*/metrics.json")):
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    row = {"name": metrics_path.parent.name}
    for key in keys[1:]:
        row[key] = metrics.get(key)
    rows.append(row)

csv_path = root / "triage_summary.csv"
with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)

md_path = root / "triage_summary.md"
with md_path.open("w", encoding="utf-8") as f:
    f.write("# SFT Baseline Triage Summary\n\n")
    f.write(f"- Output: `{root}`\n")
    f.write(f"- Runs: `{len(rows)}`\n\n")
    f.write("| name | road_acc | highway_f1 | event_acc | ue_precision | ue_recall | ue_f1 | event_change_f1 |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        def pct(key):
            value = row.get(key)
            return "NA" if value is None else f"{float(value) * 100:.2f}%"
        f.write(
            f"| {row['name']} | {pct('road_acc')} | {pct('highway_f1')} | "
            f"{pct('event_acc')} | {pct('ue_precision')} | {pct('ue_recall')} | "
            f"{pct('ue_f1')} | {pct('event_change_f1')} |\n"
        )

print(f"[triage] wrote {csv_path}")
print(f"[triage] wrote {md_path}")
PY

echo "[triage] done"
echo "[triage] package this directory for review:"
echo "tar -czf /tmp/$(basename "${OUT_ROOT}").tgz -C \"$(dirname "${OUT_ROOT}")\" \"$(basename "${OUT_ROOT}")\""
