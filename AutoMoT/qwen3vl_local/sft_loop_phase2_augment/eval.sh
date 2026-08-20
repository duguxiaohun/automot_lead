#!/usr/bin/env bash
# Phase2 augment 一键评测：base production/audit -> LoRA production/audit -> error audit -> <=30MB 审计包。
#
# 从 AutoMoT/ 目录运行：
#   ADAPTER_DIR=checkpoints/sft_loop_phase2_augment_runs/latest/best_generation bash qwen3vl_local/sft_loop_phase2_augment/eval.sh
# 或：
#   bash qwen3vl_local/sft_loop_phase2_augment/eval.sh checkpoints/sft_loop_phase2_augment_runs/latest/best_generation

set -euo pipefail

ulimit -S -c 0 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

PHASE_NAME="sft_loop_phase2_augment"
EVAL_PY="qwen3vl_local/sft_loop_phase2_augment/eval.py"
AUDIT_PY="qwen3vl_local/sft_loop_phase2_augment/audit_eval_cases.py"
VISUAL_AUDIT_PY="qwen3vl_local/sft_loop_phase2_augment/visual_audit.py"
ADAPTER_CONFIG_NAME="sft_loop_phase2_augment_adapter_config.json"

MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
INDEX="${INDEX:-checkpoints/sft_loop_phase2_augment_data/frame_index.jsonl}"
DATA_ROOT="${DATA_ROOT:-lead_data}"
REQUESTED_HISTORY_RGB_MODE="${HISTORY_RGB_MODE:-}"
SPLIT="${SPLIT:-test}"
CASES_PER_BIN="${CASES_PER_BIN:-64}"
MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
AUDIT_PER_TARGET="${AUDIT_PER_TARGET:-8}"
RUN_VISUAL_AUDIT="${RUN_VISUAL_AUDIT:-1}"
RUN_AUDIT_PROMPT_EVAL="${RUN_AUDIT_PROMPT_EVAL:-1}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/sft_loop_phase2_augment_eval_review/${TIMESTAMP}}"
BUNDLE_MAX_MB="${BUNDLE_MAX_MB:-30}"
ADAPTER_INPUT="${ADAPTER_DIR:-${CKPT_DIR:-${1:-}}}"

if [[ -z "${ADAPTER_INPUT}" ]]; then
  echo "Usage: ADAPTER_DIR=<lora-adapter-or-run-dir> bash qwen3vl_local/sft_loop_phase2_augment/eval.sh" >&2
  exit 2
fi

resolve_adapter_dir() {
  local input="$1"
  local candidate
  for candidate in "${input}/best_generation" "${input}/best_val" "${input}/final" "${input}"; do
    if [[ -f "${candidate}/${ADAPTER_CONFIG_NAME}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "Cannot resolve ${PHASE_NAME} adapter from ${input}; expected ${ADAPTER_CONFIG_NAME} in the adapter dir or best_generation/best_val/final." >&2
  return 1
}

ADAPTER_DIR="$(resolve_adapter_dir "${ADAPTER_INPUT}")"
if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "Adapter dir not found: ${ADAPTER_DIR}" >&2
  exit 2
fi

read_adapter_history_rgb_mode() {
  local adapter_dir="$1"
  python - "${adapter_dir}" <<'PY'
import json, pathlib, sys
adapter = pathlib.Path(sys.argv[1])
for name in ("sft_loop_phase2_augment_adapter_config.json", "sft_loop_phase2_adapter_config.json", "sft_loop_phase1_adapter_config.json"):
    path = adapter / name
    if path.is_file():
        print(str(json.loads(path.read_text(encoding="utf-8")).get("history_rgb_mode", "4rgb")))
        break
else:
    print("4rgb")
PY
}

BASE_HISTORY_RGB_MODE="${REQUESTED_HISTORY_RGB_MODE:-$(read_adapter_history_rgb_mode "${ADAPTER_DIR}")}"
case "${BASE_HISTORY_RGB_MODE}" in
  4rgb|2rgb_endpoints) ;;
  *)
    echo "Unknown BASE_HISTORY_RGB_MODE=${BASE_HISTORY_RGB_MODE}. Use 4rgb or 2rgb_endpoints." >&2
    exit 2
    ;;
esac

GPU_IDS="${GPU_IDS:-0,1,2,3}"
export GPU_IDS
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
NPROC="$(awk -F',' '{print NF}' <<< "${GPU_IDS}")"
if [[ "${NPROC}" -ne 4 ]]; then
  echo "eval.sh defaults to four-card evaluation; set exactly four GPU ids, got GPU_IDS=${GPU_IDS}" >&2
  exit 2
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

mkdir -p "${OUTPUT_ROOT}"
LOG_PATH="${OUTPUT_ROOT}/eval.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

find_free_master_port() {
  python -c 'import socket
s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null || echo "$((20000 + RANDOM % 20000))"
}

run_eval() {
  local title="$1"
  shift
  echo
  echo "========== ${title} =========="
  torchrun --nproc_per_node=4 \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="$(find_free_master_port)" \
    "${EVAL_PY}" "$@"
}

build_bundle() {
  PHASE_NAME="${PHASE_NAME}" OUTPUT_ROOT="${OUTPUT_ROOT}" ADAPTER_DIR="${ADAPTER_DIR}" \
  BUNDLE_MAX_MB="${BUNDLE_MAX_MB}" python - <<'PY'
import json, os, pathlib, shutil, tarfile

root = pathlib.Path(os.environ["OUTPUT_ROOT"])
phase = os.environ["PHASE_NAME"]
adapter_dir = os.environ["ADAPTER_DIR"]
adapter_path = pathlib.Path(adapter_dir)
limit_bytes = int(float(os.environ.get("BUNDLE_MAX_MB", "30")) * 1024 * 1024)
archive = root / f"{phase}_audit_bundle.tar.gz"
bundle = root / "audit_bundle"
text_suffixes = {".json", ".jsonl", ".md", ".txt", ".log", ".csv"}
weight_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}

try:
    from PIL import Image
except Exception:
    Image = None

def copy_text(src: pathlib.Path, dst: pathlib.Path, max_jsonl_lines: int = 500) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".jsonl":
        with src.open("r", encoding="utf-8", errors="replace") as f, dst.open("w", encoding="utf-8") as g:
            for idx, line in enumerate(f):
                if idx >= max_jsonl_lines:
                    g.write(json.dumps({"truncated_after_lines": max_jsonl_lines, "source": str(src)}, ensure_ascii=False) + "\n")
                    break
                g.write(line)
        return
    if src.stat().st_size > 8 * 1024 * 1024:
        dst.write_text(f"Skipped large text file: {src} size={src.stat().st_size}\n", encoding="utf-8")
        return
    shutil.copy2(src, dst)

def copy_image(src: pathlib.Path, dst: pathlib.Path, max_side: int, quality: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Image is None:
        shutil.copy2(src, dst)
        return
    with Image.open(src) as im:
        im = im.convert("RGB")
        scale = min(1.0, float(max_side) / float(max(im.size)))
        if scale < 1.0:
            im = im.resize((max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale))))
        im.save(dst.with_suffix(".jpg"), quality=quality, optimize=True)

def copy_adapter_metadata() -> list[str]:
    copied: list[str] = []
    sources: list[tuple[pathlib.Path, str, bool]] = []
    if adapter_path.exists():
        sources.append((adapter_path, "adapter", True))
        if adapter_path.parent.exists():
            sources.append((adapter_path.parent, "run_root", False))
    for base, label, recursive in sources:
        iterator = base.rglob("*") if recursive else base.iterdir()
        for src in sorted(iterator):
            if not src.is_file():
                continue
            if src.suffix.lower() in weight_suffixes:
                continue
            if any(part in {"tb", "__pycache__"} or part.startswith("checkpoint-") for part in src.relative_to(base).parts):
                continue
            if src.suffix.lower() not in text_suffixes:
                continue
            rel = pathlib.Path("adapter_metadata") / label / src.relative_to(base)
            copy_text(src, bundle / rel)
            copied.append(str(rel))
    return copied

def selected_case_dirs(case_limit: int, audit_limit: int) -> list[pathlib.Path]:
    selected: list[pathlib.Path] = []
    for eval_name in ("base_production", "base_audit_prompt", "lora_production", "lora_audit_prompt"):
        err = root / eval_name / "error_cases"
        if err.is_dir():
            by_group: dict[str, list[pathlib.Path]] = {}
            for case in sorted([p for p in err.rglob("case_*") if p.is_dir()]):
                group = case.parent.relative_to(err).as_posix()
                by_group.setdefault(group, []).append(case)
            for group in sorted(by_group):
                selected.extend(by_group[group][:case_limit])
    for audit_root in sorted(root.glob("audit_*")):
        by_target: dict[str, list[pathlib.Path]] = {}
        for case in sorted([p for p in audit_root.rglob("case_*") if p.is_dir()]):
            by_target.setdefault(case.parent.name, []).append(case)
        for cases in by_target.values():
            selected.extend(cases[:audit_limit])
    return selected

def build_attempt(case_limit: int, audit_limit: int, max_side: int, quality: int) -> int:
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    manifest = {
        "phase": phase,
        "source_root": str(root),
        "adapter_dir": adapter_dir,
        "case_limit_per_eval": case_limit,
        "audit_limit_per_target": audit_limit,
        "image_max_side": max_side,
        "image_quality": quality,
        "error_case_limit_per_group": case_limit,
        "jsonl_copy_policy": "copy at most 500 lines per jsonl and append a truncation marker; metrics.json/report.md remain the full-run source of truth",
        "adapter_metadata_policy": "copy adapter/run-root text metadata only; exclude weights, checkpoints, TensorBoard, and binary artifacts",
        "bundle_contract": "metrics/reports/case jsonl plus sampled downscaled error RGB; designed for prompt/code audit under 30MB",
    }
    (bundle / "BUNDLE_README.md").write_text(
        f"# {phase} eval audit bundle\n\n"
        f"- source_root: `{root}`\n"
        f"- adapter_dir: `{adapter_dir}`\n"
        "- metrics.json/report.md are full-run summaries.\n"
        "- copied .jsonl files are capped at 500 lines with a truncation marker.\n",
        encoding="utf-8",
    )
    for src in root.rglob("*"):
        if not src.is_file() or bundle in src.parents or src == archive:
            continue
        if src.suffix.lower() in text_suffixes and "/rgb/" not in src.as_posix():
            copy_text(src, bundle / src.relative_to(root))
    manifest["adapter_metadata_files"] = copy_adapter_metadata()
    cases = selected_case_dirs(case_limit, audit_limit)
    manifest["selected_rgb_case_dirs"] = [str(p.relative_to(root)) for p in cases if p.exists()]
    for case in cases:
        for img in sorted((case / "rgb").glob("*")):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                copy_image(img, bundle / img.relative_to(root), max_side=max_side, quality=quality)
    (bundle / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(bundle, arcname=bundle.name)
    return archive.stat().st_size

for attempt in [(24, 4, 768, 60), (16, 3, 640, 55), (10, 2, 560, 50), (6, 1, 448, 45), (3, 1, 384, 40)]:
    size = build_attempt(*attempt)
    if size <= limit_bytes:
        print(json.dumps({"archive": str(archive), "bundle_dir": str(bundle), "bytes": size, "max_bytes": limit_bytes}, ensure_ascii=False, indent=2))
        break
else:
    raise SystemExit("could not build a <= limit audit bundle")
PY
}

echo "[eval] phase=${PHASE_NAME}"
echo "[eval] output_root=${OUTPUT_ROOT}"
echo "[eval] adapter_dir=${ADAPTER_DIR}"
echo "[eval] base_history_rgb_mode=${BASE_HISTORY_RGB_MODE}"
echo "[eval] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

BASE_EVAL_DIR="${OUTPUT_ROOT}/base_production"
BASE_AUDIT_EVAL_DIR="${OUTPUT_ROOT}/base_audit_prompt"
LORA_EVAL_DIR="${OUTPUT_ROOT}/lora_production"
LORA_AUDIT_EVAL_DIR="${OUTPUT_ROOT}/lora_audit_prompt"

run_eval "base production" \
  --model-dir "${MODEL_DIR}" --index "${INDEX}" --split "${SPLIT}" \
  --history-rgb-mode "${BASE_HISTORY_RGB_MODE}" --cases-per-bin "${CASES_PER_BIN}" \
  --max-frames "${MAX_EVAL_FRAMES}" --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-dir "${BASE_EVAL_DIR}" --no-timestamp-output --overwrite --save-error-rgb --no-save-all-rgb

if [[ "${RUN_AUDIT_PROMPT_EVAL}" == "1" ]]; then
  run_eval "base audit-prompt" \
    --model-dir "${MODEL_DIR}" --index "${INDEX}" --split "${SPLIT}" \
    --history-rgb-mode "${BASE_HISTORY_RGB_MODE}" --cases-per-bin "${CASES_PER_BIN}" \
    --max-frames "${MAX_EVAL_FRAMES}" --max-new-tokens "${MAX_NEW_TOKENS}" \
    --audit-prompt --output-dir "${BASE_AUDIT_EVAL_DIR}" --no-timestamp-output --overwrite --save-error-rgb --no-save-all-rgb
fi

run_eval "LoRA production" \
  --model-dir "${MODEL_DIR}" --index "${INDEX}" --split "${SPLIT}" \
  --adapter-dir "${ADAPTER_DIR}" --cases-per-bin "${CASES_PER_BIN}" \
  --max-frames "${MAX_EVAL_FRAMES}" --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-dir "${LORA_EVAL_DIR}" --no-timestamp-output --overwrite --save-error-rgb --no-save-all-rgb

if [[ "${RUN_AUDIT_PROMPT_EVAL}" == "1" ]]; then
  run_eval "LoRA audit-prompt" \
    --model-dir "${MODEL_DIR}" --index "${INDEX}" --split "${SPLIT}" \
    --adapter-dir "${ADAPTER_DIR}" --cases-per-bin "${CASES_PER_BIN}" \
    --max-frames "${MAX_EVAL_FRAMES}" --max-new-tokens "${MAX_NEW_TOKENS}" \
    --audit-prompt --output-dir "${LORA_AUDIT_EVAL_DIR}" --no-timestamp-output --overwrite --save-error-rgb --no-save-all-rgb
fi

python "${AUDIT_PY}" --eval-dir "${BASE_EVAL_DIR}" --output-dir "${OUTPUT_ROOT}/audit_base_production" \
  --data-root "${DATA_ROOT}" --per-target "${AUDIT_PER_TARGET}" --overwrite
python "${AUDIT_PY}" --eval-dir "${LORA_EVAL_DIR}" --output-dir "${OUTPUT_ROOT}/audit_lora_production" \
  --data-root "${DATA_ROOT}" --per-target "${AUDIT_PER_TARGET}" --overwrite

if [[ "${RUN_VISUAL_AUDIT}" == "1" ]]; then
  python "${VISUAL_AUDIT_PY}" --output "${OUTPUT_ROOT}/visual_audit_manifest.json"
fi

build_bundle

echo
echo "[done] eval root: ${OUTPUT_ROOT}"
echo "[done] audit bundle: ${OUTPUT_ROOT}/${PHASE_NAME}_audit_bundle.tar.gz"
