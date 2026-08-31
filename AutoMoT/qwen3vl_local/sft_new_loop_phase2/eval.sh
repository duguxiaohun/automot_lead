#!/usr/bin/env bash
# 新 Phase2 一键评测：base production/audit -> LoRA production/audit -> error audit -> <=30MB 审计包。
#
# 从 AutoMoT/ 目录运行：
#   ADAPTER_DIR=checkpoints/sft_new_loop_phase2_runs/latest/best_generation bash qwen3vl_local/sft_new_loop_phase2/eval.sh
# 或：
#   bash qwen3vl_local/sft_new_loop_phase2/eval.sh checkpoints/sft_new_loop_phase2_runs/latest/best_generation

set -euo pipefail

ulimit -S -c 0 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AUTOMOT_ROOT}"

PHASE_NAME="sft_new_loop_phase2"
EVAL_PY="qwen3vl_local/sft_new_loop_phase2/eval.py"
AUDIT_PY="qwen3vl_local/sft_new_loop_phase2/audit_eval_cases.py"
VISUAL_AUDIT_PY="qwen3vl_local/sft_new_loop_phase2/visual_audit.py"
ADAPTER_CONFIG_NAME="sft_new_loop_phase2_adapter_config.json"

MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
INDEX="${INDEX:-checkpoints/sft_new_loop_phase2_data/frame_index.jsonl}"
DATA_ROOT="${DATA_ROOT:-lead_data}"
SPLIT="${SPLIT:-test}"
CASES_PER_BIN="${CASES_PER_BIN:-64}"
EXCLUDE_CASES_JSONL="${EXCLUDE_CASES_JSONL:-}"
EXPECTED_EXCLUDED_CASES="${EXPECTED_EXCLUDED_CASES:-0}"
EXPECTED_TOTAL_CASES="${EXPECTED_TOTAL_CASES:-0}"
MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
AUDIT_PER_TARGET="${AUDIT_PER_TARGET:-8}"
RUN_VISUAL_AUDIT="${RUN_VISUAL_AUDIT:-1}"
SCAN_VISUAL_RISKS="${SCAN_VISUAL_RISKS:-1}"
RUN_AUDIT_PROMPT_EVAL="${RUN_AUDIT_PROMPT_EVAL:-1}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/sft_new_loop_phase2_eval_review/${TIMESTAMP}}"
BUNDLE_MAX_MB="${BUNDLE_MAX_MB:-30}"
REQUESTED_BUNDLE_BASENAME="${BUNDLE_BASENAME:-}"
ADAPTER_INPUT="${ADAPTER_DIR:-${CKPT_DIR:-${1:-}}}"

EXCLUSION_ARGS=()
if [[ -n "${EXCLUDE_CASES_JSONL}" ]]; then
  for exclusion_path in ${EXCLUDE_CASES_JSONL}; do
    EXCLUSION_ARGS+=(--exclude-cases-jsonl "${exclusion_path}")
  done
fi
EXCLUSION_ARGS+=(
  --expected-excluded-cases "${EXPECTED_EXCLUDED_CASES}"
  --expected-total-cases "${EXPECTED_TOTAL_CASES}"
)

if [[ -z "${ADAPTER_INPUT}" ]]; then
  echo "Usage: ADAPTER_DIR=<lora-adapter-or-run-dir> bash qwen3vl_local/sft_new_loop_phase2/eval.sh" >&2
  exit 2
fi

resolve_adapter_dir() {
  local input="$1"
  local candidate
  for candidate in "${input}/best_generation" "${input}"; do
    if [[ -f "${candidate}/${ADAPTER_CONFIG_NAME}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "Cannot resolve ${PHASE_NAME} adapter from ${input}; pass an exact adapter or a run containing production-ready best_generation." >&2
  echo "Pass best_val/final/fallback_generation explicitly only for diagnostic evaluation." >&2
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
path = adapter / "sft_new_loop_phase2_adapter_config.json"
if not path.is_file():
    raise SystemExit(f"missing adapter config: {path}")
config = json.loads(path.read_text(encoding="utf-8"))
if not config.get("history_rgb_mode"):
    raise SystemExit(f"adapter config has no history_rgb_mode: {path}")
print(str(config["history_rgb_mode"]))
PY
}

BASE_HISTORY_RGB_MODE="$(read_adapter_history_rgb_mode "${ADAPTER_DIR}")"
case "${BASE_HISTORY_RGB_MODE}" in
  4rgb|2rgb_endpoints) ;;
  *)
    echo "Unknown BASE_HISTORY_RGB_MODE=${BASE_HISTORY_RGB_MODE}. Use 4rgb or 2rgb_endpoints." >&2
    exit 2
    ;;
esac
BUNDLE_BASENAME="${REQUESTED_BUNDLE_BASENAME:-${PHASE_NAME}_${TIMESTAMP}_${BASE_HISTORY_RGB_MODE}_audit_bundle}"

pick_idle_gpus() {
  local want_count="$1"
  local selected
  if command -v nvidia-smi >/dev/null 2>&1; then
    selected="$(
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); print $2, $3, $1}' \
        | sort -n -k1,1 -k2,2 \
        | head -n "${want_count}" \
        | awk '{print $3}' \
        | paste -sd, -
    )"
    if [[ -n "${selected}" ]]; then echo "${selected}"; return 0; fi
  fi
  if [[ "${want_count}" -le 1 ]]; then echo "0"; else seq -s, 0 "$((want_count - 1))"; fi
}

EVAL_GPU_COUNT="${EVAL_GPU_COUNT:-4}"
if [[ -z "${GPU_IDS:-}" ]]; then
  GPU_SELECTION_SOURCE="automatic"
  GPU_IDS="$(pick_idle_gpus "${EVAL_GPU_COUNT}")"
else
  GPU_SELECTION_SOURCE="explicit"
fi
export GPU_IDS
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
NPROC="$(awk -F',' '{print NF}' <<< "${GPU_IDS}")"
if [[ "${GPU_SELECTION_SOURCE}" == "automatic" && "${NPROC}" -ne "${EVAL_GPU_COUNT}" ]]; then
  echo "automatic selection requested ${EVAL_GPU_COUNT} GPUs but found GPU_IDS=${GPU_IDS}" >&2
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
  torchrun --nproc_per_node="${NPROC}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="$(find_free_master_port)" \
    "${EVAL_PY}" "$@"
}

build_bundle() {
  PHASE_NAME="${PHASE_NAME}" OUTPUT_ROOT="${OUTPUT_ROOT}" ADAPTER_DIR="${ADAPTER_DIR}" \
  ADAPTER_INPUT="${ADAPTER_INPUT}" ADAPTER_CONFIG_NAME="${ADAPTER_CONFIG_NAME}" \
  BUNDLE_MAX_MB="${BUNDLE_MAX_MB}" BUNDLE_BASENAME="${BUNDLE_BASENAME}" \
  TIMESTAMP="${TIMESTAMP}" MODEL_DIR="${MODEL_DIR}" INDEX="${INDEX}" SPLIT="${SPLIT}" \
  HISTORY_RGB_MODE="${BASE_HISTORY_RGB_MODE}" EVAL_SCRIPT="${EVAL_PY}" \
  RUN_AUDIT_PROMPT_EVAL="${RUN_AUDIT_PROMPT_EVAL}" RUN_VISUAL_AUDIT="${RUN_VISUAL_AUDIT}" python - <<'PY'
import datetime, json, os, pathlib, shutil, subprocess, tarfile
from typing import Iterable

root = pathlib.Path(os.environ["OUTPUT_ROOT"])
phase = os.environ["PHASE_NAME"]
run_timestamp = os.environ["TIMESTAMP"]
adapter_dir = os.environ["ADAPTER_DIR"]
adapter_input = os.environ.get("ADAPTER_INPUT", "")
adapter_config_name = os.environ.get("ADAPTER_CONFIG_NAME", "")
adapter_path = pathlib.Path(adapter_dir)
limit_bytes = int(float(os.environ.get("BUNDLE_MAX_MB", "30")) * 1024 * 1024)
bundle_name = os.environ["BUNDLE_BASENAME"]
archive = root / f"{bundle_name}.tar.gz"
bundle = root / bundle_name
text_suffixes = {".json", ".jsonl", ".md", ".txt", ".log", ".csv"}
weight_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}

try:
    from PIL import Image
except Exception:
    Image = None

def _git(args: list[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=pathlib.Path.cwd(),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except Exception:
        return None

def git_metadata() -> dict:
    status = _git(["status", "--short"]) or ""
    return {
        "root": _git(["rev-parse", "--show-toplevel"]),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines()[:300],
    }

def read_json(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def validate_expected_artifacts() -> list[str]:
    """审计包必须覆盖 direct EVENT 评测真正要复核的结果。"""

    expected = [
        "base_production/metrics.json",
        "lora_production/metrics.json",
        "audit_base_production/summary.json",
        "audit_lora_production/summary.json",
    ]
    if os.environ.get("RUN_AUDIT_PROMPT_EVAL", "1") == "1":
        expected.extend([
            "base_audit_prompt/metrics.json",
            "lora_audit_prompt/metrics.json",
        ])
    if os.environ.get("RUN_VISUAL_AUDIT", "1") == "1":
        expected.append("visual_audit_manifest.json")
    missing = [rel for rel in expected if not (root / rel).is_file()]
    if missing:
        raise SystemExit(f"refuse incomplete new Phase2 audit bundle; missing={missing}")
    return expected

def adapter_identity() -> dict:
    cfg_path = adapter_path / adapter_config_name if adapter_config_name else None
    if cfg_path is None or not cfg_path.is_file():
        candidates = sorted(adapter_path.glob("*_adapter_config.json"))
        cfg_path = candidates[0] if candidates else None
    cfg = read_json(cfg_path) if cfg_path is not None and cfg_path.is_file() else {}
    weight_files = []
    if adapter_path.is_dir():
        weight_files = [
            str(path.relative_to(adapter_path))
            for path in sorted(adapter_path.iterdir())
            if path.is_file() and path.suffix.lower() in weight_suffixes
        ]
    weight_slot = adapter_path.name if adapter_path.name in {"best_generation", "best_val", "final"} else "direct_adapter_dir"
    run_root = adapter_path.parent if weight_slot != "direct_adapter_dir" else adapter_path
    history_mode = cfg.get("history_rgb_mode")
    default_indices = {"4rgb": [0, 1, 2, 3], "2rgb_endpoints": [0, 3]}.get(history_mode)
    selected_indices = cfg.get("history_rgb_selected_indices") or default_indices
    return {
        "input": adapter_input,
        "resolved_dir": adapter_dir,
        "run_root": str(run_root),
        "weight_slot": weight_slot,
        "weight_files": weight_files,
        "config_path": str(cfg_path) if cfg_path is not None else None,
        "config_schema": cfg.get("schema"),
        "prompt_name": cfg.get("prompt_name"),
        "production_prompt_sha256": cfg.get("production_prompt_sha256"),
        "global_step": cfg.get("global_step"),
        "base_model_dir": cfg.get("base_model_dir"),
        "history_rgb_mode": history_mode,
        "history_rgb_count": cfg.get("history_rgb_count") or (len(selected_indices) if selected_indices else None),
        "history_rgb_selected_indices": selected_indices,
    }

def eval_identity() -> dict:
    wanted = (
        "prompt_name",
        "prompt_mode",
        "production_prompt_sha256",
        "eval_prompt_sha256",
        "adapter_dir",
        "adapter_dir_resolve_source",
        "adapter_production_prompt_sha256",
        "adapter_prompt_matches_current_production",
        "history_rgb_mode",
        "exact_match_accuracy",
        "total_cases",
    )
    per_eval = {}
    for eval_name in ("base_production", "base_audit_prompt", "lora_production", "lora_audit_prompt"):
        metrics_path = root / eval_name / "metrics.json"
        if metrics_path.is_file():
            data = read_json(metrics_path)
            per_eval[eval_name] = {key: data.get(key) for key in wanted if key in data}
    prompt_names = sorted({str(item["prompt_name"]) for item in per_eval.values() if item.get("prompt_name")})
    production_hashes = sorted({str(item["production_prompt_sha256"]) for item in per_eval.values() if item.get("production_prompt_sha256")})
    expected_mode = os.environ.get("HISTORY_RGB_MODE", "")
    mismatched_modes = {
        name: item.get("history_rgb_mode")
        for name, item in per_eval.items()
        if item.get("history_rgb_mode") != expected_mode
    }
    if mismatched_modes:
        raise SystemExit(
            f"refuse mixed-RGB-mode Phase2 bundle: expected={expected_mode} got={mismatched_modes}"
        )
    return {
        "prompt_name": prompt_names[0] if len(prompt_names) == 1 else prompt_names,
        "production_prompt_sha256": production_hashes[0] if len(production_hashes) == 1 else production_hashes,
        "per_eval": per_eval,
    }

def bundle_identity() -> dict:
    adapter = adapter_identity()
    eval_meta = eval_identity()
    return {
        "phase": phase,
        "run_timestamp": run_timestamp,
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "bundle_name": bundle_name,
        "archive": str(archive),
        "source_root": str(root),
        "eval_script": os.environ.get("EVAL_SCRIPT", ""),
        "model_dir": os.environ.get("MODEL_DIR", ""),
        "index": os.environ.get("INDEX", ""),
        "split": os.environ.get("SPLIT", ""),
        "history_rgb_mode": os.environ.get("HISTORY_RGB_MODE", ""),
        "prompt_name": eval_meta.get("prompt_name") or adapter.get("prompt_name"),
        "production_prompt_sha256": eval_meta.get("production_prompt_sha256") or adapter.get("production_prompt_sha256"),
        "adapter": adapter,
        "eval": eval_meta,
        "git": git_metadata(),
    }

def skip_existing_bundle_artifact(src: pathlib.Path) -> bool:
    if not src.is_file() or bundle in src.parents or src == archive:
        return True
    try:
        first = src.relative_to(root).parts[0]
    except Exception:
        return False
    return first == "audit_bundle" or first.endswith("_audit_bundle") or src.name.endswith("_audit_bundle.tar.gz")

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

def copy_dataset_metadata() -> list[str]:
    """保留六类采样、RE/INVALID 与视觉风险统计，不复制大 frame index。"""

    copied: list[str] = []
    index_path = pathlib.Path(os.environ.get("INDEX", ""))
    for name in ("manifest.json", "visual_audit_manifest.json"):
        src = index_path.parent / name
        if not src.is_file():
            continue
        rel = pathlib.Path("dataset_metadata") / name
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
        **bundle_identity(),
        "validated_expected_files": validate_expected_artifacts(),
        "bundle_max_bytes": limit_bytes,
        "review_focus": [
            "production exact/format and base-to-LoRA delta",
            "UE1/UE3/UE5/UE6 per-question precision recall F1 and class exact",
            "applicable RE with highway R3 and local-junction hard-negative slices",
            "INVALID all-UE-NO joint guard plus source/true-RS/wrong-domain/joint-signature exact",
            "production versus audit-prompt consistency and strict evidence format",
            "sampled RGB errors with newest-frame and model-versus-label adjudication",
            "adapter prompt hash, base model, checkpoint step, and checkpoint-owned RGB mode",
        ],
        "case_limit_per_eval": case_limit,
        "audit_limit_per_target": audit_limit,
        "image_max_side": max_side,
        "image_quality": quality,
        "error_case_limit_per_group": case_limit,
        "adapter_metadata_policy": "copy adapter/run-root text metadata only; exclude weights, checkpoints, TensorBoard, and binary artifacts",
        "bundle_contract": "metrics/reports/case jsonl plus sampled downscaled error RGB; designed for prompt/code audit under 30MB",
    }
    (bundle / "BUNDLE_README.md").write_text(
        f"# {bundle_name}\n\n"
        f"- phase: `{phase}`\n"
        f"- test_time: `{run_timestamp}`\n"
        f"- source_root: `{root}`\n"
        f"- adapter_input: `{adapter_input}`\n"
        f"- adapter_dir: `{adapter_dir}`\n"
        f"- prompt_name: `{manifest.get('prompt_name')}`\n"
        f"- production_prompt_sha256: `{manifest.get('production_prompt_sha256')}`\n"
        f"- history_rgb_mode_from_ckpt: `{manifest.get('history_rgb_mode')}`\n"
        f"- history_rgb_selected_indices: `{manifest.get('adapter', {}).get('history_rgb_selected_indices')}`\n"
        f"- bundle_max_bytes: `{limit_bytes}`\n"
        f"- git_commit: `{manifest.get('git', {}).get('commit')}`\n"
        f"- image_max_side: `{max_side}`\n- image_quality: `{quality}`\n",
        encoding="utf-8",
    )
    for src in root.rglob("*"):
        if skip_existing_bundle_artifact(src):
            continue
        if src.suffix.lower() in text_suffixes and "/rgb/" not in src.as_posix():
            copy_text(src, bundle / src.relative_to(root))
    manifest["adapter_metadata_files"] = copy_adapter_metadata()
    manifest["dataset_metadata_files"] = copy_dataset_metadata()
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

attempts = [(24, 4, 768, 60), (16, 3, 640, 55), (10, 2, 560, 50), (6, 1, 448, 45)]
final_size = 0
for attempt in attempts:
    final_size = build_attempt(*attempt)
    if final_size <= limit_bytes:
        break
else:
    final_size = build_attempt(3, 1, 384, 40)
    if final_size > limit_bytes:
        raise SystemExit(f"bundle still exceeds limit: {final_size} > {limit_bytes}")
print(json.dumps({"archive": str(archive), "bundle_dir": str(bundle), "bytes": final_size, "max_bytes": limit_bytes}, ensure_ascii=False, indent=2))
PY
}

echo "[eval] phase=${PHASE_NAME}"
echo "[eval] output_root=${OUTPUT_ROOT}"
echo "[eval] adapter_dir=${ADAPTER_DIR}"
echo "[eval] history_rgb_mode=${BASE_HISTORY_RGB_MODE} (authoritative adapter config)"
echo "[eval] exclude_cases_jsonl=${EXCLUDE_CASES_JSONL:-none} expected_excluded=${EXPECTED_EXCLUDED_CASES} expected_total=${EXPECTED_TOTAL_CASES}"
echo "[eval] GPU_IDS=${GPU_IDS} NPROC=${NPROC} source=${GPU_SELECTION_SOURCE} (EVAL_GPU_COUNT only controls automatic selection)"
echo "[eval] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

BASE_EVAL_DIR="${OUTPUT_ROOT}/base_production"
BASE_AUDIT_EVAL_DIR="${OUTPUT_ROOT}/base_audit_prompt"
LORA_EVAL_DIR="${OUTPUT_ROOT}/lora_production"
LORA_AUDIT_EVAL_DIR="${OUTPUT_ROOT}/lora_audit_prompt"

run_eval "base production" \
  --model-dir "${MODEL_DIR}" --index "${INDEX}" --data-root "${DATA_ROOT}" --split "${SPLIT}" \
  --history-rgb-mode "${BASE_HISTORY_RGB_MODE}" --cases-per-bin "${CASES_PER_BIN}" \
  "${EXCLUSION_ARGS[@]}" \
  --max-frames "${MAX_EVAL_FRAMES}" --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-dir "${BASE_EVAL_DIR}" --no-timestamp-output --overwrite --save-error-rgb --no-save-all-rgb

if [[ "${RUN_AUDIT_PROMPT_EVAL}" == "1" ]]; then
  run_eval "base audit-prompt" \
    --model-dir "${MODEL_DIR}" --index "${INDEX}" --data-root "${DATA_ROOT}" --split "${SPLIT}" \
    --history-rgb-mode "${BASE_HISTORY_RGB_MODE}" --cases-per-bin "${CASES_PER_BIN}" \
    "${EXCLUSION_ARGS[@]}" \
    --max-frames "${MAX_EVAL_FRAMES}" --max-new-tokens "${MAX_NEW_TOKENS}" \
    --audit-prompt --output-dir "${BASE_AUDIT_EVAL_DIR}" --no-timestamp-output --overwrite --save-error-rgb --no-save-all-rgb
fi

run_eval "LoRA production" \
  --model-dir "${MODEL_DIR}" --index "${INDEX}" --data-root "${DATA_ROOT}" --split "${SPLIT}" \
  --adapter-dir "${ADAPTER_DIR}" --cases-per-bin "${CASES_PER_BIN}" \
  "${EXCLUSION_ARGS[@]}" \
  --max-frames "${MAX_EVAL_FRAMES}" --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-dir "${LORA_EVAL_DIR}" --no-timestamp-output --overwrite --save-error-rgb --no-save-all-rgb

if [[ "${RUN_AUDIT_PROMPT_EVAL}" == "1" ]]; then
  run_eval "LoRA audit-prompt" \
    --model-dir "${MODEL_DIR}" --index "${INDEX}" --data-root "${DATA_ROOT}" --split "${SPLIT}" \
    --adapter-dir "${ADAPTER_DIR}" --cases-per-bin "${CASES_PER_BIN}" \
    "${EXCLUSION_ARGS[@]}" \
    --max-frames "${MAX_EVAL_FRAMES}" --max-new-tokens "${MAX_NEW_TOKENS}" \
    --audit-prompt --output-dir "${LORA_AUDIT_EVAL_DIR}" --no-timestamp-output --overwrite --save-error-rgb --no-save-all-rgb
fi

python "${AUDIT_PY}" --eval-dir "${BASE_EVAL_DIR}" --output-dir "${OUTPUT_ROOT}/audit_base_production" \
  --data-root "${DATA_ROOT}" --per-target "${AUDIT_PER_TARGET}" --overwrite
python "${AUDIT_PY}" --eval-dir "${LORA_EVAL_DIR}" --output-dir "${OUTPUT_ROOT}/audit_lora_production" \
  --data-root "${DATA_ROOT}" --per-target "${AUDIT_PER_TARGET}" --overwrite

if [[ "${RUN_VISUAL_AUDIT}" == "1" ]]; then
  VISUAL_AUDIT_ARGS=(--output "${OUTPUT_ROOT}/visual_audit_manifest.json")
  if [[ "${SCAN_VISUAL_RISKS}" == "1" ]]; then
    VISUAL_AUDIT_ARGS+=(--scan-frame-risks)
  fi
  python "${VISUAL_AUDIT_PY}" "${VISUAL_AUDIT_ARGS[@]}"
fi

build_bundle

echo
echo "[done] eval root: ${OUTPUT_ROOT}"
echo "[done] audit bundle: ${OUTPUT_ROOT}/${BUNDLE_BASENAME}.tar.gz"
