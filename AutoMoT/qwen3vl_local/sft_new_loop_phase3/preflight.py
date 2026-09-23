"""训练前验证新索引、物理路线分割及本地模型文件；绝不下载模型。"""
import argparse
import hashlib
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.sft_new_loop_phase3.build_dataset import physical_route_group, development_route_groups
from qwen3vl_local.sft_new_loop_phase3.source_mapping import validate_mapping_contract
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import validate_action_rule
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS, CONTEXT_BY_ID
from qwen3vl_local.sft_new_loop_phase3.choice_semantics import validate_choice_row
from qwen3vl_local.sft_new_loop_phase3.quality_guards import same_rs_coverage
from qwen3vl_local.sft_new_loop_phase3.prompts import DEFAULT_ACTION_OUTPUT_MODE, PROMPT_NAME, action_prompt_sha256


def training_pool_path(index):
    """读取绑定到 manifest 的完整训练池；缺失/被修改必须重建，不能静默回退子集。"""
    index = Path(index)
    manifest = json.loads(index.with_name("manifest.json").read_text())
    info = manifest.get("training_pool", {})
    if info.get("file") != "train_sampling_pool.jsonl" or not info.get("sha256"):
        raise ValueError("smooth_cap requires a hashed training_pool; rebuild the Phase3 index")
    path = index.with_name(info["file"])
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != info["sha256"]:
        raise ValueError("training_pool hash mismatch; rebuild the Phase3 index")
    return path


def audit_input_rows(index, coverage, *, include_training_pool=None):
    """审计实际训练池和原holdout；帧IO去重不能丢掉不同context/前提的监督。

    返回去掉完全相同副本的行，冲突副本仍各自送往核验器。各来源分别记录读入行、
    物理帧和语义case数；跨来源完全相同的行复用核验结果，不掩盖仅新池出现的帧。
    """
    index = Path(index)
    manifest_path = index.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    use_pool = bool(manifest.get("training_pool")) if include_training_pool is None else include_training_pool
    pool = training_pool_path(index) if use_pool else None
    coverage.update(scope="training_pool_plus_original_holdout" if use_pool else "original_index",
                    training_pool=manifest.get("training_pool") if use_pool else None,
                    sources={}, unique_frames=0, row_variants_verified=0, identical_rows_reused=0)
    seen_rows, all_frames = set(), set()
    source_frames, source_cases = defaultdict(set), defaultdict(set)
    paths = [("training_pool", pool)] if pool else []
    paths.append(("index", index))
    for source, path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if source == "index" and use_pool and row["split"] == "train":
                    continue
                if source == "training_pool" and row["split"] != "train":
                    raise ValueError("non-train row entered training_pool audit")
                label = source if source == "training_pool" else "index_" + row["split"]
                frame = (row["scenario"], row["route_id"], int(row["frame_id"]))
                case = (*frame, row["context_id"], row["prompt_road_structure"], row["invalid_reason"])
                source_frames[label].add(frame)
                source_cases[label].add(case)
                all_frames.add(frame)
                stats = coverage["sources"].setdefault(label, dict(rows=0, unique_frames=0, unique_cases=0))
                stats.update(rows=stats["rows"] + 1, unique_frames=len(source_frames[label]),
                             unique_cases=len(source_cases[label]))
                coverage["unique_frames"] = len(all_frames)
                fingerprint = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).digest()
                if fingerprint in seen_rows:
                    coverage["identical_rows_reused"] += 1
                    continue
                seen_rows.add(fingerprint)
                coverage["row_variants_verified"] += 1
                yield row, label


def check_index(path, action_output_mode="binary"):
    """全索引检查；不加载 Qwen 或读取未来状态作为模型条件。"""
    path = Path(path)
    manifest_path = path.with_name("manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prompt_contract = manifest.get("prompt_contract")
        if not isinstance(prompt_contract, dict):
            raise ValueError(
                f"{manifest_path}: missing prompt_contract; rebuild this index for the current prompt"
            )
        if prompt_contract.get("prompt_name") != PROMPT_NAME:
            raise ValueError(
                f"{manifest_path}: prompt_name mismatch: {prompt_contract.get('prompt_name')!r}; "
                "rebuild the index for the current prompt contract"
            )
        expected_hashes = {
            output_mode: {
                history_mode: action_prompt_sha256(
                    audit=False, history_rgb_mode=history_mode, action_output_mode=output_mode,
                )
                for history_mode in ("4rgb", "2rgb_endpoints")
            }
            for output_mode in ("binary", "choice")
        }
        if prompt_contract.get("production_prompt_sha256") != expected_hashes:
            raise ValueError(
                f"{manifest_path}: prompt hash mismatch; rebuild this index for the current prompt contract"
            )
    pool_path = training_pool_path(path) if manifest_path.is_file() and manifest.get("training_pool") else None
    counts = Counter()
    groups = {}
    unique = defaultdict(set)
    same_rs_groups = defaultdict(set)
    for line in path.open():
        row = json.loads(line)
        validate_action_rule(row)
        validate_mapping_contract(row)
        validate_choice_row(row)
        speed = row.get("current_speed_mps")
        if speed is None or not math.isfinite(speed) or speed < 0:
            raise ValueError("current_speed_mps must be measured, finite and nonnegative")
        group = physical_route_group(row["scenario"], row["route_id"])
        split = row["split"]
        if group in development_route_groups() and split != "train":
            raise ValueError(f"development route in holdout: {group}")
        if group in groups and groups[group] != split:
            raise ValueError(f"physical route leakage: {group}")
        groups[group] = split
        if row["invalid_action_context"] and row.get("invalid_reason") == "same_rs_wrong_event":
            same_rs_groups[split].add(group)
        cls = "INVALID" if row["invalid_action_context"] else row["context_id"]
        counts[f"{split}/{cls}"] += 1
        unique[split].add((row["scenario"], row["route_id"], row["frame_id"],
                           row["context_id"], row["prompt_road_structure"], row["invalid_reason"]))
    if pool_path is not None:
        for line in pool_path.open():
            row = json.loads(line)
            validate_action_rule(row)
            validate_mapping_contract(row)
            validate_choice_row(row)
            group = physical_route_group(row["scenario"], row["route_id"])
            if row["split"] != "train" or groups.get(group, "train") != "train":
                raise ValueError(f"physical route leakage in training_pool: {group}")
    for split in ("train", "val", "test"):
        missing = [key for key in (*CONTEXT_IDS, "INVALID") if not counts[f"{split}/{key}"]]
        if missing:
            raise ValueError(f"{split}: missing contexts {missing}")
    support = {split: same_rs_coverage(len(same_rs_groups[split]))
               for split in ("train", "val", "test")}
    if action_output_mode == "binary":
        for split in ("val", "test"):
            if not support[split]["supported"]:
                print(f"[phase3-preflight] {split}: same-RS event rejection has "
                      f"{len(same_rs_groups[split])} independent routes; insufficient_support. "
                      "Training allowed; this subgroup is excluded from checkpoint guards.",
                      file=sys.stderr)
    return dict(counts_scope="original_index", counts=dict(counts), unique_cases={k: len(v) for k, v in unique.items()},
                same_rs_physical_routes={s: len(same_rs_groups[s]) for s in ("train", "val", "test")},
                same_rs_evaluation=support,
                physical_routes=len(groups), physical_route_overlap=0,
                training_pool=manifest.get("training_pool") if manifest_path.is_file() else None)


def check_model(path):
    """接受本地完整 HF 权重；缺分片不能等到开始训练才尝试联网。"""
    root = Path(path).expanduser().resolve()
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"local Qwen model missing: {root}/config.json; downloading is disabled")
    indices = list(root.glob("*.index.json"))
    shards = set()
    for index in indices:
        shards.update(json.loads(index.read_text()).get("weight_map", {}).values())
    if shards:
        missing = [name for name in shards if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"missing local model shards: {missing}")
    elif not any((root / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError(f"no local model weights: {root}")
    return dict(model_dir=str(root), local_weight_shards=len(shards) or 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--action-output-mode", choices=("binary", "choice"), default=DEFAULT_ACTION_OUTPUT_MODE)
    args = parser.parse_args()
    report = {}
    try:
        if args.index:
            report["index"] = check_index(args.index, args.action_output_mode)
        if args.model_dir:
            report["model"] = check_model(args.model_dir)
        report["ok"] = True
    except (ValueError, FileNotFoundError) as exc:
        report.update(ok=False, error=str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
