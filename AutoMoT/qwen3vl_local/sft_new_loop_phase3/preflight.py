"""训练前验证新索引、物理路线分割及本地模型文件；绝不下载模型。"""
import argparse
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
from qwen3vl_local.sft_new_loop_phase3.primary_action import PRIMARY_ACTION_VERSION, primary_action
from qwen3vl_local.sft_new_loop_phase3.quality_guards import MIN_SAME_RS_PHYSICAL_ROUTES
from qwen3vl_local.sft_new_loop_phase3.prompts import DEFAULT_ACTION_OUTPUT_MODE, PROMPT_NAME, action_prompt_sha256


def check_index(path, action_output_mode="binary"):
    """全索引检查；不加载 Qwen 或读取未来状态作为模型条件。"""
    path = Path(path)
    manifest_path = path.with_name("manifest.json")
    primary_contract = False
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        primary_contract = manifest.get("primary_action_version") == PRIMARY_ACTION_VERSION
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
    counts = Counter()
    groups = {}
    unique = defaultdict(set)
    same_rs_groups = defaultdict(set)
    for line in path.open():
        row = json.loads(line)
        validate_action_rule(row)
        validate_mapping_contract(row)
        if primary_contract or "primary_action_version" in row:
            expected = None if row["invalid_action_context"] else primary_action(
                row["answers"], CONTEXT_BY_ID[row["context_id"]].action_keys)
            if row.get("primary_action_version") != PRIMARY_ACTION_VERSION or row.get("primary_action") != expected:
                raise ValueError("primary action label/version mismatch; rebuild index")
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
    for split in ("train", "val", "test"):
        missing = [key for key in (*CONTEXT_IDS, "INVALID") if not counts[f"{split}/{key}"]]
        if missing:
            raise ValueError(f"{split}: missing contexts {missing}")
    if action_output_mode == "binary":
        for split in ("val", "test"):
            count = len(same_rs_groups[split])
            if count < MIN_SAME_RS_PHYSICAL_ROUTES:
                raise ValueError(
                    f"{split}: same_rs_wrong_event has {count} independent physical routes; "
                    f"requires >= {MIN_SAME_RS_PHYSICAL_ROUTES}. Review additional unseen routes "
                    "and rebuild the index; increasing sample repetitions cannot fix missing coverage."
                )
    return dict(counts=dict(counts), unique_cases={k: len(v) for k, v in unique.items()},
                same_rs_physical_routes={s: len(same_rs_groups[s]) for s in ("train", "val", "test")},
                physical_routes=len(groups), physical_route_overlap=0)


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
