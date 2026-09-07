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
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS


def check_index(path):
    """全索引检查；不加载 Qwen 或读取未来状态作为模型条件。"""
    counts = Counter()
    groups = {}
    unique = defaultdict(set)
    for line in Path(path).open():
        row = json.loads(line)
        validate_action_rule(row)
        validate_mapping_contract(row)
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
        cls = "INVALID" if row["invalid_action_context"] else row["context_id"]
        counts[f"{split}/{cls}"] += 1
        unique[split].add((row["scenario"], row["route_id"], row["frame_id"],
                           row["context_id"], row["prompt_road_structure"], row["invalid_reason"]))
    for split in ("train", "val", "test"):
        missing = [key for key in (*CONTEXT_IDS, "INVALID") if not counts[f"{split}/{key}"]]
        if missing:
            raise ValueError(f"{split}: missing contexts {missing}")
    return dict(counts=dict(counts), unique_cases={k: len(v) for k, v in unique.items()},
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
    args = parser.parse_args()
    report = {}
    try:
        if args.index:
            report["index"] = check_index(args.index)
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
