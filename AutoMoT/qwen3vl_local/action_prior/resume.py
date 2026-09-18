#!/usr/bin/env python3
"""从原 run 的 config/plan 恢复完整参数，不用人工重复 LR 和索引路径。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.action_prior.config import DEFAULTS


def main():
    """不加载 checkpoint tensor，先用旁边的可读配置重建原启动参数。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    cli, extra = p.parse_known_args()
    checkpoint = Path(cli.checkpoint).resolve()
    cfg = json.loads((checkpoint.parent / "config.json").read_text())
    # 新 run 恢复保存的开关；旧 run 缺字段时保留原摘要语义，执行指纹仍严格检查。
    cfg.setdefault("generate_analysis", True)
    cfg.setdefault("high_level_planning", False)
    cfg.setdefault("high_level_action_prior", False)
    if "HIGH_LEVEL_ACTION_PRIOR" in os.environ:
        value = os.environ["HIGH_LEVEL_ACTION_PRIOR"]
        if value not in ("0", "1"):
            raise ValueError("HIGH_LEVEL_ACTION_PRIOR must be 0 or 1")
        cfg["high_level_action_prior"] = value == "1"
    if "HIGH_LEVEL_ACTION_INDEX" in os.environ:
        cfg["high_level_action_index"] = os.environ["HIGH_LEVEL_ACTION_INDEX"]
    if "HIGH_LEVEL_PLANNING" in os.environ:
        value = os.environ["HIGH_LEVEL_PLANNING"]
        if value not in ("0", "1"):
            raise ValueError("HIGH_LEVEL_PLANNING must be 0 or 1")
        cfg["high_level_planning"] = value == "1"
    if "GENERATE_ANALYSIS" in os.environ:
        value = os.environ["GENERATE_ANALYSIS"]
        if value not in ("0", "1"):
            raise ValueError("GENERATE_ANALYSIS must be 0 or 1")
        cfg["generate_analysis"] = value == "1"
    plan = json.loads((checkpoint.parent / "training_plan.json").read_text())
    selected = json.loads((checkpoint.parent / "selected_priors.json").read_text())
    if selected.get("phase1"):
        from qwen3vl_local.action_prior.lora_bundle import restore_paths
        local_paths = restore_paths(selected, checkpoint)
        cfg["phase1_adapter"] = local_paths["phase1"]
        cfg["phase2_adapter"] = local_paths["phase2"]
    args = []
    for k, v in cfg.items():
        if k not in DEFAULTS or k in ("resume", "output_dir", "selection_output", "selection_manifest", "lora_bundle"):
            continue
        key = k.replace("_", "-")
        if isinstance(v, list):
            if v:
                args.extend([f"--{key}", *map(str, v)])
            continue
        args.extend(
            [f"--{key}" if v else f"--no-{key}"]
            if isinstance(v, bool)
            else [f"--{key}", str(v)]
        )
    os.environ["RESUME"] = str(checkpoint)
    os.environ.setdefault("DDP_GPU_COUNT", str(plan["world_size"]))
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("launch.py")),
            "train",
            *args,
            *extra,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
