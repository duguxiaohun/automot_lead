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
    plan = json.loads((checkpoint.parent / "training_plan.json").read_text())
    selected = json.loads((checkpoint.parent / "selected_priors.json").read_text())
    cfg["phase1_adapter"] = selected["phase1"]["path"]
    cfg["phase2_adapter"] = selected["phase2"]["path"]
    args = []
    for k, v in cfg.items():
        if k not in DEFAULTS or k in ("resume", "output_dir"):
            continue
        key = k.replace("_", "-")
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
