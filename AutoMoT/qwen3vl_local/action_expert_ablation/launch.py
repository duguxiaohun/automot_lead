#!/usr/bin/env python3
"""Launcher for action expert ablation train/eval jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3vl_local.action_expert_ablation.common import VARIANTS
from qwen3vl_local.action_prior.launch import ensure_gpu, prepare_run_directory, run_logged


def resolve_train_launch_args(variant: str, extra: list[str]) -> tuple[Path, str, list[str]]:
    """Merge CLI/env output-dir and resume before protecting the run directory."""

    cleaned = []
    cli_output = ""
    cli_resume = ""
    i = 0
    while i < len(extra):
        item = extra[i]
        if item == "--output-dir":
            if i + 1 >= len(extra):
                raise ValueError("--output-dir requires a value")
            cli_output = extra[i + 1]
            i += 2
            continue
        if item.startswith("--output-dir="):
            cli_output = item.split("=", 1)[1]
            i += 1
            continue
        if item == "--resume":
            if i + 1 >= len(extra):
                raise ValueError("--resume requires a value")
            cli_resume = extra[i + 1]
            i += 2
            continue
        if item.startswith("--resume="):
            cli_resume = item.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(item)
        i += 1

    base = Path(cli_output or os.environ.get("OUTPUT_DIR", VARIANTS[variant]["output_dir"]))
    resume = cli_resume or os.environ.get("RESUME", "")
    return base, resume, cleaned


def restore_resume_world_size_default(resume: str) -> None:
    """Use the checkpoint run's original world size unless the caller overrode it."""

    if (
        not resume
        or os.environ.get("GPU_IDS")
        or os.environ.get("DDP_GPU_COUNT")
        or os.environ.get("NPROC_PER_NODE")
    ):
        return
    plan_path = Path(resume).expanduser().resolve().parent / "training_plan.json"
    if not plan_path.is_file():
        return
    with plan_path.open("r", encoding="utf-8") as f:
        plan = json.load(f)
    world = int(plan.get("world_size", 0))
    if world > 0:
        os.environ["DDP_GPU_COUNT"] = str(world)


def main() -> None:
    """Select GPUs, protect run directories, and dispatch the variant module."""

    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval", "preflight"])
    p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    known, extra = p.parse_known_args()
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    module = f"qwen3vl_local.action_expert_ablation.{known.variant}." + (
        "train" if known.mode == "preflight" else known.mode
    )
    if known.mode == "preflight":
        subprocess.run([sys.executable, "-m", module, "--preflight", *extra], check=True)
        return
    if known.mode == "train":
        base, resume, extra = resolve_train_launch_args(known.variant, extra)
        if resume:
            resume = str(Path(resume).expanduser().resolve())
            restore_resume_world_size_default(resume)
    count = ensure_gpu(
        int(
            os.environ.get(
                "DDP_GPU_COUNT",
                os.environ.get("NPROC_PER_NODE", "4" if known.mode == "train" else "1"),
            )
        )
    )
    if known.mode == "train":
        out = prepare_run_directory(base, resume)
        os.environ["ACTION_ABLATION_RUN_READY"] = "1"
        extra = ["--output-dir", str(out), *(["--resume", resume] if resume else []), *extra]
        log_path = out / "train.log"
    else:
        log_parser = argparse.ArgumentParser(add_help=False)
        log_parser.add_argument("--checkpoint", required=True)
        log_parser.add_argument("--output-dir", default="")
        log_parser.add_argument("--split", default="test")
        log_args, _ = log_parser.parse_known_args(extra)
        log_dir = Path(log_args.output_dir or str(Path(log_args.checkpoint).parent / f"eval_{log_args.split}"))
        log_path = log_dir / "eval.log"
    command = (
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={count}",
            "-m",
            module,
        ]
        if count > 1
        else [sys.executable, "-m", module]
    )
    run_logged(command + extra, log_path)


if __name__ == "__main__":
    main()
