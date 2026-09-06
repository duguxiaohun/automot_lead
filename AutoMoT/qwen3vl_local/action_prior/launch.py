#!/usr/bin/env python3
"""统一自动选卡和 run 目录管理；先选 GPU 再导入任何 torch。"""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys


def ensure_gpu(count=1):
    """GPU_IDS 显式选卡；torchrun 子进程继承同一 mask。"""
    if os.environ.get("ACTION_PRIOR_GPU_READY") == "1":
        return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    explicit = os.environ.get("GPU_IDS", "").strip()
    if explicit:
        ids = [s.strip() for s in explicit.split(",")]
        if not all(s.isdigit() for s in ids) or len(set(ids)) != len(ids):
            raise ValueError("GPU_IDS must contain distinct integer GPU ids")
    else:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        available = sorted(
            (
                tuple(int(s.strip()) for s in line.split(","))
                for line in result.stdout.splitlines()
            ),
            key=lambda v: (v[1], v[2]),
        )
        if len(available) < count:
            raise ValueError(f"requested {count} GPUs, found {len(available)}")
        ids = [str(v[0]) for v in available[:count]]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
    os.environ["ACTION_PRIOR_GPU_READY"] = "1"
    print(f"[GPU] selected {ids}", flush=True)
    return len(ids)


def prepare_run_directory(base, resume=""):
    """shell 与直接 Python 入口共用防覆盖规则。"""
    base = Path(base)
    tag = os.environ.get("RUN_TAG", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if resume:
        path = Path(resume).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        out = path.parent
    else:
        out = base if os.environ.get("NO_RUN_SUBDIR") == "1" else base / f"run_{tag}"
        out.mkdir(parents=True, exist_ok=False)
        base.mkdir(parents=True, exist_ok=True)
        if out != base:
            link = base / "latest"
            if link.exists() and not link.is_symlink():
                raise FileExistsError(f"{link}: expected symlink")
            link.unlink(missing_ok=True)
            link.symlink_to(out.resolve(), target_is_directory=True)
    os.environ.setdefault("HF_HOME", str(base.resolve() / ".hf_cache"))
    return out


def main():
    """所有 worker 使用唯一 run tag；resume 使用原 run 目录。"""
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval", "probe", "preflight"])
    known, extra = p.parse_known_args()
    root = Path(__file__).resolve().parents[2]
    os.environ["PYTHONPATH"] = str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    module = "qwen3vl_local.action_prior." + (
        "train"
        if known.mode == "preflight"
        else "eval" if known.mode == "probe" else known.mode
    )
    if known.mode == "preflight":
        subprocess.run(
            [sys.executable, "-m", module, "--preflight", *extra], check=True
        )
        return
    count = ensure_gpu(
        int(
            os.environ.get(
                "DDP_GPU_COUNT",
                os.environ.get("NPROC_PER_NODE", "4" if known.mode == "train" else "1"),
            )
        )
    )
    if known.mode == "train":
        base = Path(os.environ.get("OUTPUT_DIR", "checkpoints/action_prior"))
        resume = os.environ.get("RESUME", "")
        out = prepare_run_directory(base, resume)
        os.environ["ACTION_PRIOR_RUN_READY"] = "1"
        extra = [
            "--output-dir",
            str(out),
            *(["--resume", resume] if resume else []),
            *extra,
        ]
        if any(x == "--output-dir" for x in extra[2:]):
            raise ValueError(
                "use OUTPUT_DIR env so launcher can protect run directories"
            )
    if known.mode == "probe":
        probe_parser = argparse.ArgumentParser(add_help=False)
        probe_parser.add_argument("--checkpoint", required=True)
        probe_parser.add_argument("--output-dir", default="")
        probe_parser.add_argument("--split", default="test")
        probe_args, _ = probe_parser.parse_known_args(extra)
        if not probe_args.output_dir:
            extra += [
                "--output-dir",
                str(Path(probe_args.checkpoint).parent / f"probe_{probe_args.split}"),
            ]
        extra = ["--max-samples", "24", "--dump-cases", *extra]
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
    subprocess.run(command + extra, check=True)
    if known.mode == "probe":
        ep = argparse.ArgumentParser(add_help=False)
        ep.add_argument("--checkpoint", required=True)
        ep.add_argument("--output-dir", default="")
        ep.add_argument("--split", default="test")
        ea, _ = ep.parse_known_args(extra)
        directory = (
            Path(ea.output_dir)
            if ea.output_dir
            else Path(ea.checkpoint).parent / f"eval_{ea.split}"
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "qwen3vl_local.action_prior.plot_probe",
                str(directory / "cases"),
            ],
            check=True,
        )

        from qwen3vl_local.action_prior.audit_bundle import pack
        pack(directory)


if __name__ == "__main__":
    main()
