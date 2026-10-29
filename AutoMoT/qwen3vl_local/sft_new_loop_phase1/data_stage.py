"""Separate, observable CPU dataset admission for ONE phase; no model loading."""
from __future__ import annotations

import argparse
import importlib
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

from . import run_integrity as integrity
from .local_assets import ensure_data, inspect_index


def run(command, status, *, timeout=7200, heartbeat=10, env=None):
    """Own only the child process group. Keep full logs and failed outputs."""
    if not math.isfinite(timeout) or timeout <= 0 or heartbeat <= 0:
        raise ValueError("Finite positive stage timeout/heartbeat required")
    status = Path(status)
    status.parent.mkdir(parents=True, exist_ok=True)
    # Each attempt needs a new status path; do not erase prior failure evidence.
    with status.open("x"):
        pass
    started = time.monotonic()
    record = dict(status="starting", command=command, supervisor_pid=os.getpid(),
                  timeout_seconds=timeout, log=str(status.with_suffix(".log")))
    child = None
    last_output = started
    def save(state, **extra):
        record.update(status=state, elapsed_seconds=time.monotonic()-started,
                      silent_seconds=time.monotonic()-last_output, **extra)
        integrity.write(status, record)
    def terminate():
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=3)
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(*_):
        raise InterruptedError("CPU stage supervisor terminated")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        save("starting")
        with status.with_suffix(".log").open("xb") as log:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=env)
            record["child_pid"] = child.pid
            save("running")
            next_heartbeat = started + heartbeat
            while True:
                now = time.monotonic()
                if now-started >= timeout:
                    raise TimeoutError(f"CPU stage exceeded {timeout}s; inspect {status} and its log")
                if now >= next_heartbeat:
                    save("running")
                    print(f"[cpu-stage] {status.name} pid={child.pid} elapsed={now-started:.1f}s silent={now-last_output:.1f}s", flush=True)
                    next_heartbeat = now + heartbeat
                readable, _, _ = select.select([child.stdout], [], [], min(.25, timeout-(now-started)))
                if readable:
                    chunk = os.read(child.stdout.fileno(), 65536)
                    if chunk:
                        last_output = time.monotonic()
                        log.write(chunk); log.flush()
                        sys.stdout.write(chunk.decode("utf-8", errors="replace")); sys.stdout.flush()
                    elif child.poll() is not None:
                        break
                elif child.poll() is not None:
                    break
            code = child.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        save("completed", returncode=0)
        return record
    except BaseException as exc:
        terminate()
        save("timed_out" if isinstance(exc, TimeoutError) else "failed", error=f"{type(exc).__name__}: {exc}",
             returncode=child.returncode if child else None)
        raise
    finally:
        if child is not None and child.stdout:
            child.stdout.close()
        signal.signal(signal.SIGTERM, previous)


def native_arguments(phase, arguments):
    module = importlib.import_module(f"qwen3vl_local.sft_new_loop_phase{phase}.build_dataset")
    old = sys.argv
    try:
        sys.argv = ["build_dataset", *arguments]
        return module.parse_args()
    finally:
        sys.argv = old


def prepare(args):
    if (not math.isfinite(args.timeout) or args.timeout <= 0 or args.world_size < 1
            or args.focus_balance_count < 0 or args.cases_per_bin < 0
            or len(set(args.modes)) != len(args.modes)):
        raise ValueError("Invalid CPU admission timeout/world size/quota/modes")
    native = native_arguments(args.phase, args.builder_args)
    index = Path(native.output_dir) / "frame_index.jsonl"
    action, _ = ensure_data(index, args.phase, args.run_build)
    if action == "build":
        if not Path(native.data_root).is_dir() or not list(Path(native.collection_dir).glob("*_result.json")):
            raise ValueError("Missing local RGB root or collection results BEFORE dataset scanning")
        required = [native.answer_table] if args.phase == 1 else [native.highway_ue3_rgb_decisions]
        if not Path(native.review_root).is_dir():
            required.append(native.coverage_manifest)
        for name in required:
            if not Path(name).is_file():
                raise ValueError(f"Missing dataset input BEFORE scanning: {name}")
    output = Path(args.audit_dir)
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="preparing", phase=args.phase, dataset=str(index), model_loaded=False)
    integrity.write(output / "status.json", state)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "",
           "WORLD_SIZE": str(args.world_size), "RANK": "0", "LOCAL_RANK": "0"}
    env.pop("GPU_IDS", None)
    source = integrity.sources(args.phase)
    try:
        if action == "build":
            run([sys.executable, "-B", "-m", f"qwen3vl_local.sft_new_loop_phase{args.phase}.build_dataset", *args.builder_args],
                output / "build.json", timeout=args.timeout, env=env)
        receipt = inspect_index(index, args.phase)
        integrity.write(output / "dataset_receipt.json", receipt)
        for mode in args.modes:
            run([sys.executable, "-B", "-m", "qwen3vl_local.sft_new_loop_phase1.pipeline_preflight",
                 "--phase", str(args.phase), "--index", str(index), "--data-root", native.data_root,
                 "--history-rgb-mode", mode, "--data-only", "--device", "cpu",
                 "--cases-per-bin", str(args.cases_per_bin), "--output", str(output / f"{mode}_data.json")],
                output / f"{mode}_images.json", timeout=args.timeout, env=env)
            command = [sys.executable, "-B", "-m", f"qwen3vl_local.sft_new_loop_phase{args.phase}.train",
                       "--plan-only", "--device", "cpu", "--index", str(index), "--data-root", native.data_root,
                       "--history-rgb-mode", mode, "--output-dir", str(output / f"{mode}_plan"),
                       "--focus-balance-count", str(args.focus_balance_count or (768 if args.phase == 1 else 1024)),
                       "--num-epochs", "3", "--action-index", args.action_index,
                       "--epoch-step-limit", str(args.epoch_step_limit),
                       "--total-step-limit", str(args.total_step_limit),
                       "--epoch-hours-limit", str(args.epoch_hours_limit)]
            run(command, output / f"{mode}_sampling.json", timeout=args.timeout, env=env)
        if inspect_index(index, args.phase) != receipt:
            raise ValueError("Dataset changed during CPU admission")
        integrity.verify_sources(source)
        state.update(status="ready", dataset_receipt=receipt, modes=args.modes, source_sha256=source,
                     note="CPU data/plan admission only. Target machine still checks local RGB and Qwen3.5 runtime; no GPU certification.")
        state["evidence_sha256"] = {str(p.relative_to(output)): integrity.sha(p) for p in sorted(output.rglob("*"))
                                    if p.is_file() and p != output / "status.json"}
    except BaseException as exc:
        state.update(status="rejected", reason=f"{type(exc).__name__}: {exc}")
        integrity.write(output / "status.json", state)
        raise
    integrity.write(output / "status.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    monitor = commands.add_parser("run")
    monitor.add_argument("--status", required=True)
    monitor.add_argument("--timeout", type=float, default=7200)
    monitor.add_argument("child", nargs=argparse.REMAINDER)
    cpu = commands.add_parser("prepare")
    cpu.add_argument("--phase", type=int, choices=(1,2), required=True)
    cpu.add_argument("--audit-dir", required=True)
    cpu.add_argument("--run-build", choices=("auto", "0", "1"), default="auto")
    cpu.add_argument("--timeout", type=float, default=7200)
    cpu.add_argument("--world-size", type=int, default=4)
    cpu.add_argument("--modes", nargs="+", choices=("4rgb", "2rgb_endpoints"), default=["4rgb"])
    cpu.add_argument("--cases-per-bin", type=int, default=64)
    cpu.add_argument("--focus-balance-count", type=int, default=0)
    cpu.add_argument("--action-index", default="auto")
    cpu.add_argument("--epoch-step-limit", type=int, default=4096)
    cpu.add_argument("--total-step-limit", type=int, default=12288)
    cpu.add_argument("--epoch-hours-limit", type=float, default=10.0)
    cpu.add_argument("builder_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "run":
        child = args.child[1:] if args.child[:1] == ["--"] else args.child
        run(child, args.status, timeout=args.timeout)
    else:
        if args.builder_args[:1] == ["--"]:
            args.builder_args = args.builder_args[1:]
        prepare(args)


if __name__ == "__main__":
    main()
