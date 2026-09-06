"""action_prior 的 Bench2Drive 220 闭环 launcher；正式 test 不用于训练期选优。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qwen3vl_local.action_prior.benchmark_report import (
    routes,
    route_record,
    sha,
    report,
)
from qwen3vl_local.action_prior.audit_bundle import pack


def environment(gpu, output, checkpoint):
    """当前 Python 环境直接启动，不隐式切换到另一套 conda 包版本。"""
    env = os.environ.copy()
    paths = [
        ROOT,
        ROOT / "leaderboard",
        ROOT / "leaderboard/team_code",
        ROOT / "scenario_runner",
        ROOT / "Automot",
        ROOT / "Automot/mot",
    ]
    carla = Path(env["CARLA_ROOT"])
    paths += [carla / "PythonAPI", carla / "PythonAPI/carla"]
    env["PYTHONPATH"] = (
        os.pathsep.join(map(str, paths)) + os.pathsep + env.get("PYTHONPATH", "")
    )
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        GPU_RANK=str(gpu),
        IS_BENCH2DRIVE="True",
        LEADMOT_CKPT=str(checkpoint),
        SAVE_PATH=str(output / "rollouts"),
        ROUTES=str(ROOT / "leaderboard/data/bench2drive220.xml"),
        SCENARIO_RUNNER_ROOT=str(ROOT / "scenario_runner"),
        LEADERBOARD_ROOT=str(ROOT / "leaderboard"),
        PLANNER_TYPE="only_traj",
        STEP_STRIDE="5",
        SENSOR_PROFILE="3cam",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    for key in (
        "RECORD_INPUT",
        "RECORD_DEBUG",
        "RECORD_DEMO",
        "RECORD_GRID",
        "RECORD_BEV_DEBUG",
    ):
        env.setdefault(key, "0")
    return env


def command(cli, rid, gpu, worker, output):
    """每 worker 使用独立 RPC/TM 端口槽；仅调用现有只读 evaluator。"""
    return [
        sys.executable,
        str(ROOT / "leaderboard/leaderboard/leaderboard_evaluator.py"),
        "--routes",
        str(Path(cli.routes).resolve()),
        "--routes-subset",
        str(rid),
        "--repetitions",
        "1",
        "--track",
        "SENSORS",
        "--agent",
        str(Path(__file__).with_name("carla_agent.py")),
        "--agent-config",
        str(Path(cli.checkpoint).resolve()) + "+route" + str(rid),
        "--checkpoint",
        str(output / "eval_per_route" / f"eval_{rid}.json"),
        "--debug-checkpoint",
        str(output / "logs" / f"live_{rid}.txt"),
        "--port",
        str(cli.port + worker * 50),
        "--traffic-manager-port",
        str(cli.port + worker * 50 + 20),
        "--traffic-manager-seed",
        str(cli.seed),
        "--gpu-rank",
        str(gpu),
        "--timeout",
        str(cli.timeout),
    ]


def validate_checkpoint(cli):
    """先在 CPU 核验模型来源合同，缺权重或错版时不要逐路线启动 CARLA 才失败。"""
    import torch
    from qwen3vl_local.action_prior.config import build_contract, validate_args
    from qwen3vl_local.action_prior.contracts import require_contract

    state = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    if state.get("schema") != "action_prior_checkpoint_v2":
        raise ValueError("requires action_prior_checkpoint_v2")
    args = argparse.Namespace(**state["args"])
    args.selection_manifest = ""
    args.selection_output = ""
    args.lora_bundle = ""
    from qwen3vl_local.action_prior.lora_bundle import restore_paths
    local_paths = restore_paths(state["qwen_backbone"], cli.checkpoint)
    for key in ("phase1", "phase2"):
        setattr(args, key + "_adapter", local_paths[key])
    for key in (
        "model_dir",
        "lead_bev_ckpt",
        "phase1_adapter",
        "phase2_adapter",
        "phase1_training_index",
        "phase2_training_index",
    ):
        if getattr(cli, key):
            setattr(args, key, getattr(cli, key))
    validate_args(args)
    contract = build_contract(args)
    require_contract(state["qwen_backbone"], contract)
    return contract["identity"]


def main():
    """完整 220 默认；显式单路线仅作 smoke，报告始终保留计划分母。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument(
        "--routes", default=str(ROOT / "leaderboard/data/bench2drive220.xml")
    )
    p.add_argument(
        "--num-gpus", type=int, default=int(os.environ.get("EVAL_GPU_COUNT", "1"))
    )
    p.add_argument("--route-id", action="append", default=[])
    p.add_argument("--scenario", action="append", default=[])
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--port", type=int, default=20000)
    p.add_argument("--timeout", type=float, default=1200)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只读 XML 并打印计划，不加载模型或启动 CARLA",
    )
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--raw", action="store_true", help="默认 EMA")
    for key in (
        "model-dir",
        "lead-bev-ckpt",
        "phase1-adapter",
        "phase2-adapter",
        "phase1-training-index",
        "phase2-training-index",
    ):
        p.add_argument("--" + key, default="")
    cli = p.parse_args()
    if cli.num_gpus < 1 or cli.port < 1024 or cli.port + cli.num_gpus * 50 > 65535:
        p.error("invalid GPU count or port range")
    catalog = routes(cli.routes)
    chosen = [
        rid
        for rid, row in catalog.items()
        if (not cli.route_id or rid in cli.route_id)
        and (not cli.scenario or row["scenario"] in cli.scenario)
    ]
    if not chosen or set(cli.route_id) - set(catalog):
        p.error("empty subset or unknown route IDs")
    if (
        not cli.route_id
        and not cli.scenario
        and (
            len(catalog) != 220 or len({r["scenario"] for r in catalog.values()}) != 44
        )
    ):
        p.error("full benchmark requires 220 routes and 44 scenario types")
    if cli.dry_run:
        print(
            json.dumps(
                dict(
                    route_count=len(chosen),
                    scenario_count=len({catalog[k]["scenario"] for k in chosen}),
                    routes_sha256=sha(cli.routes),
                    route_ids=chosen,
                    mode="plan only: no model load, no CARLA",
                ),
                indent=2,
            )
        )
        return
    if (cli.resume or cli.report_only) and not cli.output_dir:
        p.error("--resume/--report-only requires --output-dir")
    output = Path(
        cli.output_dir
        or str(
            Path(cli.checkpoint).resolve().parent
            / ("bench2drive_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        )
    ).resolve()
    if cli.report_only:
        report(output, cli.routes)
        pack(output)
        return
    carla_root = Path(os.environ.get("CARLA_ROOT", ""))
    if not (carla_root / "CarlaUE4.sh").is_file():
        p.error("set CARLA_ROOT to the installed CARLA 0.9.15 directory")
    for key in (
        "model_dir",
        "lead_bev_ckpt",
        "phase1_adapter",
        "phase2_adapter",
        "phase1_training_index",
        "phase2_training_index",
    ):
        if getattr(cli, key):
            os.environ["ACTION_" + key.upper()] = str(Path(getattr(cli, key)).resolve())
    generation_identity = validate_checkpoint(cli)
    identity = dict(
        generation_identity=generation_identity,
        sensor_settings={
            key: os.environ.get(key, default)
            for key, default in {
                "USE_RADAR": "1",
                "JPEG_QUALITY": "85",
                "LIDAR_REMOVE_GROUND": "1",
                "LIDAR_GROUND_Z": "-1.4",
                "USE_UKF": "1",
            }.items()
        },
        checkpoint_sha256=sha(cli.checkpoint),
        routes_sha256=sha(cli.routes),
        route_ids=chosen,
        seed=cli.seed,
        ema=not cli.raw,
        code={
            str(path.relative_to(ROOT)): sha(path)
            for path in [
                Path(__file__),
                Path(__file__).with_name("carla_agent.py"),
                Path(__file__).with_name("carla_runtime.py"),
                Path(__file__).with_name("benchmark_report.py"),
                ROOT / "tools/ability_benchmark.py",
                ROOT / "tools/efficiency_smoothness_benchmark.py",
                ROOT / "qwen3vl_local/eval_carla/agent.py",
                ROOT / "qwen3vl_local/eval_carla/safety.py",
                ROOT / "leaderboard/leaderboard/leaderboard_evaluator.py",
                ROOT / "leaderboard/leaderboard/utils/statistics_manager.py",
            ]
        },
    )
    manifest = dict(
        **identity,
        checkpoint=str(Path(cli.checkpoint).resolve()),
        routes=str(Path(cli.routes).resolve()),
        protocol="Bench2Drive220",
        motion_telemetry_hz=10,
        training_selection="best offline validation; formal routes are test only",
    )
    if cli.resume:
        previous = json.loads((output / "run_manifest.json").read_text())
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError(
                "resume benchmark identity differs; use a new output directory"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    for name in ("eval_per_route", "logs", "rollouts"):
        (output / name).mkdir(exist_ok=True)
    from qwen3vl_local.action_prior.launch import ensure_gpu

    ensure_gpu(cli.num_gpus)
    ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    os.environ["LEADMOT_USE_EMA"] = "0" if cli.raw else "1"
    pending = []
    for rid in chosen:
        path = output / "eval_per_route" / f"eval_{rid}.json"
        if cli.resume and path.is_file():
            try:
                record = route_record(path, rid)
                if record["status"] in ("Completed", "Perfect") or record[
                    "status"
                ].startswith("Failed"):
                    continue  # 正常驾驶失败也是结果，不通过重跑挑好成绩。
            except (ValueError, OSError, KeyError, TypeError):
                pass
            path.rename(path.with_suffix(".incomplete.json"))
        pending.append(rid)

    def worker(index, gpu):
        """各 worker 顺序处理本分片；失败留记录，其余路线继续。"""
        env = environment(gpu, output, Path(cli.checkpoint).resolve())
        env["ROUTES"] = str(Path(cli.routes).resolve())
        for rid in pending[index :: len(ids)]:
            env["ROUTES_SUBSET"] = rid
            with (output / "logs" / f"route_{rid}.log").open("a") as log:
                proc = subprocess.run(
                    command(cli, rid, gpu, index, output),
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            print(f"[Bench2Drive] route={rid} exit={proc.returncode}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=len(ids)) as pool:
            jobs = [pool.submit(worker, i, gpu) for i, gpu in enumerate(ids)]
            for job in jobs:
                job.result()
    finally:
        try:
            summary = report(output, cli.routes)
        finally:
            pack(output)
    if not summary["full_220_records"] and len(chosen) == 220:
        raise SystemExit(
            "Incomplete benchmark: inspect missing_routes and logs; audit.zip was saved"
        )


if __name__ == "__main__":
    main()
