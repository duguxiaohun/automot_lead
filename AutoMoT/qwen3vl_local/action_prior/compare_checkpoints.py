#!/usr/bin/env python3
"""多训练目录、同帧同噪声的 event/action 离线可视化对比。"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import secrets
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.action_prior.comparison_cases import (
    AUTOMOT_ROOT, case_id, identity, read_json, write_json, resolve_path,
    select_checkpoint, select_cases, require_same_frames,
    parse_category_counts, EVENT_NAMES, ACTION_NAMES,
)


def resolve_sampling_seed(value):
    """auto按本次时间和系统随机源生成；整数可复现案例选择及顺序。"""
    if str(value).lower() == "auto":
        return (time.time_ns() ^ secrets.randbits(63)) & ((1 << 63) - 1)
    try:
        seed = int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("sampling-seed 必须为 auto 或非负整数") from exc
    if seed < 0 or str(seed) != str(value).strip():
        raise ValueError("sampling-seed 必须为 auto 或非负整数")
    return seed


def ordered_cases(rows, seed, split):
    """稳定输入归一后打乱；每个split独立，同次各模型顺序一致。"""
    result = sorted(rows, key=identity)
    random.Random(f"{seed}:{split}:comparison-order").shuffle(result)
    return result


def prepare(cli, out):
    """在任何模型加载前校验来源、split、选优记录并固定所有 case。"""
    import copy
    import torch
    from qwen3vl_local.action_prior.comparison_runtime import restore_args, check_contract
    from qwen3vl_local.action_prior.config import read_rows
    from qwen3vl_local.action_prior.action_token import token_source
    from qwen3vl_local.action_prior.contracts import file_hash
    from qwen3vl_local.action_prior.comparison_progress import PreflightProgress
    progress = PreflightProgress(out)
    sampling_seed = cli.sampling_seed
    overrides = {key: getattr(cli, key) for key in (
        "data_root", "data_dir", "model_dir", "lead_bev_ckpt", "event_balance_index",
        "high_level_action_index", "prior_labels", "phase1_training_index", "phase2_training_index") if getattr(cli, key)}
    entries, namespaces, jobs = [], [], []
    for index, run in enumerate(cli.runs):
        print(f"[preflight] model {index + 1}/{len(cli.runs)}: {run}", flush=True)
        with progress.stage(f"model {index+1}/{len(cli.runs)} select/load checkpoint (CPU)"):
            checkpoint, state, selection = select_checkpoint(run, lambda p: torch.load(p, map_location="cpu", weights_only=False))
        with progress.stage(f"model {index+1}/{len(cli.runs)} restore paths and validate contract / full map / action candidates"):
            args, variant = restore_args(state, checkpoint, overrides)
            contract = check_contract(state, args, variant)
        with progress.stage(f"model {index+1}/{len(cli.runs)} checkpoint SHA256"):
            checkpoint_sha256 = file_hash(checkpoint)
        entry = dict(id=f"model_{index+1:02d}", label=(cli.names[index] if cli.names else f"{index+1}:{variant}:{checkpoint.parent.name}"),
            checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha256, variant=variant,
            selection=selection, ema=True, contract_identity=contract["identity"], dataset_hashes=state["dataset_hashes"],
            high_level_action_token=bool(args.high_level_action_token), high_level_action_prior=bool(getattr(args, "high_level_action_prior", False)),
            rgb_frame_count=args.rgb_frame_count, trained_seed=args.seed,
            saved_args=state["args"], effective_args=vars(args).copy())
        # 轨迹形状/缩放/时间点必须相同，才有同一 eps/t 和可解释的指标。
        signature = {key: getattr(args, key) for key in (
            "route_points", "waypoint_points", "smooth_route", "frame_interval_s", "flow_route_coordinate_scale_m",
            "flow_waypoint_coordinate_scale_m", "flow_sample_steps", "route_loss_weight", "waypoint_loss_weight",
            "tp_mode", "target_point_lookahead_s", "next_target_point_lookahead_s", "tp_min_lookahead_m")}
        entry["comparison_signature"] = signature
        if entries and (entry["dataset_hashes"] != entries[0]["dataset_hashes"] or signature != entries[0]["comparison_signature"]):
            raise ValueError("模型索引或轨迹/导航评估口径不同，不能直接配对")
        entries.append(entry)
        namespaces.append(args)
        jobs.append(dict(checkpoint=str(checkpoint), overrides=overrides, seed=cli.seed, workers=cli.workers,
                         checkpoint_sha256=entry["checkpoint_sha256"],
                         output=str(out / "_models" / entry["id"]), rows={}, row_hashes={}))
        del state
    # 无 token 的模型也用同源标签分 action 类，但这些标签绝不注入其模型输入。
    source_args = next((args for args in namespaces if getattr(args, "event_balance_index", "")), None)
    if source_args is None and not cli.label_index:
        raise ValueError("event/action 分类需要原 full map，请传 --label-index（不改变模型条件）")
    source_args = copy.copy(source_args or namespaces[0])
    if cli.label_index:
        source_args.event_balance_index = str(resolve_path(cli.label_index))
    with progress.stage("shared action labels: full map / candidate / split hash verification"):
        source = token_source(source_args)
        for args in namespaces:
            if getattr(args, "event_balance_index", ""):
                if file_hash(args.event_balance_index) != source.full.source.sha256:
                    raise ValueError("多个模型或分类用 full map 内容不同")
    manifest = dict(schema="action_checkpoint_comparison_v1", seed=cli.seed, per_category=cli.cases_per_category,
        sampling_seed=sampling_seed, sampling_seed_mode=cli.sampling_seed_mode,
        evaluation_seed=cli.seed, case_order="shuffled_per_split_shared_across_models",
        error_filter=dict(enabled=getattr(cli, "error_only", False), threshold_m=getattr(cli, "error_threshold_m", 1.0)),
        category_counts=cli.category_counts, camera_config=cli.camera_configuration,
        min_frame_gap=cli.min_frame_gap, label_source=source.identity, models=entries, splits={},
        interpretation="分层抽样的离线同帧 EMA 对比；不是全量 test、闭环或泛化结论。event 可重叠。")
    physical = {}
    for split in ("train", "test"):
        with progress.stage(f"{split}: read model 1 effective pool / route filtering"):
            reference = read_rows(namespaces[0], split)
            physical[split] = {r["route_group"] for r in reference}
        with progress.stage(f"{split}: annotate event/action labels ({len(reference)} frames)"):
            labels = [dict(scenario=r["scenario"], run_id=r["run_id"], anchor=r["anchor"], route_group=r["route_group"], split=split) for r in reference]
            source.full.annotate(labels)
            source.annotate(labels)
        with progress.stage(f"{split}: stratified case selection ({len(reference)} frames)"):
            picked, groups, coverage = select_cases(labels, per_category=cli.cases_per_category, seed=sampling_seed,
                min_frame_gap=cli.min_frame_gap, category_counts=cli.category_counts, split=split)
            picked = ordered_cases(picked, sampling_seed, split)
            order = {identity(row): index for index, row in enumerate(picked)}
            id_order = {case_id(row): index for index, row in enumerate(picked)}
            for categories in groups.values():
                for ids in categories.values():
                    ids.sort(key=id_order.__getitem__)
        # 只保留被选中的标签，避免读第二模型全量池时还占用一整份标签字典。
        del labels
        print(f"[preflight] {split}: selected {len(picked)} unique cases / {len(reference)} effective frames", flush=True)
        wanted = {identity(r) for r in picked}
        manifest["splits"][split] = dict(effective_frames=len(reference), selected_unique_frames=len(picked), groups=groups, coverage=coverage,
            cases={case_id(r): r for r in picked})
        for index, args in enumerate(namespaces):
            with progress.stage(f"{split}: model {index+1}/{len(namespaces)} effective pool and paired frames"):
                actual = reference if index == 0 else read_rows(args, split)
                require_same_frames(reference, actual, split)
                rows = [r for r in actual if identity(r) in wanted]
                rows.sort(key=lambda row: order[identity(row)])
            # 注入开启模型的 token 必须和独立分类标签逐帧完全一致。
            expected = {identity(r): r["action_token"] for r in picked}
            for row in rows:
                if args.high_level_action_token and row["action_token"] != expected[identity(row)]:
                    raise ValueError(f"实际 action token 与审计标签不同: {identity(row)}")
            path = out / "_plan" / f"{entries[index]['id']}_{split}.json"
            write_json(path, rows)
            jobs[index]["rows"][split] = str(path)
            jobs[index]["row_hashes"][split] = file_hash(path)
            if index:
                del actual
        # actual在index=0时也是reference的别名，单独释放避免跨split残留。
        del reference
    if physical["train"] & physical["test"]:
        raise ValueError("train/test 物理路线交叉")
    if not any(plan["selected_unique_frames"] for plan in manifest["splits"].values()):
        raise ValueError("所有类别均未选到案例，请检查采样数量/类别支持")
    # 把条件之外的差异写入清单，允许用户比较不同模型组合，但不冒称严格单因素消融。
    keys = set().union(*(set(entry["saved_args"]) for entry in entries))
    manifest["training_config_differences"] = {key: [entry["saved_args"].get(key) for entry in entries]
        for key in sorted(keys) if any(entry["saved_args"].get(key) != entries[0]["saved_args"].get(key) for entry in entries[1:])}
    write_json(out / "manifest.json", manifest)
    for index, job in enumerate(jobs):
        write_json(out / "_plan" / f"job_{index:02d}.json", job)
    return manifest, jobs


def main():
    """按GPU数量并发跑模型，输出目录始终新建时间戳子目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--cases-per-category", type=int, default=8)
    parser.add_argument("--error-only", action=argparse.BooleanOptionalAction, default=False,
                        help="仅绘制任一route/waypoint模型-GT或模型间终点距离超过阈值的采样case")
    parser.add_argument("--error-threshold-m", type=float, default=1.0, help="终点距离阈值，单位米，严格大于；默认1.0")
    parser.add_argument("--event-cases", action="append", default=[], help="逐类数量，例如 UE1=12,UE4=20,test/UE7=6；可重复，0跳过")
    parser.add_argument("--action-cases", action="append", default=[], help="例如 STOP=12,LANE_CHANGE_LEFT=20；可重复，0跳过")
    parser.add_argument("--camera-config", default="", help="显式覆盖显示标定JSON；默认优先同帧meta，缺失回退LEAD名义标定")
    parser.add_argument("--min-frame-gap", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026, help="配对模型推理噪声seed，独立于案例采样，默认2026")
    parser.add_argument("--sampling-seed", default="auto", help="auto每次按时间+系统随机源重新采样/排序；填整数可复现")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=4, help="自动选卡上限，默认4；不足时自动减少，GPU_IDS显式卡数优先")
    parser.add_argument("--output-root", default=str(AUTOMOT_ROOT / "test"))
    parser.add_argument("--plan-only", action="store_true", help="CPU 完成选优/合同/选帧，不运行模型")
    parser.add_argument("--label-index", default="", help="仅用于 event/action 分组的 full map，不改变无 token 模型")
    parser.add_argument("--worker-job", default="", help=argparse.SUPPRESS)
    for key in ("data-root", "data-dir", "model-dir", "lead-bev-ckpt", "event-balance-index", "high-level-action-index",
                "prior-labels", "phase1-training-index", "phase2-training-index"):
        parser.add_argument("--" + key, default="")
    cli = parser.parse_args()
    if cli.worker_job:
        from qwen3vl_local.action_prior.comparison_runtime import evaluate_worker
        evaluate_worker(read_json(cli.worker_job))
        return
    try:
        cli.sampling_seed_mode = "auto" if cli.sampling_seed.lower() == "auto" else "fixed"
        cli.sampling_seed = resolve_sampling_seed(cli.sampling_seed)
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(cli.error_threshold_m) or cli.error_threshold_m <= 0:
        parser.error("error-threshold-m 必须为有限正数（米）")
    if len(cli.runs) < 2 or (cli.names and len(cli.names) != len(cli.runs)):
        parser.error("至少两个训练目录；--names 若指定必须与目录数量相同")
    if cli.cases_per_category < 0 or cli.min_frame_gap < 1 or cli.workers < 0 or cli.gpus < 1:
        parser.error("cases-per-category/workers 不得为负，min-frame-gap/gpus 须为正整数")
    try:
        cli.category_counts = dict(event=parse_category_counts(cli.event_cases, EVENT_NAMES),
                                   action=parse_category_counts(cli.action_cases, ACTION_NAMES))
        from qwen3vl_local.action_prior.comparison_scene import camera_configuration
        cli.camera_configuration = camera_configuration(resolve_path(cli.camera_config) if cli.camera_config else None)
    except ValueError as exc:
        parser.error(str(exc))
    # 用户相对路径以启动时 cwd 为准；只在解析输入后切至训练默认的 AutoMoT 根。
    cli.runs = [str(resolve_path(run)) for run in cli.runs]
    for key in ("data_root", "data_dir", "model_dir", "lead_bev_ckpt", "event_balance_index", "high_level_action_index",
                "prior_labels", "phase1_training_index", "phase2_training_index", "label_index"):
        if getattr(cli, key):
            setattr(cli, key, str(resolve_path(getattr(cli, key))))
    out = Path(cli.output_root).expanduser().resolve() / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(AUTOMOT_ROOT)
    write_json(out / "sampling.json", dict(sampling_seed=cli.sampling_seed, mode=cli.sampling_seed_mode,
                                           evaluation_seed=cli.seed))
    print(f"[comparison] sampling_seed={cli.sampling_seed} ({cli.sampling_seed_mode}); "
          f"evaluation_seed={cli.seed}; same cases/order for all models", flush=True)
    write_json(out / "status.json", dict(status="planning"))
    try:
        if not cli.plan_only:
            from qwen3vl_local.action_prior.comparison_scheduler import select_gpus, run_queue
            gpu_plan = select_gpus(cli.gpus, len(cli.runs))
            print(f"[comparison] GPU={gpu_plan['selected_ids']}; worker capacity={len(gpu_plan['selected_ids'])}; output={out}", flush=True)
        manifest, jobs = prepare(cli, out)
        if cli.plan_only:
            write_json(out / "status.json", dict(status="planned_only", models_executed=False))
        else:
            from qwen3vl_local.action_prior.comparison_shards import plan_shards
            tasks = plan_shards(jobs, manifest, gpu_plan, out)
            manifest["execution"] = gpu_plan
            write_json(out / "manifest.json", manifest)
            commands = [[sys.executable, str(Path(__file__).resolve()), "--worker-job",
                         task['job']] for task in tasks]
            run_queue(commands, gpu_plan, out)
            write_json(out / "status.json", dict(status="rendering", models_executed=True))
            print(f"[comparison] all GPU workers complete; starting CPU rendering; status: {out / 'status.json'}", flush=True)
            from qwen3vl_local.action_prior.comparison_render import publish
            publish(out, manifest)
            write_json(out / "status.json", dict(status="complete", models_executed=True))
    except BaseException as exc:
        write_json(out / "status.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
        raise
    print(f"[comparison] {'planned_only' if cli.plan_only else 'complete'}: {out}", flush=True)


if __name__ == "__main__":
    main()
