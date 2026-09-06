"""扫描 Phase1/2 best_generation，打印各项已有验证指标并推荐训练同规则权重。"""

from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.action_prior.contracts import (
    inspect_adapter,
    read_json,
    selection_score,
)
from qwen3vl_local.action_prior.lora_audit import ERRORS, audit_checks, discover


def saved_records(path, cfg, phase):
    """只读取保存 step 的验证记录，绝不借用更晚 step 或独立 test 成绩。"""
    step = int(cfg["global_step"])
    log = path.parent / "train_eval_metrics.jsonl"
    generation, teacher, notes = [], [], []
    if log.is_file():
        with log.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if int(record.get("step", -1)) != step:
                        continue
                    kind = record.get("type", record.get("kind"))
                    if kind in ("generation", "free_generation"):
                        generation.append(record)
                    elif kind == "teacher_forced":
                        teacher.append(record)
                except (ValueError, TypeError, AttributeError):
                    notes.append(
                        f"{log.name}:{number}: invalid JSON/step (training may still be writing)"
                    )
    if phase == 2 and path.name == "best_generation":
        best = read_json(path.parent / "best_generation.json")
        if int(best["step"]) != step:
            raise ValueError("best_generation.json step differs from adapter; 不展示其它 step 的指标")
        gen = best.get("generation", {})
        if not isinstance(gen, dict):
            raise ValueError("generation metrics must be an object")
        if not gen and len(generation) == 1:
            gen = generation[0]
        if not gen:
            notes.append("完整 generation 明细缺失，仅有 best 选优分数")
        tf = {
            k.removeprefix("teacher_forced_"): v
            for k, v in best.items()
            if k.startswith("teacher_forced_")
        }
        source = str(path.parent / "best_generation.json")
        guards = best.get("generation_guards", {})
    else:
        gen = generation[0] if len(generation) == 1 else {}
        tf = {}
        source = str(log)
        guards = dict(format_valid_floor=cfg.get("generation_format_valid_gate"))
    if len(teacher) == 1:
        tf.update(teacher[0])
    elif len(teacher) > 1:
        notes.append("同 step teacher-forced 记录不唯一，不选择其中一条")
    if not tf:
        notes.append("保存 step 没有 teacher-forced 指标，不用最近 step 替代")
    return dict(
        generation=gen,
        teacher_forced=tf,
        guards=guards,
        metric_source=source,
        notes=notes,
    )


def scan(root, phase, model_dir):
    """区分未发现、非 best 保存点和合同拒绝；所有检查独立报告。"""
    root = Path(root).expanduser().absolute()
    discovery = discover(root, phase)
    results, other = [], []
    for index, slot in enumerate(discovery["slots"], 1):
        path = Path(slot["path"])
        print(f'[Phase{phase} {index}/{len(discovery["slots"])}] 检查 {path}', flush=True)
        cfg, checks = audit_checks(path, phase, model_dir)
        row = dict(**slot, phase=phase, eligible=False, metadata=cfg, notes=[], checks=checks)
        row["rejection_reasons"] = [
            f'{c["name"]}: {c["detail"] or str(c["actual"]) + " != " + str(c["expected"])}'
            for c in checks if c["status"] == "fail"
        ]
        for check in checks:
            if check["name"] == "selection_score" and check["status"] == "pass":
                row["generation_exact"] = check["actual"]
        try:
            row.update(saved_records(path, cfg, phase))
        except ERRORS as exc:
            row["notes"].append(f"指标读取失败: {exc}")
        if path.name != "best_generation":
            row["rejection"] = "非 best_generation 保存点，仅审计不推荐"
            other.append(row)
            continue
        try:
            inspected = inspect_adapter(path, phase, model_dir)
            score = selection_score(path, cfg, phase)
            row.update(
                eligible=True,
                generation_exact=score,
                fingerprint=inspected["fingerprint"],
                file_sha256=inspected["file_sha256"],
            )
        except ERRORS as exc:
            row["rejection"] = str(exc)
        results.append(row)
    good = sorted(
        (r for r in results if r["eligible"]),
        key=lambda r: (
            r["generation_exact"],
            r["metadata"].get("saved_at", ""),
            r["path"],
        ),
        reverse=True,
    )
    bad = [r for r in results if not r["eligible"]]
    for rank, row in enumerate(good, 1):
        row["rank"] = rank
    return dict(
        phase=phase,
        root=str(root),
        root_exists=root.is_dir(),
        candidates=good + bad,
        other_checkpoints=other,
        discovery=discovery,
        discovery_status=("recommended" if good else "best_generation_rejected" if bad
                          else "only_non_best_slots" if other else "no_phase_checkpoint_found"),
        recommended=good[0]["path"] if good else None,
        selection_rule="compatible best_generation; validation exact descending, saved_at then path descending",
        common_holdout_verified=False,
    )


def scalar_lines(value, prefix=""):
    """递归展开每项原始指标，包括每题样本数、子类 recall、INVALID 与门槛。"""
    if isinstance(value, dict):
        for key in sorted(value):
            yield from scalar_lines(value[key], f"{prefix}/{key}" if prefix else key)
    else:
        yield prefix, value


def format_value(value):
    """缺失指标显示 N/A，浮点保留足够精度，不把缺失当 0。"""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return (
        json.dumps(value, ensure_ascii=False)
        if isinstance(value, (dict, list, bool))
        else str(value)
    )


def show(result, summary_only=False):
    """先给紧凑排名，再逐候选打印完整指标和版本来源。"""
    phase = result["phase"]
    print(f'\nPhase{phase} — {result["root"]}')
    discovery = result["discovery"]
    print(f'目标训练包: {discovery["expected_training_package"]}')
    print(f'其它训练包保存点已排除 {len(discovery["excluded_other_packages"])} 个'
          '（详情见 JSON discovery.excluded_other_packages；不计入本 Phase 排名/拒绝数）')
    print(f'发现状态: {result["discovery_status"]}; best_generation={len(result["candidates"])}; '
          f'其它保存点={len(result["other_checkpoints"])}; '
          f'Phase 压缩包={len(discovery["archives"])}; '
          f'未识别 phase 的保存点={len(discovery["unclassified_slots"])}; '
          f'扫描错误={len(discovery["errors"])}')
    print("Git 只检查训练 commit 是否存在，不要求与当前代码 commit 相等；prompt 名称/哈希必须匹配。")
    if not result["candidates"] and not result["other_checkpoints"]:
        for path in discovery["phase_directories"]:
            print(f"  疑似 Phase{phase} 目录但未找到保存点: {path}")
    for row in discovery["unclassified_slots"]:
        print(f'  phase 未识别: {row["path"]}; configs={row["metadata_files"]}; '
              '缺少新 Phase 元数据/目录线索，请核对实际训练包')
    for link in discovery["links"]:
        print(f'  软链接 [{link["status"]}]: {link["path"]} -> {link["target"]}'
              f'; {link.get("error", "按真实目标去重；目录链接会继续扫描")}')
    for error in discovery["errors"]:
        print(f'  扫描错误: {error["path"]}: {error["error"]}')
    for archive in discovery["archives"]:
        print(f'  压缩包: {archive["path"]}; {archive["reason"]}')
    print(
        "排名  状态       Exact      Format     样本数     Step       RGB              Run"
    )
    for row in result["candidates"]:
        cfg, gen = row["metadata"], row.get("generation", {})
        print(
            f'{str(row.get("rank", "-")):<5} {"可选" if row["eligible"] else "拒绝":<8} '
            f'{format_value(row.get("generation_exact")):<10} {format_value(gen.get("format_valid_rate")):<10} '
            f'{format_value(gen.get("samples")):<10} {str(cfg.get("global_step", "?")):<10} '
            f'{str(cfg.get("history_rgb_mode", "?")):<16} {Path(row["path"]).parent.name}'
        )
    if not result["candidates"]:
        print(
            "没有发现可识别的 best_generation，尚不能归因于 Git/prompt 校验；请查看保存点、软链接与扫描错误。不会使用 final 兜底。"
        )
    for row in result["candidates"] + result["other_checkpoints"]:
        cfg = row["metadata"]
        print(f'\n  路径: {row["path"]}')
        print(f'  保存点: {row["slot"]}; 元数据文件: {row["metadata_files"]}')
        if row["artifact_kind"] == "audit_metadata_only":
            print("  产物类型: 审计包中的 adapter 元数据副本，未包含实际权重，不能加载训练。")
        if row["alternative_metadata"]:
            print("  其它/旧版元数据（仅展示，不替代所需配置）:")
            for key, value in scalar_lines(row["alternative_metadata"]):
                print(f"    {key}: {format_value(value)}")
        print(f'  prompt: {cfg.get("prompt_name", "N/A")}')
        print(f'  prompt SHA: {cfg.get("production_prompt_sha256", "N/A")}')
        git = cfg.get("git") or {}
        print(
            f'  Git: {git.get("commit", "N/A") if isinstance(git, dict) else git}; saved_at: {cfg.get("saved_at", "N/A")}'
        )
        print(
            f'  RGB: {cfg.get("history_rgb_mode", "N/A")}; indices={cfg.get("history_rgb_selected_indices", "N/A")}'
        )
        print(f'  base: {cfg.get("base_model_dir", "N/A")}')
        print(f'  指标来源: {row.get("metric_source", "N/A")}')
        if not row["eligible"]:
            print(f'  拒绝原因: {row["rejection"]}')
        for check in row["checks"]:
            print(f'  [{check["status"]}] {check["name"]}: '
                  f'actual={format_value(check["actual"])}; expected={format_value(check["expected"])}'
                  f'{"; " + check["detail"] if check["detail"] else ""}')
        for note in row.get("notes", []):
            print(f"  注意: {note}")
        if not summary_only:
            for group in ("generation", "teacher_forced", "guards"):
                print(f"  [{group}]")
                for key, value in scalar_lines(row.get(group, {})):
                    print(f"    {key}: {format_value(value)}")
    print(f'\n推荐 Phase{phase}: {result["recommended"] or "无可推荐权重"}')


def main():
    """默认扫描训练入口相同 checkpoints 根；CPU 只读，不实例化 Qwen 或申请 GPU。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint-root", default=os.environ.get("CHECKPOINT_ROOT", "checkpoints")
    )
    p.add_argument("--phase1-root", default="")
    p.add_argument("--phase2-root", default="")
    p.add_argument("--phase", choices=["1", "2", "both"], default="both")
    p.add_argument("--selection-policy", choices=["available", "strict"], default="available")
    p.add_argument("--checkpoint-roots", nargs="+", default=[], help="同时搜索两台服务器已共享的目录")
    p.add_argument("--export-bundle", action=argparse.BooleanOptionalAction, default=True,
                   help="默认导出本次推荐的真实 LoRA 权重包；只审计可 --no-export-bundle")
    p.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "checkpoints/Qwen3-VL-4B-Instruct"),
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="默认 checkpoints/action_prior_lora_audit/run_<时间>，写 report.json/log.txt",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="只打印排名/来源/拒绝原因；JSON 仍含完整指标",
    )
    args = p.parse_args()
    out = Path(
        args.output_dir
        or "checkpoints/action_prior_lora_audit/run_"
        + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    out.mkdir(parents=True, exist_ok=False)
    from qwen3vl_local.run_log import install_output_log

    install_output_log(out)
    print(
        "只读已有验证结果：不重新生成、不读取 test 选优。available 允许新 Phase2 fallback，strict 保持原硬门槛。"
    )
    phases = [1, 2] if args.phase == "both" else [int(args.phase)]
    results = []
    for phase in phases:
        if args.selection_policy == "available":
            from qwen3vl_local.action_prior.available_adapters import scan_available, show_available
            phase_root = getattr(args, f"phase{phase}_root")
            roots = [phase_root] if phase_root else args.checkpoint_roots or [args.checkpoint_root]
            result = scan_available(roots, phase, args.model_dir)
            results.append(result)
            show_available(result, full_metrics=not args.summary_only)
            continue
        result = scan(
            getattr(args, f"phase{phase}_root") or args.checkpoint_root,
            phase,
            args.model_dir,
        )
        results.append(result)
        show(result, args.summary_only)
    print(
        f"\n选择策略: {args.selection_policy}；与相同 selection-policy 的 action 训练一致。"
    )
    print(
        "不同 run 可能使用不同验证样本/采样预算；此排名不代表统一 holdout 上的严格最优。Git 字段表示来源，不保证运行环境一致。"
    )
    if args.selection_policy == "strict":
        print("strict 审计额外跟随目录软链接；strict 训练扫描不跟随这些链接，请用下方显式路径固定结果。")
    else:
        print("available 审计和训练共用多目录/软链接发现；完整流水线会固定预检选出的两阶段权重。")
    if all(r["recommended"] for r in results):
        command = [
            "bash",
            "qwen3vl_local/action_prior/run_full_pipeline.sh",
            "--model-dir",
            args.model_dir,
            "--selection-policy", args.selection_policy,
        ]
        for result in results:
            command += [f'--phase{result["phase"]}-adapter', result["recommended"]]
        print("\n固定推荐权重运行（仅打印，不执行）:\n" + shlex.join(command))
    report = dict(
        schema="action_prior_available_ranking_v1" if args.selection_policy == "available" else "action_prior_lora_ranking_v3",
        selection_policy=args.selection_policy,
        model_dir=str(Path(args.model_dir).resolve()),
        phases=results,
        metrics_are_existing_validation=True,
        common_holdout_verified=False,
    )
    if args.export_bundle:
        try:
            from qwen3vl_local.action_prior.available_adapters import candidate
            from qwen3vl_local.action_prior.lora_bundle import create_bundle, archive_bundle
            from qwen3vl_local.action_prior.contracts import digest
            selected = {}
            for result in results:
                if not result["recommended"]:
                    continue
                phase = result["phase"]
                item = candidate(result["recommended"], phase, args.model_dir)
                scanned = next(r for r in result["candidates"] if r["path"] == result["recommended"])
                if item["generation_exact"] != scanned["generation_exact"] or item["metadata"] != scanned["metadata"]:
                    raise ValueError("candidate changed after ranking; rerun ranking before export")
                selected[f"phase{phase}"] = item
            if selected:
                name = "action_prior_loras_" + digest({k: v["fingerprint"] for k, v in selected.items()})[:16]
                directory = create_bundle(selected, out / name, args.selection_policy,
                                          extra_provenance={"ranking_report": "../report.json"})
                exported = archive_bundle(directory, out / f"{name}.tar.gz")
                report["weight_bundle"] = exported
                for key, item in exported["phases"].items():
                    print(f'[打包 {key}] {item["source_path"]}; step={item["global_step"]}; '
                          f'Exact={item["generation_exact"]:.6f}; prompt={item["prompt_name"]}')
                print(f'\n权重迁移压缩包: {exported["path"]}\nSHA256: {exported["sha256"]}'
                      f'\n大小: {exported["bytes"] / 1024**2:.2f} MiB（完整权重包，不限 30 MB）')
                print("解压到另一台服务器 AutoMoT/checkpoints 后，保留完整目录结构。")
                if set(selected) == {"phase1", "phase2"}:
                    print("固定使用这组 LoRA 训练:\n" + shlex.join([
                        "bash", "qwen3vl_local/action_prior/run_full_pipeline.sh", "--lora-bundle",
                        f"checkpoints/{name}"]))
                    print("本机直接使用导出目录:\n" + shlex.join([
                        "bash", "qwen3vl_local/action_prior/run_full_pipeline.sh", "--lora-bundle", str(directory)]))
                else:
                    print("此包只含有推荐的单个阶段；可解压到共享 checkpoints 与另一阶段合并搜索，尚不能单独启动双阶段训练。")
            else:
                print("没有可推荐权重，不生成空的迁移包。")
        except Exception as exc:
            report["weight_bundle_error"] = str(exc)
            (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"权重包导出失败，未发布可用压缩包: {exc}", file=sys.stderr)
            return 3
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f'\n完整报告: {out / "report.json"}')
    return 0 if all(r["recommended"] for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
