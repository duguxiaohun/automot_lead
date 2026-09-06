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
    if phase == 2:
        best = read_json(path.parent / "best_generation.json")
        gen = best.get("generation", {})
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
    """发现去重后的 best_generation，保留不兼容候选及其指标供诊断。"""
    root = Path(root).expanduser().resolve()
    filename = f"sft_new_loop_phase{phase}_adapter_config.json"
    paths = {
        p.parent.resolve()
        for p in root.rglob(filename)
        if p.parent.name == "best_generation"
    }
    # root 也允许直接指向一个 best_generation；发现不完整 slot 时仍给出拒绝原因。
    if root.name == "best_generation" and root.is_dir():
        paths.add(root)
    results = []
    for index, path in enumerate(sorted(paths), 1):
        print(f"[Phase{phase} {index}/{len(paths)}] 检查 {path}", flush=True)
        row = dict(path=str(path), phase=phase, eligible=False, metadata={}, notes=[])
        try:
            cfg = read_json(path / filename)
            if not isinstance(cfg, dict):
                raise ValueError("adapter metadata must be an object")
            row["metadata"] = cfg
            try:
                row.update(saved_records(path, cfg, phase))
            except (ValueError, KeyError, OSError, TypeError, AttributeError) as exc:
                row["notes"].append(f"指标读取失败: {exc}")
            inspected = inspect_adapter(path, phase, model_dir)
            score = selection_score(path, cfg, phase)
            row.update(
                eligible=True,
                generation_exact=score,
                fingerprint=inspected["fingerprint"],
                file_sha256=inspected["file_sha256"],
            )
        except (ValueError, KeyError, OSError, TypeError, AttributeError) as exc:
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
            "没有发现 best_generation；请检查 --checkpoint-root / --phaseN-root。不会使用 final 兜底。"
        )
    for row in result["candidates"]:
        cfg = row["metadata"]
        print(f'\n  路径: {row["path"]}')
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
        "只读已有验证结果：不重新生成、不读取 test 选优、不回退 final/best_generation_balanced。"
    )
    phases = [1, 2] if args.phase == "both" else [int(args.phase)]
    results = []
    for phase in phases:
        result = scan(
            getattr(args, f"phase{phase}_root") or args.checkpoint_root,
            phase,
            args.model_dir,
        )
        results.append(result)
        show(result, args.summary_only)
    print(
        "\n推荐依据与 action_prior 自动加载一致：兼容 best_generation 的 validation exact 最大。"
    )
    print(
        "不同 run 可能使用不同验证样本/采样预算；此排名不代表统一 holdout 上的严格最优。Git 字段表示来源，不保证运行环境一致。"
    )
    if all(r["recommended"] for r in results):
        command = [
            "bash",
            "qwen3vl_local/action_prior/run_full_pipeline.sh",
            "--model-dir",
            args.model_dir,
        ]
        for result in results:
            command += [f'--phase{result["phase"]}-adapter', result["recommended"]]
        print("\n固定推荐权重运行（仅打印，不执行）:\n" + shlex.join(command))
    report = dict(
        schema="action_prior_lora_ranking_v1",
        model_dir=str(Path(args.model_dir).resolve()),
        phases=results,
        metrics_are_existing_validation=True,
        common_holdout_verified=False,
    )
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f'\n完整报告: {out / "report.json"}')
    return 0 if all(r["recommended"] for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
