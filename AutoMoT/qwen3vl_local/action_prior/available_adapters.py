"""用户授权的现有模型选优：新 Phase1 best + 新 Phase2 best/fallback，来源缺失如实审计。"""

from __future__ import annotations
import math
import json
from pathlib import Path

from qwen3vl_local.action_prior.contracts import digest, file_hash, read_json, selection_score
from qwen3vl_local.action_prior.lora_audit import ERRORS, discover
from qwen3vl_local.action_prior.prompt_versions import prompt_module
from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices

POLICY = "available_new_phase_best_then_fallback_v1"


def inspect_available(path, phase, model_dir, *, hash_files=True):
    """保留 RGB/提示词/PEFT 硬校验；跨服务器基座路径和空 Git 作为显式来源警告。"""
    path = Path(path).expanduser().resolve()
    # 默认递归搜索到迁移包或 action 副本时也验清单，不能只在 --lora-bundle 时才防损坏。
    for parent in path.parents:
        if (parent / "bundle_manifest.json").is_file():
            from qwen3vl_local.action_prior.lora_bundle import verify_bundle
            manifest = verify_bundle(parent)
            entry = manifest["phases"].get(f"phase{phase}")
            if not entry or (parent / entry["relative_path"]).resolve() != path:
                raise ValueError("adapter is not the declared phase in its enclosing LoRA bundle")
            break
    filename = f"sft_new_loop_phase{phase}_adapter_config.json"
    cfg = read_json(path / filename)
    module = prompt_module(phase, cfg)
    indices = list(history_rgb_indices(cfg["history_rgb_mode"]))
    if cfg.get("history_rgb_selected_indices") != indices or cfg.get("history_rgb_count") != len(indices):
        raise ValueError("inconsistent RGB metadata")
    peft = read_json(path / "adapter_config.json")
    if peft.get("peft_type") != "LORA" or peft.get("bias", "none") != "none" or peft.get("modules_to_save"):
        raise ValueError("requires plain LoRA, bias=none, no modules_to_save")
    weights = [path / n for n in ("adapter_model.safetensors", "adapter_model.bin") if (path / n).is_file()]
    if len(weights) != 1 or weights[0].stat().st_size == 0:
        raise ValueError("expected exactly one nonempty adapter weight file")
    warnings = []
    commit = (cfg.get("git") or {}).get("commit")
    if not commit:
        warnings.append("missing_training_git: 原始训练 commit 未记录，保持缺失；绑定当前实际文件 SHA256")
    recorded_base = cfg.get("base_model_dir")
    if not recorded_base:
        raise ValueError("missing base_model_dir")
    target = Path(model_dir).expanduser().resolve()
    original = Path(recorded_base).expanduser()
    if original.resolve() != target:
        # 共享服务器目录迁移不应拒绝同名基座；不能把普通 Qwen 或其它尺寸当作同一基座。
        expected_name = Path(model_dir).expanduser().name
        if original.name != expected_name or expected_name != "Qwen3-VL-4B-Instruct":
            raise ValueError("base model family mismatch; expected Qwen3-VL-4B-Instruct for path remap")
        warnings.append("base_path_remapped: 同名 Qwen3-VL-4B-Instruct 按共享基座使用；原服务器权重字节一致性未验证")
    if phase == 2 and module.__name__.endswith("phase2_v3_prompts"):
        warnings.append("historical_event_v3: 使用冻结的新 Phase2 v3 提示词；不冒充 v5 高速 UE3 协议")
    item = dict(path=str(path), phase=phase, metadata=cfg, warnings=warnings,
                selection_policy=POLICY, runtime_prompt_name=module.PROMPT_NAME,
                runtime_prompt_sha256=cfg["production_prompt_sha256"],
                base_path_remapped=original.resolve() != target,
                original_base_model_dir=recorded_base, effective_base_model_dir=str(target))
    if hash_files:
        files = {p.name: file_hash(p) for p in (path / filename, path / "adapter_config.json", *weights)}
        item.update(file_sha256=files, fingerprint=digest(files))
    return item


def saved_score(path, cfg, phase):
    """只取该保存步验证结果；fallback 允许门槛失败但不能缺指标/格式不合格。"""
    path = Path(path)
    if path.name not in (("best_generation",) if phase == 1 else ("best_generation", "fallback_generation")):
        raise ValueError("only Phase1 best_generation / Phase2 best_generation or fallback_generation")
    step = int(cfg["global_step"])
    if phase == 1:
        score = selection_score(path, cfg, phase)
        records, teacher = [], []
        for line in (path.parent / "train_eval_metrics.jsonl").read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("type") == "generation" and int(record.get("step", -1)) == step:
                    records.append(record)
                if record.get("type", record.get("kind")) == "teacher_forced" and int(record.get("step", -1)) == step:
                    teacher.append(record)
        return dict(generation_exact=score, generation=records[0],
                    teacher_forced=teacher[0] if len(teacher) == 1 else {},
                    metric_source=str(path.parent / "train_eval_metrics.jsonl"),
                    guards={"format_valid_floor": cfg["generation_format_valid_gate"]}, warnings=[])
    record_path = path.parent / f"{path.name}.json"
    record = read_json(record_path)
    if int(record["step"]) != step:
        raise ValueError("selection record step differs from adapter")
    generation = record.get("generation") or {}
    if not generation:
        matches = []
        for line in (path.parent / "train_eval_metrics.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if int(r.get("step", -1)) == step and r.get("type", r.get("kind")) in ("generation", "free_generation"):
                matches.append(r)
        if len(matches) != 1:
            raise ValueError("missing/ambiguous saved-step generation metrics")
        generation = matches[0]
    if "step" in generation and int(generation["step"]) != step:
        raise ValueError("embedded generation step differs from adapter")
    score = float(record["generation_exact_accuracy"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("invalid generation exact score")
    if "exact_accuracy" in generation and not math.isclose(float(generation["exact_accuracy"]), score, abs_tol=1e-9):
        raise ValueError("selection exact differs from saved generation metrics")
    valid, floor = float(generation["format_valid_rate"]), float(cfg.get("generation_eval_min_valid_rate", 1.0))
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (valid, floor)) or valid < floor:
        raise ValueError("saved generation fails format guard")
    guards_ok = record.get("generation_guards_ok")
    if not isinstance(guards_ok, bool):
        raise ValueError("missing generation_guards_ok")
    if path.name == "best_generation" and not guards_ok:
        raise ValueError("best_generation guards failed; not a valid best slot")
    warnings = [] if guards_ok else ["fallback_guards_failed: 该保存步未通过原训练全部门槛，用户已授权用现有候选训练 action"]
    return dict(generation_exact=score, generation=generation,
                teacher_forced={k.removeprefix("teacher_forced_"): v for k, v in record.items()
                                if k.startswith("teacher_forced_")},
                guards=record.get("generation_guards", {}), generation_guards_ok=guards_ok,
                metric_source=str(record_path), warnings=warnings)


def candidate(path, phase, model_dir, *, hash_files=True):
    """将运行兼容检查与真实保存步指标组合成一个候选。"""
    item = inspect_available(path, phase, model_dir, hash_files=hash_files)
    score = saved_score(Path(item["path"]), item["metadata"], phase)
    warnings = item["warnings"] + score.pop("warnings")
    return {**item, **score, "warnings": warnings}


def scan_available(roots, phase, model_dir):
    """扫描多个共享目录及软链接；日期只作路径，不作为过滤条件。"""
    roots = [str(Path(p).expanduser().absolute()) for p in roots]
    paths, diagnostics = set(), []
    for root in roots:
        print(f"[search Phase{phase}] {root}", flush=True)
        found = discover(root, phase)
        diagnostics.append(dict(root=root, errors=found["errors"], links=found["links"],
                                excluded_other_packages=len(found["excluded_other_packages"])))
        for row in found["slots"]:
            if row["slot"] in ("best_generation", "fallback_generation"):
                paths.add(row["path"])
    rows = []
    for path in sorted(paths):
        try:
            item = candidate(path, phase, model_dir, hash_files=False)
            rows.append(dict(item, eligible=True))
        except ERRORS as exc:
            rows.append(dict(path=path, phase=phase, eligible=False, rejection=str(exc)))
    good = sorted((r for r in rows if r["eligible"]), key=lambda r: (
        Path(r["path"]).name == "best_generation", r["generation_exact"],
        str(r["metadata"].get("saved_at") or ""), r["path"]), reverse=True)
    for rank, row in enumerate(good, 1):
        row["rank"] = rank
    return dict(phase=phase, roots=roots, discovery=diagnostics,
                candidates=good + [r for r in rows if not r["eligible"]],
                recommended=good[0]["path"] if good else None, selection_policy=POLICY,
                common_holdout_verified=False)


def show_available(result, full_metrics=False):
    """预检与训练均打印实际选择和质量限制，而不是仅输出巨大合同 JSON。"""
    print(f'\nPhase{result["phase"]} 现有模型候选（best 优先，同类 validation Exact 降序）', flush=True)
    for row in result["candidates"]:
        if not row["eligible"]:
            print(f'  [拒绝] {row["path"]}: {row["rejection"]}', flush=True)
            continue
        print(f'  #{row["rank"]} Exact={row["generation_exact"]:.6f} '
              f'RGB={row["metadata"]["history_rgb_mode"]} {row["path"]}', flush=True)
        for warning in row["warnings"]:
            print(f"    [来源/质量说明] {warning}", flush=True)
        for key, value in row.get("generation", {}).items():
            if key.startswith("slice/") and ("recall" in key or key.endswith("_exact")):
                print(f"    {key}={value}", flush=True)
        if full_metrics:
            from qwen3vl_local.action_prior.rank_loras import scalar_lines
            for group in ("generation", "teacher_forced", "guards"):
                for key, value in scalar_lines(row.get(group, {})):
                    print(f"    {group}/{key}={value}", flush=True)
    for discovery in result["discovery"]:
        for error in discovery["errors"]:
            print(f'  [扫描错误] {error}', flush=True)
    print(f'推荐 Phase{result["phase"]}: {result["recommended"] or "无可推荐权重"}', flush=True)


def select_available(roots, phase, model_dir, explicit=""):
    """显式路径用于固定已选权重和恢复；同样执行全部运行兼容与保存步检查。"""
    if explicit:
        result = candidate(explicit, phase, model_dir)
        print(f'[pinned Phase{phase}] {result["path"]}; Exact={result["generation_exact"]:.6f}', flush=True)
        for warning in result["warnings"]:
            print(f"  [来源/质量说明] {warning}", flush=True)
    else:
        audit = scan_available(roots, phase, model_dir)
        show_available(audit)
        if not audit["recommended"]:
            raise ValueError(f"Phase{phase}: no usable new-package best/fallback; see search diagnostics")
        result = candidate(audit["recommended"], phase, model_dir)
        result["search_audit"] = audit
    result.update(selection="explicit_available" if explicit else POLICY,
                  is_best_generation=Path(result["path"]).name == "best_generation")
    return result
