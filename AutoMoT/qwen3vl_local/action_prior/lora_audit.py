"""只读发现 LoRA 保存点，并独立列出合同各项检查；不加载模型、不修补元数据。"""

from __future__ import annotations
import os
from pathlib import Path
import re

from qwen3vl_local.action_prior.contracts import (
    history_rgb_indices, p1, p2, read_json, selection_score,
)

ERRORS = (ValueError, KeyError, OSError, TypeError, AttributeError, RuntimeError, OverflowError)


def phase_hint(path):
    """目录名只是发现线索，不能替代新 Phase2 EVENT 合同校验。"""
    for part in reversed(Path(path).parts):
        match = re.search(r"(?:^|[_-])phase([123])(?:$|[_-])", part.lower())
        if match:
            return int(match[1])
    return None


def discover(root, phase):
    """跟随目录软链接并按真实路径去重；暴露非 best、旧配置、断链与访问失败。"""
    root = Path(root).expanduser().absolute()
    expected = f"sft_new_loop_phase{phase}_adapter_config.json"
    result = dict(slots=[], phase_directories=[], unclassified_slots=[], links=[], archives=[],
                  errors=[], visited_directories=0, follows_directory_symlinks=True)
    pending, seen = [root], set()
    while pending:
        path = pending.pop()
        try:
            resolved = path.resolve(strict=True)
            if resolved in seen:
                continue
            seen.add(resolved)
            with os.scandir(path) as scan:
                entries = sorted(scan, key=lambda e: e.name)
            result["visited_directories"] += 1
            names = {e.name for e in entries}
            # 精确元数据优先于目录命名；旧 sft_loop_phase2 仅作疑似项，不能自动改名加载。
            phases = [n for n in (1, 2, 3)
                      if f"sft_new_loop_phase{n}_adapter_config.json" in names]
            hint = phase_hint(path) or phase_hint(resolved)
            belongs = phase in phases if phases else hint == phase
            # 只记录命名入口；不把一个审计包里的数万 case 子目录重复列为 Phase 目录。
            if phase_hint(path.name) == phase or (path == root and belongs):
                result["phase_directories"].append(str(path))
            configs = sorted(n for n in names if n.endswith("adapter_config.json"))
            weights = sorted(n for n in names if n in (
                "adapter_model.safetensors", "adapter_model.bin"))
            is_slot = (bool(configs or weights) or resolved.name in (
                "best_generation", "final", "fallback_generation", "best_val",
                "best_generation_balanced") or resolved.name.startswith("checkpoint-"))
            if is_slot:
                alternatives = {}
                if expected not in names:
                    for name in configs:
                        if name == "adapter_config.json":
                            continue
                        try:
                            old = read_json(path / name)
                            alternatives[name] = {k: old.get(k) for k in (
                                "prompt_name", "production_prompt_sha256", "git",
                                "history_rgb_mode", "global_step", "base_model_dir")}
                        except ERRORS as exc:
                            alternatives[name] = {"error": str(exc)}
                row = dict(path=str(resolved), discovered_as=str(path),
                           slot=resolved.name, metadata_files=configs, weight_files=weights,
                           alternative_metadata=alternatives,
                           artifact_kind=("audit_metadata_only" if "adapter_metadata" in path.parts
                                          and not weights else "adapter_slot"),
                           expected_metadata_present=expected in names,
                           phase_evidence="exact_metadata" if phase in phases else "directory_hint")
                if belongs or (path == root and resolved.name == "best_generation" and not phases):
                    result["slots"].append(row)
                elif not phases and hint is None:
                    row["phase_evidence"] = "unknown"
                    result["unclassified_slots"].append(row)
            for entry in reversed(entries):
                child = Path(entry.path)
                try:
                    if entry.is_symlink():
                        link = dict(path=str(child), target=os.readlink(child))
                        result["links"].append(link)
                        try:
                            target = child.resolve(strict=True)
                            link.update(status="resolved", resolved=str(target))
                            if target in seen:
                                link["status"] = "already_visited"
                        except ERRORS as exc:
                            link.update(status="unavailable", error=str(exc))
                            continue
                    if entry.is_dir(follow_symlinks=True) and entry.name not in (
                        ".git", "__pycache__"):
                        pending.append(child)
                    elif phase_hint(entry.name) == phase and entry.name.endswith((
                        ".zip", ".tar", ".tar.gz", ".tgz")):
                        result["archives"].append(dict(path=str(child),
                            reason="压缩文件仅列出，未解包；不能作为本地 adapter 目录加载"))
                except ERRORS as exc:
                    result["errors"].append(dict(path=str(child), error=str(exc)))
        except ERRORS as exc:
            result["errors"].append(dict(path=str(path), error=str(exc)))
    for key in ("slots", "unclassified_slots", "links", "errors"):
        result[key].sort(key=lambda row: row["path"])
    return result


def audit_checks(path, phase, model_dir):
    """逐项检查，不因第一个 prompt/Git 错误遮住后续问题；判据沿用生产合同。"""
    path = Path(path)
    checks = []

    def add(name, actual, expected, passed, detail=""):
        checks.append(dict(name=name, status="pass" if passed else "fail",
                           actual=actual, expected=expected, detail=detail))

    def document(filename, key):
        try:
            cfg = read_json(path / filename)
            if not isinstance(cfg, dict):
                raise ValueError("JSON must be an object")
            add(key, filename, "readable JSON object", True)
            return cfg
        except ERRORS as exc:
            add(key, filename, "readable JSON object", False, str(exc))
            return {}

    cfg = document(f"sft_new_loop_phase{phase}_adapter_config.json", "phase_metadata")
    peft = document("adapter_config.json", "peft_metadata")
    module = p1 if phase == 1 else p2
    add("slot", path.name, "best_generation", path.name == "best_generation",
        "其它保存点仅审计，绝不替代 best_generation 推荐")
    add("prompt_name", cfg.get("prompt_name"), module.PROMPT_NAME,
        cfg.get("prompt_name") == module.PROMPT_NAME)
    mode = cfg.get("history_rgb_mode")
    try:
        indices = list(history_rgb_indices(mode))
        add("rgb_mode", mode, "4rgb or 2rgb_endpoints", True)
        expected_hash = (p1.phase1_prompt_sha256 if phase == 1 else p2.event_prompt_sha256)(
            history_rgb_mode=mode)
        add("prompt_hash", cfg.get("production_prompt_sha256"), expected_hash,
            cfg.get("production_prompt_sha256") == expected_hash)
        add("rgb_indices", cfg.get("history_rgb_selected_indices"), indices,
            cfg.get("history_rgb_selected_indices") == indices)
        add("rgb_count", cfg.get("history_rgb_count"), len(indices),
            cfg.get("history_rgb_count") == len(indices))
    except ERRORS as exc:
        add("rgb_mode", mode, "4rgb or 2rgb_endpoints", False, str(exc))
        for name, actual in (("prompt_hash", cfg.get("production_prompt_sha256")),
                             ("rgb_indices", cfg.get("history_rgb_selected_indices")),
                             ("rgb_count", cfg.get("history_rgb_count"))):
            checks.append(dict(name=name, status="unknown", actual=actual,
                               expected="requires valid history_rgb_mode", detail=""))
    git = cfg.get("git")
    commit = git.get("commit") if isinstance(git, dict) else None
    add("training_git_commit", commit, "nonempty training git.commit", bool(commit),
        "只要求训练 commit 有记录；不要求等于当前 checkout，不替历史权重补写 commit")
    base = cfg.get("base_model_dir")
    expected_base = str(Path(model_dir).expanduser().resolve())
    try:
        resolved_base = str(Path(base).expanduser().resolve())
    except ERRORS:
        resolved_base = None
    add("base_model_dir", base, expected_base, resolved_base == expected_base,
        f"resolved={resolved_base}; 这里只核对路径，未加载基座")
    add("plain_lora", {k: peft.get(k) for k in ("peft_type", "bias", "modules_to_save")},
        "LORA; bias absent/none; no modules_to_save",
        peft.get("peft_type") == "LORA" and peft.get("bias", "none") == "none"
        and not peft.get("modules_to_save"))
    weights = [n for n in ("adapter_model.safetensors", "adapter_model.bin")
               if (path / n).is_file()]
    add("weight_files", weights, "exactly one adapter weight file", len(weights) == 1,
        "通过生产 inspect_adapter 后另存实际文件 SHA256；不做 GPU 加载测试")
    if path.name == "best_generation":
        # guards 和 step 分开，避免两者均错时只看到第一个。
        if phase == 2:
            try:
                best = read_json(path.parent / "best_generation.json")
                add("generation_guards", best.get("generation_guards_ok"), True,
                    best.get("generation_guards_ok") is True)
                same_step = int(best.get("step", -1)) == int(cfg["global_step"])
                add("selection_step", best.get("step"), cfg.get("global_step"), same_step)
            except ERRORS as exc:
                add("selection_record", None, "readable best_generation.json with saved step",
                    False, str(exc))
        try:
            score = selection_score(path, cfg, phase)
            add("selection_score", score, "saved-step validation exact in [0,1], guards pass", True)
        except ERRORS as exc:
            add("selection_score", None, "saved-step validation exact in [0,1], guards pass",
                False, str(exc))
    else:
        checks.append(dict(name="selection_score", status="not_applicable", actual=None,
                           expected="best_generation only", detail="不借用其它 slot 的 best 分数"))
    return cfg, checks
