"""选中 LoRA 的独立副本、可迁移权重包和训练目录内恢复；绝不创建源权重软链接。"""

from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

from qwen3vl_local.action_prior.contracts import digest, file_hash, read_json

SCHEMA = "action_prior_lora_bundle_v1"


def _json(path, value):
    """写 UTF-8 来源与校验记录。"""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_path(root, name):
    """清单路径必须是包内普通文件，不能引用旧服务器或包外路径。"""
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid bundle relative path: {name}")
    result = root / path
    if result.is_symlink() or not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"bundle path escapes root: {name}")
    return result


def verify_bundle(root, expected=None):
    """逐文件核对 SHA256；源目录删除后也只读包内路径。"""
    root = Path(root).resolve()
    manifest = read_json(root / "bundle_manifest.json")
    if manifest.get("schema") != SCHEMA or not manifest.get("files"):
        raise ValueError("invalid/incomplete LoRA bundle manifest")
    for name, sha in manifest["files"].items():
        path = _safe_path(root, name)
        if not path.is_file() or file_hash(path) != sha:
            raise ValueError(f"LoRA bundle file missing/changed: {name}")
    for phase, item in manifest["phases"].items():
        path = _safe_path(root, item["relative_path"])
        if not path.is_dir() or not item.get("file_sha256"):
            raise ValueError(f"bundle missing adapter: {phase}")
        for name, sha in item["file_sha256"].items():
            rel = str(Path(item["relative_path"]) / name)
            if manifest["files"].get(rel) != sha:
                raise ValueError(f"bundle adapter fingerprint not covered: {phase}/{name}")
        if digest(item["file_sha256"]) != item["fingerprint"]:
            raise ValueError(f"bundle fingerprint mismatch: {phase}")
        if expected and item["fingerprint"] != expected[phase]["fingerprint"]:
            raise ValueError(f"local LoRA differs from selected checkpoint: {phase}")
    if expected and set(manifest["phases"]) != {k for k in expected if k in ("phase1", "phase2")}:
        raise ValueError("bundle phase set differs from selected pair")
    return manifest


def bundle_paths(root, expected=None):
    """返回已核验的包内绝对路径，供模型加载使用。"""
    manifest = verify_bundle(root, expected)
    return {key: str(_safe_path(Path(root).resolve(), item["relative_path"]))
            for key, item in manifest["phases"].items()}


def create_bundle(selected, destination, policy="available", extra_provenance=None):
    """复制选中保存点与所需旁车指标，完成后原子发布；不复制中间步骤或 optimizer。"""
    from qwen3vl_local.action_prior.available_adapters import candidate
    from qwen3vl_local.action_prior.prompt_versions import prompt_module

    destination = Path(destination).resolve()
    if destination.exists():
        verify_bundle(destination, selected)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".lora_copy_", dir=destination.parent))
    try:
        phases = {}
        for key in sorted(selected):
            phase = int(key.removeprefix("phase"))
            source_item = selected[key]
            source = Path(source_item["path"]).resolve()
            print(f"[LoRA copy {key}] {source} -> {destination / key / source.name}", flush=True)
            # 对源再校验：排名后训练端仍可能保存同名 best，不能悄悄复制成另一组。
            current = candidate(source, phase, source_item.get("effective_base_model_dir",
                                source_item["metadata"]["base_model_dir"]))
            if current["fingerprint"] != source_item["fingerprint"]:
                raise ValueError(f"source adapter changed before copy: {source}")
            run = staging / key
            target = run / source.name
            target.mkdir(parents=True)
            for name, sha in source_item["file_sha256"].items():
                shutil.copyfile(source / name, target / name)
                if file_hash(target / name) != sha:
                    raise ValueError(f"source changed during copy: {source / name}")
            # adapter 身份只包含原配置/权重；这些字节保持不变，搬迁信息写在独立清单。
            for optional in ("README.md", "training_summary.md"):
                if (source / optional).is_file():
                    shutil.copyfile(source / optional, target / optional)
            step = int(source_item["metadata"]["global_step"])
            metrics = source.parent / "train_eval_metrics.jsonl"
            saved_rows = []
            if metrics.is_file():
                with metrics.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        if int(record.get("step", -1)) == step:
                            saved_rows.append(record)
            # Phase2 有些 run 只在 slot.json 内保存详细指标，也要保留即可独立运行。
            if saved_rows:
                (run / "train_eval_metrics.jsonl").write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in saved_rows), encoding="utf-8")
            record = source.parent / f"{source.name}.json"
            if record.is_file():
                shutil.copyfile(record, run / record.name)
            # 保存 run 的轻量训练来源，数据索引本体不在权重迁移包中。
            for name in ("train_run_manifest.json", "train_config.json", "config.json"):
                if (source.parent / name).is_file():
                    shutil.copyfile(source.parent / name, run / name)
            provenance = copy.deepcopy(source_item)
            provenance.pop("search_audit", None)
            _json(run / "source_provenance.json", provenance)
            module = prompt_module(phase, source_item["metadata"])
            snapshots = staging / "prompt_sources"
            snapshots.mkdir(exist_ok=True)
            # 源码仅存档，不动态执行迁移包里的 Python；目标仓库用已支持协议重建并核对 hash。
            module_file = Path(module.__file__)
            shutil.copyfile(module_file, snapshots / f"{key}_{module_file.name}")
            # 重新从复制品读取指标/配置，证明不依赖原 run 的其它 checkpoint。
            copied = candidate(target, phase, current["effective_base_model_dir"])
            if copied["fingerprint"] != source_item["fingerprint"]:
                raise ValueError("copied adapter identity mismatch")
            phases[key] = dict(relative_path=str(target.relative_to(staging)),
                               fingerprint=source_item["fingerprint"], file_sha256=source_item["file_sha256"],
                               source_path=str(source), prompt_name=module.PROMPT_NAME,
                               prompt_sha256=source_item["metadata"]["production_prompt_sha256"],
                               git=source_item["metadata"].get("git"), global_step=step,
                               history_rgb_mode=source_item["metadata"]["history_rgb_mode"],
                               generation_exact=current["generation_exact"], slot=source.name,
                               warnings=current["warnings"])
        exporter_git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
                                      capture_output=True, text=True).stdout.strip() or None
        exporter_status = subprocess.run(["git", "status", "--short", "--untracked-files=no"],
                                        cwd=Path(__file__).resolve().parents[3],
                                        capture_output=True, text=True).stdout.strip()
        # 存档 prompt 的本地依赖；这些是源码快照，不是旧训练包权重。
        from qwen3vl_local.action_prior.provenance import execution_sources
        code_root = Path(__file__).resolve().parents[2]
        for source in execution_sources(code_root):
            if source.name in ("prompts.py", "history_rgb.py", "phase2_v3_prompts.py", "prompt_versions.py"):
                target = staging / "prompt_sources" / source.relative_to(code_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        (staging / "README.md").write_text(
            "# Action prior selected LoRA bundle\n\n"
            "解压整个目录到 AutoMoT/checkpoints。保留目录结构与所有文件。\n"
            "从 AutoMoT/ 运行：\n\n"
            "    bash qwen3vl_local/action_prior/run_full_pipeline.sh --lora-bundle checkpoints/" + destination.name + "\n\n"
            "仅包含所选 Phase1/2 LoRA；不含 Qwen 基座、BEV、数据集或 action decoder。\n"
            "目标服务器需使用包含 lora_bundle 功能及对应 prompt 协议的项目代码，准备本地 "
            "Qwen3-VL-4B-Instruct 和 BEV；可用 --model-dir / --lead-bev-ckpt 指定。\n"
            "原始 Git/配置保持不变，缺失 Git 不补造。source_provenance.json 记录来源；"
            "prompt_sources 只用于审计，不自动执行。bundle_manifest.json 校验所有包内文件。\n"
            "这是完整权重迁移包，不适用 30 MB 指标审计包限制。\n", encoding="utf-8")
        files = {str(p.relative_to(staging)): file_hash(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        manifest = dict(schema=SCHEMA, selection_policy=policy, phases=phases, files=files,
                        exporter_git_commit=exporter_git, extra_provenance=extra_provenance or {},
                        exporter_git_dirty=bool(exporter_status), exporter_tracked_changes=exporter_status.splitlines(),
                        requirements={"base_model": "Qwen3-VL-4B-Instruct", "BEV": "external local checkpoint",
                                      "project_feature": "action_prior/lora_bundle.py"})
        _json(staging / "bundle_manifest.json", manifest)
        verify_bundle(staging, selected)
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)  # 只删除此函数新建的未发布临时副本，不触碰源模型。
    return destination


def archive_bundle(root, output):
    """打包后逐成员解压流校验，确认真实权重在包内；不跟随源软链接归档。"""
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = verify_bundle(root)
    print(f"[LoRA archive] {root} -> {output}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    fd, name = tempfile.mkstemp(prefix=".lora_archive_", dir=output.parent)
    os.close(fd)
    try:
        paths = {**manifest["files"], "bundle_manifest.json": file_hash(root / "bundle_manifest.json")}
        with tarfile.open(name, "w:gz", compresslevel=1) as archive:
            for rel in sorted(paths):
                archive.add(_safe_path(root, rel), arcname=f"{root.name}/{rel}", recursive=False)
        print("[LoRA archive] 校验压缩包中所有文件的解压内容 SHA256", flush=True)
        with tarfile.open(name, "r:gz") as archive:
            seen = set()
            for member in archive:
                rel = str(Path(member.name).relative_to(root.name))
                if rel not in paths or not member.isfile() or rel in seen:
                    raise ValueError("unexpected/duplicate/non-file archive member")
                handle = archive.extractfile(member)
                sha = hashlib.sha256()
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    sha.update(block)
                if sha.hexdigest() != paths[rel]:
                    raise ValueError(f"archive verification failed: {rel}")
                seen.add(rel)
            if seen != set(paths):
                raise ValueError("archive is incomplete")
        os.replace(name, output)
    finally:
        Path(name).unlink(missing_ok=True)
    checksum = file_hash(output)
    output.with_name(output.name + ".sha256").write_text(f"{checksum}  {output.name}\n")
    return dict(path=str(output), sha256=checksum, bytes=output.stat().st_size,
                unpacked_directory=root.name, phases=manifest["phases"])


def preserve_for_training(contract, output_dir):
    """训练前复制到 action run/lora；合同内容身份不变，加载路径改为本地副本。"""
    output_dir = Path(output_dir).resolve()
    selected = {key: contract[key] for key in ("phase1", "phase2")}
    root = create_bundle(selected, output_dir / "lora", contract.get("selection_policy", "available"),
                         extra_provenance={"upstream_sources": contract.get("upstream_sources", {}),
                                           "action_git": contract.get("git_commit")})
    paths = bundle_paths(root, selected)
    result = copy.deepcopy(contract)
    result["local_lora"] = dict(directory="lora", schema=SCHEMA)
    for key, path in paths.items():
        result[key]["original_selection_path"] = result[key].get("original_selection_path", result[key]["path"])
        result[key]["path"] = path
        result[key]["local_relative_path"] = str(Path(path).relative_to(output_dir))
    print(f"[LoRA saved in action run] {root}", flush=True)
    return result


def restore_paths(contract, checkpoint):
    """优先当前 action checkpoint 旁的副本；已声明本地副本却缺失时不静默搜索上游。"""
    root = Path(checkpoint).resolve().parent / "lora"
    selected = {key: contract[key] for key in ("phase1", "phase2")}
    if root.exists() or contract.get("local_lora"):
        return bundle_paths(root, selected)
    return {key: item["path"] for key, item in selected.items()}  # 旧无副本 run 兼容。
