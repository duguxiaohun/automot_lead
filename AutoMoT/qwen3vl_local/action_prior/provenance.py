"""执行代码与上游训练候选池审计；候选池重叠不冒称实际采样命中。"""

from collections import Counter
from importlib import metadata
from pathlib import Path
from qwen3vl_local.action_prior.contracts import file_hash, read_json, digest
from qwen3vl_local.action_prior.build_dataset import route_group


# 真实 action 入口及已核对的延迟调用；不递归扫描所有实验目录。
EXECUTION_SEEDS = (
    *[
        "qwen3vl_local/action_prior/" + name + ".py"
        for name in (
            "__init__",
            "runtime",
            "prompts",
            "priors",
            "precision",
            "text_cache",
            "config",
            "contracts",
            "train",
            "build_dataset",
            "metrics",
            "provenance",
        )
    ],
    "qwen3vl_local/engine.py",
    "qwen3vl_local/mrope_utils.py",
    "qwen3vl_local/leadmot/train.py",
    "qwen3vl_local/leadmot/decoder.py",
    "qwen3vl_local/sft_new_loop_phase1/prompts.py",
    "qwen3vl_local/sft_new_loop_phase2/prompts.py",
    "lead_video_tools/abnormal_duration_filter.py",
    "leaderboard/team_code/mot_lead_offline_runner.py",
    "Automot/mot/modeling/bev_encoder/bev_encoder_utils.py",
)


def execution_sources(root):
    """展开入口的模块级本地 import 和 package initializer；延迟运行依赖由 seeds 显式声明。

    不追踪未调用方法中的导入，如 runner 的 CLI GoalGen/subgoal 分支。模块自身按全文哈希，
    因而修改真实依赖仍失效；新增执行路径时须补 seed，不能退回目录全量递归。
    """
    import ast

    root = Path(root)
    search = (
        root,
        root / "Automot",
        root / "leaderboard/team_code",
        root / "Automot/mot/modeling/bev_encoder",
    )

    def resolve(module):
        """仅检查本地 Python 路径，不 import 第三方库或模型。"""
        for base in search:
            path = base.joinpath(*module.split("."))
            for candidate in (path.with_suffix(".py"), path / "__init__.py"):
                if candidate.is_file():
                    return candidate
        return None

    def module_imports(nodes):
        """if/try 的模块初始化也执行，函数/类方法体留给显式运行入口。"""
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                yield node
            else:
                yield from module_imports(ast.iter_child_nodes(node))

    pending = [root / name for name in EXECUTION_SEEDS]
    paths = set()
    while pending:
        path = pending.pop()
        if path in paths:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"required execution source: {path}")
        paths.add(path)
        for parent in path.parents:
            if parent == root:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file() and initializer not in paths:
                pending.append(initializer)
        relative = path.relative_to(root).with_suffix("")
        package = list(relative.parts[:-1])
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in module_imports(tree.body):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            else:
                prefix = package[: len(package) - node.level + 1] if node.level else []
                base = ".".join([*prefix, *([node.module] if node.module else [])])
                names = [base] + [
                    base + "." + item.name for item in node.names if item.name != "*"
                ]
            for name in names:
                dependency = resolve(name)
                if dependency and dependency not in paths:
                    pending.append(dependency)
    return paths


def execution_fingerprint(root=None):
    """兼容身份只绑定实际入口的依赖集合和包版本；无关 Phase3 不参与。"""
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    paths = execution_sources(root)
    versions = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "numpy",
        "Pillow",
        "timm",
        "laspy",
        "opencv-python",
        "safetensors",
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return dict(
        code={str(p.relative_to(root)): file_hash(p) for p in sorted(paths)},
        packages=versions,
    )


def upstream_training_pool(adapter, explicit_index=""):
    """读所选 adapter 同 run manifest 的真实 index；无法确认的来源标为 unknown。

    SFT 未保存逐个 optimizer step 的完整采样路线，因此 index 的训练 split 是保守上界。
    不用另一套 split hash 猜训练集，不将候选池内路线声明为实际见过。
    """
    run = Path(adapter["path"]).parent
    manifest_path = run / "train_run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    for field in ("prompt_name", "production_prompt_sha256"):
        if (
            manifest.get(field)
            and adapter.get("metadata", {}).get(field)
            and manifest[field] != adapter["metadata"][field]
        ):
            raise ValueError(
                f"upstream manifest differs from selected adapter: {field}"
            )
    source = explicit_index or manifest.get("index", "")
    split = manifest.get("split", "train")
    evidence = dict(
        status="unknown",
        split=split,
        source=str(source),
        adapter_fingerprint=adapter.get("fingerprint"),
        manifest_sha256=file_hash(manifest_path) if manifest else None,
        actual_sampled_routes_verified=False,
    )
    if not source:
        return dict(evidence, reason="missing upstream index provenance", routes=[])
    raw = Path(source).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, run / raw]
    found = [p.resolve() for p in candidates if p.is_file()]
    if not found:
        if explicit_index:
            raise FileNotFoundError(explicit_index)
        return dict(
            evidence,
            reason="upstream index unavailable; supply --phaseN-training-index",
            routes=[],
        )
    if len(set(found)) > 1:
        raise ValueError(f"ambiguous upstream index: {found}")
    path = found[0]
    routes = set()
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = __import__("json").loads(line)
            expected_dataset = adapter.get("metadata", {}).get("dataset_name")
            if expected_dataset and row.get("dataset_name") != expected_dataset:
                raise ValueError(f"{path}: upstream dataset_name mismatch")
            if row.get("split") != split:
                continue
            scenario = row.get("scenario")
            run_id = row.get("route_id") or row.get("run_id")
            if not scenario or not run_id:
                raise ValueError(f"{path}: missing scenario/route identity")
            routes.add(route_group(scenario, run_id))
            rows += 1
    if not routes:
        raise ValueError(f"{path}: empty upstream {split} pool")
    return dict(
        evidence,
        status="training_pool_available",
        source=str(path),
        source_sha256=file_hash(path),
        source_override=bool(explicit_index),
        pool_rows=rows,
        routes=sorted(routes),
        pool_identity=digest(sorted(routes)),
    )


def annotate_upstream(rows, sources):
    """按真实索引训练候选池报告重叠/池外/未知，既不丢评测帧也不误称系统未见。"""
    sets = {k: set(v["routes"]) for k, v in sources.items()}
    counts = Counter()
    for row in rows:
        exposure = {}
        for phase, source in sources.items():
            exposure[phase] = (
                "unknown"
                if source["status"] == "unknown"
                else (
                    "train_pool_overlap"
                    if row["route_group"] in sets[phase]
                    else "outside_train_pool"
                )
            )
        values = exposure.values()
        exposure["combined"] = (
            "train_pool_overlap"
            if "train_pool_overlap" in values
            else "unknown" if "unknown" in values else "outside_both_train_pools"
        )
        row["upstream_exposure"] = exposure
        for phase, value in exposure.items():
            counts[f"{phase}/{value}"] += 1
    return dict(counts)


def collect_upstream_sources(adapters, args):
    """来源获取失败只降低审计可用性；不能阻断相同生成条件的训练恢复。"""
    sources = {}
    for phase, adapter in adapters.items():
        explicit = getattr(args, f"{phase}_training_index", "")
        try:
            sources[phase] = upstream_training_pool(adapter, explicit)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            sources[phase] = dict(
                status="unknown",
                routes=[],
                source=explicit,
                source_override=bool(explicit),
                error=f"{type(exc).__name__}: {exc}",
                actual_sampled_routes_verified=False,
            )
    return sources


def audit_source_changes(expected, current):
    """独立审计数据内容变化；路径、自动/显式和获取状态不属于模型兼容身份。"""
    report = {}
    for phase in sorted(set(expected) | set(current)):
        old, new = expected.get(phase, {}), current.get(phase, {})
        old_hash, new_hash = old.get("source_sha256"), new.get("source_sha256")
        if not new_hash:
            status = "current_unavailable"
        elif not old_hash:
            status = "newly_available"
        elif (old_hash, old.get("split"), old.get("pool_identity")) == (
            new_hash,
            new.get("split"),
            new.get("pool_identity"),
        ):
            status = "same_content"
        else:
            status = "changed_content"
        report[phase] = dict(
            status=status,
            original_sha256=old_hash,
            current_sha256=new_hash,
            generation_compatibility_affected=False,
        )
    return report
