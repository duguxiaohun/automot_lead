"""Phase1/2 completion evidence. Does not select, replace or append samples."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "phase12_integrity_v1"


def sha(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, k) != getattr(after, k) for k in fields):
        raise ValueError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sources(phase):
    # Include native task dependencies and the local backend, including lazy imports.
    if phase not in (1, 2):
        raise ValueError("Source bindings require an explicit phase")
    names = (f"sft_new_loop_phase{phase}", "sft_loop_phase1",
             "sft_loop_phase2", "sft_loop_phase2_augment", "sft_loop_phase3",
             "sft_v2", "sft_v3", "qwen35")
    shared = {"__init__.py", "run_integrity.py", "evaluation_integrity.py", "delivery.py",
              "training_policy.py", "exposure_sampling.py", "local_assets.py", "pipeline_preflight.py",
              "data_stage.py", "data_audit.py", "sampling_preservation_20261025.json"}
    def belongs(p):
        rel = p.relative_to(ROOT / "qwen3vl_local")
        if rel.parts[0] == f"sft_new_loop_phase{3-phase}":
            return phase == 2 and len(rel.parts) == 2 and p.name in shared
        return True
    paths = set()
    # Native Phase2 audit imports its builder, which imports this lightweight
    # helper outside qwen3vl_local. Bind/copy it so delivered code is standalone.
    for name in ('__init__.py', 'abnormal_duration_filter.py'):
        dependency = ROOT / 'lead_video_tools' / name
        if not dependency.is_file():
            raise ValueError(f'Missing native audit dependency: {dependency}')
        paths.add(dependency)
    if phase == 2:
        paths.update(p for p in (ROOT / "qwen3vl_local/sft_new_loop_phase1").iterdir()
                     if p.is_file() and p.name in shared)
    for name in names:
        for p in (ROOT / "qwen3vl_local" / name).rglob("*"):
            if p.suffix in {".py", ".sh", ".jinja", ".txt", ".json"} and p.is_file():
                if "__pycache__" not in p.parts and "tests" not in p.parts and not p.name.startswith("test_"):
                    # Only code/config dependencies; no old review reports or data products.
                    if p.suffix != ".json" or name == "qwen35" or p.name == "sampling_preservation_20261025.json":
                        paths.add(p)
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            p = Path(filename).resolve()
            if p.is_relative_to(ROOT / "qwen3vl_local") and belongs(p) and p.suffix == ".py" and not p.name.startswith("test_") and "tests" not in p.parts:
                paths.add(p)
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def verify_sources(record, root=ROOT):
    for name, expected in record.items():
        p = Path(root) / name
        if not p.is_relative_to(Path(root)) or ".." in Path(name).parts or sha(p) != expected:
            raise ValueError(f"Source binding mismatch: {name}")


def environment():
    import torch
    versions = {}
    for name in ("torch", "torchvision", "transformers", "peft", "accelerate", "tokenizers", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, packages=versions, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), gpu_count=torch.cuda.device_count(),
                gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


def data_identity(index):
    index = Path(index).resolve()
    manifest = index.parent / "manifest.json"
    return dict(index=str(index), index_sha256=sha(index), manifest=str(manifest), manifest_sha256=sha(manifest))


def same_data(left, right):
    return all(left.get(k) == right.get(k) and left.get(k) for k in ("index_sha256", "manifest_sha256"))


def collective_rank0(fn, *, rank, distributed=None):
    active = distributed is not None and distributed.is_initialized()
    result = [None]
    if rank == 0:
        try:
            result[0] = dict(ok=True, value=fn())
        except Exception as exc:
            result[0] = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
    if active:
        distributed.broadcast_object_list(result, src=0)
    if not result[0]["ok"]:
        raise ValueError(result[0]["error"])
    return result[0]["value"]


def begin_training(args, *, phase, rank, distributed=None):
    args._integrity_updates = 0
    def capture():
        value = dict(schema=SCHEMA, phase=phase, data=data_identity(args.index), source_sha256=sources(phase),
                     arguments={k: v for k, v in vars(args).items() if not k.startswith("_")},
                     environment=environment(), budget=args._budget_plan,
                     sampling_policy="unchanged_coverage_task_action_v2",
                     completion_policy="all_planned_presentations_applied_no_skips_no_extra_steps")
        write(Path(args.output_dir) / "training_inputs.json", value)
        return value
    args._integrity_inputs = collective_rank0(capture, rank=rank, distributed=distributed)


def finite_loss(loss, *, supervised=True, distributed=None):
    import torch
    bad = ((~torch.isfinite(loss.detach())).any() | (not supervised)).to(dtype=torch.int32)
    if distributed is not None and distributed.is_initialized():
        distributed.all_reduce(bad, op=distributed.ReduceOp.MAX)
    if bad.item():
        raise ValueError("Non-finite training loss; refusing optimizer update and completion")


def require_packed(packed, *, device, distributed=None):
    import torch
    bad = torch.tensor(int(packed is None), device=device)
    if distributed is not None and distributed.is_initialized():
        distributed.all_reduce(bad, op=distributed.ReduceOp.MAX)
    if bad.item():
        raise ValueError("Training input exceeds context limit; refusing skipped presentations or automatic extra steps")


def validate_consumption(output, budget):
    output = Path(output)
    world, total = int(budget["world_size"]), int(budget["total_steps"])
    if world < 1 or total < 1:
        raise ValueError("Invalid completion budget")
    paths = sorted((output / "exposure").glob("rank_*.json"))
    if {p.name for p in paths} != {f"rank_{r}.json" for r in range(world)}:
        raise ValueError("Missing or unexpected exposure ranks")
    records = [read(output / "exposure" / f"rank_{r}.json") for r in range(world)]
    for r, record in enumerate(records):
        updates = read(output / "exposure" / f"optimizer_{r}.json")
        if updates != dict(optimizer_updates=int(budget["optimizer_updates"]), loop_steps=total):
            raise ValueError(f"Optimizer update budget mismatch at rank {r}")
        counts = [sum(int(item[-1]) for item in record[k]) for k in ("frames", "tasks")]
        if (record["step"] != total or record["skipped"] != 0 or record["pending_backward"] != 0
                or record["applied_backward"] != total or counts != [total, total]):
            raise ValueError(f"Incomplete effective training at rank {r}: expected {total} applied presentations, no skips")
        for key in ("frames", "tasks"):
            ids = [tuple(x[:-1]) for x in record[key]]
            if len(set(ids)) != len(ids) or any(type(x[-1]) is not int or x[-1] <= 0 for x in record[key]):
                raise ValueError(f"Invalid exposure counters at rank {r}")
    summary = read(output / "exposure_summary.json")
    if any(summary[k] != value for k, value in dict(step=total, successful_backward=world*total,
                                                   applied_backward=world*total, pending_backward=0, skipped=0).items()):
        raise ValueError("Exposure summary differs from actual rank consumption")
    epochs = read(output / "exposure_epochs.json")
    if sum(x["actual_presentations"] for x in epochs) != world * total or epochs[-1]["step"] != total:
        raise ValueError("Epoch consumption does not match total budget")
    return summary


def before_final(args, *, rank, distributed=None):
    write(Path(args.output_dir) / "exposure" / f"optimizer_{rank}.json",
          dict(optimizer_updates=args._integrity_updates, loop_steps=args._budget_plan["total_steps"]))
    if distributed is not None and distributed.is_initialized():
        distributed.barrier()
    def check():
        try:
            return validate_consumption(args.output_dir, args._budget_plan)
        except Exception as exc:
            write(Path(args.output_dir) / "training_rejected.json", dict(schema=SCHEMA, status="rejected", reason=str(exc)))
            raise
    return collective_rank0(check, rank=rank, distributed=distributed)


def adapter_files(path):
    path = Path(path)
    weights = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in {".safetensors", ".bin"})
    if not weights or any(p.stat().st_size == 0 for p in weights):
        raise ValueError(f"Missing adapter weights: {path}")
    files = weights + sorted(p for p in path.glob("*.json") if p.name != "phase12_adapter_identity.json")
    return {p.name: sha(p) for p in files}


def stamp_adapter(path):
    value = dict(schema=SCHEMA, files=adapter_files(path))
    write(Path(path) / "phase12_adapter_identity.json", value)
    return value


def verify_adapter(path):
    record = read(Path(path) / "phase12_adapter_identity.json")
    if record.get("schema") != SCHEMA or record["files"] != adapter_files(path):
        raise ValueError(f"Adapter content identity mismatch: {path}")
    return record


def validate_saved_adapter(path, inputs):
    path = Path(path)
    phase = inputs["phase"]
    cfg = read(path / f"sft_new_loop_phase{phase}_adapter_config.json")
    step = cfg.get("global_step")
    total = inputs["budget"]["total_steps"]
    if (type(step) is not int or not 0 < step <= total or (path.name == "final" and step != total)
            or cfg.get("training_budget") != inputs["budget"]
            or cfg.get("history_rgb_mode") != inputs["arguments"]["history_rgb_mode"]
            or cfg.get("sampling_policy") != "coverage_task_action_v2"):
        raise ValueError(f"Adapter step/budget/method mismatch: {path}")


def finish_training(args, *, rank, distributed=None):
    def finish():
        root = Path(args.output_dir)
        inputs = read(root / "training_inputs.json")
        if inputs != args._integrity_inputs or inputs["data"] != data_identity(args.index):
            raise ValueError("Training input data changed during run")
        verify_sources(inputs["source_sha256"])
        summary = validate_consumption(root, inputs["budget"])
        adapters = {p.name: verify_adapter(p) for p in sorted(root.iterdir()) if p.is_dir()
                    and (p / "phase12_adapter_identity.json").is_file() and not p.name.startswith("checkpoint-")}
        if not adapters or "budget_stop" in adapters:
            raise ValueError("No completed adapter to publish")
        for slot in adapters:
            validate_saved_adapter(root / slot, inputs)
        files = [root / "training_inputs.json", root / "training_budget.json", root / "exposure_summary.json",
                 root / "exposure_epochs.json", *sorted((root / "exposure").glob("*.json"))]
        files += sorted((root / "balance").glob("*.json"))
        files += [p for p in sorted(root.glob("*.json")) if p not in files and p.name not in {"training_complete.json", "training_rejected.json"}]
        value = dict(schema=SCHEMA, status="completed", phase=inputs["phase"], budget=inputs["budget"],
                     exposure=summary, adapters=adapters,
                     evidence_sha256={str(p.relative_to(root)): sha(p) for p in files})
        write(root / "training_complete.json", value)
        return value
    return collective_rank0(finish, rank=rank, distributed=distributed)


def verify_training(root):
    root = Path(root)
    result = read(root / "training_complete.json")
    if result.get("status") != "completed" or result.get("schema") != SCHEMA:
        raise ValueError("No valid training completion receipt")
    for name, expected in result["evidence_sha256"].items():
        if ".." in Path(name).parts or Path(name).is_absolute() or sha(root / name) != expected:
            raise ValueError(f"Training evidence mismatch: {name}")
    inputs = read(root / "training_inputs.json")
    verify_sources(inputs["source_sha256"])
    if result["budget"] != inputs["budget"] or result["phase"] != inputs["phase"]:
        raise ValueError("Completion/input contract mismatch")
    if validate_consumption(root, result["budget"]) != result["exposure"]:
        raise ValueError("Completion exposure mismatch")
    for name, identity in result["adapters"].items():
        if Path(name).name != name or verify_adapter(root / name) != identity:
            raise ValueError(f"Completed adapter mismatch: {name}")
        validate_saved_adapter(root / name, inputs)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    args = p.parse_args()
    print(json.dumps(verify_training(args.run), indent=2))
