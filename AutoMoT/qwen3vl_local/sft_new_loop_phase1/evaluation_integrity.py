"""Freeze native evaluation membership and replay its scorer without a model.

Replay feeds saved raw answers through the unchanged native evaluate function.
It never selects a different sample, downloads a model or changes any label.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
import importlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

from . import run_integrity as integrity


def module_for(phase):
    return importlib.import_module(f"qwen3vl_local.sft_new_loop_phase{phase}.eval")


def item_from_json(phase, value):
    module = module_for(phase)
    prompts = importlib.import_module(f"qwen3vl_local.sft_new_loop_phase{phase}.prompts")
    spec = dict(value["spec"])
    if phase == 1:
        nested = dict(spec["phase2_spec"])
        native = prompts.phase2_prompts
        nested["questions"] = tuple(native.QuestionSpec(**q) for q in nested["questions"])
        spec["phase2_spec"] = native.PromptSpec(**nested)
        spec["phase1_answers"] = tuple(tuple(x) for x in spec["phase1_answers"])
        spec["phase1_output_keys"] = tuple(spec["phase1_output_keys"])
    else:
        spec["questions"] = tuple(prompts.QuestionSpec(**q) for q in spec["questions"])
    return module.WorkItem(**{**value, "row": module.FrameRow(**value["row"]), "spec": prompts.PromptSpec(**spec)})


def begin(args, phase, cases, output, mode, *, rank, distributed=None):
    def capture():
        module = module_for(phase)
        image_hashes = {}
        items = []
        for item in cases:
            selected = module.select_history_rgb_paths(item.row.history_rgb_paths, mode)
            for filename in selected:
                if filename not in image_hashes:
                    image_hashes[filename] = integrity.sha(filename)
            items.append(asdict(item))
        value = dict(schema=integrity.SCHEMA, phase=phase,
                     arguments={k: v for k, v in vars(args).items() if not k.startswith("_")},
                     data=integrity.data_identity(args.index), source_sha256=integrity.sources(phase),
                     history_rgb_mode=mode, items=items, image_sha256=image_hashes,
                     adapter_identity=integrity.verify_adapter(args.adapter_dir) if args.adapter_dir else None,
                     adapter_metadata=integrity.read(Path(args.adapter_dir) / f"sft_new_loop_phase{phase}_adapter_config.json") if args.adapter_dir else None,
                     expected_total=len(cases), environment=integrity.environment())
        integrity.write(Path(output) / "evaluation_inputs.json", value)
        return True
    integrity.collective_rank0(capture, rank=rank, distributed=distributed)


def record_base(output, bundle, *, rank, distributed=None):
    def capture():
        from qwen3vl_local.qwen35.identity import require_weight_identity
        assets = getattr(bundle.model.config, "local_asset_sha256", None)
        require_weight_identity(assets)
        path = Path(output) / "evaluation_inputs.json"
        record = integrity.read(path)
        record["base_asset_sha256"] = assets
        record["attention_implementation"] = getattr(bundle.model.config, "_attn_implementation", None)
        integrity.write(path, record)
        return True
    integrity.collective_rank0(capture, rank=rank, distributed=distributed)


class GenerationTimer:
    """Synchronized fresh-prefill+decode timing; image disk loading is excluded."""
    def __init__(self, bundle):
        import torch
        self.device = bundle.device
        self.cuda = self.device.type == "cuda"
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()

    def finish(self, input_tokens, output_tokens):
        import torch
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return dict(seconds=time.perf_counter() - self.started, input_tokens=int(input_tokens),
                    generated_tokens=int(output_tokens),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.cuda else None,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if self.cuda else None)


def performance_record(bundle, case_index):
    record = getattr(bundle, "_generation_measurement", None)
    return {"case_index": case_index, **record} if record is not None else None


def write_performance(output, rank, records, *, args, bundle):
    try:
        dtype = str(next(bundle.model.parameters()).dtype)
    except (AttributeError, StopIteration):
        dtype = None
    integrity.write(Path(output) / f"performance_rank{rank}.json", dict(
        schema=integrity.SCHEMA, rank=rank, records=records, warmup_cases_per_rank=2,
        scope="fresh_processor_prefill_decode_cuda_synchronized_excludes_image_disk_io",
        dtype=dtype, batch_size=1, merge_lora=bool(args.merge_lora), max_new_tokens=int(args.max_new_tokens),
        attention_implementation=getattr(bundle.model.config, "_attn_implementation", None),
        environment=integrity.environment()))


def performance_summary(output, world):
    reports = [integrity.read(Path(output) / f"performance_rank{r}.json") for r in range(world)]
    conditions = ("scope", "dtype", "merge_lora", "max_new_tokens", "attention_implementation", "environment")
    if any(any(report.get(k) != reports[0].get(k) for k in conditions) for report in reports):
        raise ValueError("Performance conditions differ between ranks")
    records = [x for report in reports for x in report["records"][2:] if x is not None]
    if not records:
        return dict(status="unmeasured_insufficient_post_warmup_cases", measured_cases=0)
    values = sorted(x["seconds"] for x in records)
    if any(not (0 < x < float("inf")) for x in values):
        raise ValueError("Invalid performance measurement")
    def quantile(q):
        pos = (len(values) - 1) * q
        lo = int(pos)
        return values[lo] + (values[min(lo + 1, len(values)-1)] - values[lo]) * (pos-lo)
    duration = sum(values)
    peak = [x["peak_allocated_bytes"] for x in records if x["peak_allocated_bytes"] is not None]
    return dict(status="measured", measured_cases=len(records), warmup_cases_per_rank=2,
                latency_seconds_p50=quantile(.5), latency_seconds_p95=quantile(.95),
                serial_equivalent_cases_per_second=len(records)/duration,
                generated_tokens_per_active_device_second=sum(x["generated_tokens"] for x in records)/duration,
                input_tokens=sum(x["input_tokens"] for x in records),
                generated_tokens=sum(x["generated_tokens"] for x in records),
                peak_allocated_bytes=max(peak) if peak else None,
                scope=reports[0]["scope"], dtype=reports[0]["dtype"], batch_size=1,
                merge_lora=reports[0]["merge_lora"], max_new_tokens=reports[0]["max_new_tokens"],
                attention_implementation=reports[0].get("attention_implementation"),
                world_size=world, environment=reports[0]["environment"],
                note="Per-device active inference rate, not aggregate job throughput or training ETA")


def read_cases(output, world):
    output = Path(output)
    names = ["cases.jsonl"] if world == 1 else [f"cases_rank{r}.jsonl" for r in range(world)]
    if {p.name for p in output.glob("cases*.jsonl")} != set(names):
        raise ValueError("Missing/unexpected evaluation shards")
    records = []
    for rank, name in enumerate(names):
        with (output / name).open() as handle:
            for local, line in enumerate(handle):
                case = json.loads(line)
                if type(case.get("case_index")) is not int or case["case_index"] != rank + local * world:
                    raise ValueError("Invalid, truncated, duplicate or misplaced evaluation case")
                records.append(case)
    records.sort(key=lambda x: x["case_index"])
    if [x["case_index"] for x in records] != list(range(len(records))):
        raise ValueError("Non-contiguous evaluation membership")
    return records, names


def score_fields(metrics):
    # These are all native scoring outputs, including conditional diagnostics.
    keys = ("total_cases", "exact_match_accuracy", "per_question", "task_reports", "variant_reports",
            "focus_counts", "answer_pattern_diagnostics", "audit_parser_diagnostics",
            "answer_only_diagnostics", "slice_reports", "invalid_contract", "invalid_subgroup_accuracy")
    return {k: metrics[k] for k in keys if k in metrics}


def replay(inputs, records):
    """No inference. Reconstruct native specs and recompute every scoring field."""
    phase = inputs["phase"]
    module = module_for(phase)
    items = [item_from_json(phase, item) for item in inputs["items"]]
    if not items or len(items) != len(records) or len(items) != inputs["expected_total"]:
        raise ValueError("Evaluation cases do not match frozen native membership")
    rows = list({item.row.idx: item.row for item in items}.values())
    iterator = iter(records)
    def output(*args, **kwargs):
        return next(iterator)["raw_output"]
    config = SimpleNamespace(**inputs["arguments"])
    config._integrity_replay = True
    config.device = "cpu"
    config.adapter_dir = None
    config.history_rgb_mode = inputs["history_rgb_mode"]
    config.save_error_rgb = config.save_all_rgb = False
    config.timestamp_output = config.overwrite = False
    with tempfile.TemporaryDirectory(prefix="phase12_score_") as tmp, ExitStack() as stack:
        config.output_dir = str(Path(tmp) / "eval")
        overrides = dict(setup_distributed=lambda: (0,0,1), cleanup_distributed=lambda: None,
                         _read_rows=lambda *a, **k: rows, _balanced_cases=lambda *a, **k: items,
                         _load_images=lambda *a: [], load_eval_bundle=lambda *a, **k: None,
                         _generate=output)
        if phase == 2:
            overrides["_exclude_prior_cases"] = lambda rows, *a, **k: (rows, {})
        for name, fn in overrides.items():
            stack.enter_context(patch.object(module, name, fn))
        with redirect_stdout(io.StringIO()):
            metrics = module.evaluate(config)
        replayed, _ = read_cases(config.output_dir, 1)
    for original, regenerated in zip(records, replayed):
        # Native reconstruction independently checks identity, GT, spec, parser and scores.
        if original != regenerated:
            differences = [k for k in set(original) | set(regenerated) if original.get(k) != regenerated.get(k)]
            raise ValueError(f"Case {original['case_index']} differs from native replay: {differences}")
    return score_fields(metrics)


def verify_evaluation(output, *, verify_current_sources=True):
    output = Path(output)
    inputs = integrity.read(output / "evaluation_inputs.json")
    if inputs.get("schema") != integrity.SCHEMA:
        raise ValueError("Missing evaluation input contract")
    if verify_current_sources:
        integrity.verify_sources(inputs["source_sha256"])
    metrics = integrity.read(output / "metrics.json")
    records, names = read_cases(output, int(metrics["world_size"]))
    scores = replay(inputs, records)
    if scores != score_fields(metrics):
        raise ValueError("Metrics differ from native raw-output replay")
    return dict(schema=integrity.SCHEMA, status="verified", phase=inputs["phase"], total_cases=len(records),
                inputs_sha256=integrity.sha(output / "evaluation_inputs.json"),
                metrics_sha256=integrity.sha(output / "metrics.json"),
                case_files_sha256={name: integrity.sha(output / name) for name in names},
                scores=scores)


def finish(args, output, *, world):
    output = Path(output)
    inputs = integrity.read(output / "evaluation_inputs.json")
    if inputs["data"] != integrity.data_identity(args.index):
        raise ValueError("Evaluation index changed during inference")
    if any(integrity.sha(p) != h for p, h in inputs["image_sha256"].items()):
        raise ValueError("Evaluation RGB changed during inference")
    if args.adapter_dir and inputs["adapter_identity"] != integrity.verify_adapter(args.adapter_dir):
        raise ValueError("Adapter changed during inference")
    from qwen3vl_local.qwen35.identity import base_asset_hashes
    if inputs["base_asset_sha256"] != base_asset_hashes(args.model_dir):
        raise ValueError("Base assets changed during inference")
    report = verify_evaluation(output)
    report["performance"] = performance_summary(output, world)
    # Include performance files in the same tamper-evident evidence contract.
    report["performance_files_sha256"] = {f"performance_rank{r}.json": integrity.sha(output / f"performance_rank{r}.json") for r in range(world)}
    integrity.write(output / "evaluation_complete.json", report)
    return report
