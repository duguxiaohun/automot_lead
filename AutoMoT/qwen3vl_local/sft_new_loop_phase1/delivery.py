"""Strict local Phase1/2 delivery and paired comparisons; never uploads assets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil

from . import evaluation_integrity as evaluation
from . import run_integrity as integrity
from .training_policy import physical_group


def verified_eval(path):
    path = Path(path)
    saved = integrity.read(path / "evaluation_complete.json")
    actual = evaluation.verify_evaluation(path)
    if any(saved.get(k) != v for k, v in actual.items()):
        raise ValueError(f"Evaluation receipt mismatch: {path}")
    for name, expected in saved["performance_files_sha256"].items():
        if Path(name).name != name or integrity.sha(path / name) != expected:
            raise ValueError("Performance evidence mismatch")
    world = int(integrity.read(path / "metrics.json")["world_size"])
    if saved["performance"] != evaluation.performance_summary(path, world):
        raise ValueError("Performance summary mismatch")
    return saved


def normalized_members(inputs):
    """Ignore machine-local paths, preserve exact questions, labels and image bytes."""
    result = []
    for raw in inputs["items"]:
        item = evaluation.item_from_json(inputs["phase"], raw)
        row = dict(raw["row"])
        for key in ("idx", "history_rgb_paths", "latest_rgb_path"):
            row.pop(key, None)
        paths = evaluation.module_for(inputs["phase"]).select_history_rgb_paths(
            item.row.history_rgb_paths, inputs["history_rgb_mode"])
        result.append(dict(row=row, spec=raw["spec"], balance_key=raw["balance_key"],
                           focus=raw.get("focus"), rgb=[inputs["image_sha256"][p] for p in paths]))
    return result


def prepare(root, phase, adapter, index, *, dataset_receipt=None, preflight=None):
    root, adapter = Path(root), Path(adapter).resolve()
    run = adapter.parent
    completion = integrity.verify_training(run)
    inputs = integrity.read(run / "training_inputs.json")
    integrity.verify_sources(inputs["source_sha256"])
    data = integrity.data_identity(index)
    if not integrity.same_data(data, inputs["data"]):
        raise ValueError("Delivery data differs from completed training dataset")
    if completion["phase"] != phase or adapter.name not in completion["adapters"]:
        raise ValueError("Selected adapter is not owned by this completed phase/run")
    identity = integrity.verify_adapter(adapter)
    if identity != completion["adapters"][adapter.name]:
        raise ValueError("Selected adapter changed since completion")
    reports, eval_inputs = {}, {}
    names = [name for name in ("base_production", "lora_production", "base_audit_prompt", "lora_audit_prompt")
             if (root / name).exists()]
    if not {"base_production", "lora_production"}.issubset(names):
        raise ValueError("Missing base/LoRA production evaluation")
    for name in names:
        reports[name] = verified_eval(root / name)
        current = integrity.read(root / name / "evaluation_inputs.json")
        if current["phase"] != phase or not integrity.same_data(current["data"], data):
            raise ValueError("Evaluation phase/dataset differs from delivery")
        expected_adapter = identity if name.startswith("lora_") else None
        if current["adapter_identity"] != expected_adapter:
            raise ValueError("Evaluation used a different adapter")
        if name.startswith("lora_") and current["base_asset_sha256"] != integrity.read(adapter / "qwen35_base_assets.json"):
            raise ValueError("Evaluation used a different base model")
        eval_inputs[name] = current
    reference = eval_inputs["base_production"]
    for name, current in eval_inputs.items():
        if (normalized_members(current) != normalized_members(reference)
                or current["base_asset_sha256"] != reference["base_asset_sha256"]
                or current["arguments"]["split"] != reference["arguments"]["split"]
                or current["arguments"]["max_new_tokens"] != reference["arguments"]["max_new_tokens"]):
            raise ValueError(f"Unpaired evaluation membership/base: {name}")
        actual_mode = integrity.read(root / name / "metrics.json")["prompt_mode"]
        if actual_mode != ("audit" if name.endswith("audit_prompt") else "production"):
            raise ValueError(f"Wrong prompt mode in {name}")
    for path, field in ((dataset_receipt, "dataset_receipt"), (preflight, "pipeline_preflight")):
        if path:
            record = integrity.read(path)
            found = record if field == "dataset_receipt" else record["dataset"]
            if found["index_sha256"] != data["index_sha256"] or found["manifest_sha256"] != data["manifest_sha256"]:
                raise ValueError(f"Stale {field}")
    stage = root / "verified_training"
    stage.mkdir(exist_ok=True)
    files = set(completion["evidence_sha256"]) | {"training_complete.json"}
    files |= {p.name for p in run.glob("*.json") if p.name != "training_rejected.json"}
    for name in sorted(files):
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run / name, target)
    for slot, recorded in completion["adapters"].items():
        target = stage / slot
        target.mkdir(exist_ok=True)
        for name in [*recorded["files"], "phase12_adapter_identity.json"]:
            if name.endswith(".json"):
                shutil.copy2(run / slot / name, target / name)
    # Copy source evidence for future review, never execute supplied snapshots automatically.
    code = root / "code"
    for name in sorted(inputs["source_sha256"]):
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(integrity.ROOT / name, target)
    integrity.write(root / "dataset_identity.json", data)
    for path, name in ((dataset_receipt, "dataset_receipt.json"), (preflight, "pipeline_preflight.json")):
        if path:
            shutil.copy2(path, root / name)
    result = dict(schema=integrity.SCHEMA, status="verified_for_audit", phase=phase,
                  selected_slot=adapter.name, selected_adapter=identity,
                  completion_sha256=integrity.sha(run / "training_complete.json"),
                  data=data, evaluations=reports, source_sha256=inputs["source_sha256"],
                  note="Weights stay on training host; hashes bind selected and final weights. Historical comparability requires compare.")
    integrity.write(root / "delivery.json", result)
    return result


def seal(bundle):
    bundle = Path(bundle)
    files = {str(p.relative_to(bundle)): integrity.sha(p) for p in sorted(bundle.rglob("*"))
             if p.is_file() and p != bundle / "integrity_files.json"}
    if any(p.is_symlink() for p in bundle.rglob("*")):
        raise ValueError("Symlinks are not permitted in audit packages")
    integrity.write(bundle / "integrity_files.json", dict(schema=integrity.SCHEMA, files=files))
    return verify_package(bundle)


def verify_package(bundle):
    bundle = Path(bundle)
    record = integrity.read(bundle / "integrity_files.json")
    if any(p.is_symlink() for p in bundle.rglob("*")):
        raise ValueError("Symlinks are not permitted in audit packages")
    actual_names = {str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file() and p != bundle / "integrity_files.json"}
    if record.get("schema") != integrity.SCHEMA or actual_names != set(record["files"]):
        raise ValueError("Package file list mismatch")
    for name, expected in record["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or (bundle / name).is_symlink() or integrity.sha(bundle / name) != expected:
            raise ValueError(f"Package file hash mismatch: {name}")
    delivery = integrity.read(bundle / "delivery.json")
    if delivery.get("status") != "verified_for_audit":
        raise ValueError("Package has no verified delivery receipt")
    integrity.verify_sources(delivery["source_sha256"], root=bundle / "code")
    completion = integrity.read(bundle / "verified_training/training_complete.json")
    if integrity.sha(bundle / "verified_training/training_complete.json") != delivery["completion_sha256"]:
        raise ValueError("Packaged completion mismatch")
    for name, expected in completion["evidence_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or integrity.sha(bundle / "verified_training" / name) != expected:
            raise ValueError("Packaged training evidence mismatch")
    train_inputs = integrity.read(bundle / "verified_training/training_inputs.json")
    if completion["budget"] != train_inputs["budget"] or completion["phase"] != train_inputs["phase"]:
        raise ValueError("Packaged input/completion budget mismatch")
    for name, saved in completion["adapters"].items():
        if Path(name).name != name:
            raise ValueError("Invalid packaged adapter slot")
        integrity.validate_saved_adapter(bundle / "verified_training" / name, train_inputs)
        for filename, expected in saved["files"].items():
            if filename.endswith(".json") and (Path(filename).name != filename or integrity.sha(bundle / "verified_training" / name / filename) != expected):
                raise ValueError("Packaged adapter config mismatch")
    if integrity.validate_consumption(bundle / "verified_training", completion["budget"]) != completion["exposure"]:
        raise ValueError("Packaged training consumption mismatch")
    if completion["adapters"].get(delivery["selected_slot"]) != delivery["selected_adapter"]:
        raise ValueError("Packaged selected adapter is not owned by completed run")
    for name, saved in delivery["evaluations"].items():
        if Path(name).name != name or verified_eval(bundle / name) != saved:
            raise ValueError("Packaged evaluation differs from delivery")
    return dict(status="verified", phase=delivery["phase"], files=len(actual_names),
                weights_included=False, selected_slot=delivery["selected_slot"])


def case_key(case):
    spec = case.get("prompt_spec", case.get("augment_spec", {}))
    return integrity.digest([case["scenario"], case["route_id"], case["frame_id"],
                             case.get("focus_question"), case.get("question_domain"),
                             case.get("augment_variant"), list(case["gt"]),
                             [(q["output_key"], q["metric_key"], q["question_id"]) for q in spec.get("questions", [])]])


def compare(old, new):
    loaded = []
    for path in (old, new):
        path = Path(path)
        metrics = integrity.read(path / "metrics.json")
        records, _ = evaluation.read_cases(path, int(metrics.get("world_size", 1)))
        if len(records) != metrics["total_cases"]:
            raise ValueError("Historical case count is incomplete; no truncated-score comparison")
        verified = (path / "evaluation_complete.json").is_file()
        if verified:
            receipt = verified_eval(path)
            inputs = integrity.read(path / "evaluation_inputs.json")
        else:
            receipt, inputs = None, None
        groups = defaultdict(list)
        for case in records:
            groups[case_key(case)].append(case)
        loaded.append((metrics, groups, receipt, inputs))
    a, b = loaded
    if a[0]["dataset_name"] != b[0]["dataset_name"] or a[0]["prompt_mode"] != b[0]["prompt_mode"]:
        raise ValueError("Different tasks/prompt modes cannot share a comparison score")
    pairs, label_conflicts, rgb_conflicts = [], 0, 0
    for key in sorted(set(a[1]) & set(b[1])):
        if len(a[1][key]) != len(b[1][key]):
            continue
        for left, right in zip(a[1][key], b[1][key]):
            if left["gt"] != right["gt"]:
                label_conflicts += 1
                continue
            if a[3] and b[3]:
                lh = [a[3]["image_sha256"][p] for p in left["history_rgb_paths_used"]]
                rh = [b[3]["image_sha256"][p] for p in right["history_rgb_paths_used"]]
                if lh != rh:
                    rgb_conflicts += 1
                    continue
            pairs.append((left, right))
    old_correct = sum(all(x["parsed"].get(k) == v for k, v in x["gt"].items()) for x, _ in pairs)
    new_correct = sum(all(y["parsed"].get(k) == v for k, v in y["gt"].items()) for _, y in pairs)
    flips = Counter()
    losses = []
    for x, y in pairs:
        was = all(x["parsed"].get(k) == v for k, v in x["gt"].items())
        now = all(y["parsed"].get(k) == v for k, v in y["gt"].items())
        flips[f"{int(was)}_to_{int(now)}"] += 1
        if was and not now:
            losses.append(dict(scenario=x["scenario"], route_id=x["route_id"], frame_id=x["frame_id"],
                               old_case_index=x["case_index"], new_case_index=y["case_index"]))
    complete = len(pairs) == a[0]["total_cases"] == b[0]["total_cases"]
    certified = bool(a[2] and b[2])
    status = ("paired_system_comparison" if complete else "common_subset_only") if certified else "legacy_reference_unverified_rgb_and_protocol"
    performance = dict(status="not_comparable_missing_measurements")
    if certified:
        pa, pb = a[2]["performance"], b[2]["performance"]
        keys = ("scope", "dtype", "batch_size", "merge_lora", "max_new_tokens", "world_size", "attention_implementation", "environment")
        if (pa.get("status") == pb.get("status") == "measured" and complete
                and a[0]["history_rgb_mode"] == b[0]["history_rgb_mode"]
                and a[0]["eval_prompt_sha256"] == b[0]["eval_prompt_sha256"]
                and all(pa.get(k) == pb.get(k) for k in keys)):
            performance = dict(status="matched_recorded_conditions", old=pa, new=pb)
        else:
            performance = dict(status="not_comparable_conditions_or_members_differ", old=pa, new=pb)
    return dict(schema=integrity.SCHEMA, status=status, common_cases=len(pairs),
                old_total=a[0]["total_cases"], new_total=b[0]["total_cases"], label_conflicts=label_conflicts,
                rgb_conflicts=rgb_conflicts, old_correct=old_correct, new_correct=new_correct,
                exact_delta=(new_correct-old_correct)/len(pairs) if pairs else None,
                paired_transitions=dict(flips), old_correct_losses=losses,
                physical_route_groups=len({physical_group((x["scenario"], x["route_id"], x["frame_id"])) for x, _ in pairs}),
                old_metrics=evaluation.score_fields(a[0]), new_metrics=evaluation.score_fields(b[0]),
                production_prompt_changed=a[0].get("production_prompt_sha256") != b[0].get("production_prompt_sha256"),
                base_changed=(a[3]["base_asset_sha256"] != b[3]["base_asset_sha256"]) if certified else None,
                old_training=a[3].get("adapter_metadata") if a[3] else None,
                new_training=b[3].get("adapter_metadata") if b[3] else None,
                performance=performance,
                note="System comparison only; no causal attribution to base model or sampling. Historical exposed test is not a blind holdout.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("bundle")
    comparison = sub.add_parser("compare")
    comparison.add_argument("old")
    comparison.add_argument("new")
    comparison.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_package(args.bundle)
    else:
        result = compare(args.old, args.new)
        integrity.write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
