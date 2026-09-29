#!/usr/bin/env python3
"""Read-only BEV-only checkpoint diagnostics; never authorizes contract bypass."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime
from importlib import metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def differences(expected, actual, path=()):
    """Return leaf differences, retaining missing-vs-null and literal file names."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(expected.keys() | actual.keys()):
            if key not in expected or key not in actual:
                result.append(dict(path=[*path, key], saved_present=key in expected,
                                   current_present=key in actual,
                                   saved=expected.get(key), current=actual.get(key)))
            else:
                result.extend(differences(expected[key], actual[key], (*path, key)))
        return result
    if expected == actual:
        return []
    return [dict(path=list(path), saved_present=True, current_present=True,
                 saved=expected, current=actual)]


def probe_config(state, *, legacy=False):
    """Restore all saved dimensions; old full/non-interleaved RoPE is audit-only."""
    from qwen3vl_local.leadmot import LeadMoTPlanningDecoderConfig
    if (state.get("ablation_variant") != "bev_only"
            or state.get("schema") != "action_expert_ablation_checkpoint_v1"
            or state.get("trajectory_decoder") != "conditional_joint_trajectory_flow_matching_v2"):
        raise ValueError("decoder probe supports BEV-only joint FM checkpoints only")
    saved = dict(state["decoder_config"])
    if legacy:
        added = {"partial_rotary_factor", "mrope_interleaved", "qwen_full_attention_layers"}
        required = {field.name for field in fields(LeadMoTPlanningDecoderConfig)} - added
        if set(saved) != required:
            raise ValueError("legacy probe requires the complete pre-Qwen3.5 decoder config, without new fields")
        saved.update(partial_rotary_factor=1.0, mrope_interleaved=False,
                     qwen_full_attention_layers=())
    config = LeadMoTPlanningDecoderConfig(**saved)
    config.validate_qwen_kv_shape()
    return config


def synthetic_inputs(config, *, seed=2026, batch=1):
    """Fixed CPU FP32 decoder inputs, including zero Qwen prefix and all action IDs."""
    import torch
    generator = torch.Generator().manual_seed(seed)
    def randn(*shape):
        return torch.randn(*shape, generator=generator)
    count = config.num_route_queries + config.num_waypoint_queries
    inputs = dict(
        pooled_kv=[(torch.empty(batch, config.num_kv_heads, 0, config.head_dim),
                    torch.empty(batch, config.num_kv_heads, 0, config.head_dim))
                   for _ in range(config.num_layers)],
        bev=randn(batch, config.bev_channels, *config.bev_grid) if config.use_bev else None,
        speed=randn(batch), target_point=randn(batch, 2), target_point_next=randn(batch, 2),
        final_goal=randn(batch, 2) if config.use_final_goal else None,
        rope_position_offset=0, flow_state=randn(batch, count, 2),
        flow_time=torch.linspace(0.1, 0.9, batch), flow_sample_noise=randn(batch, count, 2),
        sample_trajectory=True,
    )
    if config.use_high_level_action_token:
        inputs["action_token_id"] = torch.arange(batch) % 7
    return inputs


def probe_decoder(state, *, legacy=False):
    """Strict EMA load plus finite forward/ODE smoke; no RGB/LiDAR or GPU claims."""
    import torch
    from qwen3vl_local.action_prior.flow_matching import ConditionalFlowMatchingDecoder, FlowMatchingConfig
    config = probe_config(state, legacy=legacy)
    model = ConditionalFlowMatchingDecoder(config, FlowMatchingConfig(**state["flow_config"]))
    model.load_state_dict(state["ema_state_dict"]["shadow"], strict=True)
    model.eval()
    with torch.inference_mode():
        output = model(**synthetic_inputs(config, batch=7 if config.use_high_level_action_token else 2))
    for name, tensor in output.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"nonfinite decoder output: {name}")
    return dict(status="passed", strict_ema_load=True, config=asdict(config),
                output_shapes={key: list(value.shape) for key, value in output.items()},
                scope="synthetic BEV features, CPU FP32, forward and Euler; not end-to-end compatibility")


def attempt(fn):
    try:
        return fn()
    except Exception as exc:
        return dict(status="failed", error_type=type(exc).__name__, error=str(exc))


def recorded_execution_audit(payload, root=ROOT):
    """Keep source/package diagnostics even when datasets or the runner are missing."""
    from qwen3vl_local.action_prior.contracts import file_hash
    saved = payload["execution"]
    current = dict(code={}, packages={})
    for name in saved["code"]:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid recorded source path")
        try:
            current["code"][name] = file_hash(root / relative)
        except OSError as exc:
            current["code"][name] = dict(unavailable=type(exc).__name__)
    for name in saved["packages"]:
        try:
            current["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            current["packages"][name] = None
    return dict(status="completed", differences=differences(saved, current),
                scope="recorded source/package list only; full current contract checked separately")


def audit(state, checkpoint, overrides, *, decoder_probe=False):
    from qwen3vl_local.action_prior.contracts import digest, file_hash
    from qwen3vl_local.action_prior.comparison_runtime import restore_args
    report = dict(schema="bev_checkpoint_compatibility_audit_v1", checkpoint=str(checkpoint),
                  checkpoint_sha256=file_hash(checkpoint), evaluation_authorized=False,
                  scope="diagnosis only; original checkpoint and production contract checks are unchanged")
    expected = state.get("condition_contract", {})
    payload = expected.get("identity_payload")
    report["saved_contract_integrity"] = bool(isinstance(payload, dict) and digest(payload) == expected.get("identity"))
    report["variant"] = state.get("ablation_variant")
    if report["variant"] != "bev_only":
        report["error"] = "Only BEV-only is supported; old Qwen-conditioned models cannot substitute Qwen3.5"
        return report
    report["recorded_execution"] = attempt(lambda: recorded_execution_audit(payload))

    def contract_audit():
        from qwen3vl_local.action_expert_ablation.common import build_contract, validate_args
        args, variant = restore_args(state, checkpoint, overrides)
        validate_args(args, variant)
        actual = build_contract(args, variant)
        diff = differences(payload, actual["identity_payload"])
        splits = {split: file_hash(Path(args.data_dir) / f"{split}.jsonl") == state["dataset_hashes"][split]
                  for split in ("train", "val", "test")}
        return dict(status="completed", saved_identity=expected.get("identity"),
                    current_identity=actual["identity"], differences=diff, dataset_hash_matches=splits,
                    exact_contract_match=(expected.get("schema") == actual["schema"]
                                          and expected.get("identity") == actual["identity"]))

    report["contract"] = attempt(contract_audit)
    report["current_decoder_config"] = attempt(lambda: dict(status="passed", config=asdict(probe_config(state))))
    if not report["saved_contract_integrity"]:
        report["error"] = "Missing or inconsistent saved contract payload; no decoder probes executed"
        return report
    if decoder_probe:
        report["current_decoder_probe"] = attempt(lambda: probe_decoder(state))
        added = {"partial_rotary_factor", "mrope_interleaved", "qwen_full_attention_layers"}
        if not added.intersection(state.get("decoder_config", {})):
            report["legacy_rope_candidate_probe"] = attempt(lambda: probe_decoder(state, legacy=True))
    report["remaining_checks"] = [
        "Review every source/package/asset difference; a matching tensor shape is insufficient",
        "Compare old and new environments on identical real RGB/LiDAR cases, original EMA and evaluation noise",
        "Check BEV features, decoder conditioning, trajectories and ADE/FDE with declared numerical tolerances",
        "Synthetic probes do not validate the external runner, preprocessing, CUDA/BF16 or training resume",
    ]
    return report


def main(argv=None):
    from qwen3vl_local.action_prior.comparison_runtime import PATH_FIELDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="original timestamped BEV-only training directory")
    parser.add_argument("--output", help="new JSON report path; refuses overwrite or writing inside the run")
    parser.add_argument("--probe-decoder", action="store_true", help="also strictly load EMA and run synthetic CPU FP32 inference")
    for name in PATH_FIELDS:
        parser.add_argument("--" + name.replace("_", "-"), default="")
    cli = parser.parse_args(argv)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    import torch
    from qwen3vl_local.action_prior.comparison_cases import resolve_path, select_checkpoint
    torch.set_num_threads(4)
    run = resolve_path(cli.run)
    out = Path(cli.output).expanduser().resolve() if cli.output else ROOT / "test" / (
        "compatibility_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    if out == run or run in out.parents or out.exists():
        parser.error("report must be a new file outside the original training directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the report before loading: selection/load failures remain inspectable.
    with out.open("x", encoding="utf-8") as handle:
        def run_audit():
            print("[compatibility] selecting/loading checkpoint on CPU", flush=True)
            checkpoint, state, selection = select_checkpoint(
                run, lambda p: torch.load(p, map_location="cpu", weights_only=False))
            print("[compatibility] checking contract/config" + (" and probing EMA decoder" if cli.probe_decoder else ""), flush=True)
            report = audit(state, checkpoint, {name: getattr(cli, name) for name in PATH_FIELDS if getattr(cli, name)},
                           decoder_probe=cli.probe_decoder)
            report["selection"] = selection
            return report
        report = attempt(run_audit)
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"[compatibility] diagnostic report: {out}; this does not unlock evaluation", flush=True)
    return 0 if report.get("contract", {}).get("status") == "completed" and not report.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
