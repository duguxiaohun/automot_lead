"""Reviewed migration acceptance, refusal boundaries, and actual eval/worker wiring."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from qwen3vl_local.action_prior import evaluation_compatibility as compat
from qwen3vl_local.action_prior.contracts import digest, file_hash
from qwen3vl_local.action_prior.tests.test_checkpoint_compatibility_audit import old_results


def contracts(state):
    pairs = {**compat.QWEN_SOURCE_PAIRS, **compat.ENTRY_SOURCE_PAIRS}
    before = dict(variant="bev_only", condition=dict(uses_qwen=False), base=None, bev="same-bev",
                  precision="same-precision", high_level_action_token="same-labels",
                  event_balanced_sampling="same-sampling",
                  execution=dict(code={key: pair[0] for key, pair in pairs.items()},
                                 packages=dict(torch="same-torch", transformers="4.57.3")))
    before["execution"]["code"]["external_runner.py"] = "same-runner"
    after = deepcopy(before)
    after["execution"]["code"].update({key: pair[1] for key, pair in pairs.items()})
    after["execution"]["packages"]["transformers"] = "5.3.0"
    def wrap(payload):
        return dict(schema="action_expert_ablation_condition_v1", identity=digest(payload),
                    identity_payload=payload, variant="bev_only", prior_source="none",
                    adapter_enabled=False, final_cache_model="none_zero_length_prefix_kv")
    return {**deepcopy(state), "condition_contract": wrap(before)}, wrap(after)


def refresh(contract):
    contract["identity"] = digest(contract["identity_payload"])


def test_profile_hashes_are_exact_reviewed_current_sources():
    for name, (_, current) in {**compat.QWEN_SOURCE_PAIRS, **compat.ENTRY_SOURCE_PAIRS}.items():
        assert file_hash(ROOT / name) == current, name


def test_real_report_transition_preserves_checkpoint_and_decoder(old_results):
    state, actual = contracts(old_results[("mrope", True)][0])
    before = deepcopy(state["condition_contract"])
    note = compat.require_evaluation_contract(state, actual)
    assert note["profile"] == compat.BEV_PROFILE and len(note["changes"]) == 8
    config = compat.evaluation_decoder_config(state, note)
    assert config.num_layers == state["decoder_config"]["num_layers"]
    assert config.partial_rotary_factor == 1 and config.mrope_interleaved is False
    assert state["condition_contract"] == before
    from qwen3vl_local.action_expert_ablation.common import require_contract
    with pytest.raises(ValueError, match="contract mismatch"):
        require_contract(state["condition_contract"], actual)  # resume remains strict


@pytest.mark.parametrize("kind", ["source", "source_pair", "torch", "transformers", "bev", "labels", "sampling", "precision", "inventory", "integrity"])
def test_unknown_or_semantic_changes_are_rejected(old_results, kind):
    state, actual = contracts(old_results[("mrope", False)][0])
    payload = actual["identity_payload"]
    if kind == "source":
        payload["execution"]["code"]["external_runner.py"] = "modified"
    elif kind == "source_pair":
        payload["execution"]["code"]["qwen3vl_local/leadmot/mot_block.py"] = "unreviewed"
    elif kind in ("torch", "transformers"):
        payload["execution"]["packages"][kind] = "different-version"
    elif kind == "inventory":
        payload["execution"]["code"]["extra.py"] = "new"
    else:
        payload[{"labels": "high_level_action_token", "sampling": "event_balanced_sampling"}.get(kind, kind)] = "changed"
    if kind != "integrity":
        refresh(actual)
    with pytest.raises(ValueError, match="contract mismatch"):
        compat.require_evaluation_contract(state, actual)


def test_qwen_model_cannot_use_bev_migration(old_results):
    state, actual = contracts(old_results[("mrope", False)][0])
    state["ablation_variant"] = "qwen_simple"
    for contract in (state["condition_contract"], actual):
        contract["variant"] = contract["identity_payload"]["variant"] = "qwen_simple"
        contract["identity_payload"]["condition"]["uses_qwen"] = True
        refresh(contract)
    with pytest.raises(ValueError, match="unreviewed source"):
        compat.require_evaluation_contract(state, actual)


def test_entry_only_change_keeps_current_qwen_evaluation_working(old_results):
    state, actual = contracts(old_results[("mrope", False)][0])
    state["ablation_variant"] = "qwen_simple"
    state["decoder_config"]["partial_rotary_factor"] = 0.25
    saved = deepcopy(actual)
    for name, pair in compat.ENTRY_SOURCE_PAIRS.items():
        saved["identity_payload"]["execution"]["code"][name] = pair[0]
    for contract in (saved, actual):
        contract["variant"] = contract["identity_payload"]["variant"] = "qwen_simple"
        contract["identity_payload"]["condition"]["uses_qwen"] = True
        refresh(contract)
    state["condition_contract"] = saved
    note = compat.require_evaluation_contract(state, actual)
    assert note["profile"] == compat.ENTRY_PROFILE and not note["restore_legacy_rope"]
    state["condition_contract"] = actual
    assert compat.require_evaluation_contract(state, actual) is None


def test_comparison_checks_real_split_hashes_after_transition(old_results, tmp_path, monkeypatch):
    from qwen3vl_local.action_expert_ablation import common
    from qwen3vl_local.action_prior import comparison_runtime
    state, actual = contracts(old_results[("mrope", False)][0])
    state["dataset_hashes"] = {}
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text("original\n")
        state["dataset_hashes"][split] = file_hash(path)
    monkeypatch.setattr(common, "validate_args", lambda *a: None)
    monkeypatch.setattr(common, "build_contract", lambda *a: deepcopy(actual))
    args = SimpleNamespace(data_dir=str(tmp_path))
    result = comparison_runtime.check_contract(state, args, "bev_only")
    assert result["evaluation_compatibility"]["profile"] == compat.BEV_PROFILE
    (tmp_path / "test.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="test 索引"):
        comparison_runtime.check_contract(state, args, "bev_only")


def test_single_eval_uses_restored_config_and_actual_ema(old_results, tmp_path, monkeypatch):
    from qwen3vl_local.action_expert_ablation import common
    from qwen3vl_local.action_prior import launch
    original, inputs, expected = old_results[("mrope", True)]
    state, actual = contracts(original)
    data = tmp_path / "data"
    data.mkdir()
    (data / "test.jsonl").write_text("[]\n")
    state.update(args=dict(data_dir=str(data), decoder_dtype="float32"), step=1,
                 dataset_hashes=dict(test=file_hash(data / "test.jsonl")),
                 decoder={name: torch.zeros_like(value) for name, value in state["ema_state_dict"]["shadow"].items()})
    checkpoint = tmp_path / "best.pt"
    torch.save(state, checkpoint)
    out = tmp_path / "eval"
    monkeypatch.setattr(sys, "argv", ["eval", "--checkpoint", str(checkpoint), "--output-dir", str(out)])
    monkeypatch.setattr(launch, "ensure_gpu", lambda: None)
    monkeypatch.setattr(common, "leadmot_train", lambda: SimpleNamespace(
        _init_distributed=lambda: (0, 0, 1), _dtype=lambda value: torch.float32))
    monkeypatch.setattr(common, "validate_args", lambda *a: None)
    monkeypatch.setattr(common, "build_contract", lambda *a: deepcopy(actual))
    monkeypatch.setattr(common, "read_rows", lambda *a: [])
    monkeypatch.setattr(common, "make_runtime", lambda *a: None)
    model_class = common.ConditionalFlowMatchingDecoder
    def cpu_model(*args):
        model = model_class(*args)
        model.to = lambda **kwargs: model  # CUDA orchestration only; tensors stay on CPU.
        return model
    monkeypatch.setattr(common, "ConditionalFlowMatchingDecoder", cpu_model)
    def evaluate(runtime, model, config, *args):
        assert config.partial_rotary_factor == 1.0 and config.mrope_interleaved is False
        model.eval()
        with torch.inference_mode():
            outputs = model(**inputs)
        for name in expected:
            torch.testing.assert_close(outputs[name], expected[name], atol=1e-6, rtol=1e-6)
        return {"samples": 7}
    monkeypatch.setattr(common, "evaluate", evaluate)
    common.eval_main("bev_only")
    report = json.loads((out / "metrics.json").read_text())
    assert report["evaluation_compatibility"]["saved_identity"] == state["condition_contract"]["identity"]
    assert report["ema"] and report["metrics"]["samples"] == 7


def test_comparison_worker_loads_legacy_ema_after_its_own_contract_check(old_results, tmp_path, monkeypatch):
    from qwen3vl_local.action_expert_ablation import common
    from qwen3vl_local.action_prior import comparison_runtime as runtime, flow_matching, launch
    original, inputs, expected = old_results[("mrope", False)]
    state, actual = contracts(original)
    state["dataset_hashes"] = {}
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text("[]\n")
        state["dataset_hashes"][split] = file_hash(path)
    checkpoint = tmp_path / "best.pt"
    torch.save(state, checkpoint)
    cases = tmp_path / "cases.json"
    cases.write_text('[{"anchor": 4}]')
    monkeypatch.setattr(launch, "ensure_gpu", lambda: None)
    args = SimpleNamespace(data_dir=str(tmp_path), seed=99, decoder_dtype="float32")
    monkeypatch.setattr(runtime, "restore_args", lambda *a: (args, "bev_only"))
    monkeypatch.setattr(common, "validate_args", lambda *a: None)
    monkeypatch.setattr(common, "build_contract", lambda *a: deepcopy(actual))
    model_class = flow_matching.ConditionalFlowMatchingDecoder
    def cpu_model(*args):
        model = model_class(*args)
        model.to = lambda **kwargs: model
        return model
    monkeypatch.setattr(flow_matching, "ConditionalFlowMatchingDecoder", cpu_model)
    monkeypatch.setattr(common, "make_runtime", lambda *a: SimpleNamespace(
        runner=SimpleNamespace(_prepare_inference_inputs=lambda *a: None)))
    def evaluate(capture, model, config, rows, *args):
        assert config.partial_rotary_factor == 1 and config.mrope_interleaved is False
        model.eval()
        with torch.inference_mode():
            outputs = model(**inputs)
        for name in expected:
            torch.testing.assert_close(outputs[name], expected[name], atol=1e-6, rtol=1e-6)
        return {"samples": len(rows)}
    monkeypatch.setattr(common, "evaluate", evaluate)
    job = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint), overrides={},
               seed=2026, workers=0, output=str(tmp_path / "worker"), rows={"test": str(cases)},
               row_hashes={"test": file_hash(cases)})
    runtime.evaluate_worker(job)
    assert json.loads((tmp_path / "worker/test/metrics.json").read_text())["samples"] == 1
