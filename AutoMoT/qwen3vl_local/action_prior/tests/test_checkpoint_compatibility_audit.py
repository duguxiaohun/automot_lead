"""Audit diagnostics and numerical replay against the actual pre-migration code."""
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from qwen3vl_local.action_prior import audit_checkpoint_compatibility as audit
from qwen3vl_local.action_prior.flow_matching import ConditionalFlowMatchingDecoder, FlowMatchingConfig
from qwen3vl_local.leadmot import LeadMoTPlanningDecoderConfig

REFERENCE = "59a5d7fe21de66e1bae95d95fd7848cccfe64f7b"


@pytest.fixture(scope="module")
def old_results(tmp_path_factory):
    """Isolated interpreter, exact old source, actual old EMA tensors and outputs."""
    folder = tmp_path_factory.mktemp("pre_qwen35_decoder")
    archive = subprocess.run(["git", "archive", REFERENCE, "AutoMoT/qwen3vl_local"],
                             cwd=ROOT.parent, capture_output=True)
    if archive.returncode:
        pytest.skip("Numerical reference requires local Git object " + REFERENCE)
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".py"):
                target = folder / member.name
                assert target.resolve().is_relative_to(folder)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
    script = folder / "reference.py"
    script.write_text('''
from dataclasses import asdict
import importlib.util
import sys
import torch
from qwen3vl_local.leadmot import LeadMoTPlanningDecoderConfig
from qwen3vl_local.action_prior.flow_matching import ConditionalFlowMatchingDecoder, FlowMatchingConfig
spec = importlib.util.spec_from_file_location("input_generator", sys.argv[1])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
torch.set_num_threads(1)
results = {}
for rope in ("mrope", "mhrope", "none"):
    for token in (False, True):
        torch.manual_seed(1234)
        config = LeadMoTPlanningDecoderConfig(hidden_size=32, num_heads=2, num_kv_heads=2,
            head_dim=16, num_layers=2, bev_channels=4, bev_grid=(2,3),
            mrope_section_dim=(2,3,3), mrope_section_head=(1,1,0), rope_type=rope,
            use_high_level_action_token=token)
        flow = FlowMatchingConfig(trajectory_heads=2, trajectory_layers=1, sample_steps=3)
        model = ConditionalFlowMatchingDecoder(config, flow).eval()
        inputs = helper.synthetic_inputs(config, batch=7 if token else 2)
        with torch.inference_mode():
            outputs = model(**inputs)
        state = dict(schema="action_expert_ablation_checkpoint_v1", ablation_variant="bev_only",
            trajectory_decoder="conditional_joint_trajectory_flow_matching_v2",
            decoder_config=asdict(config), flow_config=asdict(flow),
            ema_state_dict={"shadow": model.state_dict()})
        results[(rope, token)] = (state, inputs, outputs)
torch.save(results, sys.argv[2])
''')
    env = dict(os.environ, PYTHONPATH=str(folder / "AutoMoT"), OMP_NUM_THREADS="1")
    subprocess.run([sys.executable, str(script), str(Path(audit.__file__)), str(folder / "result.pt")],
                   cwd=folder, env=env, check=True, capture_output=True, text=True)
    return torch.load(folder / "result.pt", map_location="cpu", weights_only=False)


@pytest.mark.parametrize("rope", ["mrope", "mhrope", "none"])
@pytest.mark.parametrize("token", [False, True])
def test_legacy_candidate_matches_real_old_decoder(old_results, rope, token):
    state, inputs, expected = old_results[(rope, token)]
    config = audit.probe_config(state, legacy=True)
    model = ConditionalFlowMatchingDecoder(config, FlowMatchingConfig(**state["flow_config"])).eval()
    model.load_state_dict(state["ema_state_dict"]["shadow"], strict=True)
    with torch.inference_mode():
        actual = model(**inputs)
    assert set(actual) == set(expected)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=1e-6, rtol=1e-6)
    assert audit.probe_decoder(state, legacy=True)["strict_ema_load"]


def test_unmodified_new_defaults_reject_old_mrope(old_results):
    state, _, _ = old_results[("mrope", False)]
    with pytest.raises(ValueError, match="M-RoPE section sum"):
        audit.probe_config(state)


def test_real_old_dimensions_keep_12_layers_8_heads():
    saved = asdict(LeadMoTPlanningDecoderConfig())
    for name in ("partial_rotary_factor", "mrope_interleaved", "qwen_full_attention_layers"):
        saved.pop(name)
    saved.update(num_layers=12, num_heads=8, num_kv_heads=8, head_dim=128,
                 num_qwen_layers=36, rope_theta=5000000.0,
                 mrope_section_dim=(16,24,24), mrope_section_head=(3,3,2))
    state = dict(schema="action_expert_ablation_checkpoint_v1", ablation_variant="bev_only",
                 trajectory_decoder="conditional_joint_trajectory_flow_matching_v2", decoder_config=saved)
    restored = audit.probe_config(state, legacy=True)
    assert (restored.num_layers, restored.num_heads, restored.head_dim) == (12, 8, 128)
    assert restored.partial_rotary_factor == 1 and not restored.mrope_interleaved


def test_incomplete_or_new_config_cannot_be_reinterpreted(old_results):
    original = old_results[("mrope", False)][0]
    for config in ({**original["decoder_config"], "mrope_interleaved": True},
                   {key: value for key, value in original["decoder_config"].items() if key != "rope_theta"}):
        with pytest.raises(ValueError, match="complete pre-Qwen3.5"):
            audit.probe_config({**original, "decoder_config": config}, legacy=True)
    with pytest.raises(ValueError, match="BEV-only"):
        audit.probe_config({**original, "ablation_variant": "qwen_simple"}, legacy=True)


def test_probe_rejects_missing_weights_and_nonfinite_values(old_results):
    original = old_results[("mrope", False)][0]
    weights = dict(original["ema_state_dict"]["shadow"])
    weights.pop(next(iter(weights)))
    with pytest.raises(RuntimeError, match="Missing key"):
        audit.probe_decoder({**original, "ema_state_dict": {"shadow": weights}}, legacy=True)
    weights = {name: tensor.clone() for name, tensor in original["ema_state_dict"]["shadow"].items()}
    weights["velocity_head.3.bias"].fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        audit.probe_decoder({**original, "ema_state_dict": {"shadow": weights}}, legacy=True)


def test_diff_separates_sources_packages_and_missing():
    old = {"execution": {"code": {"a.py": "old"}, "packages": {"transformers": "4"}}, "base": None}
    new = {"execution": {"code": {"a.py": "new"}, "packages": {"transformers": "5"}}}
    diff = audit.differences(old, new)
    assert [item["path"] for item in diff] == [["base"], ["execution", "code", "a.py"],
                                               ["execution", "packages", "transformers"]]
    assert diff[0]["saved_present"] and not diff[0]["current_present"]


def test_recorded_sources_report_changes_even_if_other_files_missing(tmp_path, monkeypatch):
    from qwen3vl_local.action_prior.contracts import file_hash
    (tmp_path / "code.py").write_text("new")
    monkeypatch.setattr(audit.metadata, "version", lambda name: "5")
    payload = {"execution": {"code": {"code.py": "old", "missing.py": "old"},
                             "packages": {"transformers": "4"}}}
    report = audit.recorded_execution_audit(payload, tmp_path)
    assert len(report["differences"]) == 3
    assert report["differences"][0]["current"] == file_hash(tmp_path / "code.py")
    assert report["differences"][1]["current"] == {"unavailable": "FileNotFoundError"}


def test_contract_build_failure_still_reports_config_problem(old_results, monkeypatch, tmp_path):
    from qwen3vl_local.action_prior.contracts import digest
    from qwen3vl_local.action_prior import comparison_runtime
    def fail(*args):
        raise FileNotFoundError("missing original dataset")
    monkeypatch.setattr(comparison_runtime, "restore_args", fail)
    state = dict(old_results[("mrope", False)][0], condition_contract={"identity_payload": {}, "identity": digest({})})
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    report = audit.audit(state, checkpoint, {}, decoder_probe=True)
    assert report["contract"]["status"] == "failed"
    assert report["current_decoder_config"]["status"] == "failed"
    assert report["legacy_rope_candidate_probe"]["status"] == "passed"
    assert report["evaluation_authorized"] is False
    state["condition_contract"]["identity"] = "wrong"
    report = audit.audit(state, checkpoint, {}, decoder_probe=True)
    assert not report["saved_contract_integrity"]
    assert "legacy_rope_candidate_probe" not in report


def test_cli_writes_failure_report_without_touching_run(old_results, tmp_path):
    from qwen3vl_local.action_prior.contracts import digest
    run = tmp_path / "original run"
    run.mkdir()
    (run / "config.json").write_text("{}")
    state = dict(old_results[("mrope", False)][0], args={}, step=1, best_sampled_trajectory_score=1.0,
                 condition_contract={"identity_payload": {}, "identity": digest({})})
    checkpoint = run / "best.pt"
    torch.save(state, checkpoint)
    before = checkpoint.read_bytes()
    out = tmp_path / "audit.json"
    command = [sys.executable, audit.__file__, str(run), "--probe-decoder", "--output", str(out)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    report = json.loads(out.read_text())
    assert report["contract"]["status"] == "failed"
    assert report["legacy_rope_candidate_probe"]["status"] == "passed"
    assert report["evaluation_authorized"] is False
    assert checkpoint.read_bytes() == before
    assert sorted(path.name for path in run.iterdir()) == ["best.pt", "config.json"]
    # Neither a repeated report nor a destination within the run may be overwritten.
    report_before = out.read_bytes()
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2 and out.read_bytes() == report_before
    result = subprocess.run([*command[:-1], str(run / "audit.json")], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2 and not (run / "audit.json").exists()
