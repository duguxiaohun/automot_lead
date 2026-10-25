"""Acceptance failures, native scoring replay and unchanged sampling contracts."""
import ast
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import shutil
import os
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from qwen3vl_local.sft_new_loop_phase1 import run_integrity as integrity
from qwen3vl_local.sft_new_loop_phase1 import evaluation_integrity as evaluation
from qwen3vl_local.sft_new_loop_phase1 import delivery
from qwen3vl_local.sft_new_loop_phase1.test_training_policy import write_dataset
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35.tests.test_processor import processor


def fixture_assets(root, phase):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    index = write_dataset(root / "data", phase)
    # All raw fixture RGB references are real small local images.
    records = [json.loads(x) for x in index.read_text().splitlines()]
    for row in records:
        for name in row["history_rgb_paths"]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                Image.new("RGB", (32, 32), (40, 60, 80)).save(path)
    model = root / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"fixture base; never loaded as a real model")
    (model / "config.json").write_text('{"model_type":"qwen3_5","fixture_only":true}')
    return index, model


def train_fixture(phase, arguments, *, skip=False, nonfinite=False, max_steps=5):
    """Native trainer/optimizer/save contracts, a small Linear in place of the VLM."""
    module = importlib.import_module(f"qwen3vl_local.sft_new_loop_phase{phase}.train")
    from qwen3vl_local.qwen35.identity import base_asset_hashes
    with pytest.MonkeyPatch.context() as monkey:
        monkey.setattr(sys, "argv", ["train", *arguments,
            "--device", "cpu", "--focus-balance-count", "6", "--max-steps", str(max_steps), "--grad-accum", "2",
            "--eval-steps", "0", "--generation-eval-steps", "0", "--save-steps", "0",
            "--action-index", "off", "--no-tb", "--budget-probe-steps", "2"])
        args = module.parse_args()
        model = torch.nn.Linear(2, 1, bias=False)
        model.config = SimpleNamespace(local_asset_sha256=base_asset_hashes(args.model_dir))
        def save_pretrained(path, **kwargs):
            target = Path(path)
            target.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), target / "adapter_model.bin")
            integrity.write(target / "adapter_config.json", {"fixture_only": True})
        model.save_pretrained = save_pretrained
        bundle = SimpleNamespace(model=model, unwrap=lambda: model, lora_target_modules=[], device=torch.device("cpu"))
        monkey.setattr(module, "load_model_with_lora", lambda *a, **k: bundle)
        monkey.setattr(module, "_load_images", lambda *a: [None])
        monkey.setattr(module, "_build_inputs", lambda *a, **k: None if skip else {})
        def loss(*a, **k):
            value = (model(torch.ones(1, 2)) - 1).square().sum()
            return (value * float("nan") if nonfinite else value), {"denom": 1.0}
        monkey.setattr(module, "_loss_one", loss)
        module.train(args)
    return Path(args.output_dir)


def eval_fixture(phase, arguments):
    """Real members, images, native parser and metrics; only VLM output is substituted."""
    module = evaluation.module_for(phase)
    from qwen3vl_local.qwen35.identity import base_asset_hashes
    with pytest.MonkeyPatch.context() as monkey:
        monkey.setattr(sys, "argv", ["eval", *arguments, "--device", "cpu"])
        args = module.parse_args()
        rows = module._read_rows(Path(args.index), split=args.split, max_frames=args.max_frames, data_root=Path(args.data_root))
        kwargs = dict(cases_per_bin=args.cases_per_bin, seed=args.seed)
        if phase == 2:
            rows, _ = module._exclude_prior_cases(rows, [Path(p) for p in args.exclude_cases_jsonl], expected_excluded_cases=args.expected_excluded_cases)
            kwargs.update(highway_regular_fraction=args.highway_regular_fraction, highway_ue3_fraction=args.highway_ue3_fraction,
                          route_diverse=args.route_diverse_sampling)
        items = module._balanced_cases(rows, **kwargs)
        answers = []
        for i, item in enumerate(items):
            prompts = importlib.import_module(f"qwen3vl_local.sft_new_loop_phase{phase}.prompts")
            target = (module.build_phase1_target(item.row.answers, spec=item.spec) if phase == 1 else prompts.build_event_target(item.spec))
            answers.append(target if i % 3 else "BROKEN OUTPUT")
        iterator = iter(answers)
        model = torch.nn.Linear(2, 1)
        model.config = SimpleNamespace(local_asset_sha256=base_asset_hashes(args.model_dir), _attn_implementation="fixture")
        bundle = SimpleNamespace(model=model, device=torch.device("cpu"))
        monkey.setattr(module, "load_eval_bundle", lambda *a, **k: bundle)
        monkey.setattr(module, "setup_distributed", lambda: (0,0,1))
        def generate(*a, **k):
            # Deliberately do not invent timing numbers for the substituted VLM.
            return next(iterator)
        monkey.setattr(module, "_generate", generate)
        return module.evaluate(args)


def completed_fixture(tmp_path, phase):
    index, model = fixture_assets(tmp_path, phase)
    run = train_fixture(phase, ["--index", str(index), "--data-root", str(tmp_path),
                                "--model-dir", str(model), "--output-dir", str(tmp_path / "run"),
                                "--history-rgb-mode", "2rgb_endpoints"])
    return index, model, run


@pytest.mark.parametrize("phase", [1,2])
def test_real_cpu_training_consumption_weights_and_tamper(tmp_path, phase):
    index, model, run = completed_fixture(tmp_path, phase)
    result = integrity.verify_training(run)
    assert result["budget"]["total_steps"] == 5
    assert result["budget"]["optimizer_updates"] == 3
    assert result["exposure"]["successful_backward"] == 5
    weights = run / "final/adapter_model.bin"
    weights.write_bytes(weights.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        integrity.verify_training(run)


@pytest.mark.parametrize("phase", [1,2])
@pytest.mark.parametrize("failure", ["skip", "nan"])
def test_no_final_on_skipped_or_nonfinite_training(tmp_path, phase, failure):
    index, model = fixture_assets(tmp_path, phase)
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="context limit|Non-finite"):
        train_fixture(phase, ["--index", str(index), "--data-root", str(tmp_path),
            "--model-dir", str(model), "--output-dir", str(run)], skip=failure=="skip", nonfinite=failure=="nan")
    assert not (run / "final").exists()
    assert not (run / "training_complete.json").exists()


@pytest.mark.parametrize("phase", [1,2])
def test_native_eval_replay_missing_cases_and_metric_tamper(tmp_path, phase):
    index, model = fixture_assets(tmp_path, phase)
    out = tmp_path / "eval"
    metrics = eval_fixture(phase, ["--index", str(index), "--data-root", str(tmp_path),
        "--model-dir", str(model), "--output-dir", str(out), "--no-timestamp-output",
        "--cases-per-bin", "4", "--history-rgb-mode", "2rgb_endpoints", "--no-save-error-rgb"])
    receipt = delivery.verified_eval(out)
    assert receipt["total_cases"] == metrics["total_cases"]
    assert receipt["performance"]["status"].startswith("unmeasured")
    path = out / "metrics.json"
    metrics["exact_match_accuracy"] = .9999
    integrity.write(path, metrics)
    with pytest.raises(ValueError, match="Metrics differ"):
        evaluation.verify_evaluation(out)
    cases = out / "cases.jsonl"
    cases.write_text("\n".join(cases.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="membership"):
        evaluation.verify_evaluation(out)


@pytest.mark.parametrize("phase", [1,2])
def test_delivery_with_real_eval_and_no_weight_copy(tmp_path, phase):
    index, model, run = completed_fixture(tmp_path, phase)
    review = tmp_path / "review"
    for name in ("base_production", "lora_production"):
        arguments = ["--index", str(index), "--data-root", str(tmp_path), "--model-dir", str(model),
            "--output-dir", str(review / name), "--no-timestamp-output", "--cases-per-bin", "4",
            "--no-save-error-rgb"]
        arguments += (["--adapter-dir", str(run / "final")] if name.startswith("lora") else ["--history-rgb-mode", "2rgb_endpoints"])
        eval_fixture(phase, arguments)
    delivery.prepare(review, phase, run / "final", index)
    assert delivery.seal(review)["status"] == "verified"
    assert not list((review / "verified_training").rglob("*.bin"))
    result = delivery.compare(review / "base_production", review / "lora_production")
    assert result["status"] == "paired_system_comparison" and result["exact_delta"] == 0
    path = review / "verified_training/exposure/rank_0.json"
    path.write_text('{}')
    with pytest.raises(ValueError, match="hash mismatch"):
        delivery.verify_package(review)


@pytest.mark.parametrize("phase", [1,2])
def test_jsonl_packer_never_truncates_scoring_evidence(tmp_path, phase):
    script = Path(__file__).parents[1] / f"sft_new_loop_phase{phase}/eval.sh"
    source = script.read_text()
    start, end = source.index("def copy_text("), source.index("\ndef copy_image(")
    namespace = dict(pathlib=__import__("pathlib"), shutil=shutil)
    exec(source[start:end], namespace)
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    src.write_text("".join(json.dumps({"case_index": i})+"\n" for i in range(1024)))
    namespace["copy_text"](src,dst)
    assert src.read_bytes() == dst.read_bytes()


def test_native_sampler_implementations_and_prompt_files_preserved():
    contract = integrity.read(Path(__file__).with_name("sampling_preservation_20261025.json"))
    for name, functions in contract["functions"].items():
        tree = ast.parse((integrity.ROOT / name).read_text())
        current = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name, expected in functions.items():
            import hashlib
            assert hashlib.sha256(ast.dump(current[name], include_attributes=False).encode()).hexdigest() == expected
    for name, expected in contract["files"].items():
        assert integrity.sha(integrity.ROOT / name) == expected


@pytest.mark.parametrize("phase", [1,2])
def test_native_qwen_generate_still_uses_fresh_state_and_records_tokens(model, processor, phase):
    module = evaluation.module_for(phase)
    for key, token in [("image_token_id","<|image_pad|>"),("video_token_id","<|video_pad|>"),
                       ("vision_start_token_id","<|vision_start|>"),("vision_end_token_id","<|vision_end|>")]:
        setattr(model.config, key, processor.tokenizer.convert_tokens_to_ids(token))
    model.eval()
    bundle = module.EvalBundle(model, processor, processor.tokenizer, torch.device("cpu"))
    images = [Image.new("RGB", (32,32)) for _ in range(2)]
    if phase == 1:
        text = module._generate(bundle, images, "Report the visible traffic facts.", 2)
    else:
        from qwen3vl_local.sft_new_loop_phase1.test_training_policy import phase2_rows
        row = phase2_rows(8)[0]
        spec = module._make_all_item(row, module._target_class(row), seed=1).spec
        text = module._generate(bundle, images, spec=spec, audit=False, history_rgb_mode="2rgb_endpoints", max_new_tokens=2)
    assert isinstance(text, str)
    measured = bundle._generation_measurement
    assert measured["seconds"] > 0 and measured["input_tokens"] > 0
    assert 0 < measured["generated_tokens"] <= 2
    assert measured["peak_allocated_bytes"] is None


@pytest.mark.parametrize("phase", [1,2])
def test_replay_rejects_raw_answer_and_duplicate_case(tmp_path, phase):
    index, base = fixture_assets(tmp_path, phase)
    out = tmp_path / "eval"
    eval_fixture(phase, ["--index",str(index),"--data-root",str(tmp_path),"--model-dir",str(base),
        "--output-dir",str(out),"--no-timestamp-output","--cases-per-bin","4","--no-save-error-rgb"])
    path = out / "cases.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[1]["raw_output"] = "CORRUPT"
    path.write_text("".join(json.dumps(x)+"\n" for x in rows))
    with pytest.raises(ValueError, match="native replay"):
        evaluation.verify_evaluation(out)
    rows[1]["case_index"] = 0
    path.write_text("".join(json.dumps(x)+"\n" for x in rows))
    with pytest.raises(ValueError, match="misplaced"):
        evaluation.verify_evaluation(out)


def test_legacy_comparison_never_claims_verified_rgb(tmp_path):
    index, base = fixture_assets(tmp_path, 2)
    out = tmp_path / "new"
    eval_fixture(2, ["--index",str(index),"--data-root",str(tmp_path),"--model-dir",str(base),
        "--output-dir",str(out),"--no-timestamp-output","--cases-per-bin","4","--no-save-error-rgb"])
    old = tmp_path / "old"
    shutil.copytree(out, old)
    (old / "evaluation_complete.json").unlink()
    result = delivery.compare(old, out)
    assert result["status"] == "legacy_reference_unverified_rgb_and_protocol"
    assert result["performance"]["status"].startswith("not_comparable")


def test_completion_rejects_optimizer_counter_and_wrong_step(tmp_path):
    _, _, run = completed_fixture(tmp_path, 1)
    updates = run / "exposure/optimizer_0.json"
    integrity.write(updates, dict(optimizer_updates=2, loop_steps=5))
    with pytest.raises(ValueError, match="Optimizer"):
        integrity.validate_consumption(run, integrity.read(run / "training_budget.json"))
    cfg = run / "final/sft_new_loop_phase1_adapter_config.json"
    data = integrity.read(cfg)
    data["global_step"] = 4
    integrity.write(cfg, data)
    with pytest.raises(ValueError, match="step/budget"):
        integrity.validate_saved_adapter(run / "final", integrity.read(run / "training_inputs.json"))


def test_performance_excludes_warmup_and_records_condition(tmp_path):
    records = [dict(case_index=i, seconds=t, input_tokens=10, generated_tokens=2,
                    peak_allocated_bytes=100, peak_reserved_bytes=200) for i,t in enumerate([100,100,1,2,3])]
    integrity.write(tmp_path / "performance_rank0.json", dict(records=records, scope="fixture",
        dtype="bf16", merge_lora=True, max_new_tokens=2, environment={}))
    report = evaluation.performance_summary(tmp_path, 1)
    assert report["measured_cases"] == 3 and report["latency_seconds_p50"] == 2
    assert report["latency_seconds_p95"] == pytest.approx(2.9)
    assert report["serial_equivalent_cases_per_second"] == .5
    other = integrity.read(tmp_path / "performance_rank0.json")
    other["attention_implementation"] = "different_attention"
    integrity.write(tmp_path / "performance_rank1.json", other)
    with pytest.raises(ValueError, match="conditions differ"):
        evaluation.performance_summary(tmp_path, 2)


def test_two_gloo_ranks_agree_on_invalid_input_loss_and_rank0_failure(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text('''import torch
import torch.distributed as dist
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as integrity
dist.init_process_group('gloo')
rank=dist.get_rank()
checks=[lambda:integrity.require_packed(None if rank else {},device='cpu',distributed=dist),
        lambda:integrity.finite_loss(torch.tensor(float('nan') if rank else 1.),distributed=dist),
        lambda:integrity.collective_rank0(lambda:(_ for _ in ()).throw(ValueError('rank0 failure')),rank=rank,distributed=dist)]
for check in checks:
    rejected=False
    try:check()
    except ValueError:rejected=True
    flags=[None,None];dist.all_gather_object(flags,rejected)
    assert flags==[True,True],flags
dist.destroy_process_group()
''')
    env = dict(os.environ, PYTHONPATH=str(integrity.ROOT))
    result = subprocess.run([sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=2","--",str(script)],
                            env=env,capture_output=True,text=True,timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
