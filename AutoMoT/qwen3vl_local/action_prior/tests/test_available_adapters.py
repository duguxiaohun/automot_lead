"""跨服务器现有权重选择、历史提示词、预检固定与错误隔离；不加载 GPU 模型。"""

import json
from pathlib import Path
import subprocess
import sys

import pytest
from test_contracts import fixture_adapter
from qwen3vl_local.action_prior import phase2_v3_prompts as v3
from qwen3vl_local.action_prior.available_adapters import select_available, scan_available, candidate
from qwen3vl_local.action_prior.prompt_versions import prompt_module
from qwen3vl_local.action_prior.config import parser, build_contract


def event_adapter(root, score=.8, slot="fallback_generation", mode="4rgb"):
    """保存真实 v3 提示词哈希、独立 step 指标和未达标 guard 的合成权重。"""
    path = fixture_adapter(root, 2, slot=slot, score=score, mode=mode)
    file = path / "sft_new_loop_phase2_adapter_config.json"
    cfg = json.loads(file.read_text())
    cfg.update(prompt_name=v3.PROMPT_NAME,
               production_prompt_sha256=v3.event_prompt_sha256(history_rgb_mode=mode),
               git={"commit": None}, generation_eval_min_valid_rate=1.)
    file.write_text(json.dumps(cfg))
    record = dict(step=100, generation_exact_accuracy=score,
                  generation_guards_ok=slot == "best_generation",
                  generation_guards={"passed": {"ue3_target_recall": slot == "best_generation"}},
                  generation={"step": 100, "exact_accuracy": score, "format_valid_rate": 1.,
                              "samples": 192, "slice/ue3_target_recall": .4375})
    (path.parent / f"{slot}.json").write_text(json.dumps(record))
    return path


def test_union_of_two_servers_selects_each_phase_without_run_name_rules(tmp_path):
    servers = [tmp_path / "server_a", tmp_path / "server_b"]
    a = fixture_adapter(servers[0] / "arbitrary_name", 1, score=.82)
    fixture_adapter(servers[1] / "newest_name", 1, score=.79)
    event_adapter(servers[0] / "different_name", score=.79)
    b = event_adapter(servers[1] / "not_a_timestamp", score=.83)
    # 模拟共享路径的统一基座，不依赖训练服务器旧挂载路径。
    for server in servers:
        for file in server.rglob("sft_new_loop_phase*_adapter_config.json"):
            cfg = json.loads(file.read_text())
            cfg.update(base_model_dir="/old/server/Qwen3-VL-4B-Instruct", git={"commit": None})
            file.write_text(json.dumps(cfg))
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "a").symlink_to(servers[0], target_is_directory=True)
    (shared / "b").symlink_to(servers[1], target_is_directory=True)
    model = tmp_path / "Qwen3-VL-4B-Instruct"
    assert select_available([shared], 1, model)["path"] == str(a)
    selected = select_available(servers, 2, model)
    assert selected["path"] == str(b) and not selected["is_best_generation"]
    assert selected["runtime_prompt_name"] == v3.PROMPT_NAME
    assert selected["file_sha256"]["adapter_model.safetensors"]
    assert len(selected["warnings"]) == 4


def test_valid_best_has_priority_over_higher_scoring_fallback(tmp_path):
    best = event_adapter(tmp_path / "a", score=.7, slot="best_generation")
    event_adapter(tmp_path / "b", score=.9)
    event_adapter(tmp_path / "c", score=1., slot="final")
    selected = select_available([tmp_path], 2, tmp_path / "base")
    assert selected["path"] == str(best) and selected["is_best_generation"]


@pytest.mark.parametrize("mode,expected", [
    ("4rgb", "34f08770084117b415b6abff20d5ebdae4230182dd969702bdd070a6b463bb57"),
    ("2rgb_endpoints", "cd564634257fe0f072de70947200a820d6dd2b43375981b60120a1fe2296dd7f"),
])
def test_frozen_v3_hash_matches_remote_training_metadata(mode, expected):
    assert v3.event_prompt_sha256(history_rgb_mode=mode) == expected


def test_v3_module_drives_all_event_questions_and_rechecks():
    from qwen3vl_local.action_prior.priors import collect_priors
    from qwen3vl_local.sft_new_loop_phase1 import prompts as p1
    calls = []

    def ask(phase, spec, history):
        module = p1 if phase == 1 else v3
        render = module.build_phase1_prompt if phase == 1 else module.build_event_prompt
        prompt = render(spec=spec, history_rgb_mode="4rgb")
        if phase == 2:
            assert v3.PROMPT_NAME in prompt and "v5_highway" not in prompt
            assert isinstance(spec, v3.PromptSpec)
        calls.append((phase, spec.output_keys, history))
        return "\n".join(f"{key}: NO" for key in spec.output_keys), prompt

    collect_priors(ask, "sample", event_module=v3)
    assert [phase for phase, _, _ in calls] == [1] * 5 + [2] * 4
    assert calls[6][2] and calls[8][2]  # 两域继续问的历史确实包含第一次回答。


@pytest.mark.parametrize("damage", ["hash", "step", "format", "weights", "base", "old_package"])
def test_authorized_relaxations_do_not_bypass_real_incompatibilities(tmp_path, damage):
    path = event_adapter(tmp_path / "a")
    file = path / "sft_new_loop_phase2_adapter_config.json"
    cfg = json.loads(file.read_text())
    if damage == "hash":
        cfg["production_prompt_sha256"] = "incorrect"
    if damage == "base":
        cfg["base_model_dir"] = "/other/Qwen3-VL-8B-Instruct"
    file.write_text(json.dumps(cfg))
    if damage == "old_package":
        file.rename(path / "sft_loop_phase2_augment_adapter_config.json")
    if damage == "weights":
        (path / "adapter_model.safetensors").unlink()
    if damage in ("step", "format"):
        record = path.parent / "fallback_generation.json"
        data = json.loads(record.read_text())
        if damage == "step":
            data["step"] = 200
        else:
            data["generation"]["format_valid_rate"] = .9
        record.write_text(json.dumps(data))
    result = scan_available([tmp_path], 2, tmp_path / "base")
    assert result["recommended"] is None


def test_selection_manifest_keeps_original_pair_and_rejects_changed_weights(tmp_path):
    p1 = fixture_adapter(tmp_path / "p1", 1)
    p2 = event_adapter(tmp_path / "p2")
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base fixture")
    bev = tmp_path / "bev.pt"
    bev.write_bytes(b"BEV fixture")
    args = parser().parse_args(["--checkpoint-root", str(tmp_path), "--model-dir", str(base),
                               "--lead-bev-ckpt", str(bev)])
    contract = build_contract(args)
    sources = contract["identity_payload"]["execution"]["code"]
    assert "qwen3vl_local/action_prior/available_adapters.py" in sources
    assert "qwen3vl_local/action_prior/phase2_v3_prompts.py" in sources
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps(dict(schema="action_prior_selection_v1", selection_policy="available",
        contract_identity=contract["identity"], phase1=contract["phase1"], phase2=contract["phase2"])))
    event_adapter(tmp_path / "p2_new", score=.99)
    args.selection_manifest = str(manifest)
    pinned = build_contract(args)
    assert pinned["phase2"]["path"] == str(p2)
    assert pinned["identity"] == contract["identity"]
    (p1 / "adapter_model.safetensors").write_bytes(b"changed during preflight")
    with pytest.raises(ValueError, match="changed after preflight"):
        build_contract(args)


def test_cli_searches_both_shared_roots_and_prints_training_command(tmp_path):
    fixture_adapter(tmp_path / "p1", 1)
    event_adapter(tmp_path / "p2")
    out = tmp_path / "audit"
    cli = Path(__file__).resolve().parents[1] / "rank_loras.py"
    run = subprocess.run([sys.executable, str(cli), "--checkpoint-roots", str(tmp_path / "p1"),
        str(tmp_path / "p2"), "--model-dir", str(tmp_path / "base"), "--output-dir", str(out)],
        capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "run_full_pipeline.sh" in run.stdout and "--selection-policy available" in run.stdout
    assert all(p["recommended"] for p in json.loads((out / "report.json").read_text())["phases"])


def test_real_models_only_preflight_writes_pinned_manifest_without_gpu(tmp_path):
    fixture_adapter(tmp_path / "p1", 1)
    event_adapter(tmp_path / "p2")
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base fixture")
    bev = tmp_path / "bev.pt"
    bev.write_bytes(b"bev fixture")
    out = tmp_path / "selection.json"
    cli = Path(__file__).resolve().parents[1] / "train.py"
    run = subprocess.run([sys.executable, str(cli), "--preflight", "--models-only",
        "--checkpoint-root", str(tmp_path), "--model-dir", str(base), "--lead-bev-ckpt", str(bev),
        "--selection-output", str(out)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    manifest = json.loads(out.read_text())
    assert manifest["selection_policy"] == "available"
    assert manifest["phase2"]["runtime_prompt_name"] == v3.PROMPT_NAME
    assert manifest["phase2"]["file_sha256"] and manifest["contract_identity"]


def test_full_pipeline_passes_selection_manifest_and_continues_to_test(tmp_path):
    """运行真实 shell 流水线，用轻量子入口替身确认顺序和参数，不启动训练。"""
    import os
    import shutil
    pipeline = Path(__file__).resolve().parents[1] / "run_full_pipeline.sh"
    shutil.copy(pipeline, tmp_path / pipeline.name)
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{}")
    recorder = tmp_path / "record.py"
    recorder.write_text('''import json, os, pathlib, sys
stage = sys.argv[1]
with open(os.environ["TRACE_FILE"], "a") as f:
    f.write(json.dumps([stage, os.environ.get("ACTION_MODE"), sys.argv[2:]]) + "\\n")
if stage == "train":
    if os.environ.get("ACTION_MODE") == "preflight":
        target = pathlib.Path(sys.argv[sys.argv.index("--selection-output") + 1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture selection")
    else:
        target = pathlib.Path(sys.argv[sys.argv.index("--selection-manifest") + 1])
        assert target.read_text() == "fixture selection"
        run = pathlib.Path(os.environ["OUTPUT_DIR"]) / ("run_" + os.environ["RUN_TAG"])
        run.mkdir()
        (run / "best.pt").write_text("fixture checkpoint")
''')
    for name in ("train", "eval", "probe"):
        (tmp_path / f"{name}.sh").write_text(
            f'#!/usr/bin/env bash\nset -euo pipefail\npython "$(dirname "$0")/record.py" {name} "$@"\n')
    (tmp_path / "audit_bundle.py").write_text(
        'import runpy, sys\nfrom pathlib import Path\nsys.argv.insert(1,"audit")\n'
        'runpy.run_path(str(Path(__file__).with_name("record.py")), run_name="__main__")\n')
    env = os.environ.copy()
    for key in ("RESUME", "NO_RUN_SUBDIR", "ACTION_MODE", "BENCH2DRIVE"):
        env.pop(key, None)
    env.update(OUTPUT_DIR=str(tmp_path / "output"), RUN_TAG="test", DATA_DIR=str(data),
               TRACE_FILE=str(tmp_path / "trace.jsonl"))
    run = subprocess.run(["bash", str(tmp_path / pipeline.name), "--checkpoint-roots", "server a", "server b"],
                         env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    calls = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [c[0] for c in calls] == ["train", "train", "eval", "probe", "audit"]
    assert calls[0][1] == "preflight" and calls[1][1] is None
    assert "server a" in calls[0][2] and "server b" in calls[1][2]
    assert "--selection-output" in calls[0][2] and "--selection-manifest" in calls[1][2]
    assert "--split" in calls[2][2] and "test" in calls[2][2]
