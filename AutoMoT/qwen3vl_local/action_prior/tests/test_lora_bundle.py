"""权重包真实复制/压缩/删除原 run/跨目录恢复集成检查；不加载 GPU 模型。"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest
from test_contracts import fixture_adapter
from test_available_adapters import event_adapter
from qwen3vl_local.action_prior.available_adapters import select_available
from qwen3vl_local.action_prior.config import build_contract, parser
from qwen3vl_local.action_prior.contracts import require_contract
from qwen3vl_local.action_prior.lora_bundle import (
    create_bundle, archive_bundle, verify_bundle, bundle_paths, preserve_for_training, restore_paths,
)


def pair(tmp_path):
    """伪权重只用于验证文件身份和选择合同，布局与两服务器新包一致。"""
    source = tmp_path / "source"
    one = fixture_adapter(source / "p1", 1)
    two = event_adapter(source / "p2")
    base = source / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base fixture")
    bev = tmp_path / "bev.pt"
    bev.write_bytes(b"BEV fixture")
    # 源 run 有大量其它 checkpoint，不应被打包。
    junk = one.parent / "checkpoint-99999"
    junk.mkdir()
    (junk / "trainer_state.pt").write_bytes(b"optimizer should not travel")
    args = parser().parse_args(["--checkpoint-root", str(source), "--model-dir", str(base),
                               "--lead-bev-ckpt", str(bev)])
    contract = build_contract(args)
    return args, contract, one, two


def test_tar_roundtrip_keeps_only_selected_weights_and_loads_without_source(tmp_path):
    args, contract, one, two = pair(tmp_path)
    selected = {key: contract[key] for key in ("phase1", "phase2")}
    root = create_bundle(selected, tmp_path / "export" / "action_prior_loras_test")
    archive = archive_bundle(root, tmp_path / "pair.tar.gz")
    assert archive["bytes"] > 0 and (tmp_path / "pair.tar.gz.sha256").is_file()
    destination = tmp_path / "another_server" / "checkpoints"
    destination.mkdir(parents=True)
    with tarfile.open(archive["path"]) as handle:
        names = handle.getnames()
        assert sum(name.endswith("adapter_model.safetensors") for name in names) == 2
        assert not any("checkpoint-99999" in name or "trainer_state.pt" in name for name in names)
        assert all(member.isfile() for member in handle.getmembers())
        handle.extractall(destination, filter="data")
    shutil.rmtree(one.parent)
    shutil.rmtree(two.parent)
    shutil.rmtree(root)
    unpacked = destination / root.name
    manifest = verify_bundle(unpacked, selected)
    assert manifest["phases"]["phase2"]["git"] == {"commit": None}
    assert (unpacked / "phase1/source_provenance.json").is_file()
    assert (unpacked / "phase1/train_eval_metrics.jsonl").is_file()
    assert (unpacked / "phase2/fallback_generation.json").is_file()
    args.lora_bundle = str(unpacked)
    imported = build_contract(args)
    require_contract(contract, imported)
    assert imported["phase2"]["path"].startswith(str(unpacked))
    # 普通自动搜索同样能发现解压后的权重，不需要源服务器目录。
    found = select_available([destination], 2, args.model_dir)
    assert found["fingerprint"] == selected["phase2"]["fingerprint"]


def test_run_contains_independent_loras_and_survives_action_directory_move(tmp_path):
    args, contract, one, two = pair(tmp_path)
    run = tmp_path / "action" / "run_test"
    run.mkdir(parents=True)
    local = preserve_for_training(contract, run)
    assert local["identity"] == contract["identity"]
    assert all(Path(local[k]["path"]).is_relative_to(run) for k in ("phase1", "phase2"))
    (run / "best.pt").write_bytes(b"action fixture")
    for source in (one, two):
        copied = Path(local["phase1" if source == one else "phase2"]["path"]) / "adapter_model.safetensors"
        assert copied.stat().st_ino != (source / "adapter_model.safetensors").stat().st_ino
        assert not copied.is_symlink()
        shutil.rmtree(source.parent)
    moved = tmp_path / "moved_action_run"
    shutil.move(run, moved)
    paths = restore_paths(local, moved / "best.pt")
    args.phase1_adapter, args.phase2_adapter = paths["phase1"], paths["phase2"]
    restored = build_contract(args)
    require_contract(local, restored)
    # 再次恢复训练复用当前 run/lora，不依赖已删除来源。
    preserve_for_training(restored, moved)
    shutil.rmtree(moved / "lora")
    with pytest.raises(FileNotFoundError):
        restore_paths(local, moved / "best.pt")


def test_changed_or_incomplete_bundle_is_rejected(tmp_path):
    args, contract, one, two = pair(tmp_path)
    selected = {key: contract[key] for key in ("phase1", "phase2")}
    root = create_bundle(selected, tmp_path / "bundle")
    weight = Path(bundle_paths(root)["phase2"]) / "adapter_model.safetensors"
    weight.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="missing/changed"):
        verify_bundle(root)
    with pytest.raises(ValueError, match="missing/changed"):
        create_bundle(selected, root)
    assert not (tmp_path / "bad.tar.gz").exists()
    with pytest.raises(ValueError, match="missing/changed"):
        archive_bundle(root, tmp_path / "bad.tar.gz")
    from qwen3vl_local.action_prior.available_adapters import scan_available
    assert scan_available([root], 2, args.model_dir)["recommended"] is None


def test_ranking_exports_pair_and_import_pins_it_even_if_better_run_appears(tmp_path):
    args, contract, one, two = pair(tmp_path)
    out = tmp_path / "audit"
    cli = Path(__file__).resolve().parents[1] / "rank_loras.py"
    run = subprocess.run([sys.executable, str(cli), "--checkpoint-root", args.checkpoint_root,
                          "--model-dir", args.model_dir, "--output-dir", str(out), "--summary-only"],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads((out / "report.json").read_text())
    exported = report["weight_bundle"]
    assert Path(exported["path"]).is_file()
    assert "权重迁移压缩包:" in run.stdout and "--lora-bundle" in run.stdout
    assert "[打包 phase1]" in run.stdout and "[打包 phase2]" in run.stdout
    event_adapter(Path(args.checkpoint_root) / "better_later", score=.99)
    args.lora_bundle = str(out / exported["unpacked_directory"])
    imported = build_contract(args)
    assert imported["phase2"]["generation_exact"] == contract["phase2"]["generation_exact"]
    assert imported["phase2"]["fingerprint"] == exported["phases"]["phase2"]["fingerprint"]


def test_partial_export_is_useful_for_sharing_but_not_a_complete_training_pair(tmp_path):
    args, contract, _, _ = pair(tmp_path)
    root = create_bundle({"phase1": contract["phase1"]}, tmp_path / "phase1_only")
    archive_bundle(root, tmp_path / "phase1.tar.gz")
    args.lora_bundle = str(root)
    with pytest.raises(ValueError, match="both Phase1 and Phase2"):
        build_contract(args)
