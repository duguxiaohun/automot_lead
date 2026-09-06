"""闭环计划、报告分母和审计包的 CPU 回归，不启动 CARLA。"""

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from qwen3vl_local.action_prior.audit_bundle import pack
from qwen3vl_local.action_prior import benchmark_report as br


def test_archive_hard_cap_and_explicit_omissions(tmp_path):
    """不可压缩案例超过预算时保留核心、列出遗漏，且原文件不变。"""
    (tmp_path / "metrics.json").write_text('{"loss":1}')
    cases = tmp_path / "probe/cases"
    cases.mkdir(parents=True)
    large = cases / "large.png"
    large.write_bytes(os.urandom(16000))
    out = pack(tmp_path, max_bytes=10000)
    assert out.stat().st_size <= 10000
    with zipfile.ZipFile(out) as z:
        assert json.loads(z.read("metrics.json"))["loss"] == 1
        manifest = json.loads(z.read("AUDIT_MANIFEST.json"))
        assert manifest["omitted_optional_count"] == 1
    assert large.stat().st_size == 16000


def test_oversized_core_never_publishes_partial_archive(tmp_path):
    (tmp_path / "metrics.json").write_bytes(os.urandom(18000))
    with pytest.raises(ValueError, match="core audit"):
        pack(tmp_path, max_bytes=10000)
    assert not (tmp_path / "audit.zip").exists()


def test_formal_plan_only_never_requires_weights_or_carla():
    result = subprocess.run(
        [
            sys.executable,
            str(br.ROOT / "qwen3vl_local/action_prior/bench2drive.py"),
            "--checkpoint",
            "/does/not/exist.pt",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert data["route_count"] == 220 and data["scenario_count"] == 44


def record(rid, status="Completed", **infractions):
    return dict(
        route_id="RouteScenario_" + rid,
        status=status,
        save_name="run" + rid,
        infractions=dict(min_speed_infractions=["speed 70%"], **infractions),
        scores=dict(score_composed=80, score_route=100, score_penalty=0.8),
    )


def test_success_excludes_only_min_speed():
    assert br.success(record("1"))
    assert not br.success(record("1", red_light=["violation"]))
    assert not br.success(record("1", status="Failed - Agent crashed"))


def test_report_partial_not_paper_complete_and_traffic_sign_v004(tmp_path, monkeypatch):
    """Traffic Signs 每路线只计一次，失败但已合法过路口按 v004 补成功。"""
    xml = tmp_path / "routes.xml"
    xml.write_text(
        "<routes>"
        + "".join(
            f'<route id="{i}" town="Town01"><scenarios><scenario type="SignalizedJunctionLeftTurn"/></scenarios><waypoints><position x="0" y="0"/></waypoints></route>'
            for i in (1, 2, 3)
        )
        + "</routes>"
    )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(dict(route_ids=["1", "2", "3"]))
    )
    per = tmp_path / "eval_per_route"
    per.mkdir()
    for rid in ("1", "2"):
        rec = record(
            rid,
            status="Completed" if rid == "1" else "Failed - Collision",
            collisions_vehicle=[] if rid == "1" else ["hit"],
        )
        (per / f"eval_{rid}.json").write_text(
            json.dumps({"_checkpoint": {"records": [rec]}})
        )
        motion = tmp_path / "rollouts/sig" / ("run" + rid)
        motion.mkdir(parents=True)
        point = dict(
            acceleration=[0, 0, 0],
            angular_velocity=[0, 0, 0],
            forward_vector=[1, 0, 0],
            right_vector=[0, 1, 0],
            location=[0, 0, 0],
            rotation=[0, 0, 0],
        )
        (motion / "metric_info.json").write_text(
            json.dumps({str(k): point for k in range(10)})
        )
    monkeypatch.setattr(br, "offline_junction_threshold", lambda *args: 0.5)
    report = br.report(tmp_path, xml)
    assert report["full_220_records"] is False
    assert report["missing_routes"] == ["3"]
    assert report["overall"]["driving_score"] == pytest.approx(160 / 3)
    assert report["overall"]["success_rate"] == pytest.approx(100 / 3)
    assert report["abilities"]["Traffic_Signs"]["planned"] == 3
    assert report["abilities"]["Traffic_Signs"]["observed"] == 2
    assert report["abilities"]["Traffic_Signs"]["success_rate"] is None
    assert report["routes"][1]["traffic_sign_success"] is True
    assert report["overall"]["comfort"] == 100
    assert (tmp_path / "paper_table.md").is_file()


def test_route_record_rejects_wrong_duplicate_nan(tmp_path):
    path = tmp_path / "record.json"
    for records in ([record("2")], [record("1"), record("1")]):
        path.write_text(json.dumps({"_checkpoint": {"records": records}}))
        with pytest.raises(ValueError):
            br.route_record(path, "1")
    rec = record("1")
    rec["scores"]["score_composed"] = float("nan")
    path.write_text(json.dumps({"_checkpoint": {"records": [rec]}}))
    with pytest.raises(ValueError):
        br.route_record(path, "1")


def test_online_forward_uses_training_runtime_without_gt():
    """真实在线适配方法调用 fake runtime，确保 clip 原样透传且不读取 GT。"""
    import torch
    from types import SimpleNamespace
    from qwen3vl_local.action_prior.carla_runtime import ActionPriorRunner

    runner = object.__new__(ActionPriorRunner)
    clip = {"rgb": "live", "speed": "current"}

    def forward(sample, decoder, config, dtype, clip=None):
        assert set(sample) == {"scenario", "run_id", "anchor"}
        assert clip == {"rgb": "live", "speed": "current"}
        assert not torch.is_grad_enabled()
        return {
            "pred_route": torch.zeros(1, 10, 2),
            "pred_future_waypoints": torch.zeros(1, 8, 2),
        }

    runner.runtime = SimpleNamespace(forward_sample=forward)
    runner.latencies = []
    runner.index = 0
    runner.route_key = "123"
    runner.decoder = None
    runner.leadmot_config = None
    runner.dtype = torch.bfloat16
    out = runner.run_clip(clip)
    assert runner.index == 1 and out[0]["leadmot_future_waypoints"].shape == (1, 8, 2)


def test_actual_evaluator_rep0_route_id(tmp_path):
    rec = record("1773")
    rec["route_id"] += "_rep0"
    path = tmp_path / "route.json"
    path.write_text(json.dumps({"_checkpoint": {"records": [rec]}}))
    assert br.route_record(path, "1773") == rec


def test_nonterminal_result_is_not_counted(tmp_path):
    path = tmp_path / "partial.json"
    path.write_text(
        json.dumps({"_checkpoint": {"records": [record("1", status="Started")]}})
    )
    with pytest.raises(ValueError, match="nonterminal"):
        br.route_record(path, "1")


def test_report_refuses_different_route_source(tmp_path):
    xml = tmp_path / "routes.xml"
    xml.write_text("<routes/>")
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(dict(route_ids=[], routes_sha256="wrong"))
    )
    with pytest.raises(ValueError, match="XML differs"):
        br.report(tmp_path, xml)
