"""配对工具回归：不用权重检查选优、同帧分组、输入旁路和结果发布。"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_prior import comparison_cases as cc
from qwen3vl_local.action_prior.comparison_render import paired_cases, publish, METRICS
from qwen3vl_local.action_prior.comparison_runtime import CaptureRuntime


def row(frame=0, route="a", action="STOP"):
    """构造含并发事件的同源标签。"""
    return dict(scenario="scene", run_id=route, anchor=frame, route_group=route, split="test",
        action_token=dict(name=action, reason="complete_phase3_evidence", version="v1"), action_token_id=2,
        event_balance_status="special_eligible", event_balance_all_special_buckets=["UE1", "UE2"])


def fake_run(tmp_path):
    """不同完整 val 步的假权重载入器。"""
    config = dict(route_loss_weight=.5, waypoint_loss_weight=1., best_selection_metric="natural_ade")
    cc.write_json(tmp_path / "config.json", config)
    for step, score in [(10, 2.), (20, 1.)]:
        cc.write_json(tmp_path / "validation" / f"epoch_{step}.json", dict(optimizer_step=step,
            validation_full_epoch=True, route_ade_m=0., waypoint_ade_m=score))
    states = {"best.pt": dict(step=20, best_sampled_trajectory_score=1.),
              "latest.pt": dict(step=30, best_sampled_trajectory_score=1.),
              "step_10.pt": dict(step=10), "step_20.pt": dict(step=20)}
    return states


def test_best_uses_current_step_not_running_best(tmp_path):
    states = fake_run(tmp_path)
    (tmp_path / "latest.pt").touch()
    with pytest.raises(ValueError, match="无 best.pt"):
        cc.select_checkpoint(tmp_path, lambda p: states[p.name])
    (tmp_path / "step_10.pt").touch()
    (tmp_path / "step_20.pt").touch()
    path, _, selected = cc.select_checkpoint(tmp_path, lambda p: states[p.name])
    assert path.name == "step_20.pt" and selected["score"] == 1.


def test_best_stale_rejected(tmp_path):
    states = fake_run(tmp_path)
    (tmp_path / "best.pt").touch()
    cc.select_checkpoint(tmp_path, lambda p: states[p.name])
    states["best.pt"]["step"] = 10
    with pytest.raises(ValueError, match="最优记录不符"):
        cc.select_checkpoint(tmp_path, lambda p: states[p.name])


def test_periodic_loss_and_test_do_not_select(tmp_path):
    states = fake_run(tmp_path)
    (tmp_path / "best.pt").touch()
    cc.write_json(tmp_path / "validation/step_999.json", dict(optimizer_step=999, loss=0., route_ade_m=0., waypoint_ade_m=0.))
    cc.write_json(tmp_path / "eval_test/metrics.json", dict(loss=0.))
    assert cc.select_checkpoint(tmp_path, lambda p: states[p.name])[2]["step"] == 20


def test_validation_weighted_metric_and_nan():
    config = dict(route_loss_weight=.5, waypoint_loss_weight=1., best_selection_metric="event_balanced_ade")
    metrics = dict(event_balance_bucket_coverage_complete=1, event_balanced_route_ade_m=2., event_balanced_waypoint_ade_m=3.)
    assert cc.validation_score(metrics, config) == 4.
    metrics["event_balanced_route_ade_m"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        cc.validation_score(metrics, config)


def test_selection_route_diversity_replay_and_overlap():
    rows = [row(i, route) for route in ("a", "b", "c") for i in range(20)]
    selected, groups, cov = cc.select_cases(rows, per_category=3)
    assert cov["action"]["STOP"]["selected_physical_routes"] == 3
    assert cov["action"]["KEEP"]["shortfall"] == 3
    assert groups == cc.select_cases(list(reversed(rows)), per_category=3)[1]
    assert len(groups["event"]["UE1"]) == 3 and len(groups["event"]["UE2"]) == 3
    assert len({cc.identity(r) for r in selected}) == len(selected)


def test_gap_does_not_duplicate_rare_frames():
    rows = [row(i) for i in range(3)]
    _, _, cov = cc.select_cases(rows, per_category=8, min_frame_gap=8)
    assert cov["action"]["STOP"]["selected"] == 1
    assert cov["action"]["STOP"]["shortfall"] == 7


def test_filtered_and_unknown_are_not_regular():
    r = row()
    r["event_balance_status"] = "special_filtered"
    assert cc.event_groups(r) == ["UE1", "UE2"]
    r["event_balance_all_special_buckets"] = []
    r["event_balance_status"] = "unconfirmed"
    assert cc.event_groups(r) == ["UNCONFIRMED"]
    r["event_balance_status"] = "confirmed_regular"
    assert cc.event_groups(r) == ["REGULAR_BACKGROUND"]


def test_split_mismatch_and_duplicate_fail():
    with pytest.raises(ValueError, match="有效帧集合"):
        cc.require_same_frames([row(1)], [row(2)], "test")
    with pytest.raises(ValueError, match="有效帧集合"):
        cc.require_same_frames([row(1)], [row(1), row(1)], "test")
    with pytest.raises(ValueError, match="重复帧"):
        cc.select_cases([row(1), row(1)])


def test_path_cwd_and_migration(tmp_path, monkeypatch):
    root = tmp_path / "AutoMoT"
    target = root / "checkpoints/run a"
    target.mkdir(parents=True)
    monkeypatch.setattr(cc, "AUTOMOT_ROOT", root)
    monkeypatch.chdir(tmp_path)
    assert cc.resolve_path("AutoMoT/checkpoints/run a") == target
    assert cc.resolve_path("checkpoints/run a") == target
    assert cc.resolve_path("/old/AutoMoT/checkpoints/run a") == target


def test_capture_exact_pil_and_condition_passthrough(tmp_path):
    from PIL import Image
    class Status:
        def detach(self): return self
        def float(self): return self
        def cpu(self): return self
        def tolist(self): return [[3., 1., 2.]]
    images = [Image.new("RGB", (30, 10), color) for color in ("red", "green")]
    result = (images, [], Status())
    runner = SimpleNamespace(_prepare_inference_inputs=lambda *a, **kw: result)
    sample = row()
    original = runner._prepare_inference_inputs
    runtime = SimpleNamespace(runner=runner, device="cpu", forward_sample=lambda s, *a, **kw: runner._prepare_inference_inputs(kw["clip"]))
    capture = CaptureRuntime(runtime, tmp_path, "bev_only")
    assert capture.forward_sample(sample, clip={}) is result
    assert capture.original_prepare is original
    files = list(tmp_path.rglob("*.png"))
    assert len(files) == 1
    assert Image.open(files[0]).getpixel((0, 0)) == (0, 128, 0)
    assert cc.read_json(files[0].with_name("input.json"))["action_token"]["name"] == "STOP"


def make_outputs(tmp_path):
    """两模型合成结果走完整发布流程，不代表真实推理。"""
    from PIL import Image
    label = row(12)
    cid = cc.case_id(label)
    models = []
    metrics = {key: .1 for key in METRICS}
    for i, enabled in enumerate((False, True), 1):
        mid = f"model_{i:02d}"
        model = dict(id=mid, label=f"synthetic {i}", checkpoint=f"/fake/run{i}/best.pt", variant="bev_only",
                     high_level_action_token=enabled, selection=dict(step=20, score=.1))
        models.append(model)
        sample = dict(label, route_dir=str(tmp_path / "no_data"))
        if not enabled:
            sample.pop("action_token")
            sample.pop("action_token_id")
        trajectory = [[[0., 0.], [3., 1.]]]
        data = dict(sample=sample, metrics=metrics, pred_route=trajectory, pred_waypoints=trajectory,
                    gt_route=trajectory, gt_waypoints=trajectory)
        root = tmp_path / "_models" / mid / "test"
        cc.write_json(root / "cases/rank0_case000000.json", data)
        folder = root / "inputs" / cid
        cc.write_json(folder / "input.json", dict(action_token=sample.get("action_token"), status=[[0., 1., 2.]]))
        Image.new("RGB", (300, 100), "#446688").save(folder / "input_rgb_00.png")
    coverage = dict(available_frames=1, available_physical_routes=1, requested=8, selected=1, selected_physical_routes=1, shortfall=7)
    from qwen3vl_local.action_prior.comparison_scene import camera_configuration
    cameras = camera_configuration()
    cameras["source"] = "SYNTHETIC test cameras"
    for camera in cameras["views"]:
        camera["width"], camera["height"] = 100, 100
    manifest = dict(models=models, camera_config=cameras, interpretation="SYNTHETIC renderer test only", splits=dict(test=dict(cases={cid: label},
        groups=dict(event={"UE1": [cid]}, action={"STOP": [cid]}), coverage=dict(event={"UE1": coverage}, action={"STOP": coverage}))))
    return manifest, cid


def test_publish_json_images_gallery_and_no_token_injection(tmp_path, monkeypatch, capsys):
    from qwen3vl_local.action_prior import comparison_scene as scene
    original = scene.draw_camera
    overlays = []
    def capture(ax, rgb, camera, curves, ground):
        overlays.append([item["label"] for item in curves])
        return original(ax, rgb, camera, curves, ground)
    monkeypatch.setattr(scene, "draw_camera", capture)
    manifest, cid = make_outputs(tmp_path)
    publish(tmp_path, manifest)
    # 三个相机的主图均含GT和两个模型，后续才绘制各自单模型图。
    assert overlays[:3] == [["GT", "synthetic 1", "synthetic 2"]] * 3
    assert all(len(curves) == 2 for curves in overlays[3:])
    output = capsys.readouterr().out
    assert "[render] comparison ready (1/1):" in output
    assert "[render] complete 1/1 cases" in output
    stages = cc.read_json(tmp_path / "render.json")["completed_stages"]
    assert all(stage["status"] == "done" for stage in stages)
    assert any("case 1/1" in stage["stage"] for stage in stages)
    assert stages[-1]["stage"] == "write final report / gallery"
    folder = tmp_path / "action/test/STOP" / cid
    case = cc.read_json(folder / "case.json")
    assert case["models"][0]["action_token"] == "DISABLED"
    assert case["models"][1]["action_token"] == "STOP"
    assert (folder / "comparison.png").stat().st_size > 1000
    assert (folder / "comparison.pdf").stat().st_size > 1000
    assert (folder / "input_history.png").is_file()
    assert (folder / "model_01.png").is_file()
    assert (folder / "model_01.pdf").is_file()
    assert case["visualization"]["calibration_source"] == "SYNTHETIC test cameras"
    assert (tmp_path / "index.html").is_file()
    assert (tmp_path / "REPORT.md").is_file()
    overview = cc.read_json(tmp_path / "manifest.json")["visualization_summary"]
    assert overview == dict(unique_cases=1, calibration_counts=dict(nominal_fallback=1))
    assert "nominal_fallback=1" in (tmp_path / "REPORT.md").read_text()
    assert cc.read_json(tmp_path / "action/test/STOP/summary.json")["coverage"]["shortfall"] == 7


@pytest.mark.parametrize("problem", ["missing", "gt", "duplicate", "wrong_frame"])
def test_pairing_rejects_bad_results(tmp_path, problem):
    manifest, cid = make_outputs(tmp_path)
    path = tmp_path / "_models/model_02/test/cases/rank0_case000000.json"
    data = cc.read_json(path)
    if problem == "missing":
        path.unlink()
    elif problem == "gt":
        data["gt_route"][0][1][0] = 999.
        cc.write_json(path, data)
    elif problem == "duplicate":
        cc.write_json(path.with_name("rank0_case000001.json"), data)
    else:
        data["sample"]["anchor"] += 1
        cc.write_json(path, data)
    with pytest.raises(ValueError):
        paired_cases(tmp_path, manifest, "test")


@pytest.mark.parametrize('keep', [False, True])
def test_error_filter_publish_empty_or_pairwise_case(tmp_path, monkeypatch, keep):
    from qwen3vl_local.action_prior import comparison_render as renderer
    manifest, cid = make_outputs(tmp_path)
    manifest['error_filter'] = dict(enabled=True, ade_threshold_m=1., fde_threshold_m=3.)
    if keep:
        # Both models are <1m from GT, but their endpoints are 1.5m apart.
        for index, dx in ((1, .75), (2, -.75)):
            path = tmp_path / f'_models/model_{index:02d}/test/cases/rank0_case000000.json'
            data = cc.read_json(path)
            data['pred_waypoints'] = copy.deepcopy(data['gt_waypoints'])
            for point in data['pred_waypoints'][0]:
                point[0] += dx
            cc.write_json(path, data)
    else:
        def no_scene_reads(*args):
            raise AssertionError('excluded cases must not load scenes or render')
        monkeypatch.setattr(renderer, 'lidar_background', no_scene_reads)
    publish(tmp_path, manifest)
    audit = cc.read_json(tmp_path/'error_filter.json')['splits']['test']
    assert audit['sampled'] == 1 and audit['retained'] == int(keep)
    for style, category in (('event', 'UE1'), ('action', 'STOP')):
        folder = tmp_path / style / 'test' / category
        assert (folder / cid / 'comparison.png').exists() == keep
        group = cc.read_json(folder/'summary.json')
        assert group['means']['model_01']['loss'] == .1  # unfiltered denominator preserved
        assert bool(group['displayed_means']) == keep
        assert group['visualization']['retained'] == int(keep)
    if keep:
        detail = cc.read_json(tmp_path/'_cases/test'/cid/'case.json')['error_filter']
        assert detail['triggers'][0]['pair'] == ['model_01', 'model_02']
        assert detail['triggers'][0]['distance_m'] == 1.5
    assert (tmp_path/'REPORT.md').is_file()
    assert ('comparison.png' in (tmp_path/'index.html').read_text()) == keep


@pytest.mark.parametrize("variant", ["bev_only", "qwen_simple", "action_prior"])
@pytest.mark.parametrize("empty_train", [False, True])
def test_worker_routes_models_preserves_seed_and_cleans_capture(tmp_path, monkeypatch, variant, empty_train):
    """执行真实worker编排，模型/张量替身只验证调用合同，不冒充GPU测试。"""
    import types
    from qwen3vl_local.action_prior import comparison_runtime as cr
    from qwen3vl_local.action_prior import launch
    from qwen3vl_local.action_prior.contracts import file_hash
    from qwen3vl_local.action_prior.progress import current
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    state = dict(decoder_config={}, flow_config={}, ema_state_dict=dict(shadow={"weight": "EMA"}))
    torch = types.ModuleType("torch")
    torch.load = lambda *a, **kw: copy.deepcopy(state)
    torch.device = lambda *a: "fake_cuda"
    torch.float32, torch.float16, torch.bfloat16 = "f32", "f16", "bf16"
    monkeypatch.setitem(sys.modules, "torch", torch)
    loads, calls = [], []
    class Model:
        def __init__(self, *args): pass
        def to(self, **kwargs): return self
        def load_state_dict(self, weights, strict): loads.append((weights, strict))
    fm = types.ModuleType("qwen3vl_local.action_prior.flow_matching")
    fm.ConditionalFlowMatchingDecoder = Model
    fm.FlowMatchingConfig = lambda **kwargs: kwargs
    leadmot = types.ModuleType("qwen3vl_local.leadmot")
    leadmot.LeadMoTPlanningDecoderConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, fm.__name__, fm)
    monkeypatch.setitem(sys.modules, leadmot.__name__, leadmot)
    args = SimpleNamespace(seed=99, decoder_dtype="bfloat16", output_dir="original")
    monkeypatch.setattr(cr, "restore_args", lambda *a: (args, variant))
    monkeypatch.setattr(cr, "check_contract", lambda *a: {"upstream_sources": {}})
    monkeypatch.setattr(launch, "ensure_gpu", lambda: 1)
    original = lambda *a, **kw: None
    runtime = SimpleNamespace(runner=SimpleNamespace(_prepare_inference_inputs=original))
    def make_runtime(saved_args, *rest):
        assert saved_args.seed == 99
        assert saved_args.output_dir == str(tmp_path / "output")
        return runtime
    def evaluate(capture, model, config, rows, eval_args, dtype, rank, world, limit, dump):
        assert current() is not None
        assert eval_args.seed == 2026 and args.seed == 99
        assert capture.original_prepare is original
        assert dtype == "bf16" and (rank, world, limit) == (0, 1, 0)
        calls.append(rows)
        return {"samples": len(rows), "loss": .1}
    if variant == "action_prior":
        rmod = types.ModuleType("qwen3vl_local.action_prior.runtime")
        rmod.make_runtime = make_runtime
        tmod = types.ModuleType("qwen3vl_local.action_prior.train")
        tmod.evaluate = evaluate
        prov = types.ModuleType("qwen3vl_local.action_prior.provenance")
        prov.annotate_upstream = lambda *a: None
        for module in (rmod, tmod, prov):
            monkeypatch.setitem(sys.modules, module.__name__, module)
    else:
        mod = types.ModuleType("qwen3vl_local.action_expert_ablation.common")
        mod.make_runtime, mod.evaluate = make_runtime, evaluate
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    paths, hashes = {}, {}
    for split in ("train", "test"):
        path = tmp_path / f"{split}.json"
        cc.write_json(path, [] if empty_train and split == "train" else [dict(row(), split=split)])
        paths[split], hashes[split] = str(path), file_hash(path)
    job = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
        overrides={}, seed=2026, workers=0, output=str(tmp_path / "output"), rows=paths, row_hashes=hashes)
    cache = {}
    cr.evaluate_worker(job, cache)
    cr.evaluate_worker(job, cache)
    assert len(calls) == (2 if empty_train else 4)
    assert loads == [({"weight": "EMA"}, True)]
    assert runtime.runner._prepare_inference_inputs is original
    assert cc.read_json(tmp_path / "output/test/metrics.json")["samples"] == 1
    if empty_train:
        assert cc.read_json(tmp_path / "output/train/metrics.json")["samples"] == 0


def test_full_cycle_and_nested_final_participate_selection(tmp_path):
    states = fake_run(tmp_path)
    cc.write_json(tmp_path / "validation/cycle_003_step00000015.json", dict(optimizer_step=15,
        validation_cycle=3, validation_full_epoch=False, route_ade_m=0., waypoint_ade_m=.5))
    cc.write_json(tmp_path / "validation/final.json", dict(optimizer_step=30, split="val", weight_view="ema",
        selection_score=.75, metrics=dict(route_ade_m=0., waypoint_ade_m=.75)))
    records = cc.validation_records(tmp_path, cc.read_json(tmp_path / "config.json"))
    assert records[15]["score"] == .5 and records[30]["score"] == .75
    (tmp_path / "best.pt").touch()
    states["best.pt"] = dict(step=15, best_sampled_trajectory_score=.5)
    assert cc.select_checkpoint(tmp_path, lambda p: states[p.name])[2]["step"] == 15


def test_publish_rejects_current_image_change(tmp_path):
    from PIL import Image
    manifest, cid = make_outputs(tmp_path)
    Image.new("RGB", (300, 100), "red").save(tmp_path / "_models/model_02/test/inputs" / cid / "input_rgb_00.png")
    with pytest.raises(ValueError, match="当前RGB不同"):
        publish(tmp_path, manifest)


def test_category_overrides_split_zero_and_replay():
    overrides = dict(event=cc.parse_category_counts(["UE1=4,UE2=0", "test/UE1=2", "UE1=5"], cc.EVENT_NAMES),
                     action=cc.parse_category_counts(["STOP=3", "train/STOP=0"], cc.ACTION_NAMES))
    rows = [row(i, route) for route in ("a", "b", "c") for i in range(30)]
    train, tg, tc = cc.select_cases(rows, per_category=0, category_counts=overrides, split="train")
    test, eg, ec = cc.select_cases(rows, per_category=0, category_counts=overrides, split="test")
    assert len(tg["event"]["UE1"]) == 5 and not tg["action"]["STOP"]
    assert len(eg["event"]["UE1"]) == 2 and len(eg["action"]["STOP"]) == 3
    assert not eg["event"]["UE2"] and ec["event"]["UE2"]["shortfall"] == 0
    assert len(train) == 5 and len(test) <= 5
    assert tc["action"]["STOP"]["requested"] == 0
    assert cc.select_cases(list(reversed(rows)), per_category=0, category_counts=overrides, split="test")[1] == eg


@pytest.mark.parametrize("spec", ["UE8=2", "UE1=-1", "UE1=1.5", "val/UE1=2", "train/test/UE1=2", "UE1", "UE1="])
def test_category_override_rejects_typo(spec):
    with pytest.raises(ValueError, match="非法采样数量"):
        cc.parse_category_counts([spec], cc.EVENT_NAMES)


def test_projection_carla_axes_yaw_pitch_and_horizontal_fov():
    import numpy as np
    from qwen3vl_local.action_prior.comparison_scene import project_trajectory, carla_rotation
    camera = dict(pos=[0., 0., 2.], rot=[0., 0., 0.], width=400, height=200, fov=90.)
    pixels = project_trajectory([[10., 0.], [10., 2.], [10., -2.], [-1., 0.]], camera)
    np.testing.assert_allclose(pixels[:3], [[200, 140], [240, 140], [160, 140]])
    assert np.isnan(pixels[3]).all()
    camera["rot"] = [0., 0., -54.5]
    forward = np.array([np.cos(np.deg2rad(-54.5)), np.sin(np.deg2rad(-54.5))]) * 10
    np.testing.assert_allclose(project_trajectory([forward], camera), [[200., 140.]])
    camera.update(pos=[30., 0., 75.], rot=[0., -90., 0.])
    np.testing.assert_allclose(project_trajectory([[30, 0]], camera), [[200., 100.]])
    p = project_trajectory([[40, 0], [30, 10]], camera)
    assert p[0, 1] < 100 and p[1, 0] > 200
    rotation = carla_rotation([11., -21., 43.])
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-15)


def test_recorded_scene_rotation_calibration_and_third_person(tmp_path):
    import lzma
    import pickle
    import numpy as np
    from PIL import Image
    from qwen3vl_local.action_prior.comparison_scene import camera_configuration, load_scene, hdmap_extent
    config = camera_configuration()
    config["hdmap"] = dict(size=4, pixels_per_meter=2., pixels_ev_to_bottom=2., stored_rotation_k=-1)
    config["third_person"] = dict(name="Overhead", pos=[0, 0, 50], rot=[0, -90, 0], width=16, height=16, fov=90,
                                  image_pattern="3rd_person/{frame:04d}.jpg")
    cc.write_json(tmp_path / "comparison_camera.json", config)
    original = np.array([[0, 0, 0, 0], [0, 1, 3, 0], [0, 4, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)
    (tmp_path / "hdmap").mkdir()
    Image.fromarray(np.rot90(original, -1)).save(tmp_path / "hdmap/0001.png")
    (tmp_path / "3rd_person").mkdir()
    Image.new("RGB", (16, 16)).save(tmp_path / "3rd_person/0001.jpg")
    (tmp_path / "bboxes").mkdir()
    with lzma.open(tmp_path / "bboxes/0001.pkl", "wb") as handle:
        pickle.dump([dict(class_="car", position=[10., -4., 0.], extent=[2., 1., 1.], yaw=0., **{"class": "car"})], handle)
    scene = load_scene(tmp_path, 1)
    assert tuple(scene["hdmap"][1, 2]) == (242, 241, 224)
    assert tuple(scene["hdmap"][2, 1]) == (242, 203, 89)
    assert scene["audit"]["third_person_available"] and scene["audit"]["actor_boxes"] == 1
    assert scene["audit"]["calibration_sha256"] and len(scene["audit"]["sources"]) == 3
    np.testing.assert_allclose(hdmap_extent(config["hdmap"]), [-4/3, 4/3, -4/3, 4/3])
    config["third_person"]["image_pattern"] = "../elsewhere.jpg"
    cc.write_json(tmp_path / "comparison_camera.json", config)
    with pytest.raises(ValueError, match="相对路径"):
        camera_configuration(tmp_path / "comparison_camera.json")


def test_projection_requires_correct_input_dimensions(tmp_path):
    from qwen3vl_local.action_prior.comparison_scene import camera_configuration
    manifest, _ = make_outputs(tmp_path)
    manifest["camera_config"] = camera_configuration()
    with pytest.raises(ValueError, match="RGB尺寸"):
        publish(tmp_path, manifest)


@pytest.mark.parametrize('enabled,flag', [('false', '--no-error-only'), ('true', '--error-only')])
def test_shell_forwards_config_and_paths_with_spaces(tmp_path, enabled, flag):
    import os
    import subprocess
    script = Path(__file__).resolve().parents[1] / "compare_checkpoints.sh"
    # 模拟用户直接编辑sh配置；外部环境不覆盖脚本中的两项设置。
    content = script.read_text().replace('ERROR_ONLY=true', f'ERROR_ONLY={enabled}').replace(
        'ERROR_ADE_THRESHOLD_M=1.0', 'ERROR_ADE_THRESHOLD_M=1.25')
    script = tmp_path / 'compare_checkpoints.sh'
    script.write_text(content)
    fake = tmp_path / "fake_python"
    fake.write_text('#!/usr/bin/env python3\nimport sys,json\nprint(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    args = ["run with spaces A", "run B", "--cases-per-category", "0", "--event-cases", "UE1=3", "--names", "Method A", "Method B"]
    run = subprocess.run(["bash", str(script), *args], check=True, capture_output=True, text=True,
                         env={**os.environ, "PYTHON": str(fake), "ERROR_ONLY": "invalid_external", "ERROR_THRESHOLD_M": "999"})
    argv = json.loads(run.stdout)
    assert argv[1:3] == ["--cases-per-category", "50"] and argv[18:] == args
    assert argv[7:14] == ['--error-ade-threshold-m', '1.25', '--error-fde-threshold-m', '3.0', '--error-cases-per-category', '5', flag]
    assert argv[14:18] == ['--sampling-seed', 'auto', '--seed', '2026']
    assert argv[3] == "--output-root" and Path(argv[4]).resolve() == script.parents[2] / "test"
    assert argv[5:7] == ["--gpus", "4"]


def test_publish_with_calibrated_third_person(tmp_path):
    from PIL import Image
    manifest, cid = make_outputs(tmp_path)
    root = tmp_path / "no_data/3rd_person"
    root.mkdir(parents=True)
    Image.new("RGB", (160, 120), "#446688").save(root / "0012.jpg")
    manifest["camera_config"]["third_person"] = dict(name="Synthetic overhead", pos=[10, 0, 30], rot=[0, -90, 0],
        width=160, height=120, fov=90, image_pattern="3rd_person/{frame:04d}.jpg")
    publish(tmp_path, manifest)
    folder = tmp_path / "action/test/STOP" / cid
    assert cc.read_json(folder / "case.json")["visualization"]["third_person_available"]
    assert (folder / "comparison.pdf").stat().st_size > 1000


def write_meta_camera(tmp_path, calibration=None):
    import lzma
    import pickle
    from qwen3vl_local.action_prior.comparison_scene import camera_configuration
    if calibration is None:
        calibration = {i: camera for i, camera in enumerate(camera_configuration()["views"], 1)}
    target = tmp_path / "metas/0001.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(target, "wb") as handle:
        pickle.dump(dict(sensor_information=dict(num_cameras=3, camera_calibration=calibration)), handle)
    return target


@pytest.mark.parametrize("string_keys", [False, True])
def test_recorded_camera_calibration_used_in_stitched_order(tmp_path, string_keys):
    from qwen3vl_local.action_prior.comparison_scene import scene_configuration, camera_configuration, load_scene
    cameras = camera_configuration()["views"]
    cameras[0]["rot"][2] = -50.
    calibration = {(str(i) if string_keys else i): c for i, c in enumerate(cameras, 1)}
    source = write_meta_camera(tmp_path, calibration)
    config = scene_configuration(tmp_path, 1)
    assert [c["rot"][2] for c in config["views"]] == [-50., 0., 54.5]
    assert [c["sensor_id"] for c in config["views"]] == ["rgb_1", "rgb_2", "rgb_3"]
    assert config["configuration_file"] == str(source)
    assert config["calibration_resolution"] == "recorded_anchor_meta"
    assert load_scene(tmp_path, 1)["audit"]["calibration_sha256"]


def test_camera_configuration_precedence_and_missing_fallback(tmp_path):
    from qwen3vl_local.action_prior.comparison_scene import scene_configuration, camera_configuration
    assert scene_configuration(tmp_path, 1)["calibration_fallback_reason"] == "anchor_meta_missing"
    write_meta_camera(tmp_path)
    global_path = tmp_path / "global.json"
    config = camera_configuration()
    config["views"][0]["rot"][2] = -45.
    cc.write_json(global_path, config)
    global_config = camera_configuration(global_path)
    result = scene_configuration(tmp_path, 1, global_config)
    assert result["calibration_resolution"] == "explicit_global_json" and result["views"][0]["rot"][2] == -45.
    config["views"][0]["rot"][2] = -40.
    cc.write_json(tmp_path / "comparison_camera.json", config)
    result = scene_configuration(tmp_path, 1, global_config)
    assert result["calibration_resolution"] == "explicit_route_json" and result["views"][0]["rot"][2] == -40.


@pytest.mark.parametrize("problem", ["missing_camera", "bad_fov", "cropped"])
def test_bad_recorded_calibration_does_not_silently_fallback(tmp_path, problem):
    from qwen3vl_local.action_prior.comparison_scene import camera_configuration, scene_configuration
    cameras = {i: c for i, c in enumerate(camera_configuration()["views"], 1)}
    if problem == "missing_camera":
        cameras.pop(3)
    elif problem == "bad_fov":
        cameras[1]["fov"] = 0
    else:
        cameras[1]["cropped_height"] = 300
    write_meta_camera(tmp_path, cameras)
    with pytest.raises(ValueError):
        scene_configuration(tmp_path, 1)


def reference_functions(path, names):
    """只读抽取纯几何参考，避免导入CARLA/torch或执行整个第三方模块。"""
    import ast
    import numpy as np
    if not path.is_file():
        pytest.skip(f"本机未挂载只读参考: {path}")
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    for node in selected:
        node.decorator_list = []
    code = 'from __future__ import annotations\n' + '\n'.join(ast.unparse(node) for node in selected)
    namespace = dict(np=np)
    exec(compile(code, str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("rotation,size", [([0, 0, -54.5], [384, 384]), ([0, 0, 54.5], [384, 384]),
                                        ([0, -90, 0], [786, 786]), ([11, -17, 43], [640, 360])])
def test_projection_against_readonly_bench2drive_inverse_matrix(rotation, size):
    import numpy as np
    from qwen3vl_local.action_prior.comparison_scene import project_trajectory
    root = Path(__file__).resolve().parents[4]
    ref = reference_functions(root / "lead/3rd_party/Bench2Drive/tools/utils.py", {"get_matrix", "build_projection_matrix", "get_image_point"})
    camera = dict(pos=[.1, -.35, 2.25], rot=rotation, width=size[0], height=size[1], fov=60.)
    # get_matrix的角度参数顺序为pitch/roll/yaw，与camera配置的roll/pitch/yaw不同。
    with __import__('warnings').catch_warnings():
        __import__('warnings').simplefilter('ignore', PendingDeprecationWarning)
        camera_to_ego = np.asarray(ref["get_matrix"](camera["pos"], [rotation[1], rotation[0], rotation[2]]))
    ego_to_camera = np.linalg.inv(camera_to_ego)
    intrinsic = ref["build_projection_matrix"](*size, camera["fov"])
    points = np.random.default_rng(2026).uniform(-40, 40, (300, 2))
    actual = project_trajectory(points, camera)
    visible = 0
    for point, pixel in zip(points, actual):
        expected, depth = ref["get_image_point"]([*point, 0.], intrinsic, ego_to_camera)
        if depth > .1:
            np.testing.assert_allclose(pixel, expected, rtol=1e-10, atol=1e-8)
            visible += 1
        else:
            assert np.isnan(pixel).all()
    assert visible > 0


@pytest.mark.parametrize("meta", [[], {"sensor_information": []}, {"sensor_information": "bad"},
                                  {"sensor_information": {"num_cameras": 6}}])
def test_invalid_meta_or_known_non_three_cameras_never_fallback(tmp_path, meta):
    import lzma
    import pickle
    from qwen3vl_local.action_prior.comparison_scene import scene_configuration
    path = tmp_path / "metas/0001.pkl"
    path.parent.mkdir()
    with lzma.open(path, "wb") as handle:
        pickle.dump(meta, handle)
    with pytest.raises(ValueError):
        scene_configuration(tmp_path, 1)


def test_polyline_clips_both_offscreen_endpoints_without_sampling():
    import numpy as np
    from qwen3vl_local.action_prior.comparison_scene import project_polyline
    camera = dict(pos=[0, 0, 0], rot=[0, 0, 0], width=400, height=200, fov=90)
    pixels = project_polyline([[2, -10000], [2, 10000]], camera)
    np.testing.assert_allclose(pixels, [[-.5, 100], [399.5, 100]], atol=1e-8)
    # 完全在后方和完全在左侧的线段都不显示。
    assert project_polyline([[-3, -1], [-2, 1]], camera).shape == (0, 2)
    assert project_polyline([[2, -10], [3, -10]], camera).shape == (0, 2)


def test_polyline_breaks_at_hidden_excursion_and_clips_near_plane():
    import numpy as np
    from qwen3vl_local.action_prior.comparison_scene import project_polyline
    camera = dict(pos=[0, 0, 0], rot=[0, 0, 0], width=400, height=200, fov=90)
    pixels = project_polyline([[2, -1], [-1, 0], [2, 1]], camera)
    assert len(pixels) == 5 and np.isnan(pixels[2]).all()
    np.testing.assert_allclose(pixels[[0, 1, 3, 4]], [[100, 100], [-.5, 100], [399.5, 100], [300, 100]])
    np.testing.assert_allclose(project_polyline([[-1, 0], [10, 0]], camera), [[200, 100], [200, 100]])
    connected = project_polyline([[2, -.5], [3, 0], [2, .5]], camera)
    assert len(connected) == 3 and np.isfinite(connected).all()


def test_polyline_refuses_nonfinite_prediction():
    from qwen3vl_local.action_prior.comparison_scene import project_polyline
    camera = dict(pos=[0, 0, 0], rot=[0, 0, 0], width=400, height=200, fov=90)
    with pytest.raises(ValueError, match="非有限"):
        project_polyline([[1, 0], [float('nan'), 1]], camera)
