"""审查问题回归：真实 AMP 更新、语义拒绝、输入生效、跨进程缓存和证据分组。"""

import ast
from collections import Counter
import contextlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest
from qwen3vl_local.action_prior import config, prompts, provenance, priors
from qwen3vl_local.action_prior.precision import decoder_forward
from qwen3vl_local.action_prior.text_cache import TextCache
from qwen3vl_local.action_prior.train import metrics_from_counts
from qwen3vl_local.action_prior.metrics import grouped_counts
from test_contracts import ask_fixture


class ScalarDecoder(torch.nn.Module):
    """用真实 Linear autocast 检查 FP32 主参数保存 BF16 分辨率以下的更新。"""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
        self.linear.weight.data.fill_(1.0)

    def forward(self, x):
        return {"pred_route": self.linear(x)}


def test_fp32_master_and_adamw_states_keep_small_bf16_updates():
    decoder = ScalarDecoder()
    opt = torch.optim.AdamW(
        decoder.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01
    )
    for _ in range(100):
        opt.zero_grad()
        out = decoder_forward(
            decoder, {"x": torch.ones(1, 1)}, torch.bfloat16, torch.device("cpu")
        )
        assert out["pred_route"].dtype == torch.float32
        out["pred_route"].sum().backward()
        opt.step()
    p = decoder.linear.weight
    assert p.dtype == torch.float32 and p.item() < 0.985
    assert opt.state[p]["exp_avg"].dtype == torch.float32
    assert opt.state[p]["exp_avg_sq"].dtype == torch.float32


@pytest.mark.parametrize(
    "field", list(prompts.FACT_LABELS) + list(prompts.EVENT_LABELS)
)
def test_fallback_keeps_every_positive_prior_and_requires_review_for_generated_text(
    field,
):
    value = {"conditions": {"ROAD_STRUCTURE": "R1", field: "YES"}}
    text = prompts.fallback_analysis(value)
    label = (prompts.FACT_LABELS | prompts.EVENT_LABELS)[field]
    assert f"{label}: YES" in text
    assert prompts.analysis_format_valid(text)
    assert not prompts.valid_analysis(text, value)  # 格式不是语义通过。
    assert text not in prompts.analysis_prompt(value, "navigation")


def test_review_counterexample_is_rejected_and_unknown_not_negative():
    value = {"conditions": {"ROAD_STRUCTURE": "R1", "UE3": "YES", "VULNERABLE": None}}
    bad = "Scene: This is a signal-controlled junction.\nInteraction: No vehicle is cutting in.\nPlanning context: Follow navigation."
    assert not prompts.valid_analysis(bad, value)
    assert "vulnerable road user: UNKNOWN" in prompts.fallback_analysis(value)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--bev-frame-count", "2"),
        ("--bev-frame-step", "2"),
        ("--frame-interval-s", "0.5"),
    ],
)
def test_unsupported_input_config_fails_before_training(flag, value):
    with pytest.raises(ValueError):
        config.validate_args(config.parser().parse_args([flag, value]))


def test_navigation_cli_reaches_actual_legacy_loader(tmp_path, monkeypatch):
    route = tmp_path / "Scene" / "Town01_Rep0_route_1_route0"
    route.mkdir(parents=True)
    row = dict(
        schema="action_prior_data_v1",
        split="train",
        scenario="Scene",
        run_id=route.name,
        anchor=0,
        route_group="Scene/Town01_route_1",
        tp_mode="route_lookahead",
        rgb_frame_count=4,
        rgb_frame_step=1,
        target_point_lookahead_s=1.0,
        next_target_point_lookahead_s=2.0,
    )
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n")
    import lead_video_tools.abnormal_duration_filter as abnormal

    monkeypatch.setattr(abnormal, "is_abnormal_lead_route", lambda *a: (False, {}))
    args = config.parser().parse_args(
        [
            "--data-dir",
            str(tmp_path),
            "--data-root",
            str(tmp_path),
            "--target-point-lookahead-s",
            "1.5",
            "--next-target-point-lookahead-s",
            "3.0",
        ]
    )
    config.validate_args(args)
    rows = config.read_rows(args, "train")
    # 提取真实旧 Dataset 的方法，仅替换昂贵 build_clip/GT IO，核对实际传参。
    source = Path(config.__file__).parents[1] / "leadmot/train.py"
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "LeadMoTSampleDataset"
    )
    cls.bases = []
    seen = []
    env = dict(
        Path=Path,
        Any=object,
        argparse=SimpleNamespace(Namespace=object),
        os=os,
        contextlib=contextlib,
        build_clip_from_real_lead_route=lambda **kw: seen.append(kw),
        _extract_targets=lambda *a: (None, None),
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), env)
    result = env["LeadMoTSampleDataset"](rows, args)[0]
    assert "_error" not in result
    assert seen[0]["tp_lookahead_s"] == 1.5 and seen[0]["ntp_lookahead_s"] == 3.0


def cache_process(path, count, ready, queue):
    """两个进程竞争同一 key；shared Value 记录真正进入模型计算的次数。"""
    cache = TextCache(path)
    ready.wait()

    def compute():
        with count.get_lock():
            count.value += 1
        return {"calls": [], "conditions": {"UE3": None}}

    value, hit = cache.get_or_compute("a" * 64, compute)
    queue.put((value["conditions"]["UE3"], hit))


def test_cross_process_cache_computes_once(tmp_path):
    ctx = mp.get_context("spawn")
    count, ready, queue = ctx.Value("i", 0), ctx.Event(), ctx.Queue()
    processes = [
        ctx.Process(
            target=cache_process, args=(tmp_path / "shared", count, ready, queue)
        )
        for _ in range(2)
    ]
    for p in processes:
        p.start()
    ready.set()
    values = [queue.get(timeout=25) for _ in processes]
    for p in processes:
        p.join(25)
        assert p.exitcode == 0
    assert count.value == 1 and sorted(hit for _, hit in values) == [False, True]


def test_recheck_compare_records_both_without_claiming_accuracy():
    result = priors.collect_priors(ask_fixture(), "case", recheck_mode="compare")
    assert len(result["calls"]) == 15
    assert {x["mode"] for x in result["recheck_comparisons"]} == {
        "independent",
        "history",
    }
    assert not result["consistency_is_accuracy"]
    independent = priors.collect_priors(
        ask_fixture(), "case", recheck_mode="independent"
    )
    assert all(not x["history"] for x in independent["calls"])


def test_junction_identical_prompt_is_reported():
    from qwen3vl_local.sft_new_loop_phase1 import prompts as p1
    from qwen3vl_local.sft_new_loop_phase2 import prompts as p2

    def ask(phase, spec, history):
        text, _ = ask_fixture()(phase, spec, history)
        prompt = (p1.build_phase1_prompt if phase == 1 else p2.build_event_prompt)(
            spec=spec, history_rgb_mode="4rgb"
        )
        return text, prompt

    result = priors.collect_priors(ask, "case")
    junction = next(
        c for c in result["recheck_comparisons"] if c["scope"] == "LOCAL_JUNCTION"
    )
    assert junction["same_prompt"]


def test_upstream_pool_overlap_is_conservative_and_repeat_aware(tmp_path):
    run = tmp_path / "sft"
    run.mkdir()
    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps(
            dict(
                split="train",
                scenario="X",
                route_id="Town01_Rep1_route_1_route0_01_01_00_00_00",
            )
        )
        + "\n"
    )
    (run / "train_run_manifest.json").write_text(
        json.dumps(dict(index=str(index), split="train"))
    )
    source = provenance.upstream_training_pool({"path": str(run / "best_generation")})
    rows = [
        dict(
            route_group=provenance.route_group(
                "X", "Town01_Rep2_route_1_route0_02_02_00_00_00"
            )
        ),
        dict(route_group="X/Town01_route_2"),
    ]
    report = provenance.annotate_upstream(
        rows, {"phase1": source, "phase2": dict(status="unknown", routes=[])}
    )
    assert (
        report["combined/train_pool_overlap"] == 1 and report["combined/unknown"] == 1
    )
    assert not source["actual_sampled_routes_verified"]
    assert (
        provenance.upstream_training_pool(
            {"path": str(tmp_path / "missing/best_generation")}
        )["status"]
        == "unknown"
    )


def test_shared_source_changes_execution_identity(tmp_path):
    names = [
        "qwen3vl_local/engine.py",
        "qwen3vl_local/mrope_utils.py",
        "qwen3vl_local/leadmot/decoder.py",
        "leaderboard/team_code/mot_lead_offline_runner.py",
        "Automot/mot/modeling/bev_encoder/bev_encoder_utils.py",
    ]
    for name in set(names) | set(provenance.EXECUTION_SEEDS):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# first\n")
    for name in names:
        before = provenance.execution_fingerprint(tmp_path)
        p = tmp_path / name
        p.write_text(p.read_text() + "# mutation\n")
        after = provenance.execution_fingerprint(tmp_path)
        assert before != after and before["code"][name] != after["code"][name]


def test_group_metrics_use_own_denominators():
    totals = Counter(samples=3, loss=12)
    totals.update(
        grouped_counts({"invalid": {}, "conditions": {"UE3": "YES"}}, {}, {"loss": 2})
    )
    for _ in range(2):
        totals.update(
            grouped_counts(
                {
                    "invalid": {"UE3": "disagreement"},
                    "conditions": {"UE3": None},
                    "analysis_fallback": True,
                },
                {},
                {"loss": 5},
            )
        )
    result = metrics_from_counts(totals)
    assert result["loss"] == 4 and result["group/accepted/loss"] == 2
    assert (
        result["group/invalid/loss"] == 5
        and result["group/summary_fallback/samples"] == 2
    )
    assert result["group/condition/UE3/YES/loss"] == 2


def test_paired_ablation_uses_same_prior_groups_for_both_runs():
    from qwen3vl_local.action_prior.compare_ablation import compare

    key = ("X", "run", 3)
    row = dict(scenario="X", run_id="run", anchor=3)
    base = {
        key: dict(
            sample=row,
            gt_route=[[1, 2]],
            gt_waypoints=[[2, 3]],
            metrics={"loss": 5},
            condition_mode="base",
        )
    }
    prior = {
        key: dict(
            sample=row,
            gt_route=[[1, 2]],
            gt_waypoints=[[2, 3]],
            metrics={"loss": 2},
            conditions={"UE3": "YES"},
            invalid={},
        )
    }
    result = compare(base, prior)
    assert result["base"]["group/condition/UE3/YES/samples"] == 1
    assert result["prior_minus_base"]["group/condition/UE3/YES/loss"] == -3
    with pytest.raises(ValueError, match="identities"):
        compare(base, {})
