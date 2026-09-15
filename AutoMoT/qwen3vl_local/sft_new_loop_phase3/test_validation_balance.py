"""用合成候选复现同 RS 负例预算不足，核验自动增容及训练前 CPU 预检。"""

from collections import Counter
from dataclasses import replace
import random
import sys

import pytest

from qwen3vl_local.sft_new_loop_phase3 import train
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID, CONTEXT_IDS
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import (
    InvalidQuotaError, balanced_invalid_items, case_identity, invalid_subgroup_report,
)


def candidate_rows(reviewed_contexts=5):
    """所有维度齐全，但人工同 RS 负例集中于单个来源，模拟远端小验证桶。"""
    rows = []

    def add(source, rs, asked, reason):
        """构造不同输入身份；签名均为真实候选字段，不访问数据集。"""
        invalid = bool(reason)
        rows.append(train.FrameRow(
            scenario="synthetic", route_id=f"route_{len(rows)}", town="Town01",
            frame_id=len(rows), true_rs=rs, prompt_road_structure=rs,
            context_id=asked, question_domain=CONTEXT_BY_ID[asked].question_domain,
            action_signature="INVALID" if invalid else "DECELERATE", event="",
            split="val", goal_ego_xy=(10, 0), history_rgb_paths=["unused.jpg"] * 4,
            latest_rgb_path="unused.jpg", current_speed_mps=3,
            answers={**{key: False for key in train.ANSWER_KEYS},
                     "DECELERATE": not invalid, train.INVALID_KEY: invalid},
            invalid_source=(f"source={source}|true_rs={rs}|asked_context={asked}" if invalid else ""),
            invalid_reason=reason,
        ))

    for source in CONTEXT_IDS:
        add(source, CONTEXT_BY_ID[source].allowed_rs[0], source, "")
        for rs in ("R1", "R2", "R3", "R4", "R5"):
            for asked in CONTEXT_IDS:
                add(source, rs, asked, "wrong_road_structure")
    for asked in CONTEXT_IDS[:reviewed_contexts]:
        add(CONTEXT_IDS[0], "R1", asked, "same_rs_wrong_event")
    return rows


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("target", [31, 32])
def test_quota_remainder_goes_to_required_reviewed_source(seed, target):
    """总量32已有可行解，余数必须分给需要4条人工种子的来源，不能随机报错。"""
    rows = [row for row in candidate_rows(4) if row.invalid_source]
    sampled = balanced_invalid_items(rows, target=target, rng=random.Random(seed))
    report = invalid_subgroup_report(sampled)
    assert len(sampled) == target
    assert report["source_class"]["counts"][CONTEXT_IDS[0]] == 4
    assert report["same_rs_unique_cases"] == 4
    assert report["same_rs_max_case_repeat"] == 1
    assert all(report["guards"].values())


def test_real_shortage_reports_feasible_budget_and_retries_once():
    """五个同来源人工题至少需41个 INVALID；验证上调到每类21、INVALID42。"""
    rows = candidate_rows()
    for seed in range(12):
        with pytest.raises(InvalidQuotaError) as caught:
            balanced_invalid_items([r for r in rows if r.invalid_source], target=32,
                                   rng=random.Random(seed))
        assert caught.value.required_target == 41
        work, audit = train._validation_work(rows, target_per_bin=16, seed=seed)
        assert audit["requested"] == 16
        assert audit["effective"] == 21
        assert audit["adjusted"]
        assert audit["sampled_cases"] == 252
        assert audit["class_counts"] == {**dict.fromkeys(CONTEXT_IDS, 21), "INVALID": 42}
        report = invalid_subgroup_report(work)
        assert report["same_rs_unique_cases"] == 5
        assert report["same_rs_max_case_repeat"] == 1
        assert all(report["guards"].values())
        # 请求16后自动增容与显式21必须得到同一批题、同一顺序。
        direct = train._balanced_work(rows, target_per_bin=21, seed=seed)
        assert [case_identity(x) for x in work] == [case_identity(x) for x in direct]


def test_strict_mode_and_dataset_errors_are_not_silenced():
    """关闭自动增容应保留类型化错误；缺源桶或损坏签名不能靠增容绕过。"""
    rows = candidate_rows()
    with pytest.raises(InvalidQuotaError):
        train._validation_work(rows, target_per_bin=16, seed=1, auto_increase=False)
    incomplete = [row for row in rows if not row.invalid_source.startswith(f"source={CONTEXT_IDS[-1]}|")]
    with pytest.raises(ValueError, match="missing_sources") as caught:
        train._validation_work(incomplete, target_per_bin=16, seed=1)
    assert not isinstance(caught.value, InvalidQuotaError)
    broken = [replace(row, invalid_source="broken") if row.invalid_source else row for row in rows]
    with pytest.raises(ValueError, match="signature"):
        train._validation_work(broken, target_per_bin=16, seed=1)


@pytest.mark.parametrize("mode", ["binary", "choice"])
def test_sufficient_budget_is_unchanged(mode):
    """充足预算不增容；choice 不引入 INVALID，二者均记录去重前后的数量。"""
    rows = candidate_rows()
    work, audit = train._validation_work(rows, target_per_bin=32, seed=3, action_output_mode=mode)
    assert not audit["adjusted"]
    assert audit["effective"] == audit["requested"] == 32
    assert audit["unique_cases"] == len({case_identity(x) for x in work})
    expected = {**dict.fromkeys(CONTEXT_IDS, 32), **({"INVALID": 64} if mode == "binary" else {})}
    assert Counter(train._balance_class(item.row) for item in work) == expected


def test_full_eval_retains_all_rows_and_checks_coverage():
    """独立eval的全量模式仍不重采样，保留原顺序和覆盖守卫。"""
    rows = [row for row in candidate_rows() if row.invalid_source]
    assert balanced_invalid_items(rows, target=0, rng=random.Random(3)) == rows
    with pytest.raises(ValueError, match="missing_sources"):
        balanced_invalid_items(rows[:3], target=0, rng=random.Random(3))


@pytest.mark.parametrize("mode", ["binary", "choice"])
@pytest.mark.parametrize("rgb", ["4rgb", "2rgb_endpoints"])
def test_cpu_preflight_uses_real_worklists_without_model_or_nccl(monkeypatch, tmp_path, capsys, mode, rgb):
    """四种实验配置均走实际采样；预检不加载模型、建立进程组或写训练目录。"""
    from qwen3vl_local.sft_new_loop_phase3 import preflight

    monkeypatch.setattr(sys, "argv", ["train.py", "--sampling-only", "--focus-balance-count", "32",
                                     "--action-output-mode", mode, "--history-rgb-mode", rgb,
                                     "--output-dir", str(tmp_path / "not_created")])
    args = train.parse_args()
    assert "data_v9" in args.index
    for key, value in {"WORLD_SIZE": "4", "RANK": "0", "LOCAL_RANK": "0"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(preflight, "check_index", lambda path: {})
    monkeypatch.setattr(train, "_read_rows", lambda *a, **kw: candidate_rows())

    def forbidden(*args, **kwargs):
        """CPU采样阶段不得进入任何模型或分布式运行路径。"""
        pytest.fail("sampling preflight touched model, images or NCCL")

    for module, name in ((preflight, "check_model"), (train, "setup_distributed"),
                         (train, "load_model_with_lora"), (train, "_load_images")):
        monkeypatch.setattr(module, name, forbidden)
    train.train(args)
    assert not (tmp_path / "not_created").exists()
    assert args.validation_sampling["loss"]["effective"] == (21 if mode == "binary" else 16)
    assert args.validation_sampling["generation"]["effective"] == 32
    assert '[sampling-preflight] {"action_output_mode"' in capsys.readouterr().out


def test_strict_validation_failure_precedes_distributed_setup(monkeypatch):
    """正式训练小预算失败也必须在NCCL初始化之前，并准确指出loss阶段原因。"""
    from qwen3vl_local.sft_new_loop_phase3 import preflight

    monkeypatch.setattr(sys, "argv", ["train.py", "--focus-balance-count", "32",
                                     "--no-auto-eval-balance-count"])
    monkeypatch.setattr(preflight, "check_index", lambda path: {})
    monkeypatch.setattr(preflight, "check_model", lambda path: {})
    monkeypatch.setattr(train, "_read_rows", lambda *a, **kw: candidate_rows())
    monkeypatch.setattr(train, "setup_distributed", lambda *a: pytest.fail("NCCL initialized before sampling"))
    with pytest.raises(RuntimeError, match="periodic loss validation sampling failed.*quota shortage") as caught:
        train.train(train.parse_args())
    assert isinstance(caught.value.__cause__, InvalidQuotaError)


def test_generation_budget_can_increase_independently(monkeypatch):
    """loss预算足够但generation预算不足时，各自保留请求值并独立增容。"""
    from qwen3vl_local.sft_new_loop_phase3 import preflight

    monkeypatch.setattr(sys, "argv", ["train.py", "--sampling-only", "--focus-balance-count", "32",
                                     "--eval-balance-count", "32", "--generation-eval-balance-count", "16"])
    monkeypatch.setattr(preflight, "check_index", lambda path: {})
    monkeypatch.setattr(train, "_read_rows", lambda *a, **kw: candidate_rows())
    args = train.parse_args()
    train.train(args)
    assert args.validation_sampling["loss"]["requested"] == args.eval_balance_count == 32
    assert not args.validation_sampling["loss"]["adjusted"]
    assert args.validation_sampling["generation"]["requested"] == 16
    assert args.validation_sampling["generation"]["effective"] == args.generation_eval_balance_count == 21
