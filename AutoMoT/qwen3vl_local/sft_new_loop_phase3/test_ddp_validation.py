"""不加载 Qwen/CUDA，核验验证等待预算及进度日志不改变评分。"""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

from qwen3vl_local.sft_new_loop_phase3 import train


@pytest.mark.parametrize("legacy", [False, True])
def test_timeout_reaches_nccl_including_legacy_fallback(monkeypatch, legacy):
    """新版与不支持 device_id 的旧版初始化都必须收到用户预算。"""
    for key, value in {"WORLD_SIZE": "4", "RANK": "2", "LOCAL_RANK": "2"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    pinned, calls = [], []
    monkeypatch.setattr(torch.cuda, "set_device", pinned.append)

    def initialize(**kwargs):
        calls.append(kwargs)
        if legacy and "device_id" in kwargs:
            raise TypeError("unexpected keyword argument 'device_id'")

    monkeypatch.setattr(train.dist, "init_process_group", initialize)
    assert train.setup_distributed(2700) == (2, 2, 4)
    assert pinned == [2]
    assert len(calls) == (2 if legacy else 1)
    assert all(call["timeout"] == timedelta(seconds=2700) for call in calls)
    assert all(call["backend"] == "nccl" for call in calls)


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_timeout_rejected(value):
    """拒绝无效预算，避免重新落回库默认值。"""
    with pytest.raises(ValueError, match="must be positive"):
        train.setup_distributed(value)


def test_generation_progress_preserves_dedup_scores_and_mode(monkeypatch, capsys, tmp_path):
    """不同日志频率应生成相同去重样本、结果文件与指标。"""
    model = torch.nn.Linear(1, 1)
    bundle = SimpleNamespace(unwrap=lambda: model, processor=None, tokenizer=None, device="cpu")
    row = train.FrameRow(
        scenario="synthetic", route_id="route", town="Town01", frame_id=1,
        true_rs="R3", prompt_road_structure="R3", context_id="RAMP_MERGE_EXIT",
        question_domain="FULL_MANEUVER", action_signature="NONE", event="R-E3", split="val",
        goal_ego_xy=(10, 0), history_rgb_paths=["unused.jpg"] * 4, latest_rgb_path="unused.jpg",
        answers={key: False for key in train.ANSWER_KEYS},
    )
    item = train._make_item(row, seed=1)
    raw = "\n".join(f"{key}: NO" for key in item.spec.output_keys)
    monkeypatch.setattr(train, "_load_images", lambda paths: [])
    monkeypatch.setattr(train, "build_action_messages", lambda **kwargs: [])
    monkeypatch.setattr(train, "_kv_start_state", lambda *args: None)
    generated = []

    def generate(*args):
        generated.append(1)
        return raw, None, None

    monkeypatch.setattr(train, "_student_generate_kv", generate)
    results = []
    for log_every in (1, 10):
        results.append(train.evaluate_generation_probe(
            bundle, [item, item], history_rgb_mode="4rgb", max_new_tokens=64,
            log_every=log_every, record_path=tmp_path / f"cases_{log_every}.jsonl", step=2000,
        ))
        assert model.training
    assert len(generated) == 2
    assert results[0]["samples"] == 1
    assert results[0]["sampled_before_dedup"] == 2
    assert results[0]["exact_accuracy"] == 1
    assert (tmp_path / "cases_1.jsonl").read_text() == (tmp_path / "cases_10.jsonl").read_text()
    for result in results:
        assert result.pop("elapsed_seconds") >= 0
    assert results[0] == results[1]
    output = capsys.readouterr().out
    assert "unique_cases=1" in output
    assert "route=synthetic/route frame=1" in output
    assert "done=1/1" in output
