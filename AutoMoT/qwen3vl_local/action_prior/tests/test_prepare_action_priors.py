"""自动 Phase3 动作先验：真实内容校验、事件门控、缓存发布与默认背景行为。"""

import json
from pathlib import Path

import pytest

from qwen3vl_local.action_prior import prepare_action_priors as preparation
from qwen3vl_local.action_prior import event_balance as balance, prompts
from qwen3vl_local.action_prior.action_input import HighLevelActionIndex
from qwen3vl_local.action_prior.config import parser
from qwen3vl_local.action_prior.contracts import file_hash
from qwen3vl_local.sft_new_loop_phase3 import source_mapping
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import ACTION_RULE_VERSION, action_rule_sha256
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS


@pytest.fixture
def sources(tmp_path, monkeypatch):
    """完整小型候选/full map，不加载真实数据或模型。"""
    root, data, phase3, full_dir = (tmp_path / name for name in ("raw", "data", "phase3", "full"))
    for path in (data, phase3, full_dir, root / "S/R/rgb"):
        path.mkdir(parents=True)
    monkeypatch.setattr(source_mapping, "mapping_contract_hash", lambda: "a" * 64)
    for split, frames in (("train", [1, 3]), ("val", [2]), ("test", [4, 5])):
        (data / f"{split}.jsonl").write_text("".join(json.dumps(dict(
            schema="action_prior_data_v1", split=split, scenario="S", run_id="R", anchor=i,
        )) + "\n" for i in frames))
    (data / "manifest.json").write_text("{}")
    candidates = []
    for frame, context, actions in ((1, "LEAD_BRAKE", ["STOP", "LANE_CHANGE_LEFT"]),
                                    (2, "POST_BYPASS_RETURN", ["LANE_CHANGE_RIGHT"])):
        candidates.append(dict(scenario="S", route_id="R", frame_id=frame, context_id=context,
                               mapping_contract_hash="a" * 64, action_labels={a: a in actions for a in ACTION_KEYS},
                               action_evidence=dict(rule_version=ACTION_RULE_VERSION, rule_code_sha256=action_rule_sha256())))
    candidate_path = phase3 / "candidate_frames.jsonl"
    candidate_path.write_text("".join(json.dumps(row) + "\n" for row in candidates))
    (phase3 / "frame_index.jsonl").write_text("{}\n")
    (phase3 / "candidate_counts.json").write_text(json.dumps({"train/LEAD_BRAKE": 1, "val/POST_BYPASS_RETURN": 1}))
    (phase3 / "manifest.json").write_text(json.dumps(dict(format="sft_new_loop_phase3_frame_index_v3_current_phase",
                                                         mapping_contract_hash="a" * 64, frame_index="frame_index.jsonl")))
    records = []
    for frame, status, special, eligible, contexts in (
        (1, balance.SPECIAL_ELIGIBLE, ["UE1"], ["UE1"], ["UE1"]),
        (2, balance.SPECIAL_ELIGIBLE, ["RE2"], ["RE2"], ["RE2_NAVIGATION_TRANSITION"]),
        (3, balance.CONFIRMED_REGULAR, [], [], []),
        (4, balance.SPECIAL_FILTERED, ["UE2"], [], ["UE2"]),
        (5, balance.UNCONFIRMED, [], [], []),
    ):
        records.append(dict(schema=balance.FULL_INDEX_SCHEMA, mapping_contract_hash="a" * 64,
                            scenario="S", route_id="R", frame_id=frame, status=status,
                            special_buckets=special, eligible_buckets=eligible, scene_contexts=contexts))
    full = full_dir / "full_event_mapping.jsonl"
    full.write_text("".join(json.dumps(row) + "\n" for row in records))
    (full_dir / "manifest.json").write_text(json.dumps(dict(
        schema=balance.FULL_MANIFEST_SCHEMA, index_schema=balance.FULL_INDEX_SCHEMA,
        mapping_policy=balance.EVENT_BALANCE_MAPPING_POLICY, index_file=full.name, index_sha256=file_hash(full),
        candidate_index=str(candidate_path), candidate_sha256=file_hash(candidate_path), mapping_contract_hash="a" * 64,
        action_dataset_hashes={s: file_hash(data / f"{s}.jsonl") for s in ("train", "val", "test")},
    )))
    return root, data, full, tmp_path / "cache"


def test_prepare_reuses_phase3_labels_and_keeps_background_unchanged(sources):
    """按问题域裁掉 UE1 横向副动作，RE2 保留横向，普通两份权重与 prompt 均不变。"""
    path = preparation.prepare_actions(sources[2], sources[0], sources[1], sources[3])
    index = HighLevelActionIndex(path)
    assert index.get(("S", "R", 1)) == dict(status="selected", actions=["STOP"])
    assert index.get(("S", "R", 2))["actions"] == ["LANE_CHANGE_RIGHT"]
    assert index.planning_contexts(("S", "R", 2)) == ("RE2_NAVIGATION_TRANSITION",)
    assert index.identity["privileged_action_conditioning"] is True
    assert balance.EVENT_BALANCE_WEIGHTS[balance.REGULAR_BACKGROUND] == 2
    assert all(balance.EVENT_BALANCE_WEIGHTS[bucket] == 1 for bucket in balance.SPECIAL_BUCKETS)
    prior = dict(conditions={"ROAD_STRUCTURE": "R1"}, high_level_planning=True)
    for frame in (3, 4, 5):
        assert index.get(("S", "R", frame))["status"] == "not_applicable"
        assert index.planning_contexts(("S", "R", frame)) == ()
        scoped = dict(prior, high_level_action_prior=True, high_level_action=index.get(("S", "R", frame)))
        assert prompts.prefill_prompt(scoped, "nav") == prompts.prefill_prompt(prior, "nav")
        assert prompts.analysis_prompt(scoped, "nav") == prompts.analysis_prompt(prior, "nav")
        assert prompts.review_prompt(scoped, "nav", "draft") == prompts.review_prompt(prior, "nav", "draft")
        assert prompts.fallback_analysis(scoped) == prompts.fallback_analysis(prior)
    before = path.stat().st_mtime_ns
    assert preparation.prepare_actions(sources[2], sources[0], sources[1], sources[3]) == path
    assert path.stat().st_mtime_ns == before


def test_prepare_quarantines_corrupt_automatic_output(sources):
    """损坏的自动缓存保留后重建，不把缺失内容当动作全 NO。"""
    path = preparation.prepare_actions(sources[2], sources[0], sources[1], sources[3])
    path.write_text("{}\n")
    assert preparation.prepare_actions(sources[2], sources[0], sources[1], sources[3]) == path
    assert len(list(sources[-1].glob(".invalid-*"))) == 1
    assert HighLevelActionIndex(path).get(("S", "R", 1))["actions"] == ["STOP"]


def test_auto_prepare_without_action_index_and_resume_never_rebuilds(sources, monkeypatch):
    """开关无需手填动作路径；缺失的上游产物交给原准备器，续训不重选标签。"""
    root, data, full, cache = sources
    monkeypatch.chdir(root.parent)
    called = []

    def prepare(*args):
        """代表正式的候选/full map 复用或生成，不修改原始标签。"""
        called.append(args)
        return full

    monkeypatch.setattr(preparation, "prepare", prepare)
    args = parser().parse_args(["--high-level-planning", "--high-level-action-prior", "--data-root", str(root), "--data-dir", str(data)])
    preparation.ensure_action_inputs(args)
    assert len(called) == 1 and Path(args.high_level_action_index).is_file()
    args.resume = "original.pt"
    preparation.ensure_action_inputs(args)
    assert len(called) == 1
    args.high_level_action_index = ""
    with pytest.raises(ValueError, match="resume"):
        preparation.ensure_action_inputs(args)


def test_missing_eligible_candidate_is_not_background():
    """真实 special 丢标注要失败，不能落到占两份的普通池。"""
    with pytest.raises(ValueError, match="missing Phase3"):
        preparation.project_frame(dict(status=balance.SPECIAL_ELIGIBLE, eligible_buckets=["UE1"]), {})


def test_concurrent_candidates_must_agree():
    """多 context 标签不一致时不能随便优先选一个动作。"""
    with pytest.raises(ValueError, match="conflicting"):
        preparation.project_frame(dict(status=balance.SPECIAL_ELIGIBLE, eligible_buckets=["UE1", "UE3"]), {
            "UE1": dict(answers={"STOP": True}, planning_context="UE1"),
            "UE3": dict(answers={"STOP": False}, planning_context="UE3"),
        })


def test_oracle_contract_requires_same_full_map_and_allows_relocation(sources):
    """搬迁动作文件和 manifest 不改身份，换 full map 或 split 则拒绝。"""
    from qwen3vl_local.action_prior.action_input import action_input_contract
    from types import SimpleNamespace
    import shutil
    root, data, full, cache = sources
    path = preparation.prepare_actions(full, root, data, cache)
    args = SimpleNamespace(high_level_action_prior=True, high_level_action_index=str(path),
                           data_dir=str(data), event_balance_index=str(full))
    original = action_input_contract(args)
    moved = root.parent / "moved"
    shutil.copytree(path.parent, moved)
    args.high_level_action_index = str(moved / path.name)
    assert action_input_contract(args) == original
    other = root.parent / "other_map.jsonl"
    other.write_bytes(full.read_bytes() + b"\n")
    args.event_balance_index = str(other)
    with pytest.raises(ValueError, match="same full map"):
        action_input_contract(args)
    args.event_balance_index = str(full)
    with (data / "train.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="different action splits"):
        action_input_contract(args)


def test_missing_action_data_uses_original_builder(sources, monkeypatch):
    """底层 train 直接开启时，缺 action 索引也调用原构建器。"""
    root, data, full, cache = sources
    monkeypatch.chdir(root.parent)
    (data / "manifest.json").unlink()
    called = []

    def build(script, arguments):
        """模拟原 action 构建器发布 manifest，其余小型 split 已存在。"""
        called.append((script, arguments))
        (data / "manifest.json").write_text("{}")

    monkeypatch.setattr(preparation, "run_builder", build)
    args = parser().parse_args(["--high-level-planning", "--high-level-action-prior",
                               "--data-root", str(root), "--data-dir", str(data),
                               "--event-balance-index", str(full)])
    preparation.ensure_action_inputs(args)
    assert len(called) == 1 and called[0][0].name == "build_dataset.py"
    assert Path(args.high_level_action_index).is_file()


def test_abnormal_route_is_filtered_before_candidate_labeling(sources, monkeypatch):
    """异常 route 不产动作行；每 route 只筛选一次，早于候选标签消费。"""
    from lead_video_tools import abnormal_duration_filter
    root, data, full, cache = sources
    calls = []
    monkeypatch.setattr(abnormal_duration_filter, "is_abnormal_lead_route", lambda *args: (calls.append(args) or True, "over_90s"))
    original = preparation.candidate_actions

    def candidates(*args):
        """消费候选之前，路由过滤已完成。"""
        assert len(calls) == 1
        return original(*args)

    monkeypatch.setattr(preparation, "candidate_actions", candidates)
    with pytest.raises(ValueError, match="empty high-level action index"):
        preparation.prepare_actions(full, root, data, cache)
    assert not list(cache.glob("actions_*"))


def test_special_all_no_remains_distinct_from_regular():
    """有效全 NO 仍有特殊规划上下文，背景则不追加动作提示。"""
    action, contexts = preparation.project_frame(dict(status=balance.SPECIAL_ELIGIBLE, eligible_buckets=["UE1"]), {
        "UE1": dict(answers={"STOP": False, "RESUME": False, "DECELERATE": False}, planning_context="UE1"),
    })
    assert action == dict(status="no_action", actions=[]) and contexts == ["UE1"]
    assert preparation.project_frame(dict(status=balance.CONFIRMED_REGULAR), {}) == (dict(status="not_applicable", actions=[]), [])
