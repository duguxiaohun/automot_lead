"""自动动作索引保留门控前证据，模型/外部单选输入不伪造被压掉的动作。"""

import json
from types import SimpleNamespace

import pytest

from qwen3vl_local.action_prior import prepare_action_priors as preparation
from qwen3vl_local.action_prior.action_input import HighLevelActionIndex, gate_action
from qwen3vl_local.action_prior.contracts import file_hash
from test_action_input import record, write_index


def test_external_input_rejects_compound_and_oracle_evidence(tmp_path):
    """新外部协议只接受一个主要动作或空状态；不能把旧多标签改名后混用。"""
    for changes in (dict(actions=["STOP", "LANE_CHANGE_LEFT"]), dict(candidate_actions=["STOP"]),
                    dict(action_format="binary"), dict(action_format="choice")):
        with pytest.raises(ValueError):
            HighLevelActionIndex(write_index(tmp_path / "bad.jsonl", [record(**changes)]))


def test_automatic_index_roundtrip_keeps_raw_evidence_and_reuses_cache(tmp_path, monkeypatch):
    """真实索引发布/读取/复用，合成上游产物替代全数据扫描。"""
    from qwen3vl_local.sft_new_loop_phase3 import source_mapping
    from lead_video_tools import abnormal_duration_filter
    data, raw, cache = (tmp_path / key for key in ("data", "raw", "cache"))
    data.mkdir()
    full_path = tmp_path / "full.jsonl"
    full_path.write_text("synthetic full map")
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text("synthetic candidate file")
    records, candidates = {}, {}
    for split, run in zip(("train", "val", "test"), ("R", "V", "T")):
        (raw / "S" / run).mkdir(parents=True)
        (data / (split + ".jsonl")).write_text(json.dumps(dict(
            schema="action_prior_data_v1", split=split, scenario="S", run_id=run, anchor=2)) + "\n")
        records[("S", run, 2)] = dict(status="special_eligible", eligible_buckets=["UE1", "UE2"])
        candidates[("S", run, 2)] = {
            "UE1": dict(answers={"DECELERATE": True, "STOP": False, "RESUME": False}, planning_context="UE1"),
            "UE2": dict(answers={"DECELERATE": True, "STOP": False, "RESUME": False,
                                 "LANE_CHANGE_LEFT": True, "LANE_CHANGE_RIGHT": False}, planning_context="UE2"),
        }
    full = SimpleNamespace(records=records, manifest={"candidate_index": str(candidate_path)},
        validate_action_dataset=lambda _: None,
        source=SimpleNamespace(candidate_sha256=file_hash(candidate_path), sha256=file_hash(full_path),
            mapping_contract_hash="test-mapping", action_dataset_hashes={
                split: file_hash(data / (split + ".jsonl")) for split in ("train", "val", "test")}))
    monkeypatch.setattr(preparation, "EventBalanceIndex", lambda _: full)
    monkeypatch.setattr(preparation, "candidate_actions", lambda *args: candidates)
    monkeypatch.setattr(source_mapping, "mapping_contract_hash", lambda: "test-mapping")
    monkeypatch.setattr(abnormal_duration_filter, "is_abnormal_lead_route", lambda *args: (False, {}))
    path = preparation.prepare_actions(full_path, raw, data, cache)
    index = HighLevelActionIndex(path)
    assert index.get(("S", "R", 2))["actions"] == ["LANE_CHANGE_LEFT"]
    evidence = index.candidate_evidence(("S", "R", 2))
    assert evidence["actions"] == ["DECELERATE", "LANE_CHANGE_LEFT"]
    conditions = {"ROAD_STRUCTURE": "R1", "UE1": "YES", "STATIC_OBSTACLE": "NO",
                  "ROAD_CORRIDOR/INVALID_EVENT_CONTEXT": "NO"}
    assert gate_action(evidence, conditions, index.planning_contexts(("S", "R", 2)))[0]["actions"] == ["DECELERATE"]
    assert preparation.prepare_actions(full_path, raw, data, cache) == path
    changed = json.loads(path.read_text().splitlines()[0])
    changed["actions"] = ["DECELERATE"]
    path.write_text(json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="candidate evidence"):
        HighLevelActionIndex(path)
