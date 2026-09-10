"""全帧 event-map 均衡课程的纯 CPU 合同。"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from qwen3vl_local.action_prior import event_balance as balance
from qwen3vl_local.action_prior.config import parser, validate_args
from qwen3vl_local.action_prior.contracts import file_hash
from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapping_contract_hash


def _action_row(frame, split="train"):
    return dict(
        scenario="Scenario", run_id=f"route_{frame % 3}", anchor=frame,
        split=split, route_group=f"Scenario/route_{frame % 3}",
    )


def _write_source(tmp_path, *, mapping_hash=None):
    directory = tmp_path / "event_map"
    directory.mkdir()
    index = directory / "full_event_mapping.jsonl"
    expected = mapping_contract_hash() if mapping_hash is None else mapping_hash
    rows = []
    for frame, bucket in enumerate(balance.SPECIAL_BUCKETS):
        contexts = {"RE2": ["RE2_NAVIGATION_TRANSITION"]}.get(bucket, [bucket])
        rows.append(dict(
            schema=balance.FULL_INDEX_SCHEMA, mapping_contract_hash=expected,
            scenario="Scenario", route_id=f"route_{frame % 3}", frame_id=frame,
            source_split="train", special_buckets=[bucket], eligible_buckets=[bucket], scene_contexts=contexts,
            status=balance.SPECIAL_ELIGIBLE,
        ))
    # 一个真实并发帧保留多个事实，不依赖本次抽中的配额桶。
    rows[0]["special_buckets"].append("RE2")
    rows[0]["eligible_buckets"].append("RE2")
    rows[0]["scene_contexts"].append("RE2_PRIOR_OBSTACLE")
    for frame in range(20, 40):
        rows.append(dict(
            schema=balance.FULL_INDEX_SCHEMA, mapping_contract_hash=expected,
            scenario="Scenario", route_id=f"route_{frame % 3}", frame_id=frame,
            source_split="train", special_buckets=[], scene_contexts=[],
            status=balance.CONFIRMED_REGULAR,
        ))
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = dict(
        schema=balance.FULL_MANIFEST_SCHEMA, index_schema=balance.FULL_INDEX_SCHEMA,
        index_file=index.name, index_sha256=file_hash(index),
        mapping_contract_hash=expected, candidate_sha256="a" * 64,
        mapping_policy=balance.EVENT_BALANCE_MAPPING_POLICY,
        action_dataset_hashes={key: "b" * 64 for key in ("train", "val", "test")},
    )
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return index


def test_event_balanced_epoch_has_exact_1_to_2_contract_fixed_context_and_repeat_audit(tmp_path):
    source = balance.EventBalanceIndex(_write_source(tmp_path))
    rows = [_action_row(index) for index in range(40)]
    source.annotate(rows)
    sampled, report = balance.build_event_balanced_epoch(rows, total=96, seed=7, repeat_cap=8)
    assert len(sampled) == 96
    assert report["quotas"] == {**{key: 8 for key in balance.SPECIAL_BUCKETS}, balance.REGULAR_BACKGROUND: 16}
    assert report["sampled"] == report["quotas"]
    assert report["max_frame_repeats"] <= 8
    assert report["unique_frames"] > 0 and report["repeat_histogram"]
    concurrent = next(item for item in sampled if item["anchor"] == 0)
    assert concurrent["event_balance_scene_contexts"] == ["UE1", "RE2_PRIOR_OBSTACLE"]
    assert all(report["unique_routes"][key] >= 1 for key in report["quotas"])


def test_only_confirmed_regular_is_background_and_filtered_special_is_not_silently_normal(tmp_path):
    source = balance.EventBalanceIndex(_write_source(tmp_path))
    rows = [_action_row(index) for index in range(42)]
    source.annotate(rows)
    rows[40].update(event_balance_status=balance.UNCONFIRMED, event_balance_buckets=[])
    rows[41].update(
        event_balance_status=balance.SPECIAL_FILTERED,
        event_balance_buckets=[], event_balance_all_special_buckets=["UE3"],
    )
    counts = balance.available_counts(rows)
    assert counts[balance.REGULAR_BACKGROUND] == 20
    assert counts["status/unconfirmed"] >= 1
    assert counts["status/special_filtered"] >= 1


def test_epoch_budget_never_allows_unbounded_rare_frame_repeats(tmp_path):
    source = balance.EventBalanceIndex(_write_source(tmp_path))
    rows = [_action_row(index) for index in range(40)]
    source.annotate(rows)
    with pytest.raises(ValueError, match="infeasible.*shared-frame"):
        balance.event_balanced_total(rows, requested=12000, repeat_cap=8, world=1)
    total = balance.event_balanced_total(rows, requested=0, repeat_cap=2, world=1)
    _, audit = balance.build_event_balanced_epoch(rows, total=total, seed=3, repeat_cap=2)
    assert audit["max_frame_repeats"] <= 2


def _allocation_rows(*, ue1_extra=False):
    """UE1/UE2 共享 anchor=0；UE2 没有替代帧，测联合而非桶独立容量。"""
    rows = []
    for index, bucket in enumerate(balance.SPECIAL_BUCKETS):
        if bucket == "UE2":
            continue
        anchor = 0 if bucket in ("UE1", "UE2") else index + 10
        buckets = [bucket]
        if bucket == "UE1":
            buckets.append("UE2")
        rows.append(dict(
            scenario="S", run_id="R", anchor=anchor, route_group=f"S/R{anchor}",
            event_balance_buckets=buckets, event_balance_status=balance.SPECIAL_ELIGIBLE,
        ))
    if ue1_extra:
        rows.append(dict(
            scenario="S", run_id="R", anchor=99, route_group="S/R99",
            event_balance_buckets=["UE1"], event_balance_status=balance.SPECIAL_ELIGIBLE,
        ))
    for anchor in (200, 201):
        rows.append(dict(
            scenario="S", run_id="R", anchor=anchor, route_group=f"S/R{anchor}",
            event_balance_buckets=[], event_balance_status=balance.CONFIRMED_REGULAR,
        ))
    return rows


def test_shared_frame_auto_budget_is_jointly_feasible_before_training():
    rows = _allocation_rows()
    # 独立桶计数会误报每个 UE 均有 2 次容量；联合 cap=2 时 UE1+UE2 只能各一份。
    total = balance.event_balanced_total(rows, requested=0, repeat_cap=2, world=1)
    assert total == 12
    sampled, audit = balance.build_event_balanced_epoch(rows, total=total, seed=5, repeat_cap=2)
    assert len(sampled) == total and audit["joint_allocation"] is True
    assert audit["max_frame_repeats"] <= 2


def test_joint_assignment_reserves_shared_frame_for_the_bucket_without_an_alternative():
    rows = _allocation_rows(ue1_extra=True)
    sampled, _ = balance.build_event_balanced_epoch(rows, total=12, seed=1, repeat_cap=1)
    by_bucket = {row["event_balance_bucket"]: row["anchor"] for row in sampled}
    assert by_bucket["UE2"] == 0
    assert by_bucket["UE1"] == 99


def test_global_repeat_solver_reroutes_shared_frames_when_repeats_are_required():
    """UE1 可用 A/B，UE2 只能用 A；cap=2、各两份时必须全局重排为 UE1→B、UE2→A。"""
    rows = []
    for index, bucket in enumerate(balance.SPECIAL_BUCKETS):
        if bucket in ("UE1", "UE2"):
            continue
        anchor = 100 + index
        rows.append(dict(
            scenario="S", run_id="R", anchor=anchor, route_group=f"S/{anchor}",
            event_balance_buckets=[bucket], event_balance_status=balance.SPECIAL_ELIGIBLE,
        ))
    rows.extend([
        dict(scenario="S", run_id="R", anchor=1, route_group="S/A",
             event_balance_buckets=["UE1", "UE2"], event_balance_status=balance.SPECIAL_ELIGIBLE),
        dict(scenario="S", run_id="R", anchor=2, route_group="S/B",
             event_balance_buckets=["UE1"], event_balance_status=balance.SPECIAL_ELIGIBLE),
    ])
    for anchor in (300, 301, 302, 303):
        rows.append(dict(
            scenario="S", run_id="R", anchor=anchor, route_group=f"S/{anchor}",
            event_balance_buckets=[], event_balance_status=balance.CONFIRMED_REGULAR,
        ))
    assert balance.event_balanced_total(rows, requested=24, repeat_cap=2, world=1) == 24
    sampled, audit = balance.build_event_balanced_epoch(rows, total=24, seed=11, repeat_cap=2)
    chosen = {
        bucket: [row["anchor"] for row in sampled if row["event_balance_bucket"] == bucket]
        for bucket in ("UE1", "UE2")
    }
    assert chosen["UE2"] == [1, 1]
    assert chosen["UE1"] == [2, 2]
    assert audit["bucket_max_frame_repeats"]["UE2"] == 2
    # UE1 原本有 A/B 两帧，但 A 被 UE2 独占后只能重复 B；检查实际使用次数。
    assert audit["bucket_max_frame_repeats"]["UE1"] == 2


def test_diversity_first_uses_distinct_frames_before_any_repeat_when_candidates_are_abundant():
    rows = []
    for bucket_index, bucket in enumerate(balance.SPECIAL_BUCKETS):
        for frame_index in range(100):
            anchor = bucket_index * 1000 + frame_index
            rows.append(dict(
                scenario="S", run_id=f"{bucket}_{frame_index}", anchor=anchor,
                route_group=f"S/{bucket}_{frame_index}", event_balance_buckets=[bucket],
                event_balance_status=balance.SPECIAL_ELIGIBLE,
            ))
    for frame_index in range(100):
        rows.append(dict(
            scenario="S", run_id=f"regular_{frame_index}", anchor=20000 + frame_index,
            route_group=f"S/regular_{frame_index}", event_balance_buckets=[],
            event_balance_status=balance.CONFIRMED_REGULAR,
        ))
    sampled, audit = balance.build_event_balanced_epoch(
        rows, total=96, seed=17, route_diverse=True, repeat_cap=8
    )
    assert audit["sampled"] == audit["quotas"]
    assert audit["unique_frames"] == 96
    assert audit["max_frame_repeats"] == 1
    assert audit["max_frame_repeats"] == 1
    assert audit["diversity_first"] is True


def test_sparse_bucket_repeat_does_not_open_repeat_capacity_for_abundant_buckets():
    """UE1 的单帧稀缺不能让其它有 100 个候选的桶也重复抽同一帧。"""
    rows = [dict(
        scenario="S", run_id="only_ue1", anchor=1, route_group="S/only_ue1",
        event_balance_buckets=["UE1"], event_balance_status=balance.SPECIAL_ELIGIBLE,
    )]
    for bucket_index, bucket in enumerate(balance.SPECIAL_BUCKETS[1:], start=1):
        for frame_index in range(100):
            rows.append(dict(
                scenario="S", run_id=f"{bucket}_{frame_index}",
                anchor=bucket_index * 1000 + frame_index,
                route_group=f"S/{bucket}_{frame_index}", event_balance_buckets=[bucket],
                event_balance_status=balance.SPECIAL_ELIGIBLE,
            ))
    for frame_index in range(100):
        rows.append(dict(
            scenario="S", run_id=f"regular_{frame_index}", anchor=20000 + frame_index,
            route_group=f"S/regular_{frame_index}", event_balance_buckets=[],
            event_balance_status=balance.CONFIRMED_REGULAR,
        ))
    sampled, audit = balance.build_event_balanced_epoch(rows, total=96, seed=23, repeat_cap=8)
    by_bucket = {
        bucket: [balance._identity(row) for row in sampled if row["event_balance_bucket"] == bucket]
        for bucket in audit["quotas"]
    }
    assert audit["sampled"] == audit["quotas"]
    assert audit["unique_frames"] == 89
    assert len(set(by_bucket["UE1"])) == 1
    assert all(len(set(by_bucket[bucket])) == audit["quotas"][bucket]
               for bucket in (*balance.SPECIAL_BUCKETS[1:], balance.REGULAR_BACKGROUND))
    assert audit["bucket_max_frame_repeats"]["UE1"] == 8
    assert all(audit["bucket_max_frame_repeats"][bucket] == 1
               for bucket in (*balance.SPECIAL_BUCKETS[1:], balance.REGULAR_BACKGROUND))


def test_stale_or_candidate_source_is_rejected(tmp_path):
    source = _write_source(tmp_path, mapping_hash="obsolete")
    with pytest.raises(ValueError, match="stale"):
        balance.EventBalanceIndex(source)
    candidate = tmp_path / "candidate_frames.jsonl"
    candidate.write_text("{}\n", encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError)):
        balance.EventBalanceIndex(candidate)


def test_v1_full_map_is_rejected_even_if_its_phase3_hash_is_current(tmp_path):
    source = _write_source(tmp_path)
    index = source.parent / "legacy_full_event_mapping.jsonl"
    legacy_rows = []
    for row in source.open(encoding="utf-8"):
        value = json.loads(row)
        value["schema"] = "action_prior_event_balance_full_v1"
        legacy_rows.append(json.dumps(value) + "\n")
    index.write_text("".join(legacy_rows), encoding="utf-8")
    manifest = json.loads((source.parent / "manifest.json").read_text())
    manifest.update(
        schema="action_prior_event_balance_full_manifest_v1",
        index_schema="action_prior_event_balance_full_v1",
        index_file=index.name,
        index_sha256=file_hash(index),
    )
    manifest.pop("mapping_policy", None)
    (source.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="v1/stale"):
        balance.EventBalanceIndex(index)


def test_quarantine_or_regular_gate_failure_cannot_be_confirmed_regular():
    from qwen3vl_local.action_prior.build_event_balance_index import (
        _re2_scene_state,
        _regular_confirmation_block_reasons,
    )

    assert "rgb_quarantine" in _regular_confirmation_block_reasons(
        (), {"rgb_quarantine": {"reason": "manual"}}, {"changes": []}, ("R-E3",)
    )
    assert "annotation_repair_or_gate" in _regular_confirmation_block_reasons(
        (), {}, {"changes": ["remove_generic_trigger_only_ramp_event"]}, ("R-E1",)
    )
    assert "unresolved_special_regular_context" in _regular_confirmation_block_reasons(
        (), {}, {"changes": []}, ("R-E3",)
    )
    # 当前 source_mapping 没有“绕障已完成”的帧级证据，绝不产生强 recovery claim。
    assert _re2_scene_state("Earlier ego encountered a static blockage in its normal path.") == "RE2_PRIOR_OBSTACLE"
    assert _re2_scene_state("No static-obstacle bypass history is asserted.") == "RE2_NAVIGATION_TRANSITION"


def test_source_identity_survives_path_move_and_args_stay_serializable(tmp_path):
    source = _write_source(tmp_path)
    moved_dir = tmp_path / "moved"
    shutil.copytree(source.parent, moved_dir)
    moved = moved_dir / source.name
    relocated_manifest = json.loads((moved_dir / "manifest.json").read_text())
    relocated_manifest["candidate_index"] = "/new/location/candidate_frames.jsonl"
    (moved_dir / "manifest.json").write_text(json.dumps(relocated_manifest), encoding="utf-8")
    first, second = balance.EventBalanceIndex(source), balance.EventBalanceIndex(moved)
    assert first.source.identity_dict() == second.source.identity_dict()
    assert first.source.audit_dict()["path"] != second.source.audit_dict()["path"]
    args = parser().parse_args([
        "--sampling-mode", "event_balanced", "--event-balance-index", str(source),
    ])
    validate_args(args)
    balance.source_for_args(args)
    json.dumps(vars(args))


def test_full_map_requires_the_exact_action_split_index(tmp_path):
    source = _write_source(tmp_path)
    data_dir = tmp_path / "action_data"
    data_dir.mkdir()
    for split, frame in zip(("train", "val", "test"), (0, 1, 2)):
        row = dict(_action_row(frame, split), schema="action_prior_data_v1")
        (data_dir / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest_path = source.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["action_dataset_hashes"] = {
        split: file_hash(data_dir / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    index = balance.EventBalanceIndex(source)
    index.validate_action_dataset(data_dir)
    (data_dir / "val.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different action dataset"):
        balance.EventBalanceIndex(source).validate_action_dataset(data_dir)


def test_scene_priors_and_sampling_require_a_full_map_not_candidate(tmp_path):
    with pytest.raises(ValueError, match="event-balance-index"):
        validate_args(parser().parse_args(["--sampling-mode", "event_balanced"]))
    source = _write_source(tmp_path)
    validate_args(parser().parse_args([
        "--sampling-mode", "event_balanced", "--event-balance-index", str(source),
    ]))
    with pytest.raises(ValueError, match="dataset-only privileged"):
        validate_args(parser().parse_args([
            "--event-balanced-scene-priors", "--event-balance-index", str(source),
        ]))


def test_closed_loop_style_pure_sampling_args_can_drop_the_old_index_path():
    args = parser().parse_args(["--sampling-mode", "event_balanced"])
    args.event_balance_source_identity = {"schema": balance.FULL_INDEX_SCHEMA, "sha256": "saved"}
    # carla_runtime/bench2drive deliberately clear event_balance_index but retain this identity.
    args.event_balance_index = ""
    validate_args(args)


def _diversity_row(bucket, frame, *, buckets=None):
    """构造每帧独立路线的候选，避免身份和路线混淆。"""
    return dict(
        scenario="S", run_id=f"{bucket}_{frame}", anchor=frame,
        route_group=f"S/{bucket}_{frame}",
        event_balance_buckets=list(buckets or ([bucket] if bucket != balance.REGULAR_BACKGROUND else [])),
        event_balance_status=(balance.CONFIRMED_REGULAR if bucket == balance.REGULAR_BACKGROUND
                              else balance.SPECIAL_ELIGIBLE),
    )


def test_seven_candidates_for_eight_presentations_cover_all_seven_frames():
    rows = []
    for bucket in (*balance.SPECIAL_BUCKETS, balance.REGULAR_BACKGROUND):
        count = 7 if bucket == "UE2" else 16
        rows.extend(_diversity_row(bucket, i) for i in range(count))
    sampled, audit = balance.build_event_balanced_epoch(rows, total=96, seed=42, repeat_cap=8)
    ue2 = [balance._identity(row) for row in sampled if row["event_balance_bucket"] == "UE2"]
    assert len(ue2) == 8 and len(set(ue2)) == 7
    assert audit["unique_frames"] == audit["optimal_unique_frames"] == 95
    assert audit["repeat_presentations"] == 1


@pytest.mark.parametrize("cap", [0, -1])
def test_direct_sampler_rejects_nonpositive_repeat_cap(cap):
    with pytest.raises(ValueError, match="must be positive"):
        balance.build_event_balanced_epoch([], total=12, seed=0, repeat_cap=cap)


def test_shared_candidates_are_not_repeated_across_buckets_when_distinct_assignment_exists():
    rows = [_diversity_row("shared", i, buckets=["UE1", "UE2"]) for i in range(16)]
    for bucket in (*balance.SPECIAL_BUCKETS[2:], balance.REGULAR_BACKGROUND):
        rows.extend(_diversity_row(bucket, i) for i in range(16))
    for seed in range(8):
        sampled, audit = balance.build_event_balanced_epoch(rows, total=96, seed=seed, repeat_cap=8)
        assert audit["unique_frames"] == audit["optimal_unique_frames"] == 96
        assert audit["repeat_presentations"] == 0
        assert audit["max_frame_repeats"] == 1
        assert audit["sampled"] == audit["quotas"]
        assert sampled == balance.build_event_balanced_epoch(rows, total=96, seed=seed, repeat_cap=8)[0]


def test_compressed_flow_matches_exhaustive_frame_allocation():
    """逐帧穷举小图作为独立 oracle，核对共享容量、可行性和全局唯一帧最优值。"""
    from itertools import product
    import random

    focus = balance.SPECIAL_BUCKETS[:3]
    for seed in range(30):
        rng = random.Random(seed)
        cap = 1 + seed % 3
        memberships = [tuple(i for i in range(3) if rng.random() < 0.65) for _ in range(5)]
        memberships = [m or (rng.randrange(3),) for m in memberships]
        if set().union(*map(set, memberships)) != {0, 1, 2}:
            continue
        states = {(0, 0, 0): 0}
        rows = []
        for frame, membership in enumerate(memberships):
            rows.append(_diversity_row("shared", frame, buckets=[focus[i] for i in membership]))
            allocations = [values for values in product(range(cap + 1), repeat=3)
                           if sum(values) <= cap and all(values[i] == 0 for i in range(3) if i not in membership)]
            next_states = {}
            for current, unique in states.items():
                for values in allocations:
                    target = tuple(a + b for a, b in zip(current, values))
                    if max(target) <= 2:
                        next_states[target] = max(next_states.get(target, -1), unique + int(any(values)))
            states = next_states
        for bucket in (*balance.SPECIAL_BUCKETS[3:], balance.REGULAR_BACKGROUND):
            count = 4 if bucket == balance.REGULAR_BACKGROUND else 2
            rows.extend(_diversity_row(bucket, i) for i in range(count))
        optimum = states.get((2, 2, 2))
        if optimum is None:
            with pytest.raises(ValueError, match="infeasible"):
                balance.build_event_balanced_epoch(rows, total=24, seed=seed, repeat_cap=cap)
        else:
            _, audit = balance.build_event_balanced_epoch(rows, total=24, seed=seed, repeat_cap=cap)
            assert audit["sampled"] == audit["quotas"]
            assert audit["max_frame_repeats"] <= cap
            assert audit["unique_frames"] == audit["optimal_unique_frames"] == optimum + 18
