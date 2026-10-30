"""Phase2 训练/验证客观证据过滤（revision ``evidence_20261030``）；test 逐字节保留。

2026-10-30 对 Phase2 新训练错例逐帧回放 LEAD metas/bboxes 后确认三类标定问题：

- ``HardBreakRoute`` 的 U-E1 窗口按场景激活期整段标 YES。训练池里 67% 的 UE1 正例是
  两车早已停稳的静止队列，9% 的正例里前车匀速、无刹车、距离还在拉大；模型因此把
  UE1 学成“前方有停住的车”，真正正在急刹的帧反而漏检。其它场景中前车明显刹车或刚停住
  的帧又被整体标成 NO，与 HardBreakRoute 的 YES 互相矛盾。
- ``OppositeVehicleRunningRedLight`` 的 U-E6 窗口同样过宽：33% 的训练正例里违规车尚未出现、
  已从车后经过或早已驶离视野。
- 高速切入 v1 训练正例实际是左侧匝道车并行汇入相邻车道，客观上从未进入 ego 车道。

同日复核补充（见 EVIDENCE_20261030.md 第八节）：

- 选优用的 val 与训练口径必须一致，否则学会新口径的 checkpoint 会被旧 val 标签淘汰。
  因此 **train 与 val** 都按同一客观策略排除；val 原行另存 ``val_original.jsonl``，排除明细
  写 ``val_exclusions.jsonl``；test 逐字节不变，其争议帧只读列入 ``holdout_disputes.jsonl``
  供双口径评分。
- “前车停着、ego 还在动”不再自动视为急刹证据：前车停稳超过两秒后，只有 ego 本身仍在
  明显减速（窗口内降速 ≥2 m/s）且距离 ≤20 m 才保留，否则按“接近静止车辆”排除。
- metas/bboxes 文件缺失或无法解析是数据缺口，不是“没有目标”：派生直接阻断并列出缺项。
- 实际读取的每个 meta/bbox 文件写入 ``evidence_sources.json``（路径→SHA256），供两机重放核对。

本模块不翻转任何标签；每条排除都附客观数值。它不是对剩余标签的视觉认证，也不代表
测试集标签已经修正。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import CONTRACT

REVISION = "evidence_20261030"
POLICY = {
    "revision": REVISION,
    "policy_version": "evidence_policy_v3_20261030",
    "history_source_policy": "require_every_read_nonnegative_frame",
    "min_frame_id": 4,
    "lead_lookback_frames": 8,
    "lead_brake_signal": 0.3,
    "lead_speed_drop_mps": 1.5,
    "lead_stopped_mps": 0.5,
    "recent_stop_frames": 8,
    "ego_response_speed_drop_mps": 2.0,
    "ego_response_max_distance_m": 20.0,
    "steady_far_m": 25.0,
    "lead_lateral_m": 1.6,
    "lead_max_ahead_m": 45.0,
    "ue6_conflict_lateral_m": 6.0,
    "ue6_conflict_ahead_m": 30.0,
    "ue6_recent_frames": 8,
    "ue6_approach_lookback": 4,
    "ue6_approach_min_dy": 0.5,
    "ue6_far_lateral_m": 25.0,
    "ue6_far_ahead_m": 35.0,
    "highway_onset_max_abs_y": 3.3,
    "highway_onset_min_yaw": 5.0,
    "highway_inside_abs_y": 2.6,
    "ue1_keep": ["lead_brake_signal", "lead_speed_drop", "lead_stopped_ego_closing", "lead_stopped_ego_still_braking",
                 "recent_stop_both_stopped"],
    "ue6_keep": ["actor_in_conflict_zone", "actor_just_cleared", "actor_approaching"],
    "filtered_splits": ["train", "val"],
    "byte_identical_splits": ["test"],
}
VEHICLES = ("car", "truck", "bus", "van", "motorcycle", "bicycle")
MODES = ("2rgb_endpoints", "4rgb")


class MissingSource(Exception):
    """一个必需的 meta/bbox 文件缺失或无法解析；派生必须阻断而不是当作负证据。"""


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


class LeadStore:
    """带缓存的 metas/bboxes 读取；记录每个实际读取文件的 SHA256，缺失/损坏抛 MissingSource。"""

    def __init__(self, data_root: Path):
        self.root = Path(data_root)
        self._meta: Dict[Tuple[str, str, int], Any] = {}
        self._box: Dict[Tuple[str, str, int], Any] = {}
        self.sources: Dict[str, str] = {}
        self.missing: List[str] = []

    def _load(self, path: Path, *, required: bool) -> Any:
        rel = str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)
        if not path.is_file():
            if required:
                self.missing.append(rel)
                raise MissingSource(rel)
            return None
        try:
            with lzma.open(path, "rb") as f:
                data = f.read()
        except Exception:
            data = path.read_bytes()
        self.sources[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            obj = pickle.loads(data)
        except Exception as exc:
            self.missing.append(rel)
            raise MissingSource(f"{rel}: {type(exc).__name__}") from exc
        return obj

    def _cached(self, cache: Dict[Tuple[str, str, int], Any], key: Tuple[str, str, int], path: Path, required: bool) -> Any:
        if key not in cache:
            cache[key] = self._load(path, required=required)
        value = cache[key]
        if value is None and required:
            # 先前作为可选历史帧读取过的缺失文件，此时作为当前帧必需输入同样必须阻断。
            rel = str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)
            self.missing.append(rel)
            raise MissingSource(rel)
        return value

    def meta(self, scenario: str, route: str, frame: int, *, required: bool = True) -> Optional[Dict[str, Any]]:
        return self._cached(self._meta, (scenario, route, frame), self.root / scenario / route / "metas" / f"{frame:04d}.pkl", required)

    def boxes(self, scenario: str, route: str, frame: int, *, required: bool = True) -> Optional[List[Dict[str, Any]]]:
        return self._cached(self._box, (scenario, route, frame), self.root / scenario / route / "bboxes" / f"{frame:04d}.pkl", required)


def _lead_actor(meta: Dict[str, Any], boxes: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """优先使用采集器记录的影响车辆；否则取 ego 车道正前方最近车辆。"""

    for key in ("vehicle_affecting_id", "speed_reduced_by_obj_id"):
        aid = meta.get(key)
        if aid not in (None, -1):
            actor = next((b for b in boxes if b.get("id") == aid and b.get("class") in VEHICLES), None)
            if actor is not None:
                return actor
    ahead = [b for b in boxes if b.get("class") in VEHICLES and 0.0 < float(b["position"][0]) < POLICY["lead_max_ahead_m"]
             and abs(float(b["position"][1])) < POLICY["lead_lateral_m"]]
    return min(ahead, key=lambda b: float(b["position"][0])) if ahead else None


def _finite(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def ue1_evidence(store: LeadStore, scenario: str, route: str, frame: int) -> Tuple[str, Dict[str, Any]]:
    """最近两秒内前车是否真的在刹车/刚停下；返回类别与客观数值。

    仅跳过路线开始前的负帧号；实际读取的非负历史帧与当前帧均必须存在。
    """

    meta, boxes = store.meta(scenario, route, frame), store.boxes(scenario, route, frame)
    lead = _lead_actor(meta, boxes)
    if lead is None:
        return "no_lead", {}
    aid = lead.get("id")
    speeds: List[float] = []
    brakes: List[float] = []
    ego: List[float] = []
    for g in range(frame - POLICY["lead_lookback_frames"], frame + 1):
        if g < 0:
            continue
        m, b = store.meta(scenario, route, g), store.boxes(scenario, route, g)
        ego.append(_finite(m.get("speed")) or 0.0)
        actor = next((x for x in b if x.get("id") == aid), None)
        if actor is None:
            continue
        s, br = _finite(actor.get("speed")), _finite(actor.get("brake"))
        if s is not None:
            speeds.append(s)
        if br is not None:
            brakes.append(br)
    dist = float(lead["position"][0])
    ego_drop = round(max(ego) - ego[-1], 2) if ego else None
    info = {"lead_id": aid, "distance_m": round(dist, 2), "lead_speed_mps": round(speeds[-1], 2) if speeds else None,
            "ego_speed_mps": round(ego[-1], 2) if ego else None, "ego_speed_drop_mps": ego_drop,
            "lead_speed_series": [round(s, 2) for s in speeds], "lead_brake_series": [round(b, 2) for b in brakes]}
    if speeds and speeds[-1] < POLICY["lead_stopped_mps"]:
        stopped = 0
        for g in range(frame, frame - 2 * POLICY["recent_stop_frames"] - 1, -1):
            if g < 0:
                break
            b = store.boxes(scenario, route, g)
            actor = next((x for x in b if x.get("id") == aid), None) if b else None
            if actor is None or (_finite(actor.get("speed")) or 0.0) >= POLICY["lead_stopped_mps"]:
                break
            stopped += 1
        info["lead_stopped_frames"] = stopped
        recent = stopped <= POLICY["recent_stop_frames"]
        if ego and ego[-1] > POLICY["lead_stopped_mps"]:
            if recent:
                return "lead_stopped_ego_closing", info
            # 前车停稳已超两秒：只有 ego 本身仍在明显减速（仍在应对那次急停）且距离很近才算事件持续；
            # 否则就是普通地接近一辆静止车辆，图像上与停车/队列无法区分。
            if ego_drop is not None and ego_drop >= POLICY["ego_response_speed_drop_mps"] and dist <= POLICY["ego_response_max_distance_m"]:
                return "lead_stopped_ego_still_braking", info
            return "approaching_long_stopped_lead", info
        return ("recent_stop_both_stopped" if recent else "long_both_stopped"), info
    if any(b >= POLICY["lead_brake_signal"] for b in brakes):
        return "lead_brake_signal", info
    if len(speeds) >= 2 and max(speeds) - speeds[-1] >= POLICY["lead_speed_drop_mps"]:
        return "lead_speed_drop", info
    return ("steady_far" if dist > POLICY["steady_far_m"] else "steady_lead"), info


def ue6_evidence(store: LeadStore, scenario: str, route: str, frame: int) -> Tuple[str, Dict[str, Any]]:
    """场景违规车相对 ego 的可见冲突状态；当前帧 meta/bbox 缺失即阻断。"""

    meta = store.meta(scenario, route, frame)
    boxes = store.boxes(scenario, route, frame)
    ids = set(meta.get("scenario_actors_ids") or [])
    actors = [b for b in boxes if b.get("id") in ids]
    info: Dict[str, Any] = {"traffic_light_state": meta.get("traffic_light_state"), "ego_speed_mps": round(_finite(meta.get("speed")) or 0.0, 2),
                            "scenario_actor_ids": sorted(ids)}
    if not actors:
        return "actor_absent", info
    actor = actors[0]
    x, y = float(actor["position"][0]), float(actor["position"][1])
    info.update(actor_id=actor.get("id"), x_m=round(x, 1), y_m=round(y, 1), actor_speed_mps=round(_finite(actor.get("speed")) or 0.0, 1))
    lateral_history: List[float] = []
    recent = False
    for g in range(frame - POLICY["ue6_recent_frames"], frame):
        if g < 0:
            continue
        b = store.boxes(scenario, route, g)
        a = next((v for v in b if v.get("id") == actor.get("id")), None)
        if a is None:
            continue
        ax, ay = float(a["position"][0]), float(a["position"][1])
        if g >= frame - POLICY["ue6_approach_lookback"]:
            lateral_history.append(abs(ay))
        if -2.0 < ax < 25.0 and abs(ay) < POLICY["ue6_conflict_lateral_m"]:
            recent = True
    info["actor_abs_y_history"] = [round(v, 1) for v in lateral_history]
    approaching = bool(lateral_history) and lateral_history[0] - abs(y) > POLICY["ue6_approach_min_dy"]
    if x < -2.0:
        return "actor_behind_ego", info
    if abs(y) <= POLICY["ue6_conflict_lateral_m"] and x < POLICY["ue6_conflict_ahead_m"]:
        return "actor_in_conflict_zone", info
    if recent:
        return "actor_just_cleared", info
    if approaching and x < POLICY["ue6_far_ahead_m"] and abs(y) <= POLICY["ue6_far_lateral_m"]:
        return "actor_approaching", info
    if x >= POLICY["ue6_far_ahead_m"] or abs(y) > POLICY["ue6_far_lateral_m"]:
        return "actor_far", info
    return "actor_departed", info


def highway_evidence(store: LeadStore, scenario: str, route: str, frame: int) -> Tuple[str, Dict[str, Any]]:
    """高速切入正例必须有切入车且其正在进入/已在 ego 走廊；相邻车道并行不算。"""

    meta = store.meta(scenario, route, frame)
    ids = list(meta.get("cut_in_actors_ids") or [])
    boxes = store.boxes(scenario, route, frame)
    actor = next((b for b in boxes if ids and b.get("id") == ids[0]), None)
    if actor is None:
        return "no_cut_in_actor", {"cut_in_actor_ids": ids}
    x, y = float(actor["position"][0]), float(actor["position"][1])
    yaw = math.degrees(float(actor.get("yaw") or 0.0))
    toward = yaw * (-1.0 if y > 0 else 1.0)
    info = {"actor_id": actor.get("id"), "x_m": round(x, 1), "y_m": round(y, 2), "yaw_deg": round(yaw, 1)}
    if x <= 0.0:
        return "cut_in_actor_behind", info
    if abs(y) < POLICY["highway_inside_abs_y"] or (abs(y) < POLICY["highway_onset_max_abs_y"] and toward >= POLICY["highway_onset_min_yaw"]):
        return "cut_in_entering_or_inside", info
    return "adjacent_lane_only", info


def _is_valid_road_regular(row: Dict[str, Any]) -> bool:
    answers = row["answers"]
    return (row["question_domain"] == "ROAD_CORRIDOR" and not row.get("invalid_event_context")
            and not any(answers.get(k) for k in ("UE1", "UE3", "UE5", "UE6")))


def classify(store: LeadStore, row: Dict[str, Any]) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """返回 (规则, 证据类别, 数值) 表示该行应被排除；None 表示保留。缺源文件抛 MissingSource。"""

    scenario, route, frame = row["scenario"], row["route_id"], int(row["frame_id"])
    if frame < POLICY["min_frame_id"]:
        return "route_start_padding", "history_left_padded", {"frame_id": frame}
    answers = row["answers"]
    invalid = bool(row.get("invalid_event_context"))
    if answers.get("UE1") and not invalid:
        kind, info = ue1_evidence(store, scenario, route, frame)
        if kind not in POLICY["ue1_keep"]:
            return "ue1_positive_without_braking_evidence", kind, info
    elif _is_valid_road_regular(row):
        kind, info = ue1_evidence(store, scenario, route, frame)
        if kind in POLICY["ue1_keep"]:
            return "regular_negative_with_lead_braking", kind, info
    if answers.get("UE6") and not invalid:
        kind, info = ue6_evidence(store, scenario, route, frame)
        if kind not in POLICY["ue6_keep"]:
            return "ue6_positive_without_visible_violator", kind, info
    if answers.get("UE3") and not invalid and row.get("ue3_subtype") == "HIGHWAY_CUTIN":
        kind, info = highway_evidence(store, scenario, route, frame)
        if kind != "cut_in_entering_or_inside":
            return "highway_ue3_positive_without_corridor_entry", kind, info
    return None


def target_class(row: Dict[str, Any]) -> str:
    if row.get("invalid_event_context"):
        return "INVALID"
    for key in ("UE1", "UE3", "UE5", "UE6"):
        if row["answers"].get(key):
            return key
    return "RE"


def _record(row: Dict[str, Any], rule: str, kind: str, info: Dict[str, Any]) -> Dict[str, Any]:
    return {"split": row["split"], "scenario": row["scenario"], "route_id": row["route_id"], "frame_id": int(row["frame_id"]),
            "question_domain": row["question_domain"], "invalid_source": row.get("invalid_source", ""),
            "target_event_class": target_class(row), "rule": rule, "evidence": kind, **info}


def derive(index: Path, data_root: Path, output: Path) -> Dict[str, Any]:
    index, output, data_root = Path(index).resolve(), Path(output).resolve(), Path(data_root).resolve()
    if output.exists() or index.is_relative_to(output) or output.is_relative_to(index.parent):
        raise ValueError("Choose a fresh separate output directory")
    if output.is_relative_to(data_root) or data_root.is_relative_to(output):
        raise ValueError("Output overlaps the RGB/meta root")
    parent_sha, manifest_sha = sha(index), sha(index.parent / "manifest.json")
    raw = index.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in raw]
    store = LeadStore(data_root)
    kept: List[bytes] = []
    exclusions: Dict[str, List[Dict[str, Any]]] = {s: [] for s in POLICY["filtered_splits"]}
    disputes: List[Dict[str, Any]] = []
    reasons: Dict[str, Counter] = {s: Counter() for s in POLICY["filtered_splits"]}
    before: Dict[str, Counter] = {s: Counter() for s in POLICY["filtered_splits"]}
    after: Dict[str, Counter] = {s: Counter() for s in POLICY["filtered_splits"]}
    originals: Dict[str, List[bytes]] = {s: [] for s in POLICY["filtered_splits"]}
    test_hash, test_hash_after = hashlib.sha256(), hashlib.sha256()
    missing: List[str] = []
    for line, row in zip(raw, rows):
        split = row["split"]
        if split in POLICY["byte_identical_splits"]:
            test_hash.update(line)
            test_hash_after.update(line)
            kept.append(line)
            try:
                verdict = classify(store, row)
            except MissingSource as exc:
                missing.append(str(exc))
                continue
            if verdict is not None:
                disputes.append(_record(row, *verdict))
            continue
        if split not in exclusions:
            raise ValueError("Unexpected split in parent index: " + split)
        originals[split].append(line)
        cls = target_class(row)
        before[split][cls] += 1
        try:
            verdict = classify(store, row)
        except MissingSource as exc:
            missing.append(str(exc))
            continue
        if verdict is None:
            kept.append(line)
            after[split][cls] += 1
            continue
        rule, kind, info = verdict
        reasons[split][f"{rule}/{kind}"] += 1
        exclusions[split].append(_record(row, rule, kind, info))
    if missing:
        unique = sorted(set(missing))
        raise ValueError(f"{len(unique)} required LEAD meta/bbox sources are missing or unreadable; refusing to treat "
                         f"absence as negative evidence. First entries: {unique[:10]}")
    required = ("UE1", "UE3", "UE5", "UE6", "RE", "INVALID")
    for split in POLICY["filtered_splits"]:
        empty = [k for k in required if after[split].get(k, 0) <= 0]
        if empty:
            raise ValueError(f"Evidence filter emptied a required {split} class: " + ",".join(empty))
    if (sha(index), sha(index.parent / "manifest.json")) != (parent_sha, manifest_sha):
        raise ValueError("Input changed during derivation")
    sources_digest = hashlib.sha256(json.dumps(store.sources, sort_keys=True).encode()).hexdigest()
    receipt = dict(contract=CONTRACT, revision=REVISION, status="derived_model_pending", policy=POLICY,
                   parent_sha256=parent_sha, parent_manifest_sha256=manifest_sha,
                   rows_before=len(rows), rows_after=len(kept),
                   train_excluded=len(exclusions["train"]), val_excluded=len(exclusions["val"]),
                   exclusion_reasons=dict(sorted(reasons["train"].items())),
                   val_exclusion_reasons=dict(sorted(reasons["val"].items())),
                   train_class_counts_before=dict(sorted(before["train"].items())), train_class_counts_after=dict(sorted(after["train"].items())),
                   val_class_counts_before=dict(sorted(before["val"].items())), val_class_counts_after=dict(sorted(after["val"].items())),
                   test_disputes=len(disputes), test_dispute_rules=dict(sorted(Counter(d["rule"] for d in disputes).items())),
                   labels_flipped=0, test_byte_identical=True, val_byte_identical=False,
                   val_original_rows=len(originals["val"]), test_sha256=test_hash.hexdigest(),
                   evidence_sources=len(store.sources), evidence_sources_sha256=sources_digest,
                   training_sampler_changed=False, training_quota=1024, max_frame_repeat=10, model_inference=False,
                   note="Objective exclusion on train and val from LEAD metas/bboxes; no label flips; test untouched; "
                        "val is the selection view consistent with the training policy, originals kept in val_original.jsonl; "
                        "not a visual certification of retained labels.")
    assert receipt["test_sha256"] == test_hash_after.hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    (output / "frame_index.jsonl").write_bytes(b"".join(kept))
    (output / "val_original.jsonl").write_bytes(b"".join(originals["val"]))
    for name, items in (("evidence_exclusions.jsonl", exclusions["train"]), ("val_exclusions.jsonl", exclusions["val"]),
                        ("holdout_disputes.jsonl", disputes)):
        (output / name).write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in items), encoding="utf-8")
    (output / "evidence_sources.json").write_text(json.dumps({"data_root": str(data_root), "sha256": sources_digest,
                                                                "files": dict(sorted(store.sources.items()))}, indent=1) + "\n")
    manifest = json.loads((index.parent / "manifest.json").read_text())
    parent_stats = {k: manifest.pop(k) for k in ("sampling", "route_counts", "answer_counts", "visual_label_risk_counts", "counts") if k in manifest}
    counts: Counter = Counter()
    for line in kept:
        row = json.loads(line)
        counts[f"frames/{row['split']}"] += 1
        counts[f"frames/{row['split']}/{target_class(row)}"] += 1
    manifest.update(frame_index=str(output / "frame_index.jsonl"), counts=dict(sorted(counts.items())),
                    fresh_training_revision=receipt, parent_statistics=parent_stats)
    for name, value in (("manifest.json", manifest), ("training_revision.json", receipt), ("preservation.json", receipt)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    summary = dict(contract=CONTRACT, revision=REVISION, policy=POLICY, exclusion_reasons=receipt["exclusion_reasons"],
                   val_exclusion_reasons=receipt["val_exclusion_reasons"], test_dispute_rules=receipt["test_dispute_rules"],
                   note="Objective evidence policy; see evidence_exclusions.jsonl / val_exclusions.jsonl / holdout_disputes.jsonl.")
    (output.parent / "selected_train_reviews.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return receipt


def audit(index: Path, data_root: Path, splits: Sequence[str] = ("val", "test")) -> List[Dict[str, Any]]:
    """只读：列出 holdout 中按同一证据口径可疑的标签；缺源文件记为 source_missing，不改评分。"""

    store = LeadStore(Path(data_root))
    out = []
    for line in Path(index).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["split"] not in splits:
            continue
        try:
            verdict = classify(store, row)
        except MissingSource as exc:
            out.append(_record(row, "source_missing", "missing_meta_or_bbox", {"missing": str(exc)}))
            continue
        if verdict is not None:
            out.append(_record(row, *verdict))
    return out


def _case_key(case: Dict[str, Any]) -> Tuple[str, str, int, str]:
    return (str(case["scenario"]), str(case["route_id"]), int(case["frame_id"]), str(case["question_domain"]))


def _case_class(case: Dict[str, Any]) -> str:
    gt = case["gt"]
    if gt.get("INVALID_EVENT_CONTEXT") == "YES":
        return "INVALID"
    for key in ("UE1", "UE3", "UE5", "UE6"):
        if gt.get(key) == "YES":
            return key
    return "RE"


def _production_cases(pipeline_root: Path, mode: str):
    """Read exactly one native production output, preserving sampled duplicates.

    Native evaluation already replays raw answers before issuing its completion
    receipt. Check that receipt's bound inputs, metrics and shards here, without
    recursively counting the unpacked audit bundle beside the original output.
    """
    from qwen3vl_local.sft_new_loop_phase1 import evaluation_integrity as evaluation
    from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof

    folder = pipeline_root / mode / 'eval_review' / 'lora_production'
    try:
        metrics = proof.read(folder / 'metrics.json')
        inputs = proof.read(folder / 'evaluation_inputs.json')
        complete = proof.read(folder / 'evaluation_complete.json')
    except (OSError, ValueError) as exc:
        raise ValueError(f'Missing or unreadable native production evidence: {folder}') from exc
    world = metrics.get('world_size')
    if type(world) is not int or world < 1:
        raise ValueError(f'Invalid production world_size: {folder}')
    if (metrics.get('history_rgb_mode') != mode or inputs.get('history_rgb_mode') != mode
            or inputs.get('phase') != 2 or inputs.get('schema') != proof.SCHEMA
            or complete.get('schema') != proof.SCHEMA or complete.get('phase') != 2
            or complete.get('status') != 'verified'):
        raise ValueError(f'Production mode/phase/completion mismatch: {folder}')
    cases, names = evaluation.read_cases(folder, world)
    hashes = {name: sha(folder / name) for name in names}
    if (complete.get('case_files_sha256') != hashes
            or complete.get('metrics_sha256') != sha(folder / 'metrics.json')
            or complete.get('inputs_sha256') != sha(folder / 'evaluation_inputs.json')):
        raise ValueError(f'Production evidence changed since native verification: {folder}')
    n = len(cases)
    if (n == 0 or any(type(c.get('all_ok')) is not bool for c in cases)
            or any(type(v) is not int or v != n for v in
                   (metrics.get('total_cases'), inputs.get('expected_total'), complete.get('total_cases')))):
        raise ValueError(f'Production case count or score type mismatch: {folder}')
    exact = sum(c['all_ok'] for c in cases) / n
    if metrics.get('exact_match_accuracy') != exact or complete.get('scores') != evaluation.score_fields(metrics):
        raise ValueError(f'Production scores disagree with native receipt: {folder}')
    return cases, [folder / name for name in names], {
        'case_files_sha256': hashes,
        'metrics_sha256': complete['metrics_sha256'],
        'inputs_sha256': complete['inputs_sha256'],
        'completion_sha256': sha(folder / 'evaluation_complete.json'),
    }


def dual_score(pipeline_root: Path, disputes: Path, output: Path) -> Dict[str, Any]:
    """原评分照报，另给剔除预先冻结争议帧后的评分；只读原生 production 逐题输出。"""

    pipeline_root, disputes, output = Path(pipeline_root), Path(disputes), Path(output)
    flagged: Dict[Tuple[str, str, int, str], str] = {}
    for line in disputes.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            flagged[(d["scenario"], d["route_id"], int(d["frame_id"]), d["question_domain"])] = f"{d['rule']}/{d['evidence']}"
    report: Dict[str, Any] = {"contract": CONTRACT, "revision": REVISION, "disputes_sha256": sha(disputes),
                              "disputed_keys": len(flagged), "modes": {}}
    for mode in MODES:
        cases, mode_files, binding = _production_cases(pipeline_root, mode)
        total = Counter()
        correct = Counter()
        retained = Counter()
        retained_correct = Counter()
        disputed = Counter()
        disputed_correct = Counter()
        for c in cases:
            cls = _case_class(c)
            ok = bool(c.get("all_ok"))
            total[cls] += 1
            correct[cls] += ok
            if _case_key(c) in flagged:
                disputed[cls] += 1
                disputed_correct[cls] += ok
            else:
                retained[cls] += 1
                retained_correct[cls] += ok
        n, k = sum(total.values()), sum(correct.values())
        rn, rk = sum(retained.values()), sum(retained_correct.values())
        report["modes"][mode] = {
            "case_files": [str(p) for p in mode_files],
            "native_evidence": binding,
            "original": {"correct": k, "total": n, "exact": (k / n) if n else None, "by_class": {c: [correct[c], total[c]] for c in sorted(total)}},
            "retained_after_removing_disputed": {"correct": rk, "total": rn, "exact": (rk / rn) if rn else None,
                                                 "by_class": {c: [retained_correct[c], retained[c]] for c in sorted(retained)}},
            "disputed_subset": {"correct": sum(disputed_correct.values()), "total": sum(disputed.values()),
                                "by_class": {c: [disputed_correct[c], disputed[c]] for c in sorted(disputed)}},
            "note": "Original score is the primary report. The retained score removes frames frozen before training by the "
                    "objective dispute audit; it does not prove those labels wrong and must not be used to select checkpoints.",
        }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent")
    p.add_argument("--data-root")
    p.add_argument("--output", help="fresh output directory for the derived index")
    p.add_argument("--audit-holdouts", help="write a read-only holdout dispute list (JSONL) instead of deriving")
    p.add_argument("--dual-score", nargs=3, metavar=("PIPELINE_ROOT", "DISPUTES_JSONL", "OUTPUT_JSON"),
                   help="recompute native production scores with and without the frozen disputed frames")
    a = p.parse_args(argv)
    if a.dual_score:
        print(json.dumps({"dual_score": str(a.dual_score[2]), **{m: v["original"] for m, v in dual_score(*map(Path, a.dual_score))["modes"].items()}},
                         ensure_ascii=False), flush=True)
        return
    if not a.parent or not a.data_root:
        p.error("--parent and --data-root are required")
    if a.audit_holdouts:
        items = audit(Path(a.parent), Path(a.data_root))
        Path(a.audit_holdouts).write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
        print(json.dumps({"holdout_disputes": len(items), "by_rule": dict(Counter(i["rule"] for i in items))}, ensure_ascii=False), flush=True)
        return
    if not a.output:
        p.error("--output is required unless --audit-holdouts or --dual-score is given")
    print(json.dumps(derive(Path(a.parent), Path(a.data_root), Path(a.output)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
