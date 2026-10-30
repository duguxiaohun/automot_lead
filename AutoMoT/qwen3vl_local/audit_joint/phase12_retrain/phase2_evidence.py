"""Phase2 训练集客观证据过滤（revision ``evidence_20261030``）；holdout 逐字节保留。

2026-10-30 对 Phase2 新训练错例逐帧回放 LEAD metas/bboxes 后确认三类标定问题：

- ``HardBreakRoute`` 的 U-E1 窗口按场景激活期整段标 YES。训练池里 67% 的 UE1 正例是
  两车早已停稳的静止队列，9% 的正例里前车匀速、无刹车、距离还在拉大；模型因此把
  UE1 学成“前方有停住的车”，真正正在急刹的帧反而漏检。其它场景中前车明显刹车或刚停住
  的帧又被整体标成 NO，与 HardBreakRoute 的 YES 互相矛盾。
- ``OppositeVehicleRunningRedLight`` 的 U-E6 窗口同样过宽：33% 的训练正例里违规车尚未出现、
  已从车后经过或早已驶离视野。
- 高速切入 v1 训练正例实际是左侧匝道车并行汇入相邻车道，客观上从未进入 ego 车道。

本模块只对 **train** 行做证据过滤，不翻转任何标签、不改 val/test；每条排除都附
客观数值并写入 ``evidence_exclusions.jsonl``。排除后六类桶仍须齐全，训练采样器的 1024 配额
与单帧 10 次上限不变。它不是对剩余标签的视觉认证，也不代表测试集标签已经修正。
"""

from __future__ import annotations

import argparse
import copy
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
    "min_frame_id": 4,
    "lead_lookback_frames": 8,
    "lead_brake_signal": 0.3,
    "lead_speed_drop_mps": 1.5,
    "lead_stopped_mps": 0.5,
    "recent_stop_frames": 8,
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
    "ue1_keep": ["lead_brake_signal", "lead_speed_drop", "lead_stopped_ego_closing", "recent_stop_both_stopped"],
    "ue6_keep": ["actor_in_conflict_zone", "actor_just_cleared", "actor_approaching"],
}
VEHICLES = ("car", "truck", "bus", "van", "motorcycle", "bicycle")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


class LeadStore:
    """带缓存的 metas/bboxes 读取；缺文件返回 None，不抛错。"""

    def __init__(self, data_root: Path):
        self.root = Path(data_root)
        self._meta: Dict[Tuple[str, str, int], Any] = {}
        self._box: Dict[Tuple[str, str, int], Any] = {}

    @staticmethod
    def _load(path: Path) -> Any:
        if not path.is_file():
            return None
        try:
            with lzma.open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            with path.open("rb") as f:
                return pickle.load(f)

    def meta(self, scenario: str, route: str, frame: int) -> Optional[Dict[str, Any]]:
        key = (scenario, route, frame)
        if key not in self._meta:
            self._meta[key] = self._load(self.root / scenario / route / "metas" / f"{frame:04d}.pkl")
        return self._meta[key]

    def boxes(self, scenario: str, route: str, frame: int) -> Optional[List[Dict[str, Any]]]:
        key = (scenario, route, frame)
        if key not in self._box:
            self._box[key] = self._load(self.root / scenario / route / "bboxes" / f"{frame:04d}.pkl")
        return self._box[key]


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
    """最近两秒内前车是否真的在刹车/刚停下；返回类别与客观数值。"""

    meta, boxes = store.meta(scenario, route, frame), store.boxes(scenario, route, frame)
    if meta is None or boxes is None:
        return "no_meta", {}
    lead = _lead_actor(meta, boxes)
    if lead is None:
        return "no_lead", {}
    aid = lead.get("id")
    speeds: List[float] = []
    brakes: List[float] = []
    ego: List[float] = []
    for g in range(frame - POLICY["lead_lookback_frames"], frame + 1):
        m, b = store.meta(scenario, route, g), store.boxes(scenario, route, g)
        if m is None or b is None:
            continue
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
    info = {"lead_id": aid, "distance_m": round(dist, 2), "lead_speed_mps": round(speeds[-1], 2) if speeds else None,
            "ego_speed_mps": round(ego[-1], 2) if ego else None}
    if speeds and speeds[-1] < POLICY["lead_stopped_mps"]:
        stopped = 0
        for g in range(frame, frame - 2 * POLICY["recent_stop_frames"] - 1, -1):
            b = store.boxes(scenario, route, g)
            actor = next((x for x in b if x.get("id") == aid), None) if b else None
            if actor is None or (_finite(actor.get("speed")) or 0.0) >= POLICY["lead_stopped_mps"]:
                break
            stopped += 1
        info["lead_stopped_frames"] = stopped
        if ego and ego[-1] > POLICY["lead_stopped_mps"]:
            return "lead_stopped_ego_closing", info
        return ("recent_stop_both_stopped" if stopped <= POLICY["recent_stop_frames"] else "long_both_stopped"), info
    if any(b >= POLICY["lead_brake_signal"] for b in brakes):
        return "lead_brake_signal", info
    if len(speeds) >= 2 and max(speeds) - speeds[-1] >= POLICY["lead_speed_drop_mps"]:
        return "lead_speed_drop", info
    return ("steady_far" if dist > POLICY["steady_far_m"] else "steady_lead"), info


def ue6_evidence(store: LeadStore, scenario: str, route: str, frame: int) -> Tuple[str, Dict[str, Any]]:
    """场景违规车相对 ego 的可见冲突状态。"""

    meta = store.meta(scenario, route, frame)
    if meta is None:
        return "no_meta", {}
    boxes = store.boxes(scenario, route, frame) or []
    ids = set(meta.get("scenario_actors_ids") or [])
    actors = [b for b in boxes if b.get("id") in ids]
    info: Dict[str, Any] = {"traffic_light_state": meta.get("traffic_light_state"), "ego_speed_mps": round(_finite(meta.get("speed")) or 0.0, 2)}
    if not actors:
        return "actor_absent", info
    actor = actors[0]
    x, y = float(actor["position"][0]), float(actor["position"][1])
    info.update(actor_id=actor.get("id"), x_m=round(x, 1), y_m=round(y, 1), actor_speed_mps=round(_finite(actor.get("speed")) or 0.0, 1))
    lateral_history: List[float] = []
    recent = False
    for g in range(frame - POLICY["ue6_recent_frames"], frame):
        b = store.boxes(scenario, route, g) or []
        a = next((v for v in b if v.get("id") == actor.get("id")), None)
        if a is None:
            continue
        ax, ay = float(a["position"][0]), float(a["position"][1])
        if g >= frame - POLICY["ue6_approach_lookback"]:
            lateral_history.append(abs(ay))
        if -2.0 < ax < 25.0 and abs(ay) < POLICY["ue6_conflict_lateral_m"]:
            recent = True
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
    if meta is None:
        return "no_meta", {}
    ids = list(meta.get("cut_in_actors_ids") or [])
    boxes = store.boxes(scenario, route, frame) or []
    actor = next((b for b in boxes if ids and b.get("id") == ids[0]), None)
    if actor is None:
        return "no_cut_in_actor", {}
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
    """返回 (规则, 证据类别, 数值) 表示该 train 行应被排除；None 表示保留。"""

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
    exclusions: List[Dict[str, Any]] = []
    reasons: Counter = Counter()
    before, after = Counter(), Counter()
    heldout = {s: hashlib.sha256() for s in ("val", "test")}
    heldout_after = {s: hashlib.sha256() for s in ("val", "test")}
    for line, row in zip(raw, rows):
        split = row["split"]
        if split in heldout:
            heldout[split].update(line)
            heldout_after[split].update(line)
            kept.append(line)
            continue
        cls = target_class(row)
        before[cls] += 1
        verdict = classify(store, row)
        if verdict is None:
            kept.append(line)
            after[cls] += 1
            continue
        rule, kind, info = verdict
        reasons[f"{rule}/{kind}"] += 1
        exclusions.append({"scenario": row["scenario"], "route_id": row["route_id"], "frame_id": int(row["frame_id"]),
                           "question_domain": row["question_domain"], "invalid_source": row.get("invalid_source", ""),
                           "target_event_class": cls, "rule": rule, "evidence": kind, **info})
    missing = [k for k in ("UE1", "UE3", "UE5", "UE6", "RE", "INVALID") if after.get(k, 0) <= 0]
    if missing:
        raise ValueError("Evidence filter emptied a required training class: " + ",".join(missing))
    if (sha(index), sha(index.parent / "manifest.json")) != (parent_sha, manifest_sha):
        raise ValueError("Input changed during derivation")
    receipt = dict(contract=CONTRACT, revision=REVISION, status="derived_model_pending", policy=POLICY,
                   parent_sha256=parent_sha, parent_manifest_sha256=manifest_sha,
                   rows_before=len(rows), rows_after=len(kept), train_excluded=len(exclusions),
                   exclusion_reasons=dict(sorted(reasons.items())),
                   train_class_counts_before=dict(sorted(before.items())), train_class_counts_after=dict(sorted(after.items())),
                   labels_flipped=0, holdouts_byte_identical=True,
                   heldout_sha256={s: h.hexdigest() for s, h in heldout.items()},
                   training_sampler_changed=False, training_quota=1024, max_frame_repeat=10, model_inference=False,
                   note="Train-only objective exclusion from LEAD metas/bboxes; no label flips; val/test untouched; not a visual certification of retained labels.")
    assert receipt["heldout_sha256"] == {s: h.hexdigest() for s, h in heldout_after.items()}
    output.mkdir(parents=True, exist_ok=False)
    (output / "frame_index.jsonl").write_bytes(b"".join(kept))
    (output / "evidence_exclusions.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in exclusions), encoding="utf-8")
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
                   note="Objective evidence policy; see evidence_exclusions.jsonl for every excluded train row.")
    (output.parent / "selected_train_reviews.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return receipt


def audit(index: Path, data_root: Path, splits: Sequence[str] = ("val", "test")) -> List[Dict[str, Any]]:
    """只读：列出 holdout 中按同一证据口径可疑的标签，供单列报告，不改评分。"""

    store = LeadStore(Path(data_root))
    out = []
    for line in Path(index).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["split"] not in splits:
            continue
        verdict = classify(store, row)
        if verdict is not None:
            rule, kind, info = verdict
            out.append({"split": row["split"], "scenario": row["scenario"], "route_id": row["route_id"], "frame_id": int(row["frame_id"]),
                        "question_domain": row["question_domain"], "target_event_class": target_class(row), "rule": rule, "evidence": kind, **info})
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", help="fresh output directory for the derived train-filtered index")
    p.add_argument("--audit-holdouts", help="write a read-only holdout dispute list (JSONL) instead of deriving")
    a = p.parse_args(argv)
    if a.audit_holdouts:
        items = audit(Path(a.parent), Path(a.data_root))
        Path(a.audit_holdouts).write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
        print(json.dumps({"holdout_disputes": len(items), "by_rule": dict(Counter(i["rule"] for i in items))}, ensure_ascii=False), flush=True)
        return
    if not a.output:
        p.error("--output is required unless --audit-holdouts is given")
    print(json.dumps(derive(Path(a.parent), Path(a.data_root), Path(a.output)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
