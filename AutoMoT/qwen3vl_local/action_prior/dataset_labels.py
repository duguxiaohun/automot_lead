"""数据集标定真值先验：读取全覆盖标签索引并转换成与 collect_priors 同构的结构。

开启该来源后不加载、不运行任何 Phase1/Phase2 LoRA：每帧的 RS/Phase1 事实/EVENT
直接来自上游标注，base Qwen 默认只做一次分析。这是特权标签条件化，仅适用于
离线数据集帧；闭环或数据集外帧没有标签，必须显式换回 LoRA 先验。

可选的 ``PriorNoise`` 按已审计的 LoRA 错误方向注入错误/invalid 先验，让 decoder
不把标定真值当作永远正确的条件；噪声形状是按审计整形的设计值，不是实测转移矩阵。
"""

from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path

from qwen3vl_local.action_prior.contracts import digest, file_hash, read_json
from qwen3vl_local.sft_new_loop_phase1 import prompts as p1
from qwen3vl_local.sft_new_loop_phase2 import prompts as p2

LABEL_SCHEMA = "action_prior_prior_labels_v1"
PRIOR_SOURCE = "dataset_labels"
NOISE_MODEL = "audit_shaped_rs_event_noise_v1"
ROAD_STRUCTURES = ("R1", "R2", "R3", "R4", "R5")
TARGET_CLASSES = ("UE1", "UE3", "UE5", "UE6", "RE")
DOMAINS = tuple(p2.QUESTION_DOMAINS)
DOMAIN_KEYS = tuple(f"{domain}/{p2.INVALID_KEY}" for domain in DOMAINS)
EVENT_KEYS = tuple(key for domain in DOMAINS for key in p2.event_keys_for_domain(domain))
RS_KEYS = (*p1.PHASE2_ANSWER_KEYS, "ROAD_STRUCTURE", "RS_HIGHWAY")
CONDITION_KEYS = (
    *p1.PHASE1_ANSWER_KEYS,
    *p1.PHASE2_ANSWER_KEYS,
    "RS_HIGHWAY",
    "ROAD_STRUCTURE",
    *DOMAIN_KEYS,
    *EVENT_KEYS,
)

# 形状取自 2026-08 融合 LoRA 的 RS confusion 复核：RS1 边界最不稳定且 FP/FN 近似
# 对称，RS2 recall 高但过度解释对向约束，RS5 明显漏报并常退成 RS4，R3/RS_HIGHWAY
# 最稳。这是按方向整形的设计权重，不是实测逐类转移矩阵。
ROAD_STRUCTURE_CONFUSION = {
    "R1": (("R2", 0.35), ("R4", 0.25), ("R5", 0.20), ("R3", 0.20)),
    "R2": (("R1", 0.55), ("R4", 0.20), ("R5", 0.15), ("R3", 0.10)),
    "R3": (("R1", 0.60), ("R2", 0.40)),
    "R4": (("R5", 0.50), ("R1", 0.30), ("R2", 0.20)),
    "R5": (("R4", 0.45), ("R1", 0.35), ("R2", 0.20)),
}
# 形状取自 Phase2 v3/v4 逐例 RGB 复核：UE3↔RE 是最常见混淆，其次 UE1/UE3 同屏
# 互混；UE5 最稳且错时退成 RE；junction 域只有 UE6/RE 两个合法答案。
EVENT_CONFUSION = {
    (p2.ROAD_DOMAIN, "UE1"): (("RE", 0.60), ("UE3", 0.40)),
    (p2.ROAD_DOMAIN, "UE3"): (("RE", 0.65), ("UE1", 0.35)),
    (p2.ROAD_DOMAIN, "UE5"): (("RE", 0.70), ("UE3", 0.30)),
    (p2.ROAD_DOMAIN, "RE"): (("UE3", 0.50), ("UE1", 0.30), ("UE5", 0.20)),
    (p2.JUNCTION_DOMAIN, "UE6"): (("RE", 1.0),),
    (p2.JUNCTION_DOMAIN, "RE"): (("UE6", 1.0),),
}



def pack(road_structure, target_class, question_domain, invalid_domain, phase1_answers):
    """把一帧标签压成一个 int；687k 帧的完整索引才能常驻每个 rank。"""
    value = 0
    for bit, key in enumerate(p1.PHASE1_ANSWER_KEYS):
        if phase1_answers[key]:
            value |= 1 << bit
    value |= ROAD_STRUCTURES.index(road_structure) << 4
    value |= TARGET_CLASSES.index(target_class) << 7
    value |= DOMAINS.index(question_domain) << 10
    if invalid_domain:
        if invalid_domain == question_domain:
            raise ValueError("invalid domain must differ from the labeled question domain")
        value |= 1 << 11
    return value


def unpack(value):
    """还原可读标签，用于审计输出。"""
    domain = DOMAINS[value >> 10 & 1]
    return dict(
        phase1_answers={
            key: bool(value >> bit & 1) for bit, key in enumerate(p1.PHASE1_ANSWER_KEYS)
        },
        road_structure=ROAD_STRUCTURES[value >> 4 & 0b111],
        target_event_class=TARGET_CLASSES[value >> 7 & 0b111],
        question_domain=domain,
        invalid_domain=DOMAINS[1 - DOMAINS.index(domain)] if value >> 11 & 1 else None,
    )


def conditions_from_label(value):
    """按 Phase1/Phase2 的既有语义展开条件；未标注的域保持 UNKNOWN，不补默认 NO。"""
    label = unpack(value)
    conditions, invalid = _conditions(label)
    return conditions, invalid, label


def _conditions(label):
    """从已解包（可能已被噪声改写）的标签展开条件。"""
    rs = label["road_structure"]
    domain = label["question_domain"]
    conditions = {
        key: ("YES" if answer else "NO")
        for key, answer in label["phase1_answers"].items()
    }
    positive = "RS" + rs[1:]
    conditions.update(
        {key: ("YES" if key == positive else "NO") for key in p1.PHASE2_ANSWER_KEYS}
    )
    conditions["ROAD_STRUCTURE"] = rs
    # 上游 hierarchical probe 把 RS_HIGHWAY 的真值定义为“四个 RS 全 NO”，即 R3。
    conditions["RS_HIGHWAY"] = "YES" if rs == "R3" else "NO"
    invalid = {}
    for name in DOMAINS:
        domain_key = f"{name}/{p2.INVALID_KEY}"
        if name == domain:
            conditions[domain_key] = "NO"
        elif label["invalid_domain"] == name:
            conditions[domain_key] = "YES"
        else:
            conditions[domain_key] = None
            invalid[domain_key] = "dataset_domain_unlabeled"
        for key in p2.event_keys_for_domain(name):
            if name == domain:
                conditions[key] = "YES" if key == label["target_event_class"] else "NO"
            else:
                conditions[key] = None
                invalid[key] = (
                    "domain_inapplicable"
                    if conditions[domain_key] == "YES"
                    else "domain_unconfirmed"
                )
    return conditions, invalid


def _pick(weights, unit):
    """按累计权重确定性取样。"""
    total = sum(weight for _, weight in weights)
    threshold = unit * total
    accumulated = 0.0
    for value, weight in weights:
        accumulated += weight
        if threshold < accumulated:
            return value
    return weights[-1][0]


class PriorNoise:
    """按审计错误方向把一部分帧的 RS 或 EVENT 先验改成错误值或 invalid。

    同一 (帧, seed) 的抽样恒定：分析文本缓存保存的是该帧先验对应的 base 输出，
    每个 epoch 重抽会让缓存与条件不一致，也会退化成每 epoch 重新生成。
    """

    def __init__(self, rate, invalid_share=0.25, seed=0):
        if not 0.0 <= rate <= 1.0 or not 0.0 <= invalid_share <= 1.0:
            raise ValueError("prior noise rate/invalid share must be within [0, 1]")
        self.rate = float(rate)
        self.invalid_share = float(invalid_share)
        self.seed = int(seed)

    @property
    def fingerprint(self):
        """噪声抽样会改变实际先验文本条件，因此 seed 也必须进入合同身份。"""
        return dict(
            model=NOISE_MODEL,
            rate=self.rate,
            invalid_share=self.invalid_share,
            seed=self.seed,
        )

    def _unit(self, key, salt):
        digested = hashlib.sha256(f"{self.seed}:{key}:{salt}".encode()).digest()
        return int.from_bytes(digested[:8], "big") / 2**64

    def draw(self, identity, label):
        """就地改写 label 并返回噪声说明；未命中返回 None。"""
        if self.rate <= 0.0:
            return None
        key = f"{identity[0]}/{identity[1]}:{identity[2]}"
        if self._unit(key, "hit") >= self.rate:
            return None
        # RS 与 EVENT 由两个独立 LoRA 回答，真实失败也不同步；每次只破坏一个通道。
        channel = "rs" if self._unit(key, "channel") < 0.5 else "event"
        mode = "invalid" if self._unit(key, "mode") < self.invalid_share else "confusion"
        note = dict(channel=channel, mode=mode, model=NOISE_MODEL)
        if mode == "invalid":
            return note
        if channel == "rs":
            original = label["road_structure"]
            label["road_structure"] = _pick(
                ROAD_STRUCTURE_CONFUSION[original], self._unit(key, "rs")
            )
            note.update(field="ROAD_STRUCTURE", original=original,
                        corrupted=label["road_structure"])
        else:
            original = label["target_event_class"]
            label["target_event_class"] = _pick(
                EVENT_CONFUSION[(label["question_domain"], original)],
                self._unit(key, "event"),
            )
            note.update(field="target_event_class", original=original,
                        corrupted=label["target_event_class"])
        return note


def _mask_invalid(conditions, invalid, note, label):
    """invalid 噪声复刻两种真实失败：RS 向量整体无法确认；EVENT 域被误判为不适用。"""
    if note["channel"] == "rs":
        for key in RS_KEYS:
            conditions[key] = None
            invalid[key] = "noise_rs_unresolved"
        note["field"] = "ROAD_STRUCTURE"
        return
    domain = label["question_domain"]
    domain_key = f"{domain}/{p2.INVALID_KEY}"
    conditions[domain_key] = "YES"
    for key in p2.event_keys_for_domain(domain):
        conditions[key] = None
        invalid[key] = "noise_event_domain_invalid"
    note["field"] = domain_key


def _priors(conditions, invalid, label):
    """保持与 collect_priors 完全相同的返回键，便于同一套审计与缓存复用。"""
    return dict(
        conditions=conditions,
        invalid=invalid,
        invalid_counts=dict(Counter(invalid.values())),
        calls=[],
        recheck_mode=PRIOR_SOURCE,
        recheck_comparisons=[],
        recheck_mode_disagreements=[],
        condition_acceptance_policy="dataset_ground_truth",
        compare_requires_consensus=False,
        consistency_is_accuracy=False,
        prior_source=PRIOR_SOURCE,
        privileged_label_conditioning=True,
        dataset_label=label,
    )


def missing_priors(identity):
    """索引里没有这一帧时全部保持未知；不允许用默认 NO 伪造条件。"""
    conditions = {key: None for key in CONDITION_KEYS}
    invalid = {key: "dataset_label_missing" for key in CONDITION_KEYS}
    return _priors(
        conditions,
        invalid,
        dict(
            scenario=identity[0],
            run_id=identity[1],
            frame_id=identity[2],
            labeled=False,
        ),
    )


def label_source(path, noise=None):
    """记录标签索引身份，供合同固定；旁边必须有匹配 schema 的 manifest。"""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{path}: build it with build_prior_labels.py")
    manifest_path = path.parent / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    if manifest.get("schema") != LABEL_SCHEMA:
        raise ValueError(f"{manifest_path}: missing/incompatible {LABEL_SCHEMA} manifest")
    files = {path.name: file_hash(path), manifest_path.name: file_hash(manifest_path)}
    noise_fingerprint = noise.fingerprint if noise and noise.rate > 0 else None
    return dict(
        path=str(path),
        schema=LABEL_SCHEMA,
        prior_source=PRIOR_SOURCE,
        labeled_frames=int(manifest["counts"]["labeled_frames"]),
        sources=manifest.get("sources", {}),
        file_sha256=files,
        noise=noise_fingerprint,
        fingerprint=digest(dict(files=files, noise=noise_fingerprint)),
        privileged_label_conditioning=True,
    )


class PriorLabelIndex:
    """按 (scenario, run_id, frame) 常驻查找；命中与缺失都显式区分，不静默补默认值。"""

    def __init__(self, path, noise=None):
        self.noise = noise if noise and noise.rate > 0 else None
        self.source = label_source(path, self.noise)
        self.path = self.source["path"]
        self.routes = {}
        self.rows = 0
        with Path(self.path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                frames = self.routes.setdefault(
                    (row["scenario"], row["route_id"]), {}
                )
                frames[int(row["frame_id"])] = pack(
                    row["road_structure"],
                    row["target_event_class"],
                    row["question_domain"],
                    row["invalid_domain"],
                    row["phase1_answers"],
                )
                self.rows += 1
        if self.rows != self.source["labeled_frames"]:
            raise ValueError(f"{self.path}: row count differs from its manifest")

    def lookup(self, scenario, run_id, frame_id):
        """返回打包标签或 None。"""
        return self.routes.get((scenario, run_id), {}).get(int(frame_id))

    def priors(self, identity):
        """identity 为 (scenario, run_id, frame)；缺帧返回全 UNKNOWN 的同构先验。"""
        value = self.lookup(*identity)
        if value is None:
            return missing_priors(identity)
        label = unpack(value)
        truth = dict(road_structure=label["road_structure"],
                     target_event_class=label["target_event_class"])
        note = self.noise.draw(identity, label) if self.noise else None
        conditions, invalid = _conditions(label)
        if note and note["mode"] == "invalid":
            _mask_invalid(conditions, invalid, note, label)
        return _priors(
            conditions,
            invalid,
            dict(
                label,
                scenario=identity[0],
                run_id=identity[1],
                frame_id=int(identity[2]),
                labeled=True,
                noise=note,
                # 被噪声改写时保留原标签，便于 probe/case dump 直接看到注入了什么。
                clean_label=truth if note else None,
            ),
        )

    def coverage(self, rows):
        """训练前报告实际命中率；缺帧会让该样本退化成完全无条件。"""
        splits = rows.items() if isinstance(rows, dict) else [("rows", rows)]
        counts = Counter()
        for split, items in splits:
            for row in items:
                counts[f"{split}/frames"] += 1
                hit = self.lookup(row["scenario"], row["run_id"], row["anchor"])
                counts[f"{split}/" + ("labeled" if hit is not None else "missing")] += 1
        return dict(counts, labeled_frames=self.rows, source=self.path)
