"""Phase3 high-level 动作输入：默认自动准备离线标注，后续预测复用 normalize_action。

这里只消费显式给出的动作，不从 RS/EVENT 或未来轨迹推导动作。预测质量及来源声明
由上游负责；文件内容和来源进入 checkpoint 合同，路径只用于定位。
"""

from collections import Counter
import json
from pathlib import Path

ACTION_INPUT_VERSION = "scoped_phase3_high_level_action_v2"
ACTION_TEXT = {
    "DECELERATE": "reduce speed",
    "STOP": "reach or remain at a sustained near-stop, including continued waiting",
    "RESUME": "sustain a speed increase; a previous stop is not required",
    "LANE_CHANGE_LEFT": "make the first upcoming lane-boundary crossing to the left relative to ego's heading",
    "LANE_CHANGE_RIGHT": "make the first upcoming lane-boundary crossing to the right relative to ego's heading",
}
_SPEED = {"DECELERATE", "STOP", "RESUME"}
_LANE = {"LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"}


def normalize_action(value):
    """校验外部动作，保留二值 Phase3 的纵横并行动作；缺失不当作全 NO。"""
    if value is None:
        return {"status": "unavailable", "actions": []}
    if not isinstance(value, dict) or set(value) != {"status", "actions"}:
        raise ValueError("high-level action requires exactly status and actions")
    status, actions = value["status"], value["actions"]
    if status not in ("selected", "no_action", "unavailable", "not_applicable"):
        raise ValueError("invalid high-level action status")
    if not isinstance(actions, list) or any(not isinstance(a, str) or a not in ACTION_TEXT for a in actions):
        raise ValueError("unknown high-level action; expected Phase3 action names")
    if len(actions) != len(set(actions)) or len(set(actions) & _SPEED) > 1 or len(set(actions) & _LANE) > 1:
        raise ValueError("conflicting or duplicate high-level actions")
    if bool(actions) != (status == "selected"):
        raise ValueError("selected needs actions; no_action/unavailable/not_applicable need an empty list")
    return {"status": status, "actions": [a for a in ACTION_TEXT if a in actions]}


def action_sentence(value):
    """仅固定动作释义进入提示词，不注入上游任意文本或来源元数据。"""
    value = normalize_action(value)
    if value["status"] in ("unavailable", "not_applicable"):
        return ""
    if value["status"] == "no_action":
        return "No high-level action is selected in the supplied event context; use the scene and navigation for the trajectory."
    return "The supplied upcoming high-level action is to " + "; and to ".join(
        ACTION_TEXT[a] for a in value["actions"]
    ) + ". Use this action to guide near-term trajectory planning with the images and navigation."


class HighLevelActionIndex:
    """读取逐帧标注或外部预测，严格身份对齐，缺帧记 unavailable 并报告覆盖。"""

    def __init__(self, path):
        """一次读取并核验来源；不访问 RGB、meta、Phase3 权重或未来轨迹。"""
        self.path = str(Path(path).resolve())
        self.records = {}
        self.contexts = {}
        self.source = None
        self.manifest = None
        raw = Path(path).read_bytes()
        for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != ACTION_INPUT_VERSION:
                raise ValueError(f"{path}:{number}: wrong high-level action schema")
            source = {key: row.get(key) for key in ("source_kind", "source_id")}
            if source["source_kind"] not in ("phase3_oracle", "prediction", "provided") or not isinstance(source["source_id"], str) or not source["source_id"].strip():
                raise ValueError("action input needs phase3_oracle/prediction/provided source_kind and a nonempty source_id")
            if self.source is not None and source != self.source:
                raise ValueError("mixed high-level action sources in one index")
            self.source = source
            if any(not isinstance(row.get(key), str) or not row[key] for key in ("scenario", "run_id")):
                raise ValueError("action input needs exact scenario/run_id strings")
            if type(row.get("anchor")) is not int or row["anchor"] < 0:
                raise ValueError("action anchor must be a nonnegative integer frame id")
            key = (row["scenario"], row["run_id"], row["anchor"])
            if key in self.records:
                raise ValueError(f"duplicate high-level action frame: {key}")
            action = normalize_action({"status": row.get("status"), "actions": row.get("actions")})
            from qwen3vl_local.action_prior.event_balance import SPECIAL_BUCKETS, SPECIAL_ELIGIBLE
            from qwen3vl_local.action_prior.prompts import EVENT_BALANCED_CONTEXT_DESCRIPTIONS, _MANEUVER_CONTEXTS
            buckets = row.get("event_buckets", [])
            contexts = row.get("planning_contexts", [])
            status = row.get("event_status")
            if (not isinstance(buckets, list) or any(b not in SPECIAL_BUCKETS for b in buckets)
                    or not isinstance(contexts, list) or any(c not in EVENT_BALANCED_CONTEXT_DESCRIPTIONS for c in contexts)):
                raise ValueError("invalid high-level action event scope")
            if status == SPECIAL_ELIGIBLE:
                if not buckets or not contexts or action["status"] == "not_applicable":
                    raise ValueError("eligible special frame needs scoped action status")
                if {"RE2" if c.startswith("RE2_") else c for c in contexts} != set(buckets):
                    raise ValueError("planning contexts differ from eligible event buckets")
                # 域投影与 Phase3 一致：只有机动域可提供横向动作。
                if set(action["actions"]) & _LANE and not set(contexts) & _MANEUVER_CONTEXTS:
                    raise ValueError("longitudinal event cannot supply lane-change labels")
            elif status in ("confirmed_regular", "special_filtered", "unconfirmed"):
                if buckets or contexts or action["status"] != "not_applicable":
                    raise ValueError("background/filtered/unconfirmed frames must not carry action planning")
            else:
                raise ValueError("missing/invalid high-level action event status")
            self.records[key] = action
            self.contexts[key] = tuple(contexts)
        if not self.records:
            raise ValueError("empty high-level action index")
        import hashlib
        self.identity = {"version": ACTION_INPUT_VERSION, **self.source,
                         "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(self.records)}
        if self.source["source_kind"] == "phase3_oracle":
            from qwen3vl_local.action_prior.contracts import digest
            from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapping_contract_hash
            manifest = json.loads(Path(path).with_name("manifest.json").read_text())
            if (manifest.get("schema") != ACTION_INPUT_VERSION
                    or manifest.get("source_kind") != self.source["source_kind"]
                    or manifest.get("privileged_action_conditioning") is not True
                    or manifest.get("index_sha256") != self.identity["sha256"]
                    or manifest.get("rows") != len(self.records)
                    or manifest.get("mapping_contract_hash") != mapping_contract_hash()
                    or manifest.get("source_id") != self.source["source_id"]):
                raise ValueError("stale/incomplete Phase3 action index manifest; rebuild with current rules")
            source_fields = ("schema", "source_kind", "mapping_contract_hash", "candidate_sha256",
                             "full_map_sha256", "action_dataset_hashes", "builder_sha256")
            source = {key: manifest[key] for key in source_fields}
            if digest(source) != self.source["source_id"]:
                raise ValueError("Phase3 action source identity mismatch")
            self.manifest = manifest
            self.identity.update(privileged_action_conditioning=True, source_contract=source)

    def validate_action_dataset(self, data_dir):
        """自动标注必须绑定原 action 全量 split；绝对路径不进入内容合同。"""
        if self.manifest:
            from qwen3vl_local.action_prior.contracts import file_hash
            actual = {s: file_hash(Path(data_dir) / f"{s}.jsonl") for s in ("train", "val", "test")}
            if self.manifest["action_dataset_hashes"] != actual:
                raise ValueError("Phase3 action index belongs to different action splits")

    def planning_contexts(self, identity):
        """只为通过门控的 special 帧返回条件性规划域；普通背景恒为空。"""
        return self.contexts.get(tuple(identity), ())

    def get(self, identity):
        """只做精确帧命中；不得最近邻借用前帧动作或从事件猜动作。"""
        if identity is None:
            raise ValueError("high-level action lookup requires sample identity")
        return normalize_action(self.records.get(tuple(identity)))

    def coverage(self, rows):
        """区分文件缺帧、显式不可用、全 NO 和具体动作，便于离线实验审计。"""
        result = {}
        for split, samples in rows.items():
            counts = Counter(total=len(samples))
            for sample in samples:
                key = (sample["scenario"], sample["run_id"], int(sample["anchor"]))
                counts["matched" if key in self.records else "missing"] += 1
                counts[self.get(key)["status"]] += 1
            result[split] = dict(counts)
        return result


def action_input_contract(args):
    """内容合同与绝对路径分离；关闭时不读取动作文件。"""
    if not getattr(args, "high_level_action_prior", False):
        return None
    index = HighLevelActionIndex(args.high_level_action_index)
    index.validate_action_dataset(args.data_dir)
    if index.manifest:
        from qwen3vl_local.action_prior.contracts import file_hash
        if file_hash(args.event_balance_index) != index.manifest["full_map_sha256"]:
            raise ValueError("Phase3 action labels and event mapping must share the same full map")
    return index.identity
