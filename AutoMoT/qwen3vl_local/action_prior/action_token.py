"""三条 Action 路径共用的离线 Phase3 动作 token；不改文字先验或采样权重。"""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

from qwen3vl_local.action_prior.contracts import file_hash

TOKEN_VERSION = "phase3_primary_action_token_v1"
ACTION_TOKEN_NAMES = (
    "UNCOND", "DECELERATE", "STOP", "RESUME", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT", "KEEP",
)


def conditioning_env_args():
    """只转发显式环境值，后续 CLI 覆盖；恢复不注入新默认值。"""
    result = []
    if "HIGH_LEVEL_ACTION_TOKEN" in os.environ:
        value = os.environ["HIGH_LEVEL_ACTION_TOKEN"]
        if value not in ("0", "1"):
            raise ValueError("HIGH_LEVEL_ACTION_TOKEN must be 0 or 1")
        result.append("--high-level-action-token" if value == "1" else "--no-high-level-action-token")
    if "RGB_FRAME_COUNT" in os.environ:
        value = os.environ["RGB_FRAME_COUNT"]
        if value not in ("1", "4"):
            raise ValueError("RGB_FRAME_COUNT must be 1 or 4")
        result.extend(["--rgb-frame-count", value])
    return result


def ensure_token_inputs(args):
    """新训自动准备 full map；恢复只能使用保存的文件。"""
    if not getattr(args, "high_level_action_token", False):
        return
    if not getattr(args, "event_balance_index", ""):
        if getattr(args, "resume", ""):
            raise ValueError("resume requires the saved action-token full mapping; do not regenerate labels")
        from qwen3vl_local.action_prior.prepare_action_priors import ensure_full_mapping
        ensure_full_mapping(args)


def project_token(record, candidates):
    """只用完整且可用的域内证据；普通/隔离帧不伪造 KEEP。"""
    from qwen3vl_local.action_prior.event_balance import SPECIAL_ELIGIBLE
    from qwen3vl_local.sft_new_loop_phase3.choice_semantics import primary_choice
    if record is None:
        # full map 可以明确没有收录某些 action 背景路线，不借邻帧标签。
        return dict(name="UNCOND", reason="outside_phase3_mapping")
    if record["status"] != SPECIAL_ELIGIBLE:
        return dict(name="UNCOND", reason=record["status"])
    answers = {}
    for bucket in record["eligible_buckets"]:
        if bucket not in candidates:
            raise ValueError(f"eligible action-token frame missing Phase3 candidate: {bucket}")
        for action, value in candidates[bucket].items():
            if type(value) is not bool:
                raise ValueError("action token requires complete boolean evidence")
            if action in answers and answers[action] != value:
                raise ValueError("conflicting concurrent Phase3 action-token evidence")
            answers[action] = value
    return dict(name=primary_choice(answers, tuple(answers)), reason="complete_phase3_evidence")


class ActionTokenSource:
    """读取原始候选而非均衡题库；source 内容和 mapping 合同绑定 checkpoint。"""
    def __init__(self, args):
        from qwen3vl_local.action_prior.event_balance import EventBalanceIndex
        from qwen3vl_local.action_prior.build_event_balance_index import _candidate_membership, CONTEXT_TO_BUCKET
        from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID
        from qwen3vl_local.sft_new_loop_phase3.choice_semantics import choice_annotation
        from qwen3vl_local.sft_new_loop_phase3.trajectory_action import validate_action_rule

        self.full = EventBalanceIndex(args.event_balance_index)
        self.full.validate_action_dataset(args.data_dir)
        # 支持整目录搬迁后使用同名 sibling；不修改 manifest 的内容合同。
        candidate = Path(self.full.manifest["candidate_index"])
        if not candidate.is_file():
            candidate = self.full.path.parent / candidate.name
        if not candidate.is_file() or file_hash(candidate) != self.full.source.candidate_sha256:
            raise ValueError("action-token Phase3 candidate missing or hash mismatch")
        _candidate_membership(candidate, self.full.source.mapping_contract_hash)
        candidates = defaultdict(dict)
        with candidate.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("invalid_action_context") is True:
                    continue
                validate_action_rule(row)
                if row.get("visual_label_risk"):
                    raise ValueError("action token cannot consume visual-risk candidates")
                context = CONTEXT_BY_ID[row["context_id"]]
                labels = row["action_labels"]
                # KEEP 必须通过 Phase3 的完整速度/横向证据检查，旧 NONE 不直接翻译。
                choice_annotation(labels, context.context_id, row["action_evidence"])
                key = (row["scenario"], row["route_id"], int(row["frame_id"]))
                bucket = CONTEXT_TO_BUCKET[context.context_id]
                if bucket in candidates[key]:
                    raise ValueError(f"duplicate action-token candidate: {key}/{bucket}")
                candidates[key][bucket] = {key: labels[key] for key in context.action_keys}
        self.tokens = {
            key: project_token(record, candidates.get(key, {}))
            for key, record in self.full.records.items()
        }
        self.identity = dict(
            version=TOKEN_VERSION, vocabulary=list(ACTION_TOKEN_NAMES),
            source_kind="phase3_oracle", privileged_action_conditioning=True,
            full_map=self.full.source.identity_dict(),
            projection="embedding_then_sequence_concat_after_bev", keep_scope="merged",
            source_sha256=file_hash(__file__),
            projection_sources={name: file_hash(Path(__file__).parents[1] / "sft_new_loop_phase3" / name)
                                for name in ("choice_semantics.py", "primary_action.py", "context_taxonomy.py")},
        )

    def annotate(self, rows):
        """动作 ID 只送 decoder；原始原因随样本进入审计。"""
        for row in rows:
            key = (row["scenario"], row["run_id"], int(row["anchor"]))
            token = self.tokens.get(key, dict(name="UNCOND", reason="outside_phase3_mapping"))
            row["action_token"] = dict(token, version=TOKEN_VERSION)
            row["action_token_id"] = ACTION_TOKEN_NAMES.index(token["name"])


_SOURCES = {}


def token_source(args):
    """按文件内容缓存；重新哈希避免进程内文件更换后沿用旧标签。"""
    path = Path(args.event_balance_index).resolve()
    manifest = json.loads(path.with_name("manifest.json").read_text())
    candidate = Path(manifest["candidate_index"])
    if not candidate.is_file():
        candidate = path.parent / candidate.name
    key = (str(path), file_hash(path), file_hash(path.with_name("manifest.json")),
           file_hash(candidate), *(file_hash(Path(args.data_dir) / f"{s}.jsonl") for s in ("train", "val", "test")))
    if key not in _SOURCES:
        _SOURCES[key] = ActionTokenSource(args)
    return _SOURCES[key]


def token_contract(args):
    return token_source(args).identity if getattr(args, "high_level_action_token", False) else None


def annotate_tokens(args, rows):
    if getattr(args, "high_level_action_token", False):
        token_source(args).annotate(rows)


def token_tensor(sample, config, device):
    """禁止开启后缺标签静默退化；普通 RE 的 UNCOND 是显式索引结果。"""
    import torch
    if not getattr(config, "use_high_level_action_token", False):
        return None
    token = sample.get("action_token", {})
    value = sample.get("action_token_id")
    if (type(value) is not int or not 0 <= value < len(ACTION_TOKEN_NAMES)
            or token.get("version") != TOKEN_VERSION or token.get("name") != ACTION_TOKEN_NAMES[value]):
        raise ValueError("missing/inconsistent action-token annotation")
    return torch.tensor([value], device=device, dtype=torch.long)


def token_coverage(rows):
    return {split: {**dict.fromkeys(ACTION_TOKEN_NAMES, 0),
                    **Counter(row["action_token"]["name"] for row in samples)}
            for split, samples in rows.items()}


def token_support(rows):
    """区分呈现次数、不同帧与物理路线支持；不把重复采样当新标签。"""
    from qwen3vl_local.action_prior.build_dataset import route_group
    result = {}
    for split, samples in rows.items():
        counts = Counter()
        frames = {name: set() for name in ACTION_TOKEN_NAMES}
        routes = {name: set() for name in ACTION_TOKEN_NAMES}
        reasons = Counter()
        for row in samples:
            token = row.get("action_token", {})
            name = token.get("name")
            if (name not in frames or type(row.get("action_token_id")) is not int
                    or row["action_token_id"] != ACTION_TOKEN_NAMES.index(name)
                    or token.get("version") != TOKEN_VERSION):
                raise ValueError("missing/inconsistent action token in support audit")
            counts[name] += 1
            frames[name].add((row["scenario"], row["run_id"], int(row["anchor"])))
            routes[name].add(route_group(row["scenario"], row["run_id"]))
            if name == "UNCOND":
                reasons[token.get("reason", "unspecified")] += 1
        result[split] = dict(
            presentations={name: counts[name] for name in ACTION_TOKEN_NAMES},
            unique_frames={name: len(frames[name]) for name in ACTION_TOKEN_NAMES},
            physical_routes={name: len(routes[name]) for name in ACTION_TOKEN_NAMES},
            missing_actions=[name for name in ACTION_TOKEN_NAMES[1:] if not frames[name]],
            conditioned_presentations=sum(counts[name] for name in ACTION_TOKEN_NAMES[1:]),
            unconditioned_reasons=dict(reasons),
        )
    train_names = {name for name in ACTION_TOKEN_NAMES[1:]
                   if result.get("train", {}).get("unique_frames", {}).get(name, 0)}
    for split, report in result.items():
        report["actions_without_train_support"] = [
            name for name in ACTION_TOKEN_NAMES[1:]
            if split != "train" and report["unique_frames"][name] and name not in train_names
        ]
    return result


def require_conditioned_training(report, *, stage):
    """开启 oracle token 却没有一个动作标签时拒绝；不强求每类达到任意配额。"""
    train = report.get("train", {})
    if not train.get("conditioned_presentations", 0):
        raise ValueError(f"{stage}: high-level action token has no conditioned training frames; "
                         f"UNCOND reasons={train.get('unconditioned_reasons', {})}. "
                         "Check full-map coverage, filtering and effective split; "
                         "do not replace missing labels with KEEP.")
