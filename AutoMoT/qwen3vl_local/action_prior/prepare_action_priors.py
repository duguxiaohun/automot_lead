"""自动复用 Phase3 候选动作真值，为 UE1–7/RE2/RE3/RE5 生成离线动作先验。

候选缺失不是普通背景；背景身份只取全帧 map。构建直接复用 Phase3 数据准备和
动作域定义，不另写速度/横向阈值，不读取其均衡抽样后的 frame_index 作为全量标签。
"""

from collections import Counter, defaultdict
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import time

from qwen3vl_local.action_prior.action_input import ACTION_INPUT_VERSION, HighLevelActionIndex, normalize_action
from qwen3vl_local.action_prior.contracts import digest, file_hash
from qwen3vl_local.action_prior.event_balance import EventBalanceIndex, SPECIAL_ELIGIBLE
from qwen3vl_local.action_prior.prepare_event_balance import prepare, _publish, _reuse_or_quarantine, run_builder
from qwen3vl_local.action_prior.build_event_balance_index import CONTEXT_TO_BUCKET, _candidate_membership, _re2_scene_state
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS, CONTEXT_BY_ID
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import validate_action_rule


def candidate_actions(path, mapping_hash):
    """保留完整候选，按 Phase3 三/五动作域投影标签，不包含 synthetic invalid。"""
    _candidate_membership(Path(path), mapping_hash)
    result = defaultdict(dict)
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("invalid_action_context") is True:
                continue
            validate_action_rule(row)
            context = CONTEXT_BY_ID[row["context_id"]]
            labels = row["action_labels"]
            if any(type(labels.get(key)) is not bool for key in ACTION_KEYS):
                raise ValueError("Phase3 candidate must contain all five boolean action_labels")
            if row.get("visual_label_risk"):
                raise ValueError("action prior requires default Phase3 candidates without visual-risk frames")
            key = (row["scenario"], row["route_id"], int(row["frame_id"]))
            bucket = CONTEXT_TO_BUCKET[context.context_id]
            if bucket in result[key]:
                raise ValueError(f"duplicate Phase3 context candidate: {key}/{bucket}")
            result[key][bucket] = {
                "answers": {action: labels[action] for action in context.action_keys},
                "planning_context": _re2_scene_state(row.get("context_detail", "")) if bucket == "RE2" else bucket,
            }
    return result


def project_frame(record, candidates):
    """普通/未确认/过滤帧不注入动作；并发 special 共用同帧动作，冲突立即拒绝。"""
    if record is None or record["status"] != SPECIAL_ELIGIBLE:
        return dict(status="not_applicable", actions=[]), []
    answers, contexts = {}, []
    for bucket in record["eligible_buckets"]:
        candidate = candidates.get(bucket)
        if candidate is None:
            raise ValueError(f"eligible full-map bucket missing Phase3 candidate: {bucket}")
        for action, value in candidate["answers"].items():
            if action in answers and answers[action] != value:
                raise ValueError(f"conflicting concurrent Phase3 action labels: {action}")
            answers[action] = value
        contexts.append(candidate["planning_context"])
    actions = [action for action in ACTION_KEYS if answers.get(action) is True]
    return normalize_action(dict(status="selected" if actions else "no_action", actions=actions)), contexts


def prepare_actions(full_path, data_root, data_dir, cache_root):
    """全帧门控与 Phase3 动作标签合并，锁内校验、缓存复用并原子发布。"""
    full = EventBalanceIndex(full_path)
    full.validate_action_dataset(data_dir)
    candidate_path = Path(full.manifest["candidate_index"])
    if file_hash(candidate_path) != full.source.candidate_sha256:
        raise ValueError("Phase3 candidate changed since full-map build")
    source = dict(
        schema=ACTION_INPUT_VERSION, source_kind="phase3_oracle",
        mapping_contract_hash=full.source.mapping_contract_hash,
        candidate_sha256=full.source.candidate_sha256, full_map_sha256=full.source.sha256,
        action_dataset_hashes=dict(full.source.action_dataset_hashes),
        builder_sha256=file_hash(__file__),
    )
    source_id = digest(source)
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / ("actions_" + source_id)

    def validate(directory):
        """缓存必须完整、来源相同且对应当前 action 三 split。"""
        index = HighLevelActionIndex(directory / "high_level_actions.jsonl")
        if index.source != {"source_kind": "phase3_oracle", "source_id": source_id}:
            raise ValueError("automatic high-level action source mismatch")
        index.validate_action_dataset(data_dir)

    with (cache_root / ".action_prepare.lock").open("a") as lock:
        print("[action prepare] waiting for preparation lock", file=sys.stderr, flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _reuse_or_quarantine(destination, validate):
            return destination / "high_level_actions.jsonl"
        from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route, print_progress

        routes, counts, seen = {}, Counter(), set()
        print("[action prepare] discover action frames; filter abnormal routes before labeling", file=sys.stderr, flush=True)
        for split in ("train", "val", "test"):
            with (Path(data_dir) / f"{split}.jsonl").open() as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        routes[(row["scenario"], row["run_id"])] = False
            print(f"[action prepare] discover {split}: {len(routes)} routes", file=sys.stderr, flush=True)
        started = time.time()
        print_progress(0, len(routes), started, prefix="action route filter")
        for number, route in enumerate(routes, 1):
            run = Path(data_root).joinpath(*route)
            if not run.is_dir():
                raise FileNotFoundError(run)
            routes[route] = is_abnormal_lead_route(run, route[0])[0]
            if number % 100 == 0 or number == len(routes):
                print_progress(number, len(routes), started, prefix="action route filter", last="/".join(route))
        candidates = candidate_actions(candidate_path, full.source.mapping_contract_hash)
        with tempfile.TemporaryDirectory(prefix=".actions-", dir=cache_root) as temporary:
            staging = Path(temporary) / "index"
            staging.mkdir()
            output = staging / "high_level_actions.jsonl"
            with output.open("w") as target:
                for split in ("train", "val", "test"):
                    with (Path(data_dir) / f"{split}.jsonl").open() as handle:
                        for line in handle:
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            if row.get("schema") != "action_prior_data_v1" or row.get("split") != split:
                                raise ValueError("invalid action dataset split row")
                            route = (row["scenario"], row["run_id"])
                            if routes[route]:
                                counts["abnormal_excluded"] += 1
                                continue
                            key = (*route, int(row["anchor"]))
                            if key in seen:
                                raise ValueError(f"duplicate action frame across splits: {key}")
                            seen.add(key)
                            record = full.records.get(key)
                            action, contexts = project_frame(record, candidates.get(key, {}))
                            status = record["status"] if record else "unconfirmed"
                            result = dict(
                                schema=ACTION_INPUT_VERSION, source_kind="phase3_oracle", source_id=source_id,
                                scenario=key[0], run_id=key[1], anchor=key[2], **action,
                                event_status=status,
                                event_buckets=list(record["eligible_buckets"]) if record else [],
                                planning_contexts=contexts,
                            )
                            target.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                            counts[f"{split}/{status}/{action['status']}"] += 1
            manifest = dict(source, source_id=source_id, index_sha256=file_hash(output),
                            rows=len(seen), counts=dict(counts), privileged_action_conditioning=True,
                            event_balance_index=str(Path(full_path).resolve()))
            (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            _publish(staging, destination, validate, (output.name,))
    return destination / "high_level_actions.jsonl"


def ensure_action_inputs(args):
    """新训练缺索引时自动准备；续训只恢复保存产物，禁止悄悄重新标定。"""
    if not getattr(args, "high_level_action_prior", False):
        return
    if not args.high_level_planning or args.condition_mode != "prior":
        raise ValueError("--high-level-action-prior requires --high-level-planning and condition-mode prior")
    if getattr(args, "high_level_action_index", ""):
        index = HighLevelActionIndex(args.high_level_action_index)
        index.validate_action_dataset(args.data_dir)
        if index.manifest and not args.event_balance_index:
            args.event_balance_index = index.manifest["event_balance_index"]
        return
    if getattr(args, "resume", ""):
        raise ValueError("resume must restore its saved high-level action index; do not regenerate labels")
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not all((data_dir / name).is_file() and (data_dir / name).stat().st_size
                   for name in ("manifest.json", "train.jsonl", "val.jsonl", "test.jsonl")):
            run_builder(Path(__file__).with_name("build_dataset.py"), [
                "--data-root", args.data_root, "--output-dir", args.data_dir,
            ])
    cache_root = Path("checkpoints/action_prior_prepared")
    if not args.event_balance_index:
        args.event_balance_index = str(prepare(args.data_root, args.data_dir,
                                              "keyframe_filter/collection_output", cache_root))
    args.high_level_action_index = str(prepare_actions(args.event_balance_index, args.data_root,
                                                      args.data_dir, cache_root))
