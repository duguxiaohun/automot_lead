"""可审计的 action-prior UE/RE 均衡课程。

输入必须是 ``build_event_balance_index.py`` 生成的全帧语义映射，而不是 Phase3 已经过
动作窗口/视觉过滤的 ``candidate_frames.jsonl``。全帧映射把 special、确认的普通、未确认、
以及 special 但不适合 Phase3 动作问答的帧分开；只有确认普通帧可以进入两份背景池。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from qwen3vl_local.action_prior.contracts import file_hash

SAMPLING_MODE_UNIFORM = "uniform"
SAMPLING_MODE_EVENT_BALANCED = "event_balanced"
SAMPLING_MODES = (SAMPLING_MODE_UNIFORM, SAMPLING_MODE_EVENT_BALANCED)
# v2 binds the post-quarantine normal-background rule and diversity-first allocation contract.
# v1 must be rebuilt: it could classify an empty quarantined R-E2/3/5 context as normal.
FULL_INDEX_SCHEMA = "action_prior_event_balance_full_v2"
FULL_MANIFEST_SCHEMA = "action_prior_event_balance_full_manifest_v2"
EVENT_BALANCE_MAPPING_POLICY = "v2_quarantine_gate_before_regular_diversity_first"
SPECIAL_BUCKETS: Tuple[str, ...] = (
    "UE1", "UE2", "UE3", "UE4", "UE5", "UE6", "UE7", "RE2", "RE3", "RE5",
)
REGULAR_BACKGROUND = "REGULAR_BACKGROUND"
CONFIRMED_REGULAR = "confirmed_regular"
UNCONFIRMED = "unconfirmed"
SPECIAL_FILTERED = "special_filtered"
SPECIAL_ELIGIBLE = "special_eligible"
EVENT_BALANCE_WEIGHTS = {**{key: 1 for key in SPECIAL_BUCKETS}, REGULAR_BACKGROUND: 2}
_SOURCE_CACHE: Dict[str, "EventBalanceIndex"] = {}


def _identity(row: Mapping[str, Any]) -> Tuple[str, str, int]:
    scenario = str(row.get("scenario", ""))
    route = str(row.get("route_id") or row.get("run_id") or "")
    frame = row.get("frame_id", row.get("anchor"))
    if not scenario or not route or frame is None:
        raise ValueError("event-balance row lacks scenario/route_id/frame_id")
    return scenario, route, int(frame)


def development_route_groups() -> frozenset[str]:
    """读取 Phase3 明确声明已经参与规则开发的 physical routes。"""
    from qwen3vl_local.action_prior.build_dataset import route_group

    root = Path(__file__).resolve().parents[1] / "sft_new_loop_phase3"
    groups = set()
    for name in ("development_route_groups_20260907.json", "development_route_groups_20260910.json"):
        groups.update(json.loads((root / name).read_text(encoding="utf-8"))["groups"])
    normalized = set()
    for value in groups:
        scenario, separator, run_id = str(value).partition("/")
        if not separator or not scenario or not run_id:
            raise ValueError(f"invalid Phase3 development route group: {value!r}")
        normalized.add(route_group(scenario, run_id))
    return frozenset(normalized)


@dataclass(frozen=True)
class EventBalanceSource:
    """内容身份与可迁移路径分离：path 只审计，不参与 checkpoint identity。"""
    path: str
    sha256: str
    manifest_sha256: str
    mapping_contract_hash: str
    candidate_sha256: str
    action_dataset_hashes: Mapping[str, str]
    rows: int
    schema: str

    def identity_dict(self) -> Dict[str, Any]:
        return dict(
            schema=self.schema, sha256=self.sha256, mapping_contract_hash=self.mapping_contract_hash,
            candidate_sha256=self.candidate_sha256,
            action_dataset_hashes=dict(self.action_dataset_hashes),
            mapping_policy=EVENT_BALANCE_MAPPING_POLICY,
            bucket_weights=dict(EVENT_BALANCE_WEIGHTS),
            normal_pool_policy="confirmed_regular_only; unconfirmed and special_filtered are excluded",
        )

    def audit_dict(self) -> Dict[str, Any]:
        return dict(
            path=self.path, rows=self.rows, manifest_sha256=self.manifest_sha256,
            **self.identity_dict(),
        )


class EventBalanceIndex:
    """全帧 map：采样 membership、固定 scene context 和状态均按 frame identity 保存。"""
    def __init__(self, path: str | Path):
        source = Path(path).expanduser().resolve()
        manifest_path = source.with_name("manifest.json")
        if not source.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"event balance requires {source} and sibling manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != FULL_MANIFEST_SCHEMA or manifest.get("index_schema") != FULL_INDEX_SCHEMA:
            raise ValueError(
                "event balance source is v1/stale or not a current full-frame mapping; "
                "rebuild it with build_event_balance_index.py"
            )
        if manifest.get("mapping_policy") != EVENT_BALANCE_MAPPING_POLICY:
            raise ValueError("event balance mapping policy is stale; rebuild the full-frame mapping")
        if manifest.get("index_file") != source.name or manifest.get("index_sha256") != file_hash(source):
            raise ValueError("event balance manifest/index hash mismatch")
        candidate_sha256 = str(manifest.get("candidate_sha256", ""))
        if len(candidate_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in candidate_sha256.lower()):
            raise ValueError("event balance manifest lacks the validated Phase3 candidate content hash")
        action_hashes = manifest.get("action_dataset_hashes")
        if (
            not isinstance(action_hashes, dict)
            or set(action_hashes) != {"train", "val", "test"}
            or any(len(str(value)) != 64 for value in action_hashes.values())
        ):
            raise ValueError(
                "event balance manifest lacks the action dataset split hashes; rebuild with --action-data-dir"
            )
        from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapping_contract_hash
        if manifest.get("mapping_contract_hash") != mapping_contract_hash():
            raise ValueError("event balance mapping contract is stale; rebuild full-frame mapping")
        self.path, self.manifest, self.records, self._validated_action_dirs = source, manifest, {}, set()
        for number, line in enumerate(source.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != FULL_INDEX_SCHEMA or row.get("mapping_contract_hash") != manifest["mapping_contract_hash"]:
                raise ValueError(f"{source}:{number}: invalid/stale full-frame event mapping row")
            key = _identity(row)
            if key in self.records:
                raise ValueError(f"{source}:{number}: duplicate full-frame identity {key}")
            contexts = tuple(str(x) for x in row.get("special_buckets", ()))
            eligible = tuple(str(x) for x in row.get("eligible_buckets", ()))
            status = str(row.get("status", ""))
            if (any(value not in SPECIAL_BUCKETS for value in contexts)
                    or any(value not in contexts for value in eligible)
                    or status not in (
                SPECIAL_ELIGIBLE, SPECIAL_FILTERED, CONFIRMED_REGULAR, UNCONFIRMED
            ) or (bool(contexts) != status.startswith("special_"))
                    or (bool(eligible) != (status == SPECIAL_ELIGIBLE))):
                raise ValueError(f"{source}:{number}: invalid event mapping status/context")
            self.records[key] = dict(
                special_buckets=contexts, eligible_buckets=eligible, status=status,
                scene_contexts=tuple(str(x) for x in row.get("scene_contexts", ())),
                source_split=str(row.get("source_split", "")),
            )
        if not self.records:
            raise ValueError("event balance full-frame mapping is empty")
        self.source = EventBalanceSource(
            path=str(source), sha256=file_hash(source), manifest_sha256=file_hash(manifest_path),
            mapping_contract_hash=str(manifest["mapping_contract_hash"]), rows=len(self.records),
            candidate_sha256=candidate_sha256, schema=FULL_INDEX_SCHEMA,
            action_dataset_hashes={key: str(value) for key, value in action_hashes.items()},
        )

    def validate_action_dataset(self, data_dir: str | Path) -> None:
        """full map 必须对应当前 action 三 split，拒绝半场景/旧索引误传。"""
        directory = Path(data_dir).resolve()
        if directory in self._validated_action_dirs:
            return
        actual = {split: file_hash(directory / f"{split}.jsonl") for split in ("train", "val", "test")}
        if actual != dict(self.source.action_dataset_hashes):
            raise ValueError(
                "event balance full map was built for a different action dataset index; "
                "rebuild it with this --action-data-dir"
            )
        self._validated_action_dirs.add(directory)

    def annotate(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """所有 split 固定附加同一上下文；action 的 split 负责隔离，开发路线强制 train。"""
        development = development_route_groups()
        for row in rows:
            record = self.records.get(_identity(row))
            if record is None:
                row.update(event_balance_buckets=[], event_balance_all_special_buckets=[],
                           event_balance_scene_contexts=[], event_balance_status=UNCONFIRMED,
                           event_balance_mapping_missing=True, event_balance_source_split_mismatch=False)
                continue
            row.update(
                event_balance_buckets=list(record["eligible_buckets"]),
                event_balance_all_special_buckets=list(record["special_buckets"]),
                event_balance_scene_contexts=list(record["scene_contexts"]),
                event_balance_status=record["status"], event_balance_mapping_missing=False,
                event_balance_source_split_mismatch=bool(record["source_split"] and record["source_split"] != row.get("split")),
                event_balance_development_route=bool(str(row.get("route_group", "")) in development),
            )


def source_for_args(args) -> EventBalanceIndex:
    wanted = str(Path(args.event_balance_index).expanduser().resolve())
    cached = _SOURCE_CACHE.get(wanted)
    if cached is None:
        cached = EventBalanceIndex(wanted)
        _SOURCE_CACHE[wanted] = cached
    return cached


def source_contract(args) -> Dict[str, Any] | None:
    active = getattr(args, "sampling_mode", "uniform") == SAMPLING_MODE_EVENT_BALANCED or getattr(args, "event_balanced_scene_priors", False)
    if not active:
        return None
    if getattr(args, "event_balance_index", ""):
        value = source_for_args(args).source.identity_dict()
        args.event_balance_source_identity = value
        return value
    return getattr(args, "event_balance_source_identity", None)


def source_audit(args) -> Dict[str, Any] | None:
    return source_for_args(args).source.audit_dict() if getattr(args, "event_balance_index", "") else None


def annotate_rows(args, rows: Iterable[Mapping[str, Any]]) -> None:
    active = getattr(args, "sampling_mode", "uniform") == SAMPLING_MODE_EVENT_BALANCED or getattr(args, "event_balanced_scene_priors", False)
    if active and getattr(args, "event_balance_index", ""):
        source = source_for_args(args)
        source.validate_action_dataset(args.data_dir)
        source.annotate(rows)


def _route_key(row: Mapping[str, Any]) -> Tuple[str, str]:
    return str(row.get("scenario", "")), str(row.get("route_group") or row.get("run_id", ""))


def weighted_quotas(total: int) -> Dict[str, int]:
    unit = sum(EVENT_BALANCE_WEIGHTS.values())
    if total <= 0 or int(total) % unit:
        raise ValueError(f"event-balanced epoch must be a positive multiple of {unit}")
    base = int(total) // unit
    return {key: base * EVENT_BALANCE_WEIGHTS[key] for key in (*SPECIAL_BUCKETS, REGULAR_BACKGROUND)}


def available_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for row in rows:
        for bucket in row.get("event_balance_buckets", ()): counts[bucket] += 1
        if row.get("event_balance_status") == CONFIRMED_REGULAR: counts[REGULAR_BACKGROUND] += 1
        counts[f"status/{row.get('event_balance_status', UNCONFIRMED)}"] += 1
    return dict(counts)


def _ordered_route_cycle(rows, rng, route_diverse):
    values = list(rows)
    if not route_diverse:
        rng.shuffle(values); return values
    by_route = defaultdict(list)
    for row in values: by_route[_route_key(row)].append(row)
    keys = sorted(by_route); rng.shuffle(keys)
    for items in by_route.values(): rng.shuffle(items)
    result, depth = [], 0
    while len(result) < len(values):
        for key in keys:
            if depth < len(by_route[key]): result.append(by_route[key][depth])
        depth += 1
    return result


class _MinCostFlow:
    """用势函数和最短增广路求整数最小费用流；只在压缩后的事件归属图上运行。"""

    def __init__(self, nodes):
        """初始化残量图；边记录终点、反向边位置、剩余容量和单位费用。"""
        self.graph = [[] for _ in range(nodes)]

    def add(self, start, end, capacity, cost=0):
        """添加正反向边，返回正向边位置供恢复分配使用。"""
        forward = [end, len(self.graph[end]), int(capacity), int(cost)]
        backward = [start, len(self.graph[start]), 0, -int(cost)]
        self.graph[start].append(forward)
        self.graph[end].append(backward)
        return len(self.graph[start]) - 1

    def flow(self, source, sink, requested):
        """每次按整条路径瓶颈批量增广，避免逐个 presentation 求解。"""
        from heapq import heappop, heappush

        potential = [0] * len(self.graph)
        total = cost = 0
        while total < requested:
            distance = [math.inf] * len(self.graph)
            previous = [None] * len(self.graph)
            distance[source] = 0
            queue = [(0, source)]
            while queue:
                current, node = heappop(queue)
                if current != distance[node]:
                    continue
                for index, (target, _back, capacity, price) in enumerate(self.graph[node]):
                    candidate = current + price + potential[node] - potential[target]
                    if capacity and candidate < distance[target]:
                        distance[target] = candidate
                        previous[target] = (node, index)
                        heappush(queue, (candidate, target))
            if previous[sink] is None:
                break
            for node, value in enumerate(distance):
                if value < math.inf:
                    potential[node] += value
            amount, node = requested - total, sink
            while node != source:
                parent, index = previous[node]
                amount = min(amount, self.graph[parent][index][2])
                node = parent
            node = sink
            while node != source:
                parent, index = previous[node]
                edge = self.graph[parent][index]
                edge[2] -= amount
                self.graph[node][edge[1]][2] += amount
                cost += amount * edge[3]
                node = parent
            total += amount
        return total, cost


def _groups(rows):
    groups = {key: [] for key in SPECIAL_BUCKETS}
    groups[REGULAR_BACKGROUND] = []
    for row in rows:
        for key in row.get("event_balance_buckets", ()):
            groups[key].append(row)
        if row.get("event_balance_status") == CONFIRMED_REGULAR:
            groups[REGULAR_BACKGROUND].append(row)
    missing = [key for key, values in groups.items() if not values]
    if missing:
        raise ValueError(f"event-balanced sampling missing={missing} available={available_counts(rows)}")
    return groups


def _joint_allocation(rows, quotas, *, repeat_cap, seed=0, route_diverse=True):
    """固定配额和全局重复上限下，精确最大化整个 epoch 的唯一帧数。"""
    if repeat_cap < 1:
        raise ValueError("event_balance_max_frame_repeats must be positive")
    groups = _groups(rows)
    keys = (*SPECIAL_BUCKETS, REGULAR_BACKGROUND)
    frame_rows, memberships = {}, defaultdict(set)
    for bucket in keys:
        for row in groups[bucket]:
            identity = _identity(row)
            frame_rows[identity] = row
            memberships[identity].add(bucket)
    by_membership = defaultdict(list)
    for identity, buckets in memberships.items():
        by_membership[tuple(key for key in keys if key in buckets)].append(frame_rows[identity])

    # 同一归属组的帧在配额约束下可以互换。压缩后至多 2^10-1 个 special 组加背景组，
    # 无须为几十万帧逐个构造最小费用流节点；具体帧随后按 route 轮转展开。
    rng = random.Random(f"event-balance-min-cost-v1:{seed}:{sum(quotas.values())}")
    signatures = sorted(by_membership)
    rng.shuffle(signatures)
    bucket_nodes = {key: index + 1 for index, key in enumerate(keys)}
    sink = 1 + len(keys) + len(signatures)
    flow = _MinCostFlow(sink + 1)
    edges = []
    for bucket in keys:
        flow.add(0, bucket_nodes[bucket], quotas[bucket])
    for index, signature in enumerate(signatures):
        node = 1 + len(keys) + index
        count = len(by_membership[signature])
        for bucket in signature:
            edge = flow.add(bucket_nodes[bucket], node, quotas[bucket])
            edges.append((bucket, signature, bucket_nodes[bucket], edge))
        # 每帧首次使用免费，额外使用每次费用 1。固定总流量时，最小费用恰好
        # 等价于最大化唯一帧覆盖；反向边允许跨桶撤销和重新分配共享帧。
        flow.add(node, sink, count)
        flow.add(node, sink, count * (repeat_cap - 1), cost=1)
    total = sum(quotas.values())
    sent, repeated = flow.flow(0, sink, total)
    if sent != total:
        return None
    group_assignments = defaultdict(dict)
    for bucket, signature, node, edge in edges:
        group_assignments[signature][bucket] = quotas[bucket] - flow.graph[node][edge][2]
    assigned = {key: Counter() for key in keys}
    for signature in signatures:
        ordered = _ordered_route_cycle(by_membership[signature], rng, route_diverse)
        cursor = 0
        # 同组内持续轮转，跨桶共享同一个 cursor：先覆盖所有不同帧再重复。
        # 总组容量 <= len(ordered)*repeat_cap，因此展开后每帧也满足全局上限。
        for bucket in signature:
            amount = group_assignments[signature][bucket]
            rounds, remainder = divmod(amount, len(ordered))
            if rounds:
                for row in ordered:
                    assigned[bucket][_identity(row)] += rounds
            for offset in range(remainder):
                assigned[bucket][_identity(ordered[(cursor + offset) % len(ordered)])] += 1
            cursor = (cursor + remainder) % len(ordered)
    return groups, frame_rows, assigned, dict(
        objective="maximize_global_unique_frames",
        optimal_unique_frames=total - repeated,
        repeat_presentations=repeated,
        membership_groups=len(signatures),
    )


def event_balanced_total(rows: Sequence[Mapping[str, Any]], *, requested: int, repeat_cap: int, world: int) -> int:
    """预算在加载模型前用共享 frame 的联合 capacity 求解，保证 epoch 可实际抽出。"""
    if repeat_cap < 1 or world < 1:
        raise ValueError("event_balance_max_frame_repeats and world must be positive")
    unit = sum(EVENT_BALANCE_WEIGHTS.values())
    multiple = math.lcm(unit, int(world))
    counts = available_counts(rows)
    independent_max = min(
        [int(counts.get(key, 0)) * repeat_cap for key in SPECIAL_BUCKETS]
        + [int(counts.get(REGULAR_BACKGROUND, 0)) * repeat_cap // 2]
    )
    if requested:
        total = int(requested)
        if total % multiple:
            raise ValueError(
                f"event-balanced epoch budget must be divisible by lcm(12, world_size)={multiple}"
            )
        quotas = weighted_quotas(total)
        if _joint_allocation(rows, quotas, repeat_cap=repeat_cap) is None:
            raise ValueError("event-balanced epoch budget is infeasible under shared-frame repeat caps")
        return total
    # base quota x must make 12*x divisible by world. Search only such values and use
    # max-flow, rather than independently counting each bucket and discovering conflict mid-epoch.
    scale = multiple // unit
    high = independent_max // scale
    low, feasible = 1, 0
    while low <= high:
        middle = (low + high) // 2
        total = middle * scale * unit
        if _joint_allocation(rows, weighted_quotas(total), repeat_cap=repeat_cap) is not None:
            feasible = middle
            low = middle + 1
        else:
            high = middle - 1
    if not feasible:
        raise ValueError("event-balanced epoch has no jointly feasible full DDP presentation")
    return feasible * scale * unit


def build_event_balanced_epoch(rows: Sequence[Mapping[str, Any]], *, total: int, seed: int,
                               route_diverse: bool = True, repeat_cap: int = 8):
    """固定 1:…:1:2 配额，同时限制每个 frame 在整个 epoch 的总出现次数。"""
    quotas = weighted_quotas(total)
    allocation = _joint_allocation(
        rows, quotas, repeat_cap=repeat_cap, seed=seed, route_diverse=route_diverse
    )
    if allocation is None:
        raise ValueError("event-balanced epoch quota is infeasible under shared-frame repeat caps")
    _groups_value, frame_rows, assigned, allocation_audit = allocation
    selected = []
    for bucket, assignments in assigned.items():
        for identity, amount in assignments.items():
            for _ in range(amount):
                item = dict(frame_rows[identity])
                item["event_balance_bucket"] = bucket
                selected.append(item)
    rng = random.Random(f"event-balance-v4:{seed}:{total}")
    rng.shuffle(selected)
    used = Counter(_identity(row) for row in selected)
    repeats = Counter(used.values())
    if len(used) != allocation_audit["optimal_unique_frames"] or max(used.values(), default=0) > repeat_cap:
        raise AssertionError("event-balanced allocation expansion violated the global optimum/cap")
    return selected, dict(
        schema="action_prior_event_balanced_epoch_v4", seed=int(seed), total=int(total), quotas=quotas,
        sampled=dict(Counter(row["event_balance_bucket"] for row in selected)), available=available_counts(rows),
        unique_frames=len(used), max_frame_repeats=max(used.values(), default=0),
        repeat_histogram={str(k): v for k, v in sorted(repeats.items())}, repeat_cap=int(repeat_cap),
        unique_routes={key: len({_route_key(row) for row in selected if row["event_balance_bucket"] == key}) for key in quotas},
        joint_allocation=True,
        **allocation_audit,
        bucket_unique_frames={key: len(assigned[key]) for key in quotas},
        bucket_max_frame_repeats={key: max(assigned[key].values(), default=0) for key in quotas},
        diversity_first=True,
        route_diverse=bool(route_diverse),
    )
