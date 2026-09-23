"""Phase3 的 route-aware 采样与审计工具。

动作 span 会在同一 route 内产生连续滑窗，直接 shuffle+truncate 会让单条长 span
占据过多权重。这里按 ``(scenario, route_id)`` 轮转：先从每条 route 取一帧，再取
每条 route 的第二帧，以此类推。实现与 `sft_new_loop_phase2.sampling` 同源。
"""

from __future__ import annotations

import random
import math
import json
from fractions import Fraction
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple, TypeVar


T = TypeVar("T")

SUPPORT_BALANCE_VERSION = "primary_action_capacity_return_v5_fair_cursor"


def sampling_action(answers, context_id):
    """Only six primary buckets; compound evidence is supervision, not a rare class.

    A RESUME+LEFT example shares LEFT's quota while binary keeps both YES labels.
    Never turn a rare but valid maneuver into KEEP or an unconditioned example.
    """
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID
    from qwen3vl_local.sft_new_loop_phase3.choice_semantics import primary_choice
    return primary_choice(answers, CONTEXT_BY_ID[context_id].action_keys)


def support_diagnostic(frames: int, routes: int, presentations=None) -> Dict[str, Any]:
    """只报告复查线索；100帧/10物理路线阈值不参与配额、过滤或标签。

    缺失动作不合成为 KEEP；低频也不自动撤去条件为 UNCOND。
    """
    return dict(frames=frames, physical_routes=routes,
                review_flags=(["fewer_than_100_frames"] if frames < 100 else [])
                             + (["fewer_than_10_physical_routes"] if routes < 10 else []),
                diagnostic_only=True, presentations=presentations,
                presentations_per_available_frame=(presentations / frames
                    if presentations is not None and frames else None))


def support_aware_quota(
    capacities: Mapping[str, int],
    target: int,
    *,
    repeat_cap: int = 8,
    smooth_power: float = 0.5,
    mode: str = "cycle_even",
) -> Dict[str, int]:
    """先完整覆盖自然池，再对余量做容量内均衡；稀少格子不被单独循环放大。

    整个事件不足预算时才重复完整池，单格最多 ceil(target / pool_size) 轮。
    不用 val/test 支持率决定训练取舍，也不把稀少动作重标为 KEEP/UNCOND。
    支持 mode="cycle_even" (历史默认合同) 与 mode="smooth_cap" (温和幂律提权+回流)。
    """
    if mode not in ("cycle_even", "smooth_cap"):
        raise ValueError(f"unknown sampling policy: {mode}")
    if type(repeat_cap) is not int or repeat_cap < 1 or not math.isfinite(smooth_power) or not 0 <= smooth_power <= 1:
        raise ValueError("repeat_cap must be positive and smooth_power finite in [0, 1]")
    if type(target) is not int or target < 0 or any(type(n) is not int or n < 0 for n in capacities.values()):
        raise ValueError("support-aware quotas require nonnegative integer capacities/target")
    size = sum(capacities.values())
    if not size:
        if target:
            raise ValueError("support-aware quota has no supported samples")
        return dict.fromkeys(capacities, 0)

    if mode == "smooth_cap":
        max_repeat = max(1, int(repeat_cap))
        caps = {k: int(capacities[k]) * max_repeat for k in capacities}
        total_cap = sum(caps.values())
        if target > total_cap:
            raise ValueError(
                f"smooth_cap quota infeasible: target={target} exceeds total capacity={total_cap} "
                f"under repeat_cap={max_repeat} (capacities={dict(capacities)})"
            )

        keys = sorted(capacities.keys())
        active = [k for k in keys if capacities[k] > 0]
        quotas: Dict[str, int] = {k: 0 for k in keys}
        remaining = int(target)

        while remaining > 0 and active:
            weights = {k: float(capacities[k]) ** float(smooth_power) for k in active}
            total_w = sum(weights.values())
            if total_w <= 0:
                break
            ideal = {k: remaining * weights[k] / total_w for k in active}
            saturated = [k for k in active if (quotas[k] + ideal[k]) >= caps[k]]
            if saturated:
                for k in saturated:
                    room = caps[k] - quotas[k]
                    quotas[k] += room
                    remaining -= room
                    active.remove(k)
                continue

            allocated = {k: int(ideal[k]) for k in active}
            sub_alloc = sum(allocated.values())
            for k in active:
                quotas[k] += allocated[k]
            remaining -= sub_alloc

            if remaining > 0:
                remainders = sorted(active, key=lambda k: (ideal[k] - allocated[k], capacities[k]), reverse=True)
                for k in remainders[:remaining]:
                    if quotas[k] < caps[k]:
                        quotas[k] += 1
                        remaining -= 1
                break

        if sum(quotas.values()) != target:
            raise ValueError(
                f"smooth_cap failed to satisfy target {target}: allocated {sum(quotas.values())} "
                f"(capacities={dict(capacities)}, repeat_cap={max_repeat})"
            )
        return quotas

    cycles, remainder = divmod(target, size)
    partial = even_quota_with_capacity(capacities, remainder)
    return {key: cycles * n + partial[key] for key, n in capacities.items()}


def primary_action_distribution(signature_counts: Mapping[str, int]) -> Dict[str, Any]:
    """按主要动作投影报告计数；STOP+跨线仍是STOP，INVALID单列分母。"""
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS
    from qwen3vl_local.sft_new_loop_phase3.choice_semantics import primary_choice
    counts = Counter()
    for signature, count in signature_counts.items():
        if type(count) is not int or count < 0:
            raise ValueError("signature count must be a nonnegative integer")
        if signature == "INVALID":
            counts["INVALID"] += count
            continue
        parts = set(signature.split("+"))
        if parts in ({"NONE"}, {"KEEP"}):
            parts = set()
        if not parts <= set(ACTION_KEYS):
            raise ValueError(f"unknown action signature: {signature}")
        counts[primary_choice({key: key in parts for key in ACTION_KEYS})] += count
    total = sum(counts.values())
    invalid = counts.get("INVALID", 0)
    valid = total - invalid
    return {"projection": "STOP_then_first_crossing_then_speed_else_KEEP",
            "counts": dict(counts), "total": total, "valid": valid, "invalid": invalid,
            "stop_fraction_all": counts.get("STOP", 0) / total if total else None,
            "stop_fraction_valid": counts.get("STOP", 0) / valid if valid else None}


def _route_key(item: Any) -> Tuple[str, str]:
    """按物理路线轮转，Rep和重复采集不能冒充独立支持。"""

    row = getattr(item, "row", item)
    if isinstance(row, Mapping):
        scenario, route = str(row.get("scenario", "")), str(row.get("route_group") or row.get("route_id") or row.get("run_id", ""))
    else:
        scenario, route = str(getattr(row, "scenario", "")), str(getattr(row, "route_id", ""))
    stem = re.sub(r"_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$", "", route)
    stem = re.sub(r"_route0$", "", stem)
    return scenario, re.sub(r"_Rep\d+_", "_", stem)


def _item_identity(item: Any) -> Tuple[str, str, Any]:
    """提取样本全局唯一标识 (scenario, route, frame/anchor)。"""
    row = getattr(item, "row", item)
    if isinstance(row, Mapping):
        scenario = str(row.get("scenario", ""))
        route = str(row.get("route_id") or row.get("run_id") or "")
        frame = row.get("frame_id", row.get("anchor"))
    else:
        scenario = str(getattr(row, "scenario", ""))
        route = str(getattr(row, "route_id", None) or getattr(row, "run_id", ""))
        frame = getattr(row, "frame_id", None)
        if frame is None:
            frame = getattr(row, "anchor", None)
    if not route or frame is None:
        raise ValueError("sampling requires a route and frame/anchor identity")
    return scenario, route, str(frame)


def extract_item_action_and_context(item: Any, fallback_context_id: str | None = None) -> Tuple[str, str]:
    """统一提取样本的主要动作与 context_id，兼容 dict, WorkItem, FrameRow。"""
    row = getattr(item, "row", item)
    if isinstance(row, Mapping):
        token_name = row.get("action_token", {}).get("name") if isinstance(row.get("action_token"), Mapping) else None
        if token_name:
            ctx = row.get("context_id") or (row.get("event_balance_buckets", [None])[0] if row.get("event_balance_buckets") else fallback_context_id)
            return str(token_name), str(ctx or "UNKNOWN")

    ctx = None
    if isinstance(row, Mapping):
        ctx = row.get("context_id", fallback_context_id)
    else:
        ctx = getattr(row, "context_id", fallback_context_id)
    if ctx is None:
        ctx = fallback_context_id
    if ctx is None:
        raise ValueError(f"cannot determine context_id for item of type {type(item)}")

    answers = None
    if isinstance(row, Mapping):
        answers = row.get("answers") or row.get("action_labels")
    else:
        answers = getattr(row, "answers", None) or getattr(row, "action_labels", None)

    if answers is None:
        raise ValueError(f"cannot find answers or action_labels in {type(item)}")

    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID, ACTION_KEYS
    if str(ctx) in CONTEXT_BY_ID:
        act = sampling_action(answers, str(ctx))
    else:
        from qwen3vl_local.sft_new_loop_phase3.choice_semantics import primary_choice
        act = primary_choice(answers, ACTION_KEYS)
    return str(act), str(ctx)


def route_diverse_sample(items: Sequence[T], *, target: int, rng: random.Random) -> List[T]:
    """按 route 轮转抽样；样本不足时再均匀循环已有结果。"""

    source = list(items)
    if not source or int(target) <= 0:
        return source if int(target) != 0 else []

    buckets: Dict[Tuple[str, str], List[T]] = defaultdict(list)
    for item in source:
        buckets[_route_key(item)].append(item)
    route_keys = sorted(buckets)
    rng.shuffle(route_keys)
    for key in route_keys:
        rng.shuffle(buckets[key])

    selected: List[T] = []
    depth = 0
    wanted_without_replacement = min(int(target), len(source))
    while len(selected) < wanted_without_replacement:
        added = False
        for key in route_keys:
            bucket = buckets[key]
            if depth >= len(bucket):
                continue
            selected.append(bucket[depth])
            added = True
            if len(selected) >= wanted_without_replacement:
                break
        if not added:
            break
        depth += 1

    if int(target) > len(selected):
        base = list(selected)
        if not base:
            return []
        selected.extend(base[idx % len(base)] for idx in range(int(target) - len(selected)))
    return selected


def deterministic_cycle_order(items: Sequence[T], *, cycle: int, master_seed: Any = 0) -> List[T]:
    """生成第 cycle 周期的确定性物理路线轮转全排列。在同一 cycle 内，排列完全恒定。"""
    source = sorted(items, key=_item_identity)
    n = len(source)
    if n <= 1:
        return source
    buckets: Dict[Tuple[str, str], List[T]] = defaultdict(list)
    for it in source:
        buckets[_route_key(it)].append(it)
    route_keys = sorted(buckets)
    cycle_rng = random.Random(f"{master_seed}:cycle_{cycle}")
    cycle_rng.shuffle(route_keys)
    for k in route_keys:
        cycle_rng.shuffle(buckets[k])

    ordered: List[T] = []
    depth = 0
    active = route_keys
    while active:
        remaining = []
        for key in active:
            ordered.append(buckets[key][depth])
            if depth + 1 < len(buckets[key]):
                remaining.append(key)
        active = remaining
        depth += 1
    return ordered


def route_diverse_cursor_sample(
    items: Sequence[T],
    *,
    target: int,
    cursor: int = 0,
    rng: random.Random | None = None,
    master_seed: Any = 0,
) -> Tuple[List[T], int]:
    """带有确定性周期与跨 Epoch 游标的物理路线轮转采样。

    主队列按固定种子构建，周期边界复用同一队列；
    任意连续 N 次呈现覆盖 N 帧，最终训练顺序由外层 epoch shuffle。
    """
    if target < 0 or cursor < 0:
        raise ValueError("target/cursor must be nonnegative")
    if not items:
        if target:
            raise ValueError("cannot sample an empty pool")
        return [], int(cursor)
    # 固定环形主队列。周期边界不重新洗牌，因此任意长度 <= N*cap 的切片
    # 都满足单帧上限；epoch 的最终 batch 顺序由外层独立 shuffle。
    ordered = deterministic_cycle_order(items, cycle=0, master_seed=master_seed)
    return [ordered[i % len(ordered)] for i in range(cursor, cursor + target)], cursor + target


def even_quota_with_capacity(capacities: Mapping[str, int], target: int) -> Dict[str, int]:
    """在若干子桶间尽量均分 target，并把超出容量的份额确定性地回流给其它桶。"""

    keys = sorted(str(key) for key in capacities)
    quotas: Dict[str, int] = {key: 0 for key in keys}
    remaining = int(target)
    active = [key for key in keys if int(capacities[key]) > 0]
    while remaining > 0 and active:
        share, extra = divmod(remaining, len(active))
        if share == 0:
            for idx in range(extra):
                key = active[idx % len(active)]
                if quotas[key] < int(capacities[key]):
                    quotas[key] += 1
                    remaining -= 1
            break
        progressed = False
        for key in list(active):
            room = int(capacities[key]) - quotas[key]
            take = min(room, share)
            if take > 0:
                quotas[key] += take
                remaining -= take
                progressed = True
            if quotas[key] >= int(capacities[key]):
                active.remove(key)
        if not progressed:
            break
    return quotas


def route_event_action_sample(
    items: Sequence[T], *, context_id: str, target: int, rng: random.Random,
    action_fn=None, repeat_cap: int = 8, smooth_power: float = 0.5,
    cursor_state: Dict[str, int] | None = None,
    global_frame_usage: Counter | None = None, master_seed: Any = 0,
) -> Tuple[List[T], Dict[str, Any]]:
    """单事件接口也执行真实帧约束；跨事件请使用联合入口，允许回退重分配。"""
    selected, audit = hierarchical_event_action_epoch_sample(
        {context_id: items}, {context_id: target}, repeat_cap=repeat_cap,
        smooth_power=smooth_power, cursor_state=cursor_state, master_seed=master_seed,
        action_fn=action_fn, rng=rng, initial_frame_usage=global_frame_usage,
    )
    if global_frame_usage is not None:
        global_frame_usage.update(_item_identity(it) for it in selected)
    return selected, {**audit["event_audits"][context_id], "next_cursors": audit["next_cursors"]}


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



def hierarchical_event_action_epoch_sample(
    items_by_event: Mapping[str, Sequence[T]], targets_by_event: Mapping[str, int], *,
    repeat_cap: int = 8, smooth_power: float = 0.5,
    cursor_state: Dict[str, int] | None = None, master_seed: Any = 0,
    action_fn=None, rng: random.Random | None = None,
    initial_frame_usage: Mapping | None = None, route_diverse: bool = True,
    pool_history: Mapping | None = None,
) -> Tuple[List[T], Dict[str, Any]]:
    """事件预算为硬约束，动作目标可回流，全局帧容量通过压缩最小费用流求解。

    相同 event/action 归属集合的帧共享节点，避免为几十万帧逐个求流。
    优先最少偏离动作目标，再最大化本轮不同帧数，最后按历史曝光公平分配。
    反向边允许释放共享帧。
    """
    support_aware_quota({}, 0, repeat_cap=repeat_cap, smooth_power=smooth_power, mode="smooth_cap")
    if any(type(n) is not int or n < 0 for n in targets_by_event.values()):
        raise ValueError("event targets must be nonnegative integers")
    initial = Counter(initial_frame_usage or {})
    if any(n < 0 or n > repeat_cap for n in initial.values()):
        raise ValueError("reserved frames exceed global repeat_cap")
    cursors = dict(cursor_state or {})
    if any(type(n) is not int or n < 0 for n in cursors.values()):
        raise ValueError("sampling cursors must be nonnegative integers")
    cells, memberships, objects = {}, defaultdict(set), {}
    event_audits, quotas = {}, {}
    for event, target in sorted(targets_by_event.items()):
        per_action = defaultdict(dict)
        for item in items_by_event.get(event, ()):
            ident = _item_identity(item)
            action = str(action_fn(item) if action_fn else extract_item_action_and_context(item, event)[0])
            cell = (event, action)
            if any(c[0] == event and c != cell for c in memberships[ident]):
                raise ValueError(f"conflicting primary actions for {event}: {ident}")
            memberships[ident].add(cell)
            per_action[action][ident] = item
            objects[(cell, ident)] = item
        capacities = {a: len(v) for a, v in per_action.items()}
        wanted = support_aware_quota(capacities, target, repeat_cap=repeat_cap,
                                     smooth_power=smooth_power, mode="smooth_cap")
        for action, values in sorted(per_action.items()):
            cells[(event, action)] = values
            quotas[(event, action)] = wanted[action]
        event_audits[event] = dict(context_id=event, target=target,
                                  action_capacities=capacities, action_quotas=wanted)
    # 将预留 INVALID 等外部呈现所消耗的容量纳入图；相同剩余容量才可压缩。
    pools = defaultdict(list)
    for ident, member in memberships.items():
        pools[(tuple(sorted(member)), initial[ident])].append(ident)
    def pool_key(signature):
        members, reserved = signature
        key = ("/".join(members[0]) if len(members) == 1 else json.dumps(members, separators=(",", ":")))
        return key + (f":reserved={reserved}" if reserved else "")

    history = dict(pool_history or {})
    for key, value in history.items():
        if (not isinstance(value, Mapping) or any(type(value.get(field)) is not int or value[field] < 0
                                                for field in ("presentations", "skipped_epochs"))
                or value["presentations"] != cursors.get(key, 0)):
            raise ValueError(f"pool history/cursor mismatch: {key}")
    # 游标就是累计呈现数。公平性仅打破原目标的同成本解：较久未被选中的池
    # 优先，同等等待时优先每帧历史曝光较少的池。整数费用严格分层，不以公平性牺牲
    # 动作目标或本轮唯一帧数；没有可行同成本替代时，不承诺强制该池获得配额。
    fair_order = sorted(pools, key=lambda sig: (
        Fraction(cursors.get(pool_key(sig), 0), len(pools[sig])),
        -history.get(pool_key(sig), {}).get("skipped_epochs", 0), pool_key(sig)))
    fair_rank = {signature: rank for rank, signature in enumerate(fair_order)}
    max_skipped = max((history.get(pool_key(sig), {}).get("skipped_epochs", 0) for sig in pools), default=0)
    fair_cost = {sig: (max_skipped - history.get(pool_key(sig), {}).get("skipped_epochs", 0)) * len(pools)
                 + fair_rank[sig] for sig in pools}
    events = sorted(targets_by_event)
    cell_keys, signatures = sorted(cells), sorted(pools)
    enodes = {e: i + 1 for i, e in enumerate(events)}
    cnodes = {c: len(enodes) + i + 1 for i, c in enumerate(cell_keys)}
    base = len(enodes) + len(cnodes) + 1
    sink = base + len(signatures)
    flow = _MinCostFlow(sink + 1)
    total = sum(targets_by_event.values())
    repeat_cost = total * max(fair_cost.values(), default=0) + 1
    overflow_cost = (total + 1) * repeat_cost
    for event in events:
        flow.add(0, enodes[event], targets_by_event[event])
    for cell in cell_keys:
        flow.add(enodes[cell[0]], cnodes[cell], quotas[cell])
        flow.add(enodes[cell[0]], cnodes[cell], targets_by_event[cell[0]], overflow_cost)
    edges = defaultdict(list)
    for i, signature in enumerate(signatures):
        members, used = signature
        n = len(pools[signature])
        for cell in members:
            cap = targets_by_event[cell[0]]
            edge = flow.add(cnodes[cell], base + i, cap)
            edges[signature].append((cell, cnodes[cell], edge, cap))
        if used == 0:
            flow.add(base + i, sink, n, fair_cost[signature])
            flow.add(base + i, sink, n * (repeat_cap - 1), repeat_cost + fair_cost[signature])
        else:
            flow.add(base + i, sink, n * (repeat_cap - used), repeat_cost + fair_cost[signature])
    sent, cost = flow.flow(0, sink, total)
    if sent != total:
        raise ValueError(f"insufficient shared frame capacity: target={total}, feasible={sent}, repeat_cap={repeat_cap}")
    selected, attribution, actual = [], [], Counter()
    next_cursors = dict(cursors)
    next_history = {key: dict(value) for key, value in history.items()}
    pool_audit = {}
    for signature in signatures:
        members, reserved = signature
        # 游标键描述池的语义归属，不依赖输入顺序或 epoch seed。
        key = pool_key(signature)
        assignments = [(c, cap - flow.graph[node][edge][2]) for c, node, edge, cap in edges[signature]]
        amount = sum(n for _, n in assignments)
        previous = cursors.get(key, 0)
        skipped = 0 if amount else history.get(key, {}).get("skipped_epochs", 0) + 1
        next_history[key] = dict(presentations=previous + amount, skipped_epochs=skipped)
        pool_audit[key] = dict(unique_frames=len(pools[signature]), presentations_before=previous,
                               presentations=amount, cumulative_presentations=previous + amount,
                               skipped_epochs=skipped, fairness_rank=fair_rank[signature], fairness_cost=fair_cost[signature])
        # 即使池本轮为零配额也记录起止游标/历史，恢复不能忘掉未选轮数。
        next_cursors[key] = previous + amount
        if not amount:
            continue
        representative = [objects[(members[0], ident)] for ident in pools[signature]]
        if route_diverse:
            ordered = deterministic_cycle_order(representative, cycle=0, master_seed=f"{master_seed}:{key}")
        else:
            ordered = sorted(representative, key=_item_identity)
            random.Random(f"{master_seed}:{key}").shuffle(ordered)
        cur = cursors.get(key, 0)
        for cell, count in assignments:
            for offset in range(count):
                ident = _item_identity(ordered[(cur + offset) % len(ordered)])
                selected.append(objects[(cell, ident)])
                attribution.append(cell)
            actual[cell] += count
            cur += count
        next_cursors[key] = cur
    # 一起打乱归属与样本，Action 侧可保留单次呈现的 event attribution。
    paired = list(zip(selected, attribution))
    (rng or random.Random(master_seed)).shuffle(paired)
    selected = [item for item, _ in paired]
    usage = initial + Counter(_item_identity(it) for it in selected)
    if max(usage.values(), default=0) > repeat_cap:
        raise AssertionError("joint sampling expansion violated global frame cap")
    for event, audit in event_audits.items():
        audit["sampled_actions"] = {a: actual[(event, a)] for a in audit["action_quotas"]}
    overflow = sum(max(0, actual[c] - quotas[c]) for c in quotas)
    return selected, dict(
        schema=SUPPORT_BALANCE_VERSION, policy="smooth_cap", repeat_cap=repeat_cap,
        smooth_power=smooth_power, master_seed=master_seed, event_audits=event_audits,
        cursor_start=cursors, next_cursors=next_cursors,
        pool_history_start=history, next_pool_history=next_history, pools=pool_audit,
        fairness="skipped_epoch_debt_then_normalized_cumulative_exposure_v1",
        objective="action_overflow_then_epoch_repeats_then_historical_exposure",
        selected_cells=[list(cell) for _, cell in paired], action_quota_overflow=overflow,
        unique_frames=len(usage), max_frame_repeat=max(usage.values(), default=0),
        repeat_histogram={str(k): v for k, v in sorted(Counter(usage.values()).items())},
        membership_groups=len(signatures), route_diverse=route_diverse,
    )


def _route_counts_report(items: Sequence[Any]) -> Dict[str, Any]:
    """汇总一组 case 的 route 集中度。"""

    counts = Counter(_route_key(item) for item in items)
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return {
        "route_identity": "physical_route_without_rep_or_collection_timestamp",
        "cases": len(items),
        "unique_routes": len(counts),
        "unique_scenarios": len({scenario for scenario, _ in counts}),
        "unique_towns": len({route.split("_")[0] for _, route in counts}),
        "towns": sorted({route.split("_")[0] for _, route in counts}),
        "max_cases_per_route": max(counts.values(), default=0),
        "route_case_counts": {
            f"{scenario}/{route_id}": int(count) for (scenario, route_id), count in ordered[:200]
        },
    }


def route_diversity_report(items: Sequence[Any]) -> Dict[str, Any]:
    """汇总总体及每个 balance class 的 route 集中度。"""

    report = _route_counts_report(items)
    grouped: Dict[str, List[Any]] = defaultdict(list)
    for item in items:
        key = str(item.get("balance_key", "")) if isinstance(item, Mapping) else getattr(item, "balance_key", "")
        if key:
            grouped[str(key)].append(item)
    report["by_balance_key"] = {key: _route_counts_report(grouped[key]) for key in sorted(grouped)}
    return report
