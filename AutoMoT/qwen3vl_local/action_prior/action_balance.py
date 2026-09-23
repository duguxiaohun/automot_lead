"""全局主要动作均衡；动作内温和提高小事件权重，预算对齐 event 模式。"""
from collections import Counter, defaultdict
import math
import random

from qwen3vl_local.action_prior.action_token import ACTION_TOKEN_NAMES, TOKEN_VERSION
from qwen3vl_local.action_prior.event_balance import (
    SPECIAL_BUCKETS, REGULAR_BACKGROUND, EVENT_BALANCE_WEIGHTS, CONFIRMED_REGULAR,
    SPECIAL_ELIGIBLE, _identity, _route_key, _ordered_route_cycle,
    event_balanced_total, weighted_quotas, DEFAULT_ACTION_REPEAT_CAP,
)
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_CONTEXTS
from qwen3vl_local.sft_new_loop_phase3.sampling import support_diagnostic

EVENT_ACTIONS = {c.source_event.replace("-", ""): (*c.action_keys, "KEEP") for c in ACTION_CONTEXTS}
SEMANTIC_ACTIONS = tuple(a for a in ACTION_TOKEN_NAMES if a != "UNCOND")
EVENT_BOOST_CAP = 2.0
VERSION = "action_balanced_global_soft_event_v4"


def policy_contract():
    return dict(version=VERSION,
                semantic_actions=list(SEMANTIC_ACTIONS), action_ratio="equal_global_semantic_actions",
                background="confirmed_regular_only_UNCOND", background_fraction="1/6",
                event_constraint="soft_weights_only_no_equal_event_quotas",
                event_boost=dict(formula="min(2, sqrt(max_event_frames/event_frames))",
                                 cap=EVENT_BOOST_CAP, scope="train_supported_event_memberships",
                                 concurrent="mean_not_sum"),
                rounding="semantic_remainder_rotates_by_seed_mod_6; within_one_presentation",
                automatic_budget="event_balanced_same_pool_repeat_cap_world; fail_if_action_capacity_insufficient",
                event_action_domains={e: list(a) for e, a in EVENT_ACTIONS.items()},
                out_of_domain="skip_membership_preserve_frame_token_and_event_facts",
                absent_actions="fail_missing_global_semantic_action; no_synthetic_labels",
                label_usage="sampling_only; model_input_controlled_by_high_level_action_token",
                repetition="disjoint_action_membership_pools; hard_global_frame_cap; capacity_redistribution",
                diversity="route_cycle_within_membership_pool_before_reuse")


def action_groups(rows, *, diagnostics=None):
    """分桶依据实际将输入模型的主要动作；采样不改成每个事件各自的动作。"""
    groups = {}
    available = {event: dict.fromkeys(ACTION_TOKEN_NAMES, 0)
                 for event in (*SPECIAL_BUCKETS, REGULAR_BACKGROUND)}
    seen = set()
    for row in rows:
        identity = _identity(row)
        if identity in seen:
            raise ValueError(f"action-balanced duplicate frame: {identity}")
        seen.add(identity)
        status = row.get("event_balance_status")
        if status == CONFIRMED_REGULAR:
            events = (REGULAR_BACKGROUND,)
        elif status == SPECIAL_ELIGIBLE:
            events = tuple(row.get("event_balance_buckets", ()))
            if not events or len(events) != len(set(events)) or any(e not in SPECIAL_BUCKETS for e in events):
                raise ValueError("action-balanced eligible frame has invalid event membership")
        else:
            continue
        token = row.get("action_token", {})
        action = token.get("name")
        if (action not in ACTION_TOKEN_NAMES or token.get("version") != TOKEN_VERSION
                or type(row.get("action_token_id")) is not int
                or row["action_token_id"] != ACTION_TOKEN_NAMES.index(action)):
            raise ValueError("action-balanced requires validated Phase3 primary action labels")
        if (status == CONFIRMED_REGULAR) != (action == "UNCOND"):
            raise ValueError("action-balanced requires UNCOND only for confirmed regular background")
        retained = 0
        for event in events:
            if event != REGULAR_BACKGROUND and action not in EVENT_ACTIONS[event]:
                if diagnostics is not None:
                    diagnostics[f"{event}/{action}"] += 1
                continue
            retained += 1
            available[event][action] += 1
            groups.setdefault(f"{event}/{action}", []).append(row)
        if not retained:
            raise ValueError(f"action-balanced primary action has no supporting event domain: {identity}, {action}")
    # 固定词表顺序，输入行顺序不决定配额/图节点顺序。
    groups = {f"{event}/{action}": groups[f"{event}/{action}"]
              for event in available for action in ACTION_TOKEN_NAMES if available[event][action]}
    return groups, available



def global_action_quotas(total, *, seed=0):
    """保留背景2/12，其余六类等量；整数余数随epoch seed轮转，不改变总预算。"""
    background = weighted_quotas(total)[REGULAR_BACKGROUND]
    base, extra = divmod(total - background, len(SEMANTIC_ACTIONS))
    offset = int(seed) % len(SEMANTIC_ACTIONS)
    rotated = SEMANTIC_ACTIONS[offset:] + SEMANTIC_ACTIONS[:offset]
    result = {a: base + int(a in rotated[:extra]) for a in SEMANTIC_ACTIONS}
    return {"UNCOND": background, **result}


def _sampling_pools(rows):
    diagnostics = Counter()
    groups, available = action_groups(rows, diagnostics=diagnostics)
    # 每帧只有一个主要动作；并发事件组合只建一个池，不能重复计算容量或相加权重。
    memberships, frames = defaultdict(list), {}
    for cell, values in groups.items():
        event, _ = cell.split("/")
        for row in values:
            identity = _identity(row)
            memberships[identity].append(event)
            frames[identity] = row
    pools = defaultdict(list)
    for identity, events in memberships.items():
        row = frames[identity]
        pools[(row["action_token"]["name"], tuple(sorted(events)))].append(row)
    counts = Counter()
    for (action, _), values in pools.items():
        counts[action] += len(values)
    missing = [a for a in ACTION_TOKEN_NAMES if not counts[a]]
    if missing:
        raise ValueError(f"action-balanced missing global action pools={missing}; unique_frames={dict(counts)}")
    event_counts = {e: sum(available[e].values()) for e in SPECIAL_BUCKETS}
    largest = max(event_counts.values(), default=0)
    boosts = {e: min(EVENT_BOOST_CAP, math.sqrt(largest / n)) for e, n in event_counts.items() if n}
    boosts[REGULAR_BACKGROUND] = 1.0
    return dict(groups=groups, available=available, pools=dict(sorted(pools.items())),
                counts=dict(counts), boosts=boosts, excluded=dict(diagnostics))


def _require_capacity(counts, total, repeat_cap):
    if repeat_cap < 1:
        raise ValueError("action-balanced repeat_cap must be positive")
    quotas = global_action_quotas(total)
    # 校验余数轮转的最大配额，防止首轮可行、后续epoch因多一帧才报错。
    upper = math.ceil((total - quotas["UNCOND"]) / len(SEMANTIC_ACTIONS))
    needed = {a: upper if a != "UNCOND" else quotas[a] for a in ACTION_TOKEN_NAMES}
    deficient = {a: dict(required=needed[a], unique_frames=counts.get(a, 0),
                         capacity=counts.get(a, 0) * repeat_cap,
                         minimum_repeat_cap=math.ceil(needed[a] / counts[a]) if counts.get(a) else None)
                 for a in ACTION_TOKEN_NAMES if needed[a] > counts.get(a, 0) * repeat_cap}
    if deficient:
        raise ValueError(f"action-balanced budget infeasible: total={total}, repeat_cap={repeat_cap}, "
                         f"action_capacity_deficits={deficient}. Budget was not shortened; "
                         "choose a shared smaller --event-balanced-epoch-samples or an explicit repeat cap.")


def _weighted_capacity_quota(sizes, boosts, total, repeat_cap, rng):
    """按帧数×温和权重分配，饱和池缺额回到同一动作的其它池。"""
    caps = {k: n * repeat_cap for k, n in sizes.items()}
    if sum(caps.values()) < total:
        raise ValueError("action-balanced weighted pool capacity infeasible")
    result = dict.fromkeys(sizes, 0)
    active, left = list(sizes), total
    while left and active:
        weights = {k: sizes[k] * boosts[k] for k in active}
        mass = sum(weights.values())
        ideal = {k: left * weights[k] / mass for k in active}
        saturated = [k for k in active if ideal[k] >= caps[k]]
        if saturated:
            for key in saturated:
                result[key] = caps[key]
                left -= caps[key]
                active.remove(key)
            continue
        for key in active:
            result[key] = int(ideal[key])
        remainder = left - sum(result[k] for k in active)
        # 同余数时随机打破固定事件排序偏置；依赖独立seed，恢复可重放。
        rng.shuffle(active)
        active.sort(key=lambda k: ideal[k] - result[k], reverse=True)
        for key in active[:remainder]:
            result[key] += 1
        left = 0
    if left or sum(result.values()) != total or any(result[k] > caps[k] for k in sizes):
        raise AssertionError("weighted quota failed to preserve budget/capacity")
    return result


def _layout(data, total, repeat_cap, seed):
    _require_capacity(data["counts"], total, repeat_cap)
    targets = global_action_quotas(total, seed=seed)
    quotas, event_orders = {}, {}
    for action in ACTION_TOKEN_NAMES:
        sizes = {key: len(values) for key, values in data["pools"].items() if key[0] == action}
        boosts = {key: sum(data["boosts"][e] for e in key[1]) / len(key[1]) for key in sizes}
        rng = random.Random(f"{VERSION}:quota:{seed}:{action}")
        quotas.update(_weighted_capacity_quota(sizes, boosts, targets[action], repeat_cap, rng))
        for key in sizes:
            events = list(key[1]); rng.shuffle(events)
            event_orders[key] = events
    cells = dict.fromkeys(data["groups"], 0)
    for (action, events), count in quotas.items():
        order = event_orders[(action, events)]
        base, extra = divmod(count, len(order))
        for i, event in enumerate(order):
            cells[f"{event}/{action}"] += base + int(i < extra)
    return targets, quotas, event_orders, cells


def support_audit(groups, quotas):
    cells = {key: support_diagnostic(len(values), len({_route_key(r) for r in values}),
                                     quotas[key] if quotas is not None else None)
             for key, values in groups.items()}
    events = {}
    for event in (*SPECIAL_BUCKETS, REGULAR_BACKGROUND):
        values = [r for key, rr in groups.items() if key.split("/")[0] == event for r in rr]
        count = sum(n for key, n in quotas.items() if key.split("/")[0] == event) if quotas is not None else None
        events[event] = support_diagnostic(len(values), len({_route_key(r) for r in values}), count)
    return dict(cells=cells, events=events, review_only=True,
                interpretation="event_presentations_use_single_attribution; concurrent_facts_preserved; diagnostic_only")


def action_balance_plan(rows, total=None, *, repeat_cap=DEFAULT_ACTION_REPEAT_CAP, seed=0):
    data = _sampling_pools(rows)
    targets = quotas = None
    if total is not None:
        targets, _, _, quotas = _layout(data, total, repeat_cap, seed)
    return dict(policy=policy_contract(), available=data["available"],
                action_unique_frames=data["counts"], event_boosts=data["boosts"],
                action_quotas=targets, cell_quotas=quotas, quota_seed=seed,
                quota_scope="first_epoch_seeded_plan; later_epoch_integer_remainders_rotate",
                supported_actions={e: [a for a, n in values.items() if n] for e, values in data["available"].items()},
                missing_actions={e: [a for a in EVENT_ACTIONS[e] if not data["available"][e][a]] for e in SPECIAL_BUCKETS},
                excluded_out_of_domain=data["excluded"],
                required_epoch_multiple=sum(EVENT_BALANCE_WEIGHTS.values()),
                support=support_audit(data["groups"], quotas))


def action_balanced_total(rows, *, requested, repeat_cap, world):
    """自动预算与同参数event模式一致；动作容量不足明确失败，不自动缩轮。"""
    if repeat_cap < 1 or world < 1 or requested < 0:
        raise ValueError("action-balanced needs positive repeat_cap/world and nonnegative requested budget")
    multiple = math.lcm(sum(EVENT_BALANCE_WEIGHTS.values()), world)
    if requested and requested % multiple:
        raise ValueError(f"action-balanced epoch budget {requested} must be divisible by {multiple}")
    total = requested or event_balanced_total(rows, requested=0, repeat_cap=repeat_cap, world=world)
    data = _sampling_pools(rows)
    _require_capacity(data["counts"], total, repeat_cap)
    return total


def build_action_balanced_epoch(rows, *, total, seed, route_diverse=True, repeat_cap=DEFAULT_ACTION_REPEAT_CAP):
    data = _sampling_pools(rows)
    targets, pool_quotas, event_orders, quotas = _layout(data, total, repeat_cap, seed)
    selected = []
    for key, count in pool_quotas.items():
        action, events = key
        rng = random.Random(f"{VERSION}:frames:{seed}:{action}:{events}")
        ordered = _ordered_route_cycle(data["pools"][key], rng, route_diverse)
        attribution = event_orders[key]
        for i in range(count):
            event = attribution[i % len(attribution)]
            selected.append(dict(ordered[i % len(ordered)], event_balance_bucket=event,
                                 action_balance_cell=f"{event}/{action}"))
    random.Random(f"{VERSION}:epoch:{seed}:{total}").shuffle(selected)
    used = Counter(_identity(row) for row in selected)
    sampled = Counter(row["action_balance_cell"] for row in selected)
    actions = Counter(row["action_token"]["name"] for row in selected)
    if (dict(sampled) != {k: v for k, v in quotas.items() if v} or dict(actions) != targets
            or len(selected) != total or max(used.values(), default=0) > repeat_cap):
        raise AssertionError("action-balanced expansion violated global action quotas/repeat cap")
    events = Counter(row["event_balance_bucket"] for row in selected)
    cell_frames, cell_routes, event_routes = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in selected:
        cell_frames[row["action_balance_cell"]].add(_identity(row))
        cell_routes[row["action_balance_cell"]].add(_route_key(row))
        event_routes[row["event_balance_bucket"]].add(_route_key(row))
    return selected, dict(schema=VERSION, mode="action_balanced", seed=int(seed), total=total,
        policy=policy_contract(), quotas=dict(events), cell_quotas=quotas,
        action_quotas=targets, sampled_actions=dict(actions), event_boosts=data["boosts"],
        action_unique_frames=data["counts"], excluded_out_of_domain=data["excluded"],
        support=support_audit(data["groups"], quotas), sampled=dict(events), sampled_cells=dict(sampled),
        available=data["available"], unique_frames=len(used), max_frame_repeats=max(used.values(), default=0),
        repeat_histogram={str(k): v for k, v in sorted(Counter(used.values()).items())}, repeat_cap=repeat_cap,
        cell_unique_frames={k: len(cell_frames[k]) for k in quotas},
        cell_unique_routes={k: len(cell_routes[k]) for k in quotas},
        unique_routes={k: len(v) for k, v in event_routes.items()},
        repeat_presentations=total - len(used), membership_groups=len(data["pools"]),
        event_counts_scope="single_attribution_per_presentation; all_original_events_kept_in_row",
        route_diverse=bool(route_diverse))
