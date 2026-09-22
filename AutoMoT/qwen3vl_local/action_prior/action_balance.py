"""事件内主要动作容量回流、事件间1:…:1:2的共享离线采样器。"""
from collections import Counter
import math
import random

from qwen3vl_local.action_prior.action_token import ACTION_TOKEN_NAMES, TOKEN_VERSION
from qwen3vl_local.action_prior.event_balance import (
    SPECIAL_BUCKETS, REGULAR_BACKGROUND, EVENT_BALANCE_WEIGHTS, CONFIRMED_REGULAR,
    SPECIAL_ELIGIBLE, _identity, _route_key, _joint_allocation, weighted_quotas, DEFAULT_ACTION_REPEAT_CAP,
)

from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_CONTEXTS
from qwen3vl_local.sft_new_loop_phase3.sampling import support_aware_quota, SUPPORT_BALANCE_VERSION, support_diagnostic

EVENT_ACTIONS = {c.source_event.replace("-", ""): (*c.action_keys, "KEEP") for c in ACTION_CONTEXTS}
VERSION = "action_balanced_domain_capacity_v2"


def policy_contract():
    return dict(version=VERSION, event_weights=dict(EVENT_BALANCE_WEIGHTS),
                action_weights=SUPPORT_BALANCE_VERSION,
                event_action_domains={e: list(a) for e, a in EVENT_ACTIONS.items()},
                out_of_domain="skip_membership_preserve_frame_token_and_event_facts",
                absent_actions="report_and_skip; never_synthesize",
                action_projection="same_frame_primary_token_across_all_eligible_contexts",
                background="confirmed_regular_only_UNCOND",
                label_usage="sampling_only; model_input_controlled_by_high_level_action_token",
                quota_rule="exact_event_weights_lcm_world; capacity_capped_action_targets_with_joint_return",
                repetition="global_frame_cap_across_all_event_action_cells")


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
    missing = [event for event, values in available.items() if not sum(values.values())]
    if missing:
        raise ValueError(f"action-balanced missing event pools={missing}; available={available}")
    # 固定词表顺序，输入行顺序不决定配额/图节点顺序。
    groups = {f"{event}/{action}": groups[f"{event}/{action}"]
              for event in available for action in ACTION_TOKEN_NAMES if available[event][action]}
    return groups, available


def _quotas(available, total):
    events = weighted_quotas(total)
    return {f"{event}/{action}": count
            for event, values in available.items()
            for action, count in support_aware_quota(
                {a: n for a, n in values.items() if n}, events[event]).items()}


def support_audit(groups, quotas):
    """逐动作与事件两层报告，区分事件整体重复和格子稀少；不反向改动采样。"""
    cells = {key: support_diagnostic(len(values), len({_route_key(r) for r in values}),
                                     quotas[key] if quotas is not None else None)
             for key, values in groups.items()}
    events = {}
    for event in (*SPECIAL_BUCKETS, REGULAR_BACKGROUND):
        values = [r for key, rr in groups.items() if key.split("/")[0] == event for r in rr]
        count = sum(n for key, n in quotas.items() if key.split("/")[0] == event) if quotas is not None else None
        events[event] = support_diagnostic(len(values), len({_route_key(r) for r in values}), count)
    return dict(cells=cells, events=events, review_only=True,
                interpretation="rare_event_action_cells_share_global_token_embeddings; counts_are_not_label_confidence")


def action_balance_plan(rows, total=None, *, repeat_cap=DEFAULT_ACTION_REPEAT_CAP):
    skipped = Counter()
    groups, available = action_groups(rows, diagnostics=skipped)
    desired = _quotas(available, total) if total is not None else None
    actual = None
    if total is not None:
        allocation = _joint_allocation(rows, desired, repeat_cap=repeat_cap, groups=groups,
                                       event_quotas=weighted_quotas(total))
        if allocation is None:
            raise ValueError("action-balanced plan infeasible under shared-frame repeat caps")
        actual = {key: sum(v.values()) for key, v in allocation[2].items()}
    return dict(policy=policy_contract(), available=available,
                supported_actions={event: [a for a, n in values.items() if n]
                                   for event, values in available.items()},
                missing_actions={event: [a for a in EVENT_ACTIONS[event] if not available[event][a]]
                                 for event in SPECIAL_BUCKETS},
                excluded_out_of_domain=dict(skipped),
                required_epoch_multiple=sum(EVENT_BALANCE_WEIGHTS.values()),
                desired_cell_quotas=desired, cell_quotas=actual,
                support=support_audit(groups, actual),
                unique_frames={key: len(values) for key, values in groups.items()},
                physical_routes={key: len({_route_key(r) for r in values}) for key, values in groups.items()})


def action_balanced_total(rows, *, requested, repeat_cap, world):
    """按事件联合容量确定预算；动作小桶的缺额回流，不限制整个 epoch。"""
    if repeat_cap < 1 or world < 1 or requested < 0:
        raise ValueError("action-balanced needs positive repeat_cap/world and nonnegative requested budget")
    groups, available = action_groups(rows)
    unit = sum(EVENT_BALANCE_WEIGHTS.values())
    multiple = math.lcm(unit, world)
    scale = multiple // unit
    high = min(sum(values.values()) * repeat_cap // EVENT_BALANCE_WEIGHTS[event]
               for event, values in available.items()) // scale

    def feasible(total):
        return _joint_allocation(rows, _quotas(available, total), repeat_cap=repeat_cap,
                                 groups=groups, event_quotas=weighted_quotas(total)) is not None

    if requested:
        if requested % multiple:
            raise ValueError(f"action-balanced epoch budget {requested} must be divisible by {multiple} "
                             f"(event/action quota unit={unit}, world_size={world})")
        if not feasible(requested):
            raise ValueError(f"action-balanced budget infeasible under shared-frame repeat caps; "
                             f"requested={requested}, repeat_cap={repeat_cap}, available={available}. "
                             "Use --event-balanced-epoch-samples 0 for automatic feasible budget.")
        return requested
    low, best = 1, 0
    while low <= high:
        middle = (low + high) // 2
        if feasible(middle * multiple):
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if not best:
        raise ValueError(f"action-balanced has no jointly feasible whole epoch; world={world}, "
                         f"repeat_cap={repeat_cap}, required_multiple={multiple}, available={available}")
    return best * multiple


def build_action_balanced_epoch(rows, *, total, seed, route_diverse=True, repeat_cap=DEFAULT_ACTION_REPEAT_CAP):
    skipped = Counter()
    groups, available = action_groups(rows, diagnostics=skipped)
    desired = _quotas(available, total)
    quotas = desired
    allocation = _joint_allocation(rows, quotas, repeat_cap=repeat_cap, seed=seed,
                                   route_diverse=route_diverse, groups=groups, event_quotas=weighted_quotas(total))
    if allocation is None:
        raise ValueError("action-balanced epoch infeasible under shared-frame repeat caps")
    _, frames, assigned, allocation_audit = allocation
    quotas = {key: sum(v.values()) for key, v in assigned.items()}
    selected = []
    for cell, assignments in assigned.items():
        event, action = cell.split("/")
        for identity, count in assignments.items():
            for _ in range(count):
                selected.append(dict(frames[identity], event_balance_bucket=event,
                                     action_balance_cell=cell))
    random.Random(f"{VERSION}:{seed}:{total}").shuffle(selected)
    used = Counter(_identity(row) for row in selected)
    sampled = Counter(row["action_balance_cell"] for row in selected)
    if (dict(sampled) != {k: v for k, v in quotas.items() if v} or len(used) != allocation_audit["optimal_unique_frames"]
            or max(used.values(), default=0) > repeat_cap):
        raise AssertionError("action-balanced expansion violated quotas/unique optimum/repeat cap")
    event_quotas = Counter()
    for cell, count in quotas.items():
        event_quotas[cell.split("/")[0]] += count
    return selected, dict(schema=VERSION, mode="action_balanced", seed=int(seed), total=total,
        policy=policy_contract(), quotas=dict(event_quotas), cell_quotas=quotas,
        desired_cell_quotas=desired, excluded_out_of_domain=dict(skipped),
        support=support_audit(groups, quotas),
        sampled=dict(Counter(r["event_balance_bucket"] for r in selected)), sampled_cells=dict(sampled),
        available=available, missing_actions={event: [a for a in EVENT_ACTIONS[event] if not available[event][a]]
                                            for event in SPECIAL_BUCKETS},
        unique_frames=len(used), max_frame_repeats=max(used.values(), default=0),
        repeat_histogram={str(k): v for k, v in sorted(Counter(used.values()).items())}, repeat_cap=repeat_cap,
        cell_unique_frames={key: len(values) for key, values in assigned.items()},
        cell_unique_routes={key: len({_route_key(frames[i]) for i in values}) for key, values in assigned.items()},
        unique_routes={key: len({_route_key(r) for r in selected if r["event_balance_bucket"] == key}) for key in event_quotas},
        joint_allocation=True, diversity_first=True, route_diverse=bool(route_diverse), **allocation_audit)
