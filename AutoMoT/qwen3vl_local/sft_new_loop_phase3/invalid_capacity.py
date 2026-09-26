"""binary训练的INVALID细分容量回流；正例/来源总配额和全局帧上限均为硬约束。"""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from qwen3vl_local.sft_new_loop_phase3.invalid_balance import signature_for_row
from qwen3vl_local.sft_new_loop_phase3.sampling import (
    _MinCostFlow, _item_identity, sampling_action, support_aware_quota,
)

VERSION = 'invalid_source_capacity_return_v1'


def contract():
    return dict(version=VERSION, sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                trigger='fixed_fine_quota_capacity_failure_only',
                constraints='positive_and_invalid_source_totals_frame_cap_reviewed_cases_and_fine_coverage')


def reallocate_invalid_targets(groups, targets, *, repeat_cap, reserved=None,
                               smooth_power=0.5, cursor_state=None, pool_history=None):
    """返回可行细分配额或None；不选样、不改变输入/随机状态/游标。

    每个原非空自动负例细分组至少保留一次，故来源/真实RS/asked-context/错误RS覆盖不丢失。
    同来源才可转移配额。先最少细分超额，再最少正例动作偏差，再不同帧和原历史公平费用。
    人工负例在调用方固定，并通过reserved占用真实帧容量。
    """
    reserved, cursors, history = Counter(reserved or {}), dict(cursor_state or {}), dict(pool_history or {})
    parents, parent_targets, memberships, action_items = {}, Counter(), defaultdict(set), defaultdict(set)
    for event, target in sorted(targets.items()):
        for item in groups.get(event, ()):
            row = item.row
            if row.invalid_source:
                if row.invalid_reason != 'wrong_road_structure':
                    raise ValueError('capacity return only accepts automatic wrong-road INVALID')
                expected_key = 'INVALID|' + '|'.join((row.invalid_source, row.prompt_road_structure, row.invalid_reason))
                if event != expected_key:
                    raise ValueError('INVALID capacity group does not match its row')
                source = signature_for_row(row).source_class
                if event in parents and parents[event] != source:
                    raise ValueError('mixed INVALID sources in one group')
                parents[event] = source
                action = 'INVALID'
            else:
                action = sampling_action(row.answers, row.context_id)
            ident, cell = _item_identity(item), (event, action)
            memberships[ident].add(cell)
            action_items[cell].add(ident)
        if event in parents:
            parent_targets[parents[event]] += target
    if not parents:
        return None
    floors = {event: int(targets[event] > 0) for event in parents}
    quotas = {}
    for event, target in targets.items():
        if event in parents:
            quotas[(event, 'INVALID')] = target
        else:
            capacities = {a: len(ids) for (e, a), ids in action_items.items() if e == event}
            if sum(capacities.values()) * repeat_cap < target:
                return None
            quotas.update({(event, a): n for a, n in support_aware_quota(
                capacities, target, repeat_cap=repeat_cap, smooth_power=smooth_power, mode='smooth_cap').items()})
    pools = defaultdict(list)
    for ident, members in memberships.items():
        pools[(tuple(sorted(members)), reserved[ident])].append(ident)
    def pool_key(sig):
        members, used = sig
        key = '/'.join(members[0]) if len(members) == 1 else json.dumps(members, separators=(',', ':'))
        return key + (f':reserved={used}' if used else '')
    order = sorted(pools, key=lambda sig: (
        Fraction(cursors.get(pool_key(sig), 0), len(pools[sig])),
        -history.get(pool_key(sig), {}).get('skipped_epochs', 0), pool_key(sig)))
    rank = {sig: i for i, sig in enumerate(order)}
    max_skipped = max((history.get(pool_key(sig), {}).get('skipped_epochs', 0) for sig in pools), default=0)
    fairness = {sig: (max_skipped - history.get(pool_key(sig), {}).get('skipped_epochs', 0)) * len(pools)
               + rank[sig] for sig in pools}
    total = sum(targets.values())
    repeat_cost = total * max(fairness.values(), default=0) + 1
    action_cost = (total + 1) * repeat_cost
    fine_cost = (total + 1) * action_cost
    events, cells, signatures = sorted(targets), sorted(action_items), sorted(pools)
    enodes = {e: i + 1 for i, e in enumerate(events)}
    pnodes = {p: len(enodes) + i + 1 for i, p in enumerate(sorted(parent_targets))}
    cnodes = {c: len(enodes) + len(pnodes) + i + 1 for i, c in enumerate(cells)}
    base = 1 + len(enodes) + len(pnodes) + len(cnodes)
    sink = base + len(signatures)
    flow = _MinCostFlow(sink + 1)
    for parent, budget in sorted(parent_targets.items()):
        floor_total = sum(floors[e] for e in parents if parents[e] == parent)
        flow.add(0, pnodes[parent], budget - floor_total)
    for event in events:
        if event in parents:
            floor, parent = floors[event], parents[event]
            flow.add(0, enodes[event], floor)
            flow.add(pnodes[parent], enodes[event], targets[event] - floor)
            flow.add(pnodes[parent], enodes[event], parent_targets[parent], fine_cost)
        else:
            flow.add(0, enodes[event], targets[event])
    cell_edges = {}
    for cell in cells:
        event = cell[0]
        limit = parent_targets[parents[event]] if event in parents else targets[event]
        edges = [(flow.add(enodes[event], cnodes[cell], quotas[cell]), quotas[cell]),
                 (flow.add(enodes[event], cnodes[cell], limit, action_cost if event not in parents else 0), limit)]
        cell_edges[cell] = edges
    for i, sig in enumerate(signatures):
        members, used = sig
        if used < 0 or used > repeat_cap:
            raise ValueError('reserved frames exceed global repeat_cap')
        for cell in members:
            flow.add(cnodes[cell], base + i, total)
        count = len(pools[sig])
        if used == 0:
            flow.add(base + i, sink, count, fairness[sig])
            flow.add(base + i, sink, count * (repeat_cap - 1), repeat_cost + fairness[sig])
        else:
            flow.add(base + i, sink, count * (repeat_cap - used), repeat_cost + fairness[sig])
    sent, _ = flow.flow(0, sink, total)
    if sent != total:
        return None
    adjusted = Counter()
    for (event, action), edges in cell_edges.items():
        adjusted[event] += sum(cap - flow.graph[enodes[event]][edge][2] for edge, cap in edges)
    adjusted = {event: adjusted[event] for event in events}
    actual_sources = Counter()
    for event, parent in parents.items():
        actual_sources[parent] += adjusted[event]
        if adjusted[event] < floors[event]:
            raise AssertionError('INVALID fine-group coverage lost')
    if actual_sources != parent_targets or sum(adjusted.values()) != total:
        raise AssertionError('INVALID source or total budget changed')
    if any(adjusted[e] != targets[e] for e in events if e not in parents):
        raise AssertionError('positive event budget changed')
    changes = {e: dict(requested=targets[e], actual=adjusted[e]) for e in events if adjusted[e] != targets[e]}
    return adjusted, dict(version=VERSION, shifted_presentations=sum(
        max(0, adjusted[e] - targets[e]) for e in parents), changes=changes,
        source_targets=dict(parent_targets), source_actual=dict(actual_sources),
        coverage_floors=floors, coverage_preserved=True, repeat_cap=repeat_cap)
