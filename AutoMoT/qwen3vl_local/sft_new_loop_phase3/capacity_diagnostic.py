"""Phase3容量失败的只读诊断；不修改配额、标签、上限或采样随机状态。"""
from collections import Counter, defaultdict

from qwen3vl_local.sft_new_loop_phase3.sampling import _MinCostFlow, _item_identity


class SharedFrameCapacityError(ValueError):
    """固定事件配额与物理帧上限不可同时满足；兼容原 ValueError 调用者。"""

    def __init__(self, report):
        self.report = report
        super().__init__(
            f"insufficient shared frame capacity: target={report['target']}, "
            f"feasible={report['feasible']}, repeat_cap={report['repeat_cap']}; "
            f"shortfall={report['shortfall']}; "
            f"minimum_repeat_cap_for_current_targets={report['minimum_repeat_cap']}; "
            f"bottleneck_events={report['bottleneck_events'][:8]}; "
            "targets and repeat cap were not changed; see phase3-capacity JSON for the bottleneck"
        )


def _shared_capacity_report(pools, targets, repeat_cap):
    """用无费用压缩最大流给出最小割和当前固定配额所需的最小全局上限。

    动作配额允许回流，因此容量只取决于事件归属与预留次数。此检查仅在分配失败时运行，
    不改变随机数、游标或历史；不足不等于全池缺这么多不同帧。
    """
    events, signatures = sorted(targets), sorted(pools)
    total = sum(targets.values())
    enodes = {event: i + 1 for i, event in enumerate(events)}
    base, sink = len(events) + 1, len(events) + len(signatures) + 1

    def solve(cap, detail=False):
        flow = _MinCostFlow(sink + 1)
        source_edges = {event: flow.add(0, enodes[event], targets[event]) for event in events}
        for i, signature in enumerate(signatures):
            members, used = signature
            for event in sorted({cell[0] for cell in members}):
                flow.add(enodes[event], base + i, total + 1)
            flow.add(base + i, sink, len(pools[signature]) * max(0, cap - used))
        sent, _ = flow.flow(0, sink, total)
        if not detail:
            return sent
        reachable, pending = {0}, [0]
        while pending:
            node = pending.pop()
            for target, _, capacity, _ in flow.graph[node]:
                if capacity > 0 and target not in reachable:
                    reachable.add(target)
                    pending.append(target)
        bottlenecks = [e for e in events if enodes[e] in reachable]
        affected = [sig for i, sig in enumerate(signatures) if base + i in reachable]
        return sent, dict(
            bottleneck_events=bottlenecks,
            bottleneck_target=sum(targets[e] for e in bottlenecks),
            bottleneck_unique_frames=sum(len(pools[sig]) for sig in affected),
            bottleneck_available_presentations=sum(
                len(pools[sig]) * max(0, repeat_cap - sig[1]) for sig in affected),
            unfilled_by_event={e: flow.graph[0][source_edges[e]][2] for e in events
                               if flow.graph[0][source_edges[e]][2]},
        )

    feasible, details = solve(repeat_cap, detail=True)
    low, high = repeat_cap, max(repeat_cap, total + max((sig[1] for sig in signatures), default=0))
    minimum = None
    if solve(high) == total:
        while low < high:
            middle = (low + high) // 2
            if solve(middle) == total:
                high = middle
            else:
                low = middle + 1
        minimum = low
    return dict(target=total, feasible=feasible, repeat_cap=repeat_cap,
                shortfall=total - feasible, minimum_repeat_cap=minimum,
                scope="current_event_targets_and_reserved_frames_only", **details)


def diagnose_joint_capacity(groups, targets, *, repeat_cap, initial_frame_usage=None):
    """精确检查原配额；再给出仅保留INVALID来源配额的乐观容量上界，绝不执行重分配。"""
    initial = Counter(initial_frame_usage or {})

    def pools_for(event_groups):
        memberships = defaultdict(set)
        for event, items in event_groups.items():
            for item in items:
                memberships[_item_identity(item)].add((event, 'capacity_only'))
        pools = defaultdict(list)
        for frame, members in memberships.items():
            pools[(tuple(sorted(members)), initial[frame])].append(frame)
        return pools

    report = _shared_capacity_report(pools_for(groups), targets, repeat_cap)
    report.update(schema='phase3_shared_capacity_diagnostic_v1', diagnostic_only=True,
                  automatic_changes=False, reserved_presentations=sum(initial.values()),
                  total_presentations_including_reserved=report['target'] + sum(initial.values()))
    report['bottleneck_details'] = [dict(
        event=event, target=targets[event],
        unique_frames=len({_item_identity(item) for item in groups.get(event, ())}),
    ) for event in report['bottleneck_events']]
    if any(event.startswith('INVALID|') for event in targets):
        merged_groups, merged_targets = defaultdict(list), Counter()
        for event, target in targets.items():
            # 原键 INVALID|source=X|true_rs=R*|asked_context=Y|prompt_rs|reason。
            key = event
            if event.startswith('INVALID|'):
                parts = event.split('|')
                if len(parts) != 6 or not parts[1].startswith('source='):
                    raise ValueError(f'unknown INVALID capacity group: {event}')
                key = 'INVALID_SOURCE|' + parts[1]
            merged_groups[key].extend(groups.get(event, ()))
            merged_targets[key] += target
        upper = _shared_capacity_report(pools_for(merged_groups), merged_targets, repeat_cap)
        report['invalid_source_only_upper_bound'] = dict(
            feasible=upper['feasible'], target=upper['target'], shortfall=upper['shortfall'],
            source_quotas_preserved=True, repeat_cap=repeat_cap,
            signature_prompt_rs_coverage_preserved=False,
            executable_plan=False,
            interpretation=('fine_quota_reallocation_may_help_but_coverage_must_be_checked'
                            if upper['feasible'] == upper['target'] else
                            'even_relaxed_fine_quotas_cannot_fit_current_candidate_frames'),
        )
    return report
