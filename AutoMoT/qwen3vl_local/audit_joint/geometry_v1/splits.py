"""Physical-route holdouts, inheriting all recorded development exposure."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from .sources import file_sha


def development_contract(manifests=()):
    from qwen3vl_local.sft_new_loop_phase3 import build_dataset as old
    groups = set(old.development_route_groups())
    sources = {}
    # Includes the later September and RGB-stage additions omitted by _split.
    paths = sorted(Path(old.__file__).parent.glob('development_route_groups_*.json'))
    manifests = sorted({str(Path(p).resolve(strict=True)) for p in manifests})
    for path in [*paths, *map(Path, manifests)]:
        raw = json.loads(path.read_text())
        extra = raw['additional_development'] if str(path) in manifests else raw
        values = extra['groups']
        if not isinstance(values, list) or not values or any(
                not isinstance(g, str) or g.count('/') != 1 for g in values):
            raise ValueError('invalid_development_groups: '+str(path))
        for group in values:
            scenario, route = group.split('/')
            if any(not x or Path(x).name != x or x in ('.', '..') for x in (scenario, route)):
                raise ValueError('invalid_development_group')
            groups.add(old.physical_route_group(scenario, route))
        sources[str(path.resolve())] = file_sha(path)
    return dict(groups=sorted(groups), sources=sources, manifests=manifests)


def complete(records, development, *, max_search_states=100000):
    """Only eligible geometry records determine final event support, never labels."""
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import physical_route_group
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
    from qwen3vl_local.sft_new_loop_phase3.split_coverage import complete_context_splits
    from .coverage_search import solve
    if type(max_search_states) is not int or max_search_states < 1:
        raise ValueError('positive_coverage_search_state_limit_required')
    development = set(development)
    for row in records:
        if physical_route_group(row['scenario'], row['route_id']) in development:
            row['split'] = 'train'
    eligible = [dict(r, context_id=r['event']) for r in records
                if any(t['valid'] for t in r['targets'])]
    group_rows = defaultdict(list)
    origins = {}
    # Validate ALL group members before planning, including invalid endpoints.
    for row in records:
        group = physical_route_group(row['scenario'], row['route_id'])
        if row['split'] not in ('train', 'val', 'test'): raise ValueError('unknown_split')
        if origins.setdefault(group, row['split']) != row['split']:
            raise ValueError('physical_route_split_leakage')
    for row in eligible:
        if row['event'] not in CONTEXT_IDS: raise ValueError('unknown_geometry_event')
        group_rows[physical_route_group(row['scenario'], row['route_id'])].append(row)
    report = complete_context_splits(eligible, contexts=CONTEXT_IDS,
        splits=('train', 'val', 'test'), development=set(development),
        group_of=lambda r: physical_route_group(r['scenario'], r['route_id']),
        seed=20260920, min_holdout_frames=1, min_holdout_groups=1)
    report['feasibility'] = 'feasible'
    report['search'] = dict(method='greedy', status='feasible', visited_states=0)
    if report['unresolved']:
        # The shared legacy helper mutates only our copies. Discard its partial
        # assignment and solve from ORIGINAL splits, including its moved groups.
        masks = {g: sum(1 << i for i, event in enumerate(CONTEXT_IDS)
                        if any(r['event'] == event for r in values))
                 for g, values in group_rows.items()}
        assignment, search = solve(masks, origins, development, event_count=len(CONTEXT_IDS),
                                   seed=20260920, max_states=max_search_states)
        greedy = dict(moves=report['moves'], unresolved=report['unresolved'], after=report['after'])
        # A failed/unfinished search does not publish a partial greedy assignment.
        final = assignment if assignment is not None else origins
        counts = {s: Counter() for s in ('train', 'val', 'test')}
        groups = {s: Counter() for s in counts}
        for group, values in group_rows.items():
            events = Counter(r['event'] for r in values)
            counts[final[group]].update(events); groups[final[group]].update(events.keys())
        report.update(policy='geometry_context_coverage_exact_fallback_v1',
            feasibility=search['status'], search=search, greedy_attempt=greedy,
            after={s: {e: counts[s][e] for e in CONTEXT_IDS} for s in counts},
            physical_groups_after={s: {e: groups[s][e] for e in CONTEXT_IDS} for s in counts},
            moves={g: dict(from_split='train', to_split=final[g],
                           fills=sorted({r['event'] for r in values}))
                   for g, values in group_rows.items() if final[g] != origins[g]},
            unresolved={s: [e for e in CONTEXT_IDS if not counts[s][e]]
                        for s in counts if any(not counts[s][e] for e in CONTEXT_IDS)},
            group_support={s: {e: 'supported' if groups[s][e] else 'insufficient_support'
                               for e in CONTEXT_IDS} for s in ('val', 'test')},
            group_deficits={s: {e: 1 for e in CONTEXT_IDS if not groups[s][e]} for s in ('val', 'test')})
    # Move ALL members of a physical group, including unsupported endpoints.
    for row in records:
        move = report['moves'].get(physical_route_group(row['scenario'], row['route_id']))
        if move: row['split'] = move['to_split']
    report['support_boundary'] = 'One eligible frame/group per event is coverage, not generalization certification.'
    return report


def require_coverage(report, stage):
    if report['feasibility'] == 'unknown':
        raise RuntimeError(f'{stage}_coverage_search_inconclusive: search limit reached; '
                           'not evidence of insufficient data; see split coverage report')
    if report['unresolved']:
        if report['feasibility'] != 'infeasible': raise RuntimeError('invalid_coverage_report')
        raise ValueError(f'{stage}_event_coverage_incomplete: proven infeasible under fixed '
                         f'holdout/development policy: {report["unresolved"]}')
