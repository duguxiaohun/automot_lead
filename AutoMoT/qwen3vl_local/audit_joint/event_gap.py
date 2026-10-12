"""Read existing native replay evidence for one event; never regenerate labels.

Only routes declaring this event in their receipts are opened. Their native
frame accounting and question proofs are revalidated. Other routes are not
certified by this diagnostic. Examples are restricted to training routes.
"""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

from .io import digest, file_sha, write_json


def summarize(records, event, *, example_limit=12):
    counts = defaultdict(Counter)
    examples = []
    seen_reasons = Counter()
    example_groups = defaultdict(set)
    recent = defaultdict(lambda: deque(maxlen=8))
    instances = {}
    ended = set()
    previous_route = None
    frame_window = deque(maxlen=7)
    contexts = {}
    for row in records:
        split = row['split']
        route = (row['scenario'], row['route_id'])
        if route != previous_route:
            recent.clear()  # examples never cross route boundaries
            contexts.clear()
            frame_window.clear()
        previous_route = route
        frame_window.append(dict(frame_id=row['frame_id'], sources=row.get('sources', [])))
        for q in row.get('questions', []):
            if q['episode']['event'] != event or q['rgb_mode'] != 4:
                continue
            ident = q['episode']['instance_id']
            instances[ident] = split
            contexts[ident] = {k: q[k] for k in ('episode', 'instance') if k in q}
            counts[split]['questions/' + q['edge'] + '/' + q['slice'] + '/' + q['rule_target']] += 1
            if q['edge'] != 'proceed':
                continue
            for name, value in q['facts'].items():
                counts[split]['proceed_fact/' + name + '/' + str(value)] += 1
            for reason in q['reasons']:
                counts[split]['proceed_reason/' + reason] += 1
            recent[ident].append({k: q[k] for k in ('frame_id', 'question_id', 'rule_target',
                'state_rule_target', 'slice', 'facts', 'reasons', 'ego_speed', 'input_sources')})
        for end in row.get('instance_ends', []):
            ident = end['instance_id']
            if ident not in instances:
                continue
            ended.add(ident)
            reason = end['reason']
            counts[split]['instance_end/' + reason] += 1
            # Held-out cases remain aggregate-only; no label-selected RGB review.
            group = row.get('physical_group', '/'.join(route))
            if (split == 'train' and group not in example_groups[reason]
                    and seen_reasons[reason] < 2 and len(examples) < example_limit):
                examples.append(dict(scenario=route[0], route_id=route[1], split=split,
                    frame_id=row['frame_id'], instance_end=end,
                    previous_questions=list(recent[ident]), sources=row.get('sources', []),
                    instance_context=contexts.get(ident), frame_window_sources=list(frame_window)))
                seen_reasons[reason] += 1
                example_groups[reason].add(group)
            recent.pop(ident, None)
            contexts.pop(ident, None)
    for ident, split in instances.items():
        counts[split]['instances'] += 1
        if ident not in ended:
            counts[split]['instances_without_end_in_recording'] += 1
    return dict(event=event, rgb_mode=4, counts={s: dict(c) for s, c in counts.items()},
                train_examples=examples, labels_changed=False,
                scope='native recorded conditions and censoring; no hypothetical YES or training admission')


def inspect(pool_path, index_path, *, expected_index_sha256, event='U-E5', data_root=None):
    from qwen3vl_local.sft_new_loop_phase4 import candidate_pool, teacher_replay, teacher_rules
    from qwen3vl_local.sft_new_loop_phase4.production_storage import binding, check_binding, open_route
    pool_path, index_path = Path(pool_path), Path(index_path)
    pool_binding, index_binding = binding(pool_path), binding(index_path)
    if index_binding['sha256'] != expected_index_sha256:
        raise ValueError('production index differs from received handoff SHA')
    pool = candidate_pool.validate(json.loads(pool_path.read_text()))
    index = json.loads(index_path.read_text())
    header = dict(policy=teacher_replay.POLICY, teacher=teacher_rules.identity(), candidate_pool_sha256=pool['sha256'])
    if index.get('sha256') != digest({k: v for k, v in index.items() if k != 'sha256'}) or any(index.get(k) != v for k, v in header.items()):
        raise ValueError('production identity mismatch; retain original source contract')
    routes = {(r['scenario'], r['route_id']): r for r in pool['routes']}
    keys, names, selected = set(), set(), []
    for receipt in index['routes']:
        key = receipt['scenario'], receipt['route_id']
        name = receipt['artifact']
        if key not in routes or key in keys or name in names or Path(name).name != name:
            raise ValueError('duplicate, foreign or escaping production receipt')
        if any(receipt[k] != routes[key][k] for k in ('physical_group', 'split')):
            raise ValueError('production receipt source mismatch')
        keys.add(key); names.add(name)
        if any(len(k.split('/')) == 6 and k.split('/')[1] == event and n > 0 for k, n in receipt['targets'].items()):
            selected.append((receipt, routes[key]))
    bindings = []

    def records():
        for receipt, route in selected:
            path = index_path.parent / receipt['artifact']
            before = binding(path)
            if before['sha256'] != receipt['sha256']:
                raise ValueError('production route changed')
            if teacher_replay.inspect_route(path, route, header) != receipt:
                raise ValueError('production receipt mismatch')
            with open_route(path) as stream:
                next(stream)
                for line in stream:
                    yield json.loads(line)
            check_binding(before)
            bindings.append(before)
    report = summarize(records(), event)
    raw_errors = []
    if data_root is not None:
        if event != 'U-E5':
            raise ValueError('raw departure probe currently supports only U-E5')
        from .departure_probe import inspect_example
        for example in report['train_examples']:
            try:
                example['departure_probe'] = inspect_example(example, data_root)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                error = dict(scenario=example['scenario'], route_id=example['route_id'],
                             frame_id=example['frame_id'], error=f'{type(exc).__name__}: {exc}')
                raw_errors.append(error)
                example['departure_probe'] = dict(status='invalid_raw_evidence', **error)
    check_binding(pool_binding); check_binding(index_binding)
    report.update(status='incomplete_raw_diagnostic' if raw_errors else 'verified_event_diagnostic',
        bindings=dict(pool=pool_binding, index=index_binding, inspected_routes=bindings),
        selected_routes=len(selected), declared_routes=len(keys),
        declared_complete_pool=keys == set(routes),
        diagnostic_sources={name: file_sha(Path(__file__).parent/name)
                            for name in ('event_gap.py', 'departure_probe.py')},
        raw_errors=raw_errors,
        raw_source_verification=('incomplete' if raw_errors else 'bounded_train_windows') if data_root is not None else 'not_run',
        verification_scope='native inspection of event-bearing receipts only; optional raw verification is limited to sampled training windows')
    return report


def main():
    from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--candidate-pool', type=Path, required=True)
    p.add_argument('--production-index', type=Path, required=True)
    p.add_argument('--expected-index-sha256', required=True)
    p.add_argument('--event', choices=EVENTS, default='U-E5')
    p.add_argument('--data-root', type=Path, help='Optional: verify raw sources for bounded training departure examples')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.production_index.resolve().parent):
        p.error('use a fresh report outside the production directory')
    report = inspect(a.candidate_pool, a.production_index,
                     expected_index_sha256=a.expected_index_sha256, event=a.event, data_root=a.data_root)
    if len(json.dumps(report).encode()) > 30_000_000:
        raise ValueError('diagnostic exceeds 30 MB; inputs are unchanged')
    write_json(a.output, report)
    print(json.dumps({k: report[k] for k in ('status', 'selected_routes', 'counts')}, ensure_ascii=False))
    return 0 if report['status'] == 'verified_event_diagnostic' else 2


if __name__ == '__main__':
    raise SystemExit(main())
