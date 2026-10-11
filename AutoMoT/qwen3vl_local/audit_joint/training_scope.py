"""Read-only event balance and automatic-label support audit using native loaders.

This is a distribution audit, not label certification or a training launcher.
Missing events remain in the denominator. Repeated evaluation items never count
as additional independent support. No labels, splits or sampling policy change.
"""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

from .io import digest, file_sha, write_json


def balance(records, expected):
    """records are (event, question identity, physical group); keep exposures."""
    expected = tuple(expected)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('expected events must be nonempty and unique')
    counts, identities, groups, owners = Counter(), defaultdict(set), defaultdict(set), {}
    for event, ident, group in records:
        if event not in expected or not ident or not group:
            raise ValueError('unknown event or missing question/group identity')
        if ident in owners and owners[ident] != (event, group):
            raise ValueError('question identity has conflicting event/group')
        owners[ident] = (event, group)
        counts[event] += 1
        identities[event].add(ident)
        groups[event].add(group)
    cells = {e: dict(presentations=counts[e], unique_questions=len(identities[e]),
                    physical_groups=len(groups[e])) for e in expected}
    missing = [e for e in expected if not counts[e]]
    capacity = min(len(identities[e]) for e in expected)
    return dict(events=cells, missing_events=missing,
                equal_presentations=not missing and len({counts[e] for e in expected}) == 1,
                equal_unique_questions=not missing and len({len(identities[e]) for e in expected}) == 1,
                max_equal_eval_questions_per_event_without_replacement=capacity,
                equal_eval_capacity_scope='question count only; no class, route-independence or accuracy guarantee')


def original_basis(row):
    return row.get('student_reference', {}).get('original_label_basis', row['label_basis'])


def equal_eval_selection(records, expected, *, seed=20261011):
    """Fixed, label-blind, no-replacement subset; never fill a missing event.

    Round-robin physical groups within each event avoids selecting one long
    route first. It does not certify independence between groups or tasks.
    """
    records = sorted(set(records))
    inventory = balance(records, expected)
    quota = inventory['max_equal_eval_questions_per_event_without_replacement']
    buckets = defaultdict(lambda: defaultdict(list))
    for event, ident, group in records:
        buckets[event][group].append(ident)
    selected = []
    for event in expected:
        queues = [(group, deque(sorted(ids, key=lambda ident: digest([seed, event, ident]))))
                  for group, ids in sorted(buckets[event].items(),
                                           key=lambda item: digest([seed, event, item[0]]))]
        taken = 0
        while taken < quota:
            for group, queue in queues:
                if queue and taken < quota:
                    selected.append((event, queue.popleft(), group))
                    taken += 1
    return dict(status='selected' if quota else 'blocked_missing_events', seed=seed,
                per_event=quota, missing_events=inventory['missing_events'],
                pool_sha256=digest(records), selected_ids=[r[1] for r in selected],
                ordered_ids_sha256=digest([r[1] for r in selected]),
                distribution=balance(selected, expected),
                scope='supplementary fixed subset; natural evaluation unchanged; question equality only; no label or independence approval')


def file_binding(path):
    path = Path(path).absolute()
    resolved = str(path.resolve())
    sha = file_sha(path) if path.exists() else None
    if str(path.resolve()) != resolved:
        raise ValueError(f'input path redirected during audit: {path}')
    return dict(logical_path=str(path), resolved_path=resolved, sha256=sha)


def dataset_bindings(dataset):
    """Bind actual statistical inputs, including symlinks and absent view marker.

    Native loaders still perform all admission checks. This closes the interval
    between their checks and completion of the (potentially seven-epoch) audit.
    It is not a pixel/source-geometry verification pass.
    """
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.data import parent_binding
    dataset = Path(dataset).absolute()
    paths = {dataset / 'student_view.json'}
    parent = dataset
    if (dataset / 'student_view.json').is_file():
        descriptor = json.loads((dataset / 'student_view.json').read_text())
        parent = Path(descriptor['parent']['logical_path'])
    paths.add(parent / 'manifest.json')
    manifest = json.loads((parent / 'manifest.json').read_text())
    paths.update(parent / name for name in parent_binding(parent, manifest))
    return [file_binding(path) for path in sorted(paths)]


def check_output(output, dataset, phase3_path=None):
    roots = [Path(dataset)]
    view = roots[0] / 'student_view.json'
    if view.is_file():
        roots.append(Path(json.loads(view.read_text())['parent']['logical_path']))
    if phase3_path:
        roots.append(Path(phase3_path).parent)
    if any(Path(output).resolve().is_relative_to(root.resolve()) for root in roots):
        raise ValueError('audit output must be outside source datasets (including view parent and Phase3)')


def phase4_support(data):
    from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
    from qwen3vl_local.sft_new_loop_phase4.dataset import coverage_report
    split_reports = {}
    for split, rows in data.items():
        report = balance(((r['episode']['event'], r['id'], r['physical_group']) for r in rows), EVENTS)
        report['answers'] = dict(Counter(r['target'] for r in rows))
        report['original_label_basis'] = dict(Counter(original_basis(r) for r in rows))
        if split in ('val', 'test'):
            report['equal_event_selection'] = equal_eval_selection(
                [(r['episode']['event'], r['id'], r['physical_group']) for r in rows], EVENTS)
        split_reports[split] = report
    automatic = [r for r in data['train'] if original_basis(r) in ('weak_rule_teacher', 'approved_rule_teacher')]
    # Reuse the producer's actual branch/edge definitions instead of a second taxonomy.
    coverage = coverage_report(automatic)
    cells = defaultdict(lambda: dict(rows=0, physical_groups=set()))
    for r in data['train']:
        ep = r['episode']
        key = '/'.join((ep['event'], ep.get('branch', 'default'),
                        'return' if ep.get('return_required', True) else 'no_return',
                        ep['state'], ep['longitudinal'], r['edge'], r['slice'],
                        r['target'], original_basis(r)))
        cells[key]['rows'] += 1
        cells[key]['physical_groups'].add(r['physical_group'])
    # Explicit current experiment priorities, not a replacement for the full
    # native transition matrix and not an invented YES/NO balancing objective.
    priorities = [('U-E3', 'proceed', 'YES'), ('U-E5', 'proceed', 'YES'),
                  ('U-E6', 'proceed', 'YES'), ('U-E7', 'complete', 'YES'),
                  ('U-E2', 'return', None)]
    critical = {}
    for event, edge, answer in priorities:
        supported = [r for r in automatic if r['episode']['event'] == event
                     and r['edge'] == edge and r['slice'] == 'readiness'
                     and (answer is None or r['target'] == answer)]
        critical[f'{event}/{edge}/{answer or "any_answer"}'] = dict(
            rows=len(supported), unique_questions=len({r['id'] for r in supported}),
            physical_groups=len({r['physical_group'] for r in supported}),
            status='present_not_certified' if supported else 'missing_in_supplied_pool')
    return dict(splits=split_reports,
                priority_automatic_support=critical,
                automatic_train=balance(((r['episode']['event'], r['id'], r['physical_group']) for r in automatic), EVENTS),
                automatic_readiness_support={k: v for k, v in coverage['transition_support'].items() if k.startswith('train/')},
                automatic_missing_readiness_edges=[k for k in coverage['missing_transition_edges'] if k.startswith('train/')],
                train_cells={k: dict(rows=v['rows'], physical_groups=len(v['physical_groups'])) for k, v in sorted(cells.items())},
                automatic_scope='native rule-generated rows only; per_frame_conditions/reviewed labels are not automatic rule approval')


def phase4_schedule(rows, epochs):
    from qwen3vl_local.sft_new_loop_phase4.sampling import plan
    from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
    schedules = {}
    for world in (1, 4):
        history, entries = None, []
        for epoch in range(epochs):
            indices, audit = plan(rows, epoch=epoch, world_size=world,
                                  policy='phase3_balanced', history=history)
            selected = [rows[i] for i in indices]
            measured = balance(((r['episode']['event'], r['id'], r['physical_group']) for r in selected), EVENTS)
            measured.update(epoch=epoch, ordered_ids_sha256=digest([r['id'] for r in selected]),
                            answers=dict(Counter(r['target'] for r in selected)),
                            max_question_repeat=audit['max_question_repeat'],
                            max_frame_repeat=audit['max_frame_repeat'],
                            event_repetition=audit['event_repetition'])
            entries.append(measured)
            history = audit['next_history']
        schedules[str(world)] = entries
    return dict(worlds=schedules,
                world1_world4_order_equal=all(a['ordered_ids_sha256'] == b['ordered_ids_sha256']
                    for a, b in zip(schedules['1'], schedules['4'])),
                scope='CPU native plans; no GPU execution, gradient equivalence or parameter updates')


def phase3_work_balance(work, *, evaluation=False):
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
    from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity
    from .splits import group_of
    records = [(item.row.context_id, digest(case_identity(item)),
                group_of(dict(scenario=item.row.scenario, route_id=item.row.route_id)))
               for item in work if not item.row.answers.get('INVALID_ACTION_CONTEXT', False)]
    result = balance(records, CONTEXT_IDS)
    result['invalid_presentations'] = len(work) - len(records)
    if evaluation:
        result['equal_event_selection'] = equal_eval_selection(records, CONTEXT_IDS)
    return result


def phase3_index(path, data_root, mode, variant):
    from qwen3vl_local.sft_new_loop_phase3 import train, eval as evaluate
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_index
    from qwen3vl_local.sft_new_loop_phase3.invalid_balance import unique_cases
    if variant != 'baseline':
        from qwen3vl_local.sft_new_loop_phase3.candidate_data import validate_candidate_index
        validate_candidate_index(path)
    check_index(path, action_output_mode=mode)
    result = {}
    for split in ('train', 'val', 'test'):
        rows = train._read_rows(Path(path), split, data_root=Path(data_root))
        if split == 'train':
            work = train._balanced_work(rows, target_per_bin=1024, seed=20260929, action_output_mode=mode)
            result[split] = phase3_work_balance(work)
        else:
            work, audit = train._validation_work(rows, target_per_bin=64, seed=20260929, action_output_mode=mode, equal_events=True)
            from qwen3vl_local.sft_new_loop_phase3.evaluation_balance import select_work
            natural = evaluate._balanced_cases(rows, cases_per_bin=0, seed=20260929, action_output_mode=mode)
            fixed, fixed_plan = select_work(natural, seed=20260929)
            result[split] = dict(periodic_before_dedup=phase3_work_balance(work),
                                 periodic_generation=phase3_work_balance(unique_cases(work), evaluation=True),
                                 full_evaluation=phase3_work_balance(fixed, evaluation=True),
                                 natural_evaluation=phase3_work_balance(natural),
                                 evaluation_plan=fixed_plan, native_sampling_audit=audit)
    return dict(status='measured', splits=result, action_output_mode=mode, prompt_variant=variant,
                scope='CPU native plans with explicit audit budgets: train 1024, periodic cap 64 (without replacement); no RGB decoding or GPU')


def run(dataset, *, epochs=1, phase3_path=None, data_root='lead_data', mode='choice', variant='baseline'):
    from qwen3vl_local.sft_new_loop_phase4 import dataset as native
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import data as short
    from qwen3vl_local.action_prior.event_balance import EVENT_BALANCE_WEIGHTS, weighted_quotas
    if type(epochs) is not int or epochs < 1:
        raise ValueError('epochs must be positive')
    root = Path(__file__).resolve().parents[1]
    names = ('audit_joint/training_scope.py', 'action_prior/event_balance.py',
             'action_prior/metrics.py', 'action_prior/training_core.py',
             'sft_new_loop_phase3/train.py', 'sft_new_loop_phase3/eval.py',
             'sft_new_loop_phase3/invalid_balance.py', 'sft_new_loop_phase3/preflight.py',
             'sft_new_loop_phase3/evaluation_balance.py', 'sft_new_loop_phase3/candidate_data.py', 'sft_new_loop_phase3/prompt_contract.py',
             'sft_new_loop_phase3/context_taxonomy.py', 'sft_new_loop_phase4/evaluate.py',
             'sft_new_loop_phase4/dataset.py', 'sft_new_loop_phase4/sampling.py',
             'sft_new_loop_phase4/phase3_sampling.py', 'sft_new_loop_phase4/rgb_short/data.py')
    sources = {name: file_sha(root / name) for name in names}
    dataset = Path(dataset).absolute()
    bindings = dataset_bindings(dataset)
    view = (dataset / 'student_view.json').is_file()
    entry = dataset / ('student_view.json' if view else 'manifest.json')
    initial = file_sha(entry)
    modes = {}
    for rgb in ((2, 4) if view else (None,)):
        rows, manifest = short.load_dataset(dataset, rgb_mode=rgb) if view else native.load_dataset(dataset)
        modes[str(manifest['rgb_mode'])] = dict(support=phase4_support(rows),
            schedule=phase4_schedule(rows['train'], epochs),
            pool_scope=dict(complete_causal_production=manifest.get('production_coverage', {}).get('complete_causal_production'),
                candidate_routes=manifest.get('production_coverage', {}).get('candidate_routes'),
                processed_routes=manifest.get('production_coverage', {}).get('accounting', {}).get('processed_routes'),
                complete_coverage=manifest.get('complete_coverage'),
                scope='relative to registered candidate pool only; not proof of full raw-source coverage'),
            native_trainable=manifest['trainable'], manifest_sha256=digest(manifest))
    if file_sha(entry) != initial:
        raise ValueError('dataset descriptor changed during audit')
    p3 = dict(status='missing', reason='no current Phase3 formal index supplied; diagnostic candidates are not a formal pool')
    if phase3_path:
        p3bindings = [file_binding(path) for path in (Path(phase3_path), Path(phase3_path).with_name('manifest.json'))]
        if p3bindings[0]['sha256'] is None:
            p3 = dict(status='missing', reason='specified Phase3 index does not exist')
        elif p3bindings[1]['sha256'] is None:
            p3 = dict(status='incomplete', reason='formal Phase3 index requires manifest.json')
        else:
            p3 = phase3_index(phase3_path, data_root, mode, variant)
        if p3bindings != [file_binding(b['logical_path']) for b in p3bindings]:
            raise ValueError('Phase3 input binding changed during audit')
        p3['input_bindings'] = p3bindings
        p3['index_sha256'] = p3bindings[0]['sha256']
    paired = None
    if view:
        paired = all(a['ordered_ids_sha256'] == b['ordered_ids_sha256']
                     for w in ('1', '4')
                     for a, b in zip(modes['2']['schedule']['worlds'][w], modes['4']['schedule']['worlds'][w]))
    if sources != {name: file_sha(root / name) for name in names}:
        raise ValueError('audited source changed during audit')
    if bindings != dataset_bindings(dataset):
        raise ValueError('dataset input binding changed during audit')
    return dict(schema='training_scope_audit_v2', status='measured',
        sources=sources,
        input_bindings=bindings,
        dataset=dict(logical_path=str(dataset), resolved_path=str(dataset.resolve()), entry_sha256=initial),
        phase4=modes, rgb2_rgb4_plan_order_equal=paired, phase3=p3,
        action_reference=dict(weights=EVENT_BALANCE_WEIGHTS, example_quotas=weighted_quotas(12288),
            evaluation='natural unique rows plus fixed weighted event metrics; missing buckets block balanced selection'),
        label_accuracy_certified=False, training_authorized_by_report=False,
        scope='distribution and support only; missing assets/labels cannot be repaired by quota changes')


def main():
    from qwen3vl_local.sft_new_loop_phase3.prompt_contract import PROMPT_VARIANTS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase4-dataset', required=True)
    p.add_argument('--phase3-index')
    p.add_argument('--data-root', default='lead_data')
    p.add_argument('--phase3-mode', choices=('binary', 'choice'), default='choice')
    p.add_argument('--phase3-variant', choices=PROMPT_VARIANTS, default='baseline')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    check_output(out, a.phase4_dataset, a.phase3_index)
    out.mkdir(parents=True, exist_ok=False)
    try:
        result = run(a.phase4_dataset, epochs=a.epochs, phase3_path=a.phase3_index,
                     data_root=a.data_root, mode=a.phase3_mode, variant=a.phase3_variant)
        result['audit_source_sha256'] = file_sha(__file__)
        write_json(out / 'report.json', result)
        print(json.dumps(dict(status=result['status'], phase3=result['phase3']['status'], output=str(out))))
    except Exception as exc:
        write_json(out / 'failure.json', dict(status='failed', error=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__':
    main()
