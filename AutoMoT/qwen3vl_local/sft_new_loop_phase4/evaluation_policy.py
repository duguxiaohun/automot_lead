"""Production selection contract: fixed ten-event references, explicit provenance."""
import json
import math
from pathlib import Path
from .identity import digest, file_sha
from .taxonomy import EVENTS
from qwen3vl_local.sft_new_loop_phase3.evaluation_balance import select

POLICY = 'phase4_equal_reference_selection_v1'
MODES = ('reviewed', 'rule_consistency')


def original(row, field):
    return row.get('student_reference', {}).get('original_' + field, row.get(field))


def select_rows(rows, *, seed=20261011):
    by_id = {r['id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('duplicate evaluation row ID')
    ids, report = select([(r['episode']['event'], r['id'], r['physical_group']) for r in rows], EVENTS, seed=seed)
    report['rows_sha256'] = digest(sorted(rows, key=lambda r: r['id']))
    return [by_id[i] for i in ids], report


def pack_binding(path):
    path = Path(path).absolute()
    return {name: dict(logical_path=str(path/name), resolved_path=str((path/name).resolve()), sha256=file_sha(path/name))
            for name in ('references.json', 'val.jsonl', 'test.jsonl')}


def teacher_audit_groups():
    from .dataset import producer_check_reservations, PRODUCER_CHECK_PLAN
    from .identity import ROOT
    groups = set(producer_check_reservations())  # validates frozen source/extension SHA
    plan = json.loads((ROOT/PRODUCER_CHECK_PLAN).read_text())
    extension = plan.get('instance_only_extension')
    if extension:
        groups.update(r['physical_group'] for r in json.loads((ROOT/extension['path']).read_text())['reservations'])
    return groups


def prepare(data, *, mode='reviewed', references=None, rgb_mode=4, short=False, seed=20261011, natural=False):
    """Validate references and training isolation before creating a model/run.

    Automatic mode uses only native rule training rows and explicit held-out
    rule references. Reviewed labels are not silently mixed into that score.
    """
    from .admission import training_report
    from .dataset import coverage_report
    if mode not in MODES:
        raise ValueError('unknown reference selection mode')
    if (mode == 'rule_consistency') != bool(references):
        raise ValueError('rule_consistency requires --teacher-references; reviewed mode must not supply it')
    selected_data = {s: list(data.get(s, [])) for s in ('train', 'val', 'test')}
    binding = None
    exclusions = []
    if mode == 'rule_consistency':
        from .teacher_evaluation import load
        binding = pack_binding(references)
        selected_data['train'] = [r for r in data['train'] if original(r, 'label_basis') in ('weak_rule_teacher', 'approved_rule_teacher')]
        heldout = {s: load(references, split=s, rgb_mode=4 if short else rgb_mode) for s in ('val', 'test')}
        protected = teacher_audit_groups()
        for split in ('val', 'test'):
            exclusions.extend(dict(id=r['id'],split=split,reason='reserved_independent_teacher_audit')
                              for r in heldout[split] if r['physical_group'] in protected)
            heldout[split] = [r for r in heldout[split] if r['physical_group'] not in protected]
        if short:
            from .rgb_short.data import paired_view, exposure_inventory
            exposed, exposure_files = exposure_inventory()
            heldout, short_exclusions = paired_view(heldout, rgb_mode, exposed)
            exclusions.extend(short_exclusions)
            binding['exposure_inventory'] = exposure_files
        selected_data.update(heldout)
        after = pack_binding(references)
        if any(binding[k] != v for k, v in after.items()):
            raise ValueError('teacher reference binding changed during loading')
    else:
        for split in ('val', 'test'):
            selected_data[split] = [r for r in selected_data[split] if original(r, 'reference_kind') != 'rule_teacher']
    # Check the full eligible pools, not just the small equal subsets.
    admission = training_report(selected_data, coverage_report(sum(selected_data.values(), [])))
    if not admission['trainable']:
        raise ValueError('selection input admission failed: ' + '; '.join(admission['errors']))
    if mode == 'rule_consistency':
        admission['evaluation_scope'] = 'held-out rule consistency; not independent accuracy'
    train_events = {r['episode']['event'] for r in selected_data['train']}
    if train_events != set(EVENTS):
        raise ValueError('training requires all ten events: missing=' + str(sorted(set(EVENTS)-train_events)))
    plans = {}
    for split in ('val', 'test'):
        chosen, plans[split] = select_rows(selected_data[split], seed=seed)
        if not natural:
            selected_data[split] = chosen
    return selected_data, dict(policy=POLICY, mode=mode, seed=seed, plans=plans, training_admission=admission,
        teacher_references=binding, exclusions=exclusions,
        train_ids=[r['id'] for r in selected_data['train']],
        train_ids_sha256=digest([r['id'] for r in selected_data['train']]),
        selection_metric=('ten_event_rule_consistency' if mode == 'rule_consistency' else 'ten_event_reviewed_reference_agreement'),
        independent_accuracy_certified=False,
        scope='fixed event-equal reference agreement; rule consistency is not independent accuracy')


def score_cases(cases, mode, expected_ids=None):
    from .evaluate import _reference_metrics
    if mode not in MODES:
        raise ValueError('unknown reference selection mode')
    if any((r.get('original_reference_kind', r.get('reference_kind')) == 'rule_teacher') != (mode == 'rule_consistency') for r in cases):
        raise ValueError('selection reference provenance differs from configured mode')
    if expected_ids is not None and ([r['id'] for r in cases] != list(expected_ids) or len(set(expected_ids)) != len(expected_ids)):
        raise ValueError('evaluation results do not match fixed selection IDs/order')
    measured = _reference_metrics(cases)
    counts = measured['event_counts']
    score = measured['event_macro_accuracy']
    if score is None or len(set(counts.values())) != 1 or not math.isfinite(score):
        raise ValueError('selection requires complete ten-event equal evaluation results')
    return dict(selection_agreement=score,
                selection_metric='ten_event_rule_consistency' if mode == 'rule_consistency' else 'ten_event_reviewed_reference_agreement',
                selection_mode=mode, independent_accuracy_certified=False)


def restore(data, run, *, short=False, natural=False):
    saved = run.get('training', {}).get('evaluation')
    if not saved or saved.get('policy') != POLICY:
        raise ValueError('checkpoint lacks current fixed evaluation contract; use its original code')
    pool = {r['id']:r for r in data['train']}
    if len(pool) != len(data['train']) or any(i not in pool for i in saved['train_ids']):
        raise ValueError('recorded training intersection missing from dataset')
    data = dict(data, train=[pool[i] for i in saved['train_ids']])
    binding = saved.get('teacher_references')
    path = Path(binding['references.json']['logical_path']).parent if binding else None
    selected, current = prepare(data, mode=saved['mode'], references=path,
        rgb_mode=run['observation_contract']['rgb_mode'], short=short, seed=saved['seed'], natural=natural)
    if current != saved:
        raise ValueError('saved evaluation plan/reference binding changed')
    return selected
