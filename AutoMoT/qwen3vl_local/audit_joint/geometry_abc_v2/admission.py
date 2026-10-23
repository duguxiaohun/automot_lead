"""CPU admission on the final common pool; never repair or move split members."""
from collections import Counter, defaultdict
import math
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
from . import regions
from .dataset import digest, mass, validate_splits

SCHEMA = 'geometry_abc_v2_admission_v1'
VALIDATION_PER_EVENT = None


def validation_ids(records):
    return [r['id'] for event in CONTEXT_IDS for r in sorted(
        (r for r in records if r['split'] == 'val' and r['event'] == event
         and any(t['valid'] for t in r['targets'])), key=lambda r: r['id'])]


def support(records, variant):
    counts = Counter(); groups = defaultdict(set); classes = Counter(); class_groups = defaultdict(set)
    events = {e: Counter() for e in CONTEXT_IDS}
    for r in records:
        for t in r['representations'][variant]['targets']:
            kind = 'missing'
            if t['valid']:
                q = mass(t, len(regions.names(variant)))
                if (len(q) != len(regions.names(variant)) or any(not math.isfinite(x) or x < 0 for x in q)
                        or not math.isclose(sum(q), 1., abs_tol=1e-6)):
                    raise ValueError('invalid_admission_target_distribution')
                kind = 'pure' if 1. in q else 'mixed'
                if kind == 'pure':
                    name = regions.names(variant)[q.index(1.)]
                    classes[name] += 1; class_groups[name].add(r['physical_group'])
            counts[kind] += 1; groups[kind].add(r['physical_group'])
            events.setdefault(r['event'], Counter())[kind] += 1
    return dict(anchors=len(records), physical_groups=len({r['physical_group'] for r in records}),
        queries={k: counts[k] for k in ('pure', 'mixed', 'missing')},
        query_physical_groups={k: len(groups[k]) for k in ('pure', 'mixed', 'missing')},
        class_support={n: classes[n] for n in regions.names(variant)},
        class_physical_groups={n: len(class_groups[n]) for n in regions.names(variant)},
        event_queries={e: dict(c) for e, c in events.items()}, macro_recall_defined=bool(classes))


def inspect(artifact):
    records = artifact['records']; validate_splits(records)
    errors = []
    if artifact.get('scoped_smoke_only'): errors.append('scoped_smoke_dataset_not_for_formal_training')
    for r in records:
        if set(r.get('representations', {})) != set(regions.VARIANTS):
            raise ValueError('abc_representation_payloads_required_for_admission')
        masks = [[t['valid'] for t in r['representations'][v]['targets']] for v in regions.VARIANTS]
        if any(len(m) != 4 or m != masks[0] for m in masks) or masks[0] != [t['valid'] for t in r['targets']]:
            raise ValueError('common_query_mask_mismatch')
    coverage = {}
    for split in ('train', 'val', 'test'):
        rows = [r for r in records if r['split'] == split and any(t['valid'] for t in r['targets'])]
        present = {r['event'] for r in rows}
        missing = sorted(set(CONTEXT_IDS) - present); unexpected = sorted(present - set(CONTEXT_IDS))
        coverage[split] = dict(missing_events=missing, unexpected_events=unexpected,
            events={e: dict(anchors=sum(r['event'] == e for r in rows),
                physical_groups=len({r['physical_group'] for r in rows if r['event'] == e})) for e in CONTEXT_IDS})
        if missing or unexpected: errors.append(f'{split}_event_coverage_incomplete: missing={missing}, unexpected={unexpected}')
    fixed = validation_ids(records); selected = set(fixed)
    validation = {v: support([r for r in records if r['id'] in selected], v) for v in regions.VARIANTS}
    result = dict(schema=SCHEMA, status='rejected' if errors else 'admitted', errors=errors,
        selection='all eligible validation members, event then id; no correctness-based selection',
        fixed_validation_ids=fixed, validation=validation, coverage=coverage,
        split_members_modified=False, independent_holdout_claim=False)
    return dict(result, sha256=digest(result))


def require(report):
    payload = {k: v for k, v in report.items() if k != 'sha256'}
    if report.get('schema') != SCHEMA or report.get('sha256') != digest(payload):
        raise ValueError('study_admission_binding_mismatch')
    if report.get('status') != 'admitted' or report['errors']:
        raise ValueError('study_cpu_admission_rejected: ' + '; '.join(report['errors']))
    return report
