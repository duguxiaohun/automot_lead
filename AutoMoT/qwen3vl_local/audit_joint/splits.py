"""Cross-task physical-group audit. Input pools are never re-split or sampled."""
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import re

from .io import file_sha

ROLES = {'train_pool', 'dev_val', 'final_test', 'independent_review',
         'development_exposure', 'historical_regression', 'candidate_dev_val', 'quarantine', 'raw_inventory'}
PROHIBITED_FOR_NEW_VAL = {'train_pool', 'final_test', 'independent_review',
                          'development_exposure', 'historical_regression', 'quarantine'}


@lru_cache(maxsize=None)
def physical_group(scenario, route):
    # Use the actual implementations, not a third regex copied into the auditor.
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import physical_route_group
    from qwen3vl_local.action_prior.build_dataset import route_group
    for value in (scenario, route):
        if (not isinstance(value, str) or not value or value in {'.', '..'}
                or any(c in value for c in ('/', '\\', '\x00'))):
            raise ValueError('invalid scenario/route identity')
    key = physical_route_group(scenario, route)
    if key != route_group(scenario, route):
        raise ValueError('Phase3 and Action physical identities disagree')
    return key


def group_of(row):
    if isinstance(row, str):
        parts = row.split('/')
        if len(parts) != 2 or physical_group(*parts) != row:
            raise ValueError(f'noncanonical physical group: {row!r}')
        return row
    route = row.get('route_id', row.get('run_id'))
    declared = [row[k] for k in ('physical_group', 'physical_route_group', 'route_group') if k in row]
    if route is not None:
        group = physical_group(row['scenario'], route)
    elif declared:
        group = group_of(declared[0])
    else:
        raise ValueError('row lacks source route and physical group')
    if any(g != group for g in declared):
        raise ValueError('declared physical group disagrees with source route')
    return group


def records(spec):
    path = Path(spec['path'])
    if path.suffix == '.jsonl':
        if spec.get('records_key'):
            raise ValueError('JSONL cannot have records_key')
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        value = json.loads(path.read_text())
        for part in spec.get('records_key', '').split('.'):
            if part:
                value = value[part]
        if not isinstance(value, list):
            raise ValueError('records must be a list; set records_key explicitly')
        yield from value


def role_of(row, spec):
    roles = spec.get('split_roles')
    if roles is not None:
        split = row.get('split') if isinstance(row, dict) else None
        if split not in roles:
            raise ValueError(f'unmapped split: {split!r}')
        role = roles[split]
    else:
        role = spec['role']
    if role not in ROLES:
        raise ValueError(f'unknown role: {role}')
    if spec.get('expected_split') and row.get('split') != spec['expected_split']:
        raise ValueError('row split differs from index split')
    return role


def conflict_kind(left, right):
    a, b = left.rsplit(':', 1)[1], right.rsplit(':', 1)[1]
    if 'train_pool' in (a, b) and ({a, b} & {'dev_val', 'final_test', 'independent_review'}):
        return 'training_holdout_overlap'
    if {a, b} & {'development_exposure', 'historical_regression'} and {a, b} & {'dev_val', 'final_test', 'independent_review'}:
        return 'exposed_holdout_overlap'
    if a != b and {a, b} <= {'dev_val', 'final_test', 'independent_review'}:
        return 'holdout_role_overlap'
    if 'quarantine' in (a, b) and 'train_pool' in (a, b):
        return 'quarantined_training_overlap'
    return None


def audit_sources(specs, *, required_phases=('phase3', 'phase4', 'action')):
    """Fail closed on missing/invalid sources; retain conflicts in the available subset."""
    ids = [s['id'] for s in specs]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate source id')
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', s[k]) for s in specs for k in ('id', 'phase')):
        raise ValueError('source id/phase must be plain identifiers')
    members, identities = defaultdict(dict), defaultdict(set)
    reports, full_pools, content_sources = [], set(), set()
    support = defaultdict(lambda: dict(rows=0, groups=set()))
    candidates = set()
    for spec in specs:
        path = Path(spec['path'])
        report = dict(id=spec['id'], phase=spec['phase'], path=str(path),
                      scope=spec['scope'], full_pool=spec.get('full_pool', False))
        reports.append(report)
        if not path.is_file():
            report.update(status='missing', reason='source file unavailable')
            continue
        # Stage this source so a malformed last row cannot partially pass admission.
        staged, alias_keys, cells = {}, defaultdict(set), defaultdict(lambda: dict(rows=0, groups=set()))
        counts = Counter()
        image_checks, image_failures = 0, 0
        certification_errors = {spec['validation_error']} if spec.get('validation_error') else set()
        try:
            sha = file_sha(path)
            report['sha256'] = sha
            if spec.get('expected_sha256') and sha != spec['expected_sha256']:
                certification_errors.add('source SHA256 mismatch')
            report['sha256'] = sha
            for n, row in enumerate(records(spec), 1):
                group, role = group_of(row), role_of(row, spec)
                counts[role] += 1
                key = (group, role)
                cell = staged.setdefault(key, dict(rows=0, first_record=n))
                cell['rows'] += 1
                if not isinstance(row, dict):
                    continue
                route = row.get('route_id', row.get('run_id'))
                if spec.get('data_root') and route:
                    route_path = Path(spec['data_root']) / row['scenario'] / route
                    if not route_path.is_dir():
                        certification_errors.add(f'route directory unavailable: {route_path}')
                    else:
                        alias_keys[('resolved_route', str(route_path.resolve()))].add(group)
                hashes = row.get('image_sha256', [])
                pixels = row.get('image_rgb_sha256', [])
                if hashes or pixels:
                    images = row.get('images', [])
                    if len(hashes) != len(images) or len(pixels) != len(images):
                        raise ValueError('incomplete image identity arrays')
                    if spec.get('verify_images'):
                        from qwen3vl_local.sft_new_loop_phase4.model import load_images
                        # The native loader checks source, pixel SHA and causal provenance.
                        # Asset failures invalidate certification, not readable route identities.
                        image_checks += 1
                        try:
                            load_images(row, spec['data_root'])
                        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
                            image_failures += 1
                            certification_errors.add(f'image verification record {n}: {type(error).__name__}: {error}')
                    for kind, values in [('file_sha256', hashes), ('pixel_sha256', pixels)]:
                        for value in values:
                            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                                raise ValueError('invalid image digest')
                            alias_keys[(kind, value)].add(group)
                if spec.get('phase') == 'phase4' and role == 'dev_val':
                    # Teacher agreement must never fill manual validation support.
                    basis = row.get('label_basis')
                    event = row.get('episode', {}).get('event')
                    if event and basis in {'reviewed_transition_band', 'per_frame_conditions'}:
                        from qwen3vl_local.sft_new_loop_phase4.phase3_sampling import motion
                        movement = motion(row)
                        label = '/'.join([event, row.get('edge', 'unknown'), str(row.get('target', 'UNKNOWN')), movement])
                        cells[label]['rows'] += 1
                        cells[label]['groups'].add(group)
            if not counts:
                raise ValueError('empty source is not evidence of complete coverage')
            if file_sha(path) != sha:
                raise ValueError('source changed while auditing')
            certification_errors = sorted(set(certification_errors))
            source_status = 'invalid' if certification_errors else 'audited'
            for (group, role), value in staged.items():
                bucket = f"{spec['phase']}:{role}"
                members[group][spec['id'] + ':' + role] = dict(source=spec['id'], bucket=bucket,
                                                            source_status=source_status, **value)
                if role == 'candidate_dev_val':
                    candidates.add(group)
            for key, groups in alias_keys.items():
                identities[key].update(groups)
            for key, cell in (cells.items() if not certification_errors else []):
                # Keep input configurations separate, rather than doubling support with 2/4 RGB.
                target = support[spec['id'] + '/' + key]
                target['rows'] += cell['rows']
                target['groups'].update(cell['groups'])
            if spec.get('full_pool') is True and not certification_errors:
                full_pools.update((spec['phase'], role) for role in counts)
            if image_checks and not image_failures and any(k[0] == 'pixel_sha256' for k in alias_keys):
                content_sources.add(spec['id'])
            report.update(status=source_status, observation_status='observed',
                          certification_errors=certification_errors,
                          reason='; '.join(certification_errors),
                          rows=sum(counts.values()), roles=dict(counts),
                          physical_groups=len({g for g, _ in staged}),
                          image_checks=image_checks, image_failures=image_failures,
                          image_hashes_verified=spec['id'] in content_sources)
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
            report.update(status='invalid', reason=f'{type(error).__name__}: {error}')

    # Only a resolved directory is conclusive route-alias evidence. Image
    # equality remains a review signal and must never union unrelated routes.
    parents = {group: group for group in members}
    def canonical(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group
    for (kind, _), groups in identities.items():
        if kind == 'resolved_route':
            keys = sorted({canonical(g) for g in groups})
            for key in keys[1:]:
                parents[key] = keys[0]
    components = defaultdict(set)
    for group in members:
        components[canonical(group)].add(group)
    buckets = defaultdict(set)
    for group, entries in members.items():
        for entry in entries.values():
            buckets[entry['bucket']].add(canonical(group))
    matrix, conflicts = [], []
    names = sorted(buckets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            intersection = sorted(buckets[left] & buckets[right])
            matrix.append(dict(left=left, right=right, groups=len(intersection)))
            kind = conflict_kind(left, right)
            if kind and intersection:
                evidence = [dict(physical_group=g, **entry)
                            for key in intersection for g in sorted(components[key])
                            for entry in members[g].values() if entry['bucket'] in (left, right)]
                conflicts.append(dict(evidence_status='suspected' if any(e['source_status'] != 'audited' for e in evidence) else 'observed',
                                      sources=evidence, kind=kind, left=left, right=right, physical_groups=intersection,
                                      logical_groups=sorted({g for key in intersection for g in components[key]})))
    aliases = [dict(kind=kind, identity=identity, physical_groups=sorted(groups),
                    conclusion='potential_duplicate_content' if kind != 'resolved_route' else 'same_resolved_route')
               for (kind, identity), groups in sorted(identities.items()) if len(groups) > 1]
    # A shared black frame does not prove two physical routes are identical. It still
    # needs review before a new independent reservation; do not silently union keys.
    alias_neighbors = defaultdict(set)
    for alias in aliases:
        for group in alias['physical_groups']:
            alias_neighbors[group].update(alias['physical_groups'])
    filtered = []
    for group in sorted(candidates):
        reasons = []
        neighbors = set(components[canonical(group)])
        for member in list(neighbors):
            for neighbor in alias_neighbors[member]:
                neighbors.update(components[canonical(neighbor)])
        for neighbor in sorted(neighbors):
            for entry in members[neighbor].values():
                if entry['bucket'].rsplit(':', 1)[1] in PROHIBITED_FOR_NEW_VAL or entry['source_status'] != 'audited':
                    reasons.append(dict(group=neighbor, source=entry['source'], bucket=entry['bucket'],
                                        source_status=entry['source_status'],
                                        match='physical_group' if neighbor == group else (
                                            'same_resolved_route' if canonical(neighbor) == canonical(group) else 'content_review')))
        filtered.append(dict(physical_group=group, status='excluded' if reasons else 'eligible_pending_complete_audit',
                             reasons=reasons))
    missing = [dict(phase=p, role=r) for p in required_phases
               for r in ('train_pool', 'dev_val', 'final_test') if (p, r) not in full_pools]
    complete = not missing and all(r['status'] == 'audited' for r in reports)
    from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
    observed = {key.split('/')[1] for key in support}
    report = dict(schema='joint_split_cross_audit_v1',
                  status='conflicts_found' if conflicts else ('no_declared_group_conflict' if complete else 'incomplete'),
                  full_pool_coverage_complete=complete, missing_full_pools=missing,
                  independent_admission_ready=False,
                  admission_note='Group audit alone cannot certify source/pixel coverage, exposure completeness or label validity.',
                  sources=reports, buckets={k: len(v) for k, v in sorted(buckets.items())},
                  matrix=matrix, conflicts=conflicts, aliases=aliases,
                  resolved_group_components={k: sorted(v) for k, v in sorted(components.items()) if len(v) > 1},
                  manual_val_support={k: dict(rows=v['rows'], physical_groups=len({canonical(g) for g in v['groups']})) for k, v in sorted(support.items())},
                  manual_val_observed_events=sorted(observed),
                  manual_val_missing_events=sorted(set(EVENTS) - observed),
                  support_note='Per input-source counts, native motion threshold; no weak-teacher references counted as manual.',
                  candidate_filter=filtered, reservations_written=0)
    ledger = [dict(physical_group=g, uses=sorted(entries.values(), key=lambda x: (x['bucket'], x['source'])))
              for g, entries in sorted(members.items())]
    return report, ledger
