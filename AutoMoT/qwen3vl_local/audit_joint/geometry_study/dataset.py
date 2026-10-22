"""Derive a common eligible cohort without rewriting source data or receipts."""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
from . import regions
from ..geometry_v1 import dataset as native, sources
from ..geometry_v1.dataset import digest, write, validate_splits


def code_identity():
    app = Path(__file__).resolve().parents[3]
    paths = list(Path(__file__).parent.glob('*.py')) + list(Path(__file__).parent.glob('*.sh'))
    # Bind all reused native rule/prompt/split implementations, not just imports
    # visible in this file. No runtime git checkout or historical release rewrite.
    for package in ('sft_new_loop_phase3', 'audit_joint/geometry_v1',
                    'audit_joint/phase3_single_frame', 'audit_joint/phase3_base_dense'):
        paths.extend((app/'qwen3vl_local'/package).glob('*.py'))
    paths.extend(app/'qwen3vl_local/audit_joint'/p for p in ('label_quarantine.py','versioned_data.py'))
    return {str(p.relative_to(app)): sources.file_sha(p) for p in sorted(set(paths))}


def mass(t, count):
    if not t['valid']: return [0.]*count
    if 'target' in t: return list(t['target'])
    return [float(i == t['token_id']) for i in range(count)]


def combine(record, action):
    """Same anchors and query masks; never equate the meanings of the targets."""
    r = copy.deepcopy(record)
    r['representations'] = {'six': dict(history=action['history'], targets=action['targets'])}
    for variant in ('a', 'b', 'c'):
        r['representations'][variant] = dict(
            targets=[regions.relabel(t, variant, r['road_structure']) for t in r['targets']],
            history=[regions.relabel(t, variant, r['road_structure']) for t in r['observation']['history']])
    mask = [g['valid'] and s['valid'] for g,s in zip(r['targets'], action['targets'])]
    r['native_valid_masks'] = {v: [t['valid'] for t in view['targets']]
                               for v, view in r['representations'].items()}
    for view in r['representations'].values():
        for i,t in enumerate(view['targets']):
            if t['valid'] and not mask[i]:
                t.update(valid=False, reason='other_representation_missing_query')
    r['targets'] = r['representations']['a']['targets']
    r['sampling_mass'] = {f'{v}/{i}/{c}': q for v, view in r['representations'].items()
        for i,t in enumerate(view['targets']) for c,q in enumerate(mass(t, len(regions.names(v)))) if q}
    keep = ('frame', 'action', 'native_action', 'original_action', 'reason', 'imputed', 'imputation',
            'direction_projection', 'direction_origin_frame', 'required_end_frame')
    r['six_audit'] = {key: [{k:p[k] for k in keep if k in p} for p in action['audit'][key]]
                       for key in ('history_points', 'future_points')}
    return r


def support_report(records):
    counts = Counter(); groups = defaultdict(set); mass_counts = Counter(); directions = Counter()
    for r in records:
        for kind, points in r['six_audit'].items():
            for p in points:
                directions[f'{kind}/{p.get("native_action")}->{p["action"]}'] += 1
        for v, view in r['representations'].items():
            for i,t in enumerate(view['targets']):
                base = f'{r["split"]}/{r["event"]}/{v}/{i}'
                counts[base+'/requested'] += 1
                counts[base+'/valid'] += t['valid']
                for c,q in enumerate(mass(t, len(regions.names(v)))):
                    if q:
                        key = base+'/'+regions.names(v)[c]
                        mass_counts[key] += q; groups[key].add(r['physical_group'])
    return dict(query_counts=dict(counts), target_mass=dict(mass_counts),
        physical_groups={k:len(v) for k,v in groups.items()}, direction_changes=dict(directions),
        thresholds_refitted=False, test_used_for_fitting=False,
        independent_holdout_claim=False, adjacent_frames_are_not_independent=True)


def build(geometry, maps, output):
    from . import six
    from .. import label_quarantine
    from qwen3vl_local.sft_new_loop_phase3 import trajectory_action as rule
    geometry = Path(geometry).resolve(strict=True); output = Path(output).resolve()
    artifact = native.load(geometry)
    root = Path(artifact['data_root']); maps = Path(maps).resolve(strict=True)
    for p in (root, geometry.parent, maps):
        if output.is_relative_to(p) or p.is_relative_to(output): raise ValueError('output_overlaps_input')
    output.mkdir(parents=True, exist_ok=False)
    bound = code_identity(); parent_sha = sources.file_sha(geometry)
    lanes = sources.LaneMap(maps); grouped = defaultdict(list)
    for r in artifact['records']: grouped[r['scenario'], r['route_id']].append(r)
    result = []; rejected = Counter(); candidate_counts = Counter()
    for n, ((scenario, route), records) in enumerate(sorted(grouped.items()), 1):
        trajectory = rule.load_route_trajectory(root/scenario/route)
        if trajectory is None: raise ValueError('unreadable_native_six_trajectory: '+route)
        quarantines = label_quarantine.load(root, route=(scenario,route))
        native_cache = {}; reference_cache = {}
        for r in records:
            candidate_counts[r['split']+'/'+r['event']] += 1
            if r['road_structure'] not in ('R1','R2','R3','R4','R5'):
                rejected[r['split']+'/'+r['event']+'/unknown_current_road_category'] += 1
                continue
            # Check the same current map reference before using local-origin references.
            state = sources.observed(trajectory.metas[r['frame']], route.split('_')[0], lanes)
            if state['reference'] != r['observation']['reference']:
                raise ValueError('study_map_reference_differs_from_native')
            action = six.build(r, trajectory, quarantines, lanes, native_cache=native_cache,
                               reference_cache=reference_cache)
            if action['anchor_action'] is None:
                rejected[r['split']+'/'+r['event']+'/native_six_anchor_ineligible'] += 1
                continue
            merged = combine(r, action)
            if not any(t['valid'] for t in merged['targets']):
                rejected[r['split']+'/'+r['event']+'/no_common_future_query'] += 1
                continue
            result.append(merged)
        if n % 10 == 0: print(f'[geometry study] routes={n}/{len(grouped)} common={len(result)}', flush=True)
    if not result: raise ValueError('empty_common_cohort')
    validate_splits(result)
    if bound != code_identity() or parent_sha != sources.file_sha(geometry):
        raise ValueError('study_inputs_changed_during_build')
    # Reverify raw input bytes after native six-action replay.
    native.verify(artifact)
    map_proof = {str(maps/(town+'.xodr')): sha for town,sha in lanes.hashes.items()}
    for p,sha in map_proof.items():
        if sources.file_sha(p) != sha: raise ValueError('map_changed_during_build')
    study = dict(contract={v:regions.contract(v) for v in regions.VARIANTS}, code=bound,
        data_root=str(root), parent_geometry=str(geometry), parent_sha256=parent_sha,
        map_proof=map_proof, records=result, scoped_smoke_only=artifact.get('scoped_smoke_only', False))
    write(output/'dataset.json', study)
    report = dict(dataset_sha256=sources.file_sha(output/'dataset.json'), candidates=dict(candidate_counts),
        rejected=dict(rejected), anchors=len(result), support=support_report(result),
        status='common_cohort_built_capacity_plan_required', old_artifacts_modified=False)
    write(output/'receipt.json', report)
    return study, report


def load(path, *, variant=None, check_sources=True, expected_sha256=None,
         check_budget=lambda: None, with_sha256=False):
    import hashlib
    path = Path(path); check_budget()
    receipt = json.loads(path.with_name('receipt.json').read_text())
    raw = path.read_bytes(); sha = hashlib.sha256(raw).hexdigest(); check_budget()
    if sha != receipt['dataset_sha256'] or expected_sha256 is not None and sha != expected_sha256:
        raise ValueError('study_dataset_changed')
    artifact = json.loads(raw)
    if artifact['contract'] != {v:regions.contract(v) for v in regions.VARIANTS} or artifact['code'] != code_identity():
        raise ValueError('study_contract_or_code_changed')
    validate_splits(artifact['records'])
    if check_sources:
        native.load(artifact['parent_geometry'], expected_sha256=artifact['parent_sha256'], check_budget=check_budget)
        for p, value in artifact['map_proof'].items():
            if sources.file_sha(p, check_budget=check_budget) != value: raise ValueError('study_map_changed')
    if variant is not None:
        from . import admission
        # Compute all four supports before stripping the other representations.
        # This is recomputed from verified bytes, never trusted from a receipt.
        artifact['admission'] = admission.inspect(artifact)
        regions.names(variant)
        artifact['contract'] = artifact['contract'][variant]
        artifact['variant'] = variant
        for r in artifact['records']:
            view = r['representations'][variant]
            r['variant'] = variant; r['targets'] = view['targets']
            r['observation']['history'] = view['history']
            # Drop other label payloads from model-facing records. Sampling mass
            # remains identical, and is never consumed by messages().
            del r['representations']
    return (artifact, sha) if with_sha256 else artifact
