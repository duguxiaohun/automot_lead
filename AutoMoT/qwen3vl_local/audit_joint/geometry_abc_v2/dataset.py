"""Derive a common eligible cohort without rewriting source data or receipts."""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
from . import regions
from ..geometry_v1 import dataset as native, sources
from ..geometry_v1.dataset import digest, validate_splits


def code_identity():
    app = Path(__file__).resolve().parents[3]
    paths = list(Path(__file__).parent.glob('*.py')) + list(Path(__file__).parent.glob('*.sh'))
    # Bind reused geometric/prompt/split implementations, not just imports
    # visible in this file. No runtime git checkout or historical release rewrite.
    for package in ('sft_new_loop_phase3', 'audit_joint/geometry_v1',
                    'audit_joint/phase3_single_frame', 'audit_joint/phase3_base_dense'):
        paths.extend((app/'qwen3vl_local'/package).glob('*.py'))
    paths.extend(app/'qwen3vl_local/audit_joint'/p for p in ('label_quarantine.py','versioned_data.py','phase3_comparison_policy.py','phase3_comparison_diagnostics.py'))
    return {str(p.relative_to(app)): sources.file_sha(p) for p in sorted(set(paths))}


def mass(t, count):
    if not t['valid']: return [0.]*count
    if 'target' in t: return list(t['target'])
    return [float(i == t['token_id']) for i in range(count)]


def write(path, value):
    # Stream serialization rather than duplicating a full-pool JSON string in RAM.
    with Path(path).open('w') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        f.write('\n')


def combine(record):
    """Only physical endpoint availability controls the a/b/c query masks."""
    r = copy.deepcopy(record)
    r.pop('six_audit', None)
    r['representations'] = {}
    for variant in regions.VARIANTS:
        r['representations'][variant] = dict(
            targets=[regions.relabel(t, variant, r['road_structure']) for t in r['targets']],
            history=[regions.relabel(t, variant, r['road_structure']) for t in r['observation']['history']])
    mask = [t['valid'] for t in r['targets']]
    if any([t['valid'] for t in view['targets']] != mask for view in r['representations'].values()):
        raise ValueError('geometry_validity_changed_by_region_rule')
    r['targets'] = r['representations']['a']['targets']
    r['sampling_mass'] = {f'{v}/{i}/{c}': q for v, view in r['representations'].items()
        for i,t in enumerate(view['targets']) for c,q in enumerate(mass(t, 9)) if q}
    return r


def support_report(records):
    counts = Counter(); groups = defaultdict(set); mass_counts = Counter()
    for r in records:
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
        physical_groups={k:len(v) for k,v in groups.items()}, legacy_action_filter_used=False,
        thresholds_refitted=False, test_used_for_fitting=False,
        independent_holdout_claim=False, adjacent_frames_are_not_independent=True)


def build(geometry, maps, output):
    geometry = Path(geometry).resolve(strict=True); output = Path(output).resolve()
    print('[geometry abc] verifying native geometric parent', flush=True)
    artifact = native.load(geometry)
    root = Path(artifact['data_root']); maps = Path(maps).resolve(strict=True)
    for p in (root, geometry.parent, maps):
        if output.is_relative_to(p) or p.is_relative_to(output): raise ValueError('output_overlaps_input')
    output.mkdir(parents=True, exist_ok=False)
    bound = code_identity(); parent_sha = sources.file_sha(geometry)
    map_proof = {str(maps/(town+'.xodr')): sha for town,sha in artifact['map_sha256'].items()}
    for p,sha in map_proof.items():
        if sources.file_sha(p) != sha: raise ValueError('study_map_reference_differs_from_native')
    result = []; rejected = Counter(); candidate_counts = Counter()
    for n,r in enumerate(artifact['records'],1):
        key = r['split']+'/'+r['event']; candidate_counts[key] += 1
        if r['road_structure'] not in ('R1','R2','R3','R4','R5'):
            rejected[key+'/unknown_current_road_category'] += 1
            continue
        if not any(t['valid'] for t in r['targets']):
            rejected[key+'/no_geometric_future_query'] += 1
            continue
        result.append(combine(r))
        if n % 10000 == 0:
            print(f'[geometry abc] anchors={n}/{len(artifact["records"])} retained={len(result)}', flush=True)
    if not result: raise ValueError('empty_geometry_cohort')
    validate_splits(result)
    if bound != code_identity() or parent_sha != sources.file_sha(geometry):
        raise ValueError('study_inputs_changed_during_build')
    native.verify(artifact)
    for p,sha in map_proof.items():
        if sources.file_sha(p) != sha: raise ValueError('map_changed_during_build')
    scoped = artifact.get('scoped_smoke_only', False)
    del artifact
    study = dict(contract={v:regions.contract(v) for v in regions.VARIANTS}, code=bound,
        data_root=str(root), parent_geometry=str(geometry), parent_sha256=parent_sha,
        map_proof=map_proof, records=result, scoped_smoke_only=scoped)
    write(output/'dataset.json', study)
    report = dict(dataset_sha256=sources.file_sha(output/'dataset.json'), candidates=dict(candidate_counts),
        rejected=dict(rejected), anchors=len(result), support=support_report(result),
        status='geometry_cohort_built_capacity_plan_required', legacy_action_filter_used=False,
        old_artifacts_modified=False)
    write(output/'receipt.json', report)
    return study, report


def load(path, *, variant=None, check_sources=True, expected_sha256=None,
         check_budget=lambda: None, with_sha256=False):
    import hashlib
    path = Path(path); check_budget()
    receipt = json.loads(path.with_name('receipt.json').read_text())
    raw = path.read_bytes(); check_budget()
    sha = hashlib.sha256(raw).hexdigest(); check_budget()
    if sha != receipt['dataset_sha256'] or expected_sha256 is not None and sha != expected_sha256:
        raise ValueError('study_dataset_changed')
    artifact = json.loads(raw); del raw; check_budget()
    if artifact['contract'] != {v:regions.contract(v) for v in regions.VARIANTS} or artifact['code'] != code_identity():
        raise ValueError('study_contract_or_code_changed')
    validate_splits(artifact['records'])
    if check_sources:
        native.load(artifact['parent_geometry'], expected_sha256=artifact['parent_sha256'], check_budget=check_budget)
        for p, value in artifact['map_proof'].items():
            if sources.file_sha(p, check_budget=check_budget) != value: raise ValueError('study_map_changed')
    if variant is not None:
        from . import admission
        # Compute all three supports before stripping the other representations.
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
