"""Build from unfiltered event annotations, before any legacy action eligibility.

Only the new output directory is written. Source bytes, physical split identity,
maps and rule code are bound to the artifact; old indexes are never input labels.
"""
from collections import Counter, defaultdict
import json
import lzma
import pickle
from pathlib import Path

from . import codec, sources, splits


def digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def code_identity():
    result = {p.name:sources.file_sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    # Bind reused annotation/split/quarantine policy without importing old labels.
    from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapping_contract_hash
    result['event_split_source_contract'] = mapping_contract_hash()
    from qwen3vl_local.sft_new_loop_phase3 import split_coverage
    result['split_coverage.py'] = sources.file_sha(split_coverage.__file__)
    app = Path(__file__).resolve().parents[3]
    for name in ('leadmot/config.py','leadmot/decoder.py'):
        result[name] = sources.file_sha(app/'qwen3vl_local'/name)
    return result


def annotation_routes(collection, root, *, scenarios=None, development=None):
    """Reuse event corrections and physical split policy, not action rules/quota."""
    from qwen3vl_local.sft_new_loop_phase3 import build_dataset as old
    from qwen3vl_local.sft_new_loop_phase3.annotation_repair import repair_annotation
    development = set(splits.development_contract()['groups'] if development is None else development)
    for path in sorted(Path(collection).glob('*_result.json')):
        scenario = path.stem.removesuffix('_result')
        if scenario == 'noScenarios' or (scenarios and scenario not in scenarios):
            continue
        for route in old._iter_routes_stream(path):
            rid = route.get('route_id')
            if not rid or route.get('status') == 'data_missing_skip':
                continue
            if any(Path(x).name != x or x in ('.', '..') for x in (scenario, rid)):
                raise ValueError('invalid_logical_route')
            run = Path(root)/scenario/rid
            if not run.is_dir() or old.is_abnormal_lead_route(run, scenario)[0]:
                continue
            anchors = []
            for ann in route.get('annotations', []):
                frame = int(ann['frame_id'])
                rs, primary, codes, _ = repair_annotation(scenario, rid, frame, ann,
                    old._rs_label(ann), str(ann.get('primary_event') or 'UNKNOWN'), old._event_codes(ann))
                contexts, _ = old.mapped_contexts(scenario, rid, frame, rs, primary, codes)
                # Deliberately no history-RGB minimum, label_actions, action signature,
                # old lateral completeness, event-dependent six-action mask or quota.
                for event in contexts:
                    anchors.append(dict(frame=frame, event=event, road_structure=rs))
            if anchors:
                yield dict(scenario=scenario, route_id=rid, town=rid.split('_')[0],
                    split=('train' if old.physical_route_group(scenario, rid) in development
                           else old._split(scenario, rid, 20260920, .10, .05)), anchors=anchors)


def record(route, anchor, metas, lanes, *, discontinuities=()):
    frame = anchor['frame']
    obs, state = sources.observation(metas, frame, route['town'], lanes, discontinuities)
    targets = sources.future(metas[frame], state, anchor=frame, discontinuities=discontinuities)
    key = [route['scenario'], route['route_id'], frame, anchor['event']]
    return dict(id=digest(key), scenario=route['scenario'], route_id=route['route_id'],
        split=route['split'], frame=frame, event=anchor['event'],
        road_structure=anchor['road_structure'], observation=obs, targets=targets,
        image=f"{route['scenario']}/{route['route_id']}/rgb/{frame:04d}.jpg")


def validate_splits(records, development=()):
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import physical_route_group
    development = set(development) | set(splits.development_contract()['groups'])
    groups = {}; ids = set()
    for r in records:
        group = physical_route_group(r['scenario'], r['route_id'])
        if r['split'] not in ('train', 'val', 'test'):
            raise ValueError('unknown_split')
        if group in development and r['split'] != 'train':
            raise ValueError('development_group_in_holdout')
        if groups.setdefault(group, r['split']) != r['split']:
            raise ValueError('physical_route_split_leakage')
        if r['id'] in ids:
            raise ValueError('duplicate_anchor')
        ids.add(r['id'])
        if r.get('physical_group', group) != group:
            raise ValueError('physical_group_mismatch')


def build(routes, root, lanes, output, *, input_paths=(), development_manifests=(), require_coverage=True,
          coverage_search_states=100000):
    from PIL import Image
    from qwen3vl_local.audit_joint import label_quarantine
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import physical_route_group
    development = splits.development_contract(development_manifests)
    input_paths = [*input_paths, *development['sources']]
    root = Path(root).resolve(strict=True); output = Path(output).resolve()
    protected = [root, *(Path(p).resolve() for p in input_paths)]
    if hasattr(lanes, 'root'): protected.append(lanes.root)
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('output_overlaps_input')
    output.mkdir(parents=True, exist_ok=False)
    routes = list(routes)
    if require_coverage:
        candidates = [dict(scenario=r['scenario'], route_id=r['route_id'], split=r['split'],
            event=a['event'], targets=[dict(valid=True)]) for r in routes for a in r['anchors']]
        preflight = splits.complete(candidates, development['groups'], max_search_states=coverage_search_states)
        write(output/'candidate_split_coverage.json', preflight)
        splits.require_coverage(preflight, 'candidate')
        del candidates
    bound = code_identity(); proof = {}; members = {}; targets = {}; seen = set()
    def bind(p):
        p = Path(p).absolute(); stat = p.stat()
        value = dict(sha256=sources.file_sha(p), resolved=str(p.resolve(strict=True)))
        stamp=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if stamp(stat) != stamp(p.stat()): raise RuntimeError('source_changed_during_hash')
        if str(p.resolve(strict=True))!=value['resolved']:raise RuntimeError('source_mount_changed_during_hash')
        proof[str(p)] = value
    for p in input_paths: bind(p)
    bind(label_quarantine.LEDGER)
    records = []; rejected = Counter()
    for route in routes:
        scenario, rid = route['scenario'], route['route_id']
        if any(Path(x).name != x or x in ('.', '..') for x in (scenario, rid)):
            raise ValueError('invalid_logical_route')
        run = root/scenario/rid; resolved = str(run.resolve(strict=True))
        if resolved in seen: raise ValueError('duplicate_or_aliased_route')
        seen.add(resolved); targets[str(run)] = resolved
        files = sorted((run/'metas').glob('*.pkl'))
        members[str(run/'metas')] = [p.name for p in files]
        metas = {}
        for p in files:
            bind(p)
            try: metas[int(p.stem)] = sources.read_meta(p)
            except (EOFError, ValueError, OSError, lzma.LZMAError, pickle.UnpicklingError):
                rejected['unreadable_meta'] += 1
        risks = label_quarantine.load(root, route=(scenario, rid))
        for risk in risks:
            for source in risk['sources']: bind(root/source['path'])
        discontinuities = [r['first_affected_frame'] for r in risks]
        for anchor in route['anchors']:
            try:
                r = record(route, anchor, metas, lanes, discontinuities=discontinuities)
                image = root/r['image']; bind(image)
                with Image.open(image) as im: im.load()
                r['physical_group'] = physical_route_group(scenario, rid)
            except (KeyError, ValueError, TypeError, OSError) as e:
                rejected[str(e)] += 1
                continue
            records.append(r)
    if not records: raise ValueError('no_geometry_records')
    coverage = splits.complete(records, development['groups'], max_search_states=coverage_search_states) if require_coverage else None
    if not require_coverage:
        for r in records:
            if r['physical_group'] in development['groups']: r['split'] = 'train'
    write(output/'split_coverage.json', coverage)
    if coverage: splits.require_coverage(coverage, 'geometry_after_filtering')
    validate_splits(records, development['groups'])
    if hasattr(lanes, 'root'):
        for town, sha in lanes.hashes.items():
            p = lanes.root/(town+'.xodr'); bind(p)
            if proof[str(p)]['sha256'] != sha: raise ValueError('map_changed')
    artifact = dict(contract=codec.contract(), code=bound, data_root=str(root),
        map_sha256=dict(lanes.hashes), records=records, proof=proof,
        meta_members=members, directory_targets=targets, development=development,
        split_coverage=coverage, scoped_smoke_only=not require_coverage)
    verify(artifact)
    write(output/'dataset.json', artifact)
    counts = Counter(); support = defaultdict(set)
    for r in records:
        for i, t in enumerate(r['targets']):
            key = f"{r['split']}/{r['event']}/{codec.HORIZONS[i]}/{t['name']}"
            counts[key] += 1; support[key].add(r['physical_group'])
    report = dict(status='built_not_gpu_validated', dataset_sha256=sources.file_sha(output/'dataset.json'),
        anchors=len(records), routes=len(seen), rejected=dict(rejected), counts=dict(counts),
        physical_groups={k:len(v) for k,v in support.items()},
        legacy_action_filter_used=False, training_executed=False)
    write(output/'receipt.json', report)
    return artifact, report


def verify(artifact, *, check_sources=True, check_budget=lambda: None):
    if artifact['contract'] != codec.contract() or artifact['code'] != code_identity():
        raise ValueError('geometry_contract_or_code_changed')
    if artifact['development'] != splits.development_contract(artifact['development']['manifests']):
        raise ValueError('development_contract_changed')
    validate_splits(artifact['records'], artifact['development']['groups'])
    if not check_sources: return
    for name, target in artifact['directory_targets'].items():
        check_budget()
        if str(Path(name).resolve(strict=True)) != target: raise ValueError('route_mount_changed')
    for name, expected in artifact['meta_members'].items():
        check_budget()
        if sorted(p.name for p in Path(name).glob('*.pkl')) != expected: raise ValueError('meta_members_changed')
    for name, expected in artifact['proof'].items():
        check_budget()
        p = Path(name)
        if str(p.resolve(strict=True)) != expected['resolved'] or sources.file_sha(p, check_budget=check_budget) != expected['sha256']:
            raise ValueError('source_content_changed: '+name)


def load(path, *, check_sources=True, expected_sha256=None, check_budget=lambda: None, with_sha256=False):
    import hashlib
    path = Path(path)
    receipt = json.loads(path.with_name('receipt.json').read_text())
    check_budget()
    raw = path.read_bytes(); check_budget()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != receipt['dataset_sha256'] or (expected_sha256 is not None and actual != expected_sha256):
        raise ValueError('dataset_changed')
    artifact = json.loads(raw); del raw; check_budget()
    verify(artifact, check_sources=check_sources, check_budget=check_budget)
    return (artifact, actual) if with_sha256 else artifact
