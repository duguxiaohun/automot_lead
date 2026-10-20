"""Versioned derived records, raw-source provenance, no RGB/index mutation."""
from collections import Counter, defaultdict
from pathlib import Path
import socket
import os
from datetime import datetime, timezone
from . import POLICY, history, targets
from qwen3vl_local.audit_joint import phase3_base_trial as trial, label_quarantine
from qwen3vl_local.audit_joint.phase3_base_dense import baseline, provenance
from qwen3vl_local.audit_joint.phase3_base_dense.data import Pool, premise
from qwen3vl_local.audit_joint.phase3_base_dense.trajectory_targets import classify
from qwen3vl_local.sft_new_loop_phase3 import trajectory_action as rule, train
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity


def identity(row):
    return baseline.digest(case_identity(train._make_item(row, seed=20260904, action_output_mode='choice')))


def sources():
    """Bind native dependencies plus this independent experiment without changing old releases."""
    release = trial.verified_sources()
    files = {p.name:trial.file_sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    return dict(base_release_sha256=release, experiment_sources=files)


def config(history_policy, lateral_weight, history_fill):
    if history_policy not in history.SEMANTICS:
        raise ValueError('Explicit history policy required')
    if history_fill not in history.FILL_MODES:raise ValueError('Unknown history fill mode')
    targets.distribution(['KEEP','KEEP'], lateral_weight)
    return dict(policy=POLICY, history_policy=history_policy, history_version=history.POLICY,
        history_fill=dict(policy=history.FILL_POLICY, mode=history_fill, max_missing_frames=2, max_missing_seconds=.5),
        lateral_weight=lateral_weight, future_pairs=targets.FUTURE_PAIRS,
        history_pairs=targets.HISTORY_PAIRS, native_rule_version=rule.ACTION_RULE_VERSION,
        semantics='two native upcoming-intent origins per half second, not executed future stages',
        sampling='native independent event/action anchor quotas; not future-target class balance',
        missing_policy='mask entire incomplete future pair; UNKNOWN incomplete history pair')


def make_record(row, trajectory, quarantines, *, history_policy, lateral_weight, history_fill=history.DEFAULT_FILL):
    points = []
    for delta in range(1, 9):
        frame = row.frame_id+delta
        value = classify(trajectory, frame, row.context_id)
        if any(f not in trajectory.metas for f in range(row.frame_id, frame+1)):
            value = dict(value, action=None, reason='anchor_to_target_meta_gap')
        if (label_quarantine.affected(quarantines, row.scenario, row.route_id, row.frame_id, frame) or
                label_quarantine.phase3_risks(dict(scenario=row.scenario, route_id=row.route_id,
                    frame_id=frame, context_id=row.context_id, action_evidence=value.get('evidence',{})), quarantines)):
            value = dict(value, action=None, reason='source_quarantine')
        points.append(dict(value, frame=frame,
            id=baseline.digest((row.scenario,row.route_id,row.split,frame,row.context_id))))
    cache = {}
    def native_at(frame):
        if frame in cache:return cache[frame]
        value = history.recognize(trajectory, frame, row.frame_id, row.context_id, history_policy)
        if (label_quarantine.affected(quarantines, row.scenario, row.route_id, frame, value['required_end_frame']) or
                label_quarantine.phase3_risks(dict(scenario=row.scenario, route_id=row.route_id,
                    frame_id=frame, context_id=row.context_id, action_evidence=value.get('evidence', {})), quarantines)):
            value = dict(value, action=None, reason='source_quarantine')
        cache[frame] = value
        return value
    observed = [native_at(row.frame_id+delta) for delta in range(-7,1)]
    filled = history.fill_short_gaps(observed,native_at,lambda f:f in trajectory.metas,history_fill)
    original_history = [targets.compress_history(observed[i:i+2],row.frame_id) for i in range(0,8,2)]
    resolved_history = [targets.compress_history(filled[i:i+2],row.frame_id) for i in range(0,8,2)]
    flags = [any(p['imputed'] for p in filled[i:i+2]) for i in range(0,8,2)]
    return dict(id=identity(row), event=row.context_id, split=row.split, anchor=row.frame_id,
        scenario=row.scenario, route_id=row.route_id, anchor_action=Pool.action(row),
        image=row.history_rgb_paths[-1], premise=premise(row, history=False),
        history=resolved_history, history_original=original_history, history_imputed=flags,
        history_changed=[a!=b for a,b in zip(original_history,resolved_history)],
        history_resolved_points=filled,
        history_semantics=history.SEMANTICS[history_policy]+(
            ' Short missing stretches may be interpolated from native labels on both sides.' if history_fill!='none' else ''),
        history_points=observed,
        windows=targets.window_targets(points,lateral_weight), future_points=points)


def summary(records):
    counts = Counter(); raw = Counter(); weighted = Counter(); reasons = Counter()
    unique = defaultdict(set)
    for r in records.values():
        prefix = r['split']+'/'+r['event']
        counts[prefix+'/anchors'] += 1
        counts[prefix+'/zero_valid_windows'] += not any(w['valid'] for w in r['windows'])
        for i, a in enumerate(r['history']):
            counts[f'{prefix}/history_{i}/{a}'] += 1
            counts[f'{prefix}/history_{i}/original_UNKNOWN'] += r['history_original'][i]=='UNKNOWN'
            counts[f'{prefix}/history_{i}/imputed'] += r['history_imputed'][i]
            counts[f'{prefix}/history_{i}/changed'] += r['history_changed'][i]
        for i, w in enumerate(r['windows']):
            counts[f'{prefix}/future_{i}/requested'] += 1
            counts[f'{prefix}/future_{i}/valid'] += w['valid']
            if w['valid']:
                counts[f'{prefix}/future_{i}/mixed'] += len(w['allowed_answers']) > 1
                unique[f'{prefix}/future_{i}'].add(tuple(w['source_ids']))
                for a, q, qp in zip(targets.ACTIONS,w['raw'],w['target']):
                    raw[prefix+'/'+a] += q; weighted[prefix+'/'+a] += qp
            else:
                reasons.update(f'{prefix}/{p["reason"]}' for p in r['future_points'][2*i:2*i+2] if p['action'] is None)
    return dict(counts=dict(counts), raw_target_mass=dict(raw), training_target_mass=dict(weighted),
                missing_pair_member_reasons=dict(reasons), unique_window_pairs={k:len(v) for k,v in unique.items()},
                distinct_windows_are_not_independent=True)


def build(rows, data_root, output, *, history_policy, lateral_weight=1.5, index=None, history_fill=history.DEFAULT_FILL):
    cfg = config(history_policy,lateral_weight,history_fill)
    output = Path(output).resolve()
    protected = [Path(data_root).resolve()]
    if index: protected.append(Path(index).resolve().parent)
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Output overlaps protected inputs')
    output.mkdir(parents=True, exist_ok=False)
    logical_root = Path(data_root).absolute()
    data_root = Path(data_root).resolve(strict=True)
    index_sha = trial.file_sha(index) if index else None
    bound = sources()
    native_pool = Pool(rows)  # Reject physical groups crossing train/val/test.
    groups = defaultdict(list)
    for r in native_pool.anchor_rows:
        if Pool.action(r) is not None:
            groups[r.scenario,r.route_id].append(r)
    proof = dict(sources={},meta_members={},directory_targets={})
    records = {}; physical = set()
    for number, ((scenario,route), anchors) in enumerate(sorted(groups.items()),1):
        if any(Path(x).name!=x or x in ('.','..') for x in (scenario,route)):
            raise ValueError('Invalid logical route')
        run = data_root/scenario/route
        resolved = str(run.resolve(strict=True))
        if resolved in physical:
            raise ValueError('Logical routes alias the same physical source')
        physical.add(resolved)
        for directory in (run,run/'metas'):
            proof['directory_targets'][str(directory.relative_to(data_root))] = str(directory.resolve(strict=True))
        files = sorted((run/'metas').glob('*.pkl'))
        proof['meta_members'][str((run/'metas').relative_to(data_root))] = [p.name for p in files]
        quarantines = label_quarantine.load(data_root, route=(scenario,route))
        extra = [data_root/s['path'] for q in quarantines for s in q['sources']]
        for p in files+extra:
            stamp = provenance.stamp(p); resolved_file = str(p.resolve(strict=True))
            sha = trial.file_sha(p)
            if stamp!=provenance.stamp(p) or resolved_file!=str(p.resolve(strict=True)):
                raise ValueError('Raw source changed during hashing')
            proof['sources'][str(p.relative_to(data_root))] = dict(sha256=sha,stat=stamp,resolved_path=resolved_file)
        trajectory = rule.load_route_trajectory(run)
        if trajectory is None:
            raise ValueError('Unreadable trajectory: '+str(run))
        for r in anchors:
            image = Path(r.history_rgb_paths[-1]).absolute()
            relative = None
            for prefix in (logical_root,data_root):
                if image.is_relative_to(prefix):
                    relative = image.relative_to(prefix)
                    break
            if relative is None or '..' in relative.parts:
                raise ValueError('Current RGB lacks a logical path under the configured data root')
            image = data_root/relative
            if str(relative) not in proof['sources']:
                stamp=provenance.stamp(image);resolved_image=str(image.resolve(strict=True))
                sha=trial.file_sha(image)
                if stamp!=provenance.stamp(image) or resolved_image!=str(image.resolve(strict=True)):
                    raise ValueError('RGB changed during hashing')
                proof['sources'][str(relative)] = dict(sha256=sha,stat=stamp,resolved_path=resolved_image)
            rec = make_record(r,trajectory,quarantines,history_policy=history_policy,lateral_weight=lateral_weight,history_fill=history_fill)
            if rec['id'] in records:
                raise ValueError('Duplicate anchor identity')
            records[rec['id']] = rec
        if number%10==0:
            print(f'[single-frame] routes={number}/{len(groups)} anchors={len(records)}',flush=True)
    provenance.verify_sources(data_root,proof)
    if sources()!=bound:
        raise ValueError('Experiment sources changed while building')
    if index and trial.file_sha(index)!=index_sha:
        raise ValueError('Frozen index changed during generation')
    artifact = dict(config=cfg, sources=bound, data_root=str(data_root),
        index_sha256=index_sha, records=records, raw_proof=proof)
    trial.write(output/'dataset.json',artifact)
    report = dict(status='built_not_training_admission',config=cfg,sources=bound,
        artifact_sha256=trial.file_sha(output/'dataset.json'),summary=summary(records),
        routes=len(groups),anchors=len(records),raw_source_files=len(proof['sources']),
        hostname=socket.gethostname(),server_label=os.environ.get('AUTOMOT_SERVER','unspecified'),
        created_utc=datetime.now(timezone.utc).isoformat(),native_index_rewritten=False,RGB_copied=False)
    trial.write(output/'dataset_summary.json',report)
    return artifact, report


def load(path, *, index=None):
    path = Path(path)
    report = trial.read(path.with_name('dataset_summary.json'))
    if trial.file_sha(path)!=report['artifact_sha256']:
        raise ValueError('Dataset artifact changed')
    artifact = trial.read(path)
    if artifact['config']!=report['config']:
        raise ValueError('Dataset configuration differs from receipt')
    if artifact['sources']!=sources() or report['sources']!=artifact['sources']:
        raise ValueError('Dataset source binding changed; do not silently reuse')
    if index and artifact['index_sha256']!=trial.file_sha(index):
        raise ValueError('Frozen index changed')
    provenance.verify_sources(Path(artifact['data_root']),artifact['raw_proof'])
    return artifact
