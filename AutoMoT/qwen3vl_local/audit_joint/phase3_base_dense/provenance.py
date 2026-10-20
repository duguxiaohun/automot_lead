"""Recompute supplemental targets from their actual route/frame meta.

Local receipts detect changes; they are not third-party attestations or RGB
semantic certification. Existing frozen base rows are not relabeled.
"""
from collections import defaultdict, Counter
from pathlib import Path
import json
from . import baseline
from qwen3vl_local.audit_joint import phase3_base_trial as trial
from qwen3vl_local.audit_joint import label_quarantine
from qwen3vl_local.sft_new_loop_phase3 import trajectory_action as rule, build_dataset as builder


def identity(row):
    return tuple(row[k] for k in ('scenario','route_id','split','context_id','prompt_road_structure','frame_id'))


def stamp(path):
    s=path.stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def binding(job):
    return dict(index_sha256=job['index_sha256'],aux_binding=job['aux_binding'],
        data_root=str(Path(job['data_root']).resolve()),release_sha256=job['release_sha256'],
        policy='native_frame_meta_recompute_logical_mounts_v2')


def verify_cached(job):
    root=Path(job['output']);summary=trial.read(root/'dense_aux_summary.json')
    path=root/'dense_aux_provenance.json'
    if summary['proof_sha256']!=trial.file_sha(path):raise ValueError('Auxiliary derivation receipt changed')
    proof=trial.read(path)
    if proof['binding']!=binding(job):raise ValueError('Auxiliary derivation input binding changed')
    verify_sources(Path(job['data_root']).resolve(),proof)
    return summary



def logical_path(root,relative):
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Invalid logical auxiliary path: '+str(relative))
    return root/relative


def verify_sources(root,proof):
    # A mounted scenario/route may resolve outside root. Bind both its logical
    # identity and physical target, rather than declaring external mounts invalid.
    for relative,target in proof['directory_targets'].items():
        p=logical_path(root,relative)
        if str(p.resolve(strict=True))!=target or not p.is_dir():
            raise ValueError('Auxiliary directory target changed: '+relative)
    for relative,item in proof['sources'].items():
        p=logical_path(root,relative)
        if str(p.resolve(strict=True))!=item['resolved_path'] or stamp(p)!=item['stat']:
            raise ValueError('Raw auxiliary source changed since recomputation: '+relative)
    for relative,names in proof['meta_members'].items():
        if sorted(p.name for p in logical_path(root,relative).glob('*.pkl'))!=names:
            raise ValueError('Auxiliary route meta membership changed: '+relative)

def recompute(job):
    original={identity(r) for r in map(json.loads,Path(job['index']).read_text().splitlines())
              if not r['invalid_action_context']}
    groups=defaultdict(list)
    with Path(job['aux_index']).open() as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line)
            if not r['invalid_action_context'] and identity(r) not in original:
                groups[r['scenario'],r['route_id']].append(r)
    data_root=Path(job['data_root']).resolve();sources={};meta_members={};counts=Counter();rows=[]
    directory_targets={};physical_routes={}
    def bind(path):
        # Keep the path through the logical route mount. Resolving first loses
        # the relationship that must later be checked for mount retargeting.
        p=Path(path)
        if not p.is_absolute():p=data_root/p
        if '..' in p.parts or not p.is_relative_to(data_root):
            raise ValueError('Auxiliary source outside logical data root: '+str(p))
        relative=str(p.relative_to(data_root))
        resolved=str(p.resolve(strict=True))
        if relative not in sources:
            before=stamp(p);sha=trial.file_sha(p)
            if stamp(p)!=before or str(p.resolve(strict=True))!=resolved:
                raise ValueError('Auxiliary raw source changed during hashing')
            sources[relative]=dict(sha256=sha,stat=before,resolved_path=resolved)
        return relative
    for (scenario,route),items in sorted(groups.items()):
        if any(Path(x).name!=x or x in ('.','..') for x in (scenario,route)):
            raise ValueError('Invalid raw route identity')
        run=data_root/scenario/route
        resolved=str(run.resolve(strict=True))
        if not run.is_dir():raise ValueError('Raw route is not a directory: '+str(run))
        if resolved in physical_routes:
            raise ValueError('Different logical auxiliary routes alias one physical route')
        physical_routes[resolved]=(scenario,route)
        directory_targets[str(run.relative_to(data_root))]=resolved
        directory_targets[str((run/'metas').relative_to(data_root))]=str((run/'metas').resolve(strict=True))
        paths=sorted((run/'metas').glob('*.pkl'))
        if not paths:raise ValueError(f'Missing raw meta for auxiliary route {scenario}/{route}')
        meta_members[str((run/'metas').relative_to(data_root))]=[p.name for p in paths]
        for p in paths:bind(p)
        quarantines=label_quarantine.load(data_root,route=(scenario,route))
        for record in quarantines:
            for source in record['sources']:bind(data_root/source['path'])
        trajectory=rule.load_route_trajectory(run)
        if trajectory is None:raise ValueError('Cannot load raw auxiliary trajectory')
        for row in items:
            frame=row['frame_id'];signals=trajectory.signals(frame)
            if signals is None or not signals['goal_available']:
                raise ValueError(f'Missing source frame/navigation for auxiliary target {identity(row)}')
            labels=rule.label_actions(signals)
            if labels is None or (builder.CONTEXT_BY_ID[row['context_id']].question_domain=='FULL_MANEUVER'
                                  and not signals['lateral_observation_complete']):
                raise ValueError('Raw auxiliary target lacks complete native trajectory supervision')
            history=builder._history(run,frame)
            if history is None:raise ValueError('Raw auxiliary target lacks exact RGB history')
            expected=dict(action_evidence=rule.action_evidence(signals),
                answers=builder._answers_for(row['context_id'],labels,invalid=False),
                current_speed_mps=rule.action_evidence(signals)['speed_mps'],
                goal_ego_xy=[round(signals['goal_x'],3),round(signals['goal_y'],3)],
                action_signature=builder.action_signature(labels,context_id=row['context_id']))
            for field,value in expected.items():
                if row.get(field)!=value:
                    raise ValueError(f'Auxiliary raw-frame recomputation mismatch: {identity(row)} field={field}')
            if label_quarantine.phase3_risks(row,quarantines):
                raise ValueError('Auxiliary target intersects a native source quarantine')
            resolve=lambda p:(Path(p) if Path(p).is_absolute() else data_root/p).resolve(strict=True)
            if ([resolve(p) for p in row['history_rgb_paths']]!=[Path(p).resolve() for p in history]
                    or resolve(row['latest_rgb_path'])!=Path(history[-1]).resolve()):
                raise ValueError('Auxiliary RGB history does not belong to target frame')
            for p in history:bind(p)
            rows.append((identity(row),baseline.digest(expected)));counts[row['split']]+=1
        # Do not retain all routes' unpickled metas in memory.
        del trajectory
    verify_sources(data_root,dict(sources=sources,meta_members=meta_members,directory_targets=directory_targets))
    if (trial.file_sha(job['index'])!=job['index_sha256'] or
            trial.file_sha(job['aux_index'])!=job['aux_binding']['index_sha256'] or
            trial.file_sha(Path(job['aux_index']).with_name('manifest.json'))!=job['aux_binding']['manifest_sha256']):
        raise ValueError('Index/manifest changed during auxiliary recomputation')
    proof=dict(binding=binding(job),sources=sources,meta_members=meta_members,directory_targets=directory_targets,
               recomputed_rows_sha256=baseline.digest(sorted(rows)),rows_by_split=dict(counts))
    root=Path(job['output']);trial.write(root/'dense_aux_provenance.json',proof)
    summary=dict(status='native_meta_recomputed',rows_by_split=dict(counts),routes=len(groups),
        source_files=len(sources),
        external_route_mounts=sum(not Path(p).is_relative_to(data_root) for p in physical_routes),
        source_path_policy='logical_route_and_resolved_target_bound',proof_sha256=trial.file_sha(root/'dense_aux_provenance.json'),
        recomputed_rows_sha256=proof['recomputed_rows_sha256'],
        scope='new auxiliary targets only: raw meta/action evidence/speed/goal/RGB frame paths; not visual-label certification')
    trial.write(root/'dense_aux_summary.json',summary)
    return summary
