"""Point-aligned intent targets from raw meta, never context-index interpolation.

Each forecast retains its anchor event action domain. Native frozen anchor
labels and native rules are unchanged. The artifact is local provenance, not external attestation.
"""
import ast
from collections import Counter, defaultdict
from pathlib import Path
import os
import socket
import subprocess
from datetime import datetime, timezone
from . import baseline, provenance
from .data import Pool
from qwen3vl_local.audit_joint import phase3_base_trial as trial, label_quarantine
from qwen3vl_local.sft_new_loop_phase3 import trajectory_action as rule
from qwen3vl_local.sft_new_loop_phase3.choice_semantics import choice_annotation
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID, DOMAIN_MANEUVER

POLICY = 'waypoint_aligned_anchor_event_intent_v2'
CONFIG = trial.APP/'qwen3vl_local/leadmot/config.py'


def timeline():
    tree=ast.parse(CONFIG.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LeadMoTPlanningDecoderConfig')
    values={n.target.id:ast.literal_eval(n.value) for n in cls.body
            if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)
            and n.target.id in ('num_waypoint_queries','waypoint_dt')}
    count=values['num_waypoint_queries'];dt=values['waypoint_dt']
    stride=dt/rule.FRAME_DT_SECONDS
    if type(count) is not int or not 1<=count<=16 or stride<1 or stride!=int(stride):
        raise ValueError('Waypoint timeline must have 1..16 points and integral raw-frame spacing')
    return dict(offset_frames=[i*int(stride) for i in range(count+1)],
        waypoint_count=count,waypoint_dt=dt,config_sha256=trial.file_sha(CONFIG),
        step_source='LeadMoTPlanningDecoderConfig_defaults',policy=POLICY,
        event_domain_policy='all queries retain the current anchor event action domain',
        semantics='current native query plus point-aligned future native-window intentions in the anchor event domain; not segment execution states')


def physical_key(row,frame):
    return (row.scenario,row.route_id,row.split,frame)


def target_key(row,frame):
    return (*physical_key(row,frame),row.context_id)


def binding(job):
    return dict(policy=POLICY,index_sha256=job['index_sha256'],timeline=timeline(),
        data_root=str(Path(job['data_root']).resolve()),release_sha256=job['release_sha256'])


def project(signals,context_id):
    """Apply the native anchor question domain at a new prediction origin."""
    context=CONTEXT_BY_ID[context_id]
    info=dict(anchor_context_id=context_id,question_domain=context.question_domain,
              allowed_actions=[*context.action_keys,'KEEP'])
    if signals is None:return dict(info,action=None,reason='missing_or_unreadable_target_meta')
    evidence=rule.action_evidence(signals)
    labels=rule.label_actions(signals)
    if labels is None:return dict(info,action=None,reason='incomplete_longitudinal_evidence',evidence=evidence)
    # Preserve the original domain admission, including full-maneuver STOP.
    # Longitudinal events never require proof of the absence of a lane crossing.
    if context.question_domain==DOMAIN_MANEUVER and not signals['lateral_observation_complete']:
        reason=signals.get('lateral_window_issue') or 'lateral_rgb_uncertainty'
        return dict(info,action=None,reason='incomplete_lateral_evidence:'+str(reason),evidence=evidence)
    annotation=choice_annotation(labels,context_id,evidence)
    return dict(info,action=annotation['primary_action'],reason='native_anchor_domain_action',
                keep_scope=annotation['keep_scope'],evidence=evidence)


def classify(trajectory,frame,context_id):
    return project(trajectory.signals(frame) if trajectory is not None else None,context_id)


def build(job,rows):
    root=Path(job['output']);root.mkdir(parents=True,exist_ok=True)
    dest=root/'trajectory_targets.json'
    if dest.exists() or (root/'trajectory_targets_summary.json').exists():
        raise ValueError('Trajectory target output exists; use an explicit verified --trajectory-targets artifact')
    data=Path(job['data_root']).resolve();groups=defaultdict(list);splits={}
    for row in rows:
        route=(row.scenario,row.route_id)
        if any(Path(x).name!=x or x in ('.','..') for x in route):raise ValueError('Invalid logical route')
        if route in splits and splits[route]!=row.split:raise ValueError('Trajectory route crosses splits')
        splits[route]=row.split
        if Pool.action(row) is not None:groups[route].append(row)
    sources={};members={};directories={};physical=set();targets={};bridges={};counts=Counter()
    offsets=timeline()['offset_frames'][1:]
    def bind(path):
        rel=str(path.relative_to(data));resolved=str(path.resolve(strict=True));before=provenance.stamp(path)
        sha=trial.file_sha(path)
        if before!=provenance.stamp(path) or str(path.resolve(strict=True))!=resolved:
            raise ValueError('Raw trajectory changed during hashing: '+rel)
        sources[rel]=dict(sha256=sha,stat=before,resolved_path=resolved)
    for number,((scenario,route),anchors) in enumerate(sorted(groups.items()),1):
        run=data/scenario/route;resolved=str(run.resolve(strict=True))
        if resolved in physical:raise ValueError('Different logical trajectory routes alias one physical route')
        physical.add(resolved)
        for directory in (run,run/'metas'):
            directories[str(directory.relative_to(data))]=str(directory.resolve(strict=True))
        paths=sorted((run/'metas').glob('*.pkl'))
        members[str((run/'metas').relative_to(data))]=[p.name for p in paths]
        for path in paths:bind(path)
        quarantines=label_quarantine.load(data,route=(scenario,route))
        for item in quarantines:
            for source in item['sources']:bind(data/source['path'])
        trajectory=rule.load_route_trajectory(run)
        split=splits[scenario,route]
        requested={(r.frame_id+d,r.context_id) for r in anchors for d in offsets}
        # A physical frame's native signals are computed once even when several
        # anchor events request different question domains at the same time.
        signals_by_frame={f:trajectory.signals(f) if trajectory is not None else None
                          for f in sorted({f for f,_ in requested})}
        for frame,context_id in sorted(requested):
            key=(scenario,route,split,frame,context_id)
            value=project(signals_by_frame[frame],context_id)
            if label_quarantine.phase3_risks(dict(scenario=scenario,route_id=route,
                    frame_id=frame,context_id=context_id,action_evidence=value.get('evidence',{})),quarantines):
                value=dict(value,action=None,reason='source_quarantine')
            identity=baseline.digest(key);targets[identity]=dict(key=list(key),**value)
            counts[f'{split}/{context_id}/{value["reason"]}']+=1
        for row in anchors:
            for delta in offsets:
                end=row.frame_id+delta
                reason=None
                if trajectory is None or any(f not in trajectory.metas for f in range(row.frame_id,end+1)):
                    reason='anchor_to_target_meta_gap'
                if label_quarantine.affected(quarantines,scenario,route,row.frame_id-3,end):
                    reason='anchor_to_target_quarantine'
                if reason:bridges[baseline.digest((*physical_key(row,row.frame_id),end))]=reason
        if number%100==0:print(f'[trajectory-targets] routes={number}/{len(groups)}',flush=True)
    proof=dict(binding=binding(job),sources=sources,meta_members=members,directory_targets=directories,
        targets=targets,invalid_bridges=bridges)
    provenance.verify_sources(data,proof)
    if trial.file_sha(job['index'])!=job['index_sha256']:raise ValueError('Frozen index changed during target generation')
    trial.write(dest,proof)
    try:
        git=subprocess.run(['git','rev-parse','HEAD'],cwd=trial.APP,capture_output=True,text=True,timeout=10)
        revision=git.stdout.strip() if git.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):revision=None
    summary=dict(status='native_raw_meta_targets_built',binding=proof['binding'],
        artifact_sha256=trial.file_sha(dest),routes=len(groups),domain_targets=len(targets),
        physical_targets=len({tuple(t['key'][:4]) for t in targets.values()}),
        reasons_by_split=dict(counts),invalid_bridges=len(bridges),source_files=len(sources),
        git_commit=revision,created_utc=datetime.now(timezone.utc).isoformat(),
        native_rule_version=rule.ACTION_RULE_VERSION,
        git_scope='optional checkout metadata; verified release SHA is authoritative',
        hostname=socket.gethostname(),server_label=os.environ.get('AUTOMOT_SERVER','unspecified'),
        original_labels_modified=False,RGB_copied=False,missing_imputed=False,
        scope='native anchor-event-domain intentions at future origins; not segment actions or waypoint-conditioning integration')
    trial.write(root/'trajectory_targets_summary.json',summary)
    return proof


def load(job,rows,*,create=False):
    external=job.get('trajectory_targets')
    path=Path(external) if external else Path(job['output'])/'trajectory_targets.json'
    if create and not external:
        return build(job,rows)
    summary=trial.read(path.with_name('trajectory_targets_summary.json'))
    if summary['artifact_sha256']!=trial.file_sha(path):raise ValueError('Trajectory artifact changed')
    proof=trial.read(path)
    if proof['binding']!=binding(job) or summary['binding']!=proof['binding']:
        raise ValueError('Trajectory target source/index/protocol binding changed')
    provenance.verify_sources(Path(job['data_root']).resolve(),proof)
    if external and create:
        trial.write(Path(job['output'])/'trajectory_targets_summary.json',dict(summary,reused_from=str(path)))
    return proof


class TrajectoryPool(Pool):
    def __init__(self,rows,proof):
        super().__init__(rows,offsets=timeline()['offset_frames']);self.proof=proof

    def future(self,row,offsets=None):
        offsets=self.offsets if offsets is None else offsets
        rec=super().future(row,offsets)
        rec['forecast_protocol']='trajectory';rec['target_reasons']=['frozen_native_anchor']
        rec['forecast_event_domain']=row.context_id
        allowed=(*CONTEXT_BY_ID[row.context_id].action_keys,'KEEP')
        for i,delta in enumerate(offsets[1:],1):
            identity=baseline.digest(target_key(row,row.frame_id+delta))
            target=self.proof['targets'][identity]
            if (target['key']!=list(target_key(row,row.frame_id+delta))
                    or target['anchor_context_id']!=row.context_id
                    or target['action'] is not None and target['action'] not in allowed):
                raise ValueError('Future target violates the fixed anchor event domain')
            reason=self.proof['invalid_bridges'].get(baseline.digest((*physical_key(row,row.frame_id),row.frame_id+delta)))
            rec['actions'][i]=None if reason else target['action']
            rec['source_ids'][i]=identity
            rec['target_reasons'].append(reason or target['reason'])
        return rec
