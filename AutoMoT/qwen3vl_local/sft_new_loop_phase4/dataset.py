"""从审核区间构建条件问答、因果RGB历史和严格物理路线划分。"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='qwen3vl_local.sft_new_loop_phase4'

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from .calibration import interval_facts, label_condition, reviewed_band_labels, band_summary
from .controller import Episode
from .identity import ROOT, contract, digest, file_sha, write_json, check_contract
from .prompts import prompt
from .observation import observation_contract, check_observation_contract, validate_observation
from .taxonomy import EVENTS, get_edge, template_for, event_edges, COMMON, applicable, TEMPLATE_STATES, LONGITUDINAL

EPISODE_FIELDS = {'event','instance_id','branch','state','longitudinal','direction','return_direction',
                  'return_required','target_corridor','return_corridor'}


def groups():
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import development_route_groups
    reviewed = json.loads((ROOT/'rgb_review_20260929.json').read_text())
    boundaries = json.loads((ROOT/'rgb_boundary_review_v3.json').read_text())
    reaudit = json.loads((ROOT/'reaudit_exposure_20260929.json').read_text())
    third = json.loads((ROOT/'third_audit_exposure_20260930.json').read_text())
    return (set(development_route_groups()) | set(reviewed['train_only_groups'])
            | set(boundaries['train_only_groups']) | set(reaudit['train_only_groups'])
            | set(third['train_only_groups']))


def split_for(group, exposed, seed=20260929):
    if group in exposed:
        return 'train'
    value = int(digest([seed,group])[:12],16) / 16**12
    return 'test' if value < .1 else 'val' if value < .2 else 'train'


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))


def build(annotations, data_root, output, *, rgb_mode=4, seed=20260929):
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import physical_route_group
    from qwen3vl_local.sft_new_loop_phase3.trajectory_action import _load_meta
    from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route
    if rgb_mode not in (2,4):
        raise ValueError('history mode must be 2 or 4')
    output, data_root = Path(output), Path(data_root).resolve()
    if output.exists():
        raise FileExistsError('use a fresh dataset directory')
    exposed, rows, seen = groups(), [], {}
    offsets = (-6,-4,-2,0) if rgb_mode == 4 else (-4,0)
    for ann in annotations:
        if not ann.get('reviewer') or not ann.get('evidence_id') or not ann.get('observation'):
            raise ValueError('annotation lacks reviewed evidence')
        if ann.get('label_basis') not in ('per_frame_conditions','reviewed_transition_band'):
            raise ValueError('recorded action timing is not a Phase4 permission label')
        if ann['label_basis']=='per_frame_conditions' and ann.get('slice','readiness')!='readiness':
            raise ValueError('legacy catchup requires reviewed_transition_band with per-frame conditions and conflict scope')
        ep_fields = ann['episode']
        if set(ep_fields) - EPISODE_FIELDS:
            raise ValueError('unknown episode annotation field')
        ep = Episode(**ep_fields)
        edge = get_edge(ep.event,ann['edge'],ep.branch,ep.return_required)
        run = data_root/ann['scenario']/ann['route_id']
        if not run.resolve().is_relative_to(data_root) or is_abnormal_lead_route(run,ann['scenario'])[0]:
            raise ValueError('invalid/abnormal source route')
        group = physical_route_group(ann['scenario'],ann['route_id'])
        split = split_for(group,exposed,seed)
        band = ann.get('transition_band') if ann['label_basis']=='reviewed_transition_band' else None
        if ann['label_basis']=='reviewed_transition_band' and not band:
            raise ValueError('missing frame-by-frame transition review')
        start,end = (band['review_start'],band['review_end']) if band else (ann['start'],ann['end'])
        reviewed = reviewed_band_labels(ep,edge.key,band,context_valid=ann.get('context_valid')) if band else None
        facts = interval_facts(ann.get('facts',{}),start,end,ann['evidence_id'])
        for frame in range(start,end+1):
            frames = [frame+o for o in offsets]
            if frames[0] < 4:
                continue
            paths = [run/'rgb'/f'{f:04d}.jpg' for f in frames]
            if any(not p.is_file() for p in paths):
                raise FileNotFoundError(f'missing causal history: {run}/{frame}')
            # Exact current-image identity from annotation evidence is mandatory.
            if ann['frame_sha256'].get(str(frame)) != file_sha(paths[-1]):
                raise ValueError('reviewed RGB identity mismatch')
            if reviewed is not None and frame not in reviewed:
                continue  # Audited stop: omit stale questions; never manufacture NO.
            speed = max(0.,float(_load_meta(run/'metas'/f'{frame:04d}.pkl')['speed']))
            obs = dict(frame_id=frame,history_frames=frames,speed_mps=speed)
            text = prompt(ep,edge.key,obs)
            if reviewed is not None:
                target, label_slice = reviewed[frame]
            else:
                target = label_condition(ep,edge.key,frame,facts,context_valid=ann.get('context_valid'))
            key = digest([group,ann['route_id'],frame,ep_fields,edge.key])
            row = dict(id=key,physical_group=group,scenario=ann['scenario'],route_id=ann['route_id'],
                       split=split,episode=ep_fields,edge=edge.key,observation=obs,
                       images=[str(p.relative_to(data_root)) for p in paths],
                       image_sha256=[file_sha(p) for p in paths],target=target,
                       slice=label_slice if reviewed is not None else ann.get('slice','readiness'),evidence_id=ann['evidence_id'],
                       label_basis=ann['label_basis'],transition_band=band_summary(ep,edge.key,band) if band else None,
                       frame_review=band['frames'][str(frame)] if band else None,
                       prompt_sha256=digest(text))
            if key in seen:
                if seen[key]['target'] != target or seen[key]['slice'] != row['slice']:
                    raise ValueError('conflicting interval annotations')
                continue
            seen[key] = row
            rows.append(row)
    rows=list(seen.values())
    output.mkdir(parents=True)
    review_queue=[r for r in rows if r['target'] not in ('YES','NO')]
    dump_rows(output/'review_queue.jsonl',review_queue)
    rows=[r for r in rows if r['target'] in ('YES','NO')]
    for split in ('train','val','test'):
        dump_rows(output/f'{split}.jsonl',[r for r in rows if r['split']==split])
    coverage = coverage_report(rows)
    manifest = dict(contract=contract(),rgb_mode=rgb_mode,observation_contract=observation_contract(rgb_mode),seed=seed,annotations_sha256=digest(annotations),
                    exposure_sha256=digest(sorted(exposed)),counts=dict(Counter(r['split'] for r in rows)),
                    files={f'{s}.jsonl':file_sha(output/f'{s}.jsonl') for s in ('train','val','test')},
                    coverage=coverage,ready=coverage['ready'],model_answers=['YES','NO'],
                    review_queue_sha256=file_sha(output/'review_queue.jsonl'),review_queue_count=len(review_queue),
                    calibration_policy='explicit_reviewed_frames_no_temporal_dilation')
    write_json(output/'manifest.json',manifest)
    return manifest


def coverage_report(rows):
    cells = defaultdict(lambda:dict(rows=0,groups=set(),answers=Counter()))
    for r in rows:
        c = cells[r['split']+'/'+r['episode']['event']]
        c['rows'] += 1
        c['groups'].add(r['physical_group'])
        c['answers'][r['target']] += 1
    missing = []
    for split in ('train','val','test'):
        for event in EVENTS:
            c = cells[split+'/'+event]
            if len(c['groups']) < (1 if split=='train' else 2) or not all(c['answers'][a] for a in ('YES','NO')):
                missing.append(split+'/'+event)
    edge_cells = defaultdict(Counter)
    for r in rows:
        ep = r['episode']
        template = template_for(ep['event'],ep.get('branch','default'))
        return_mode = 'return' if ep.get('return_required',True) else 'no_return'
        variant = ep.get('branch','default') + ('/'+return_mode if template=='bypass' else '')
        if r['slice'] == 'readiness':
            edge_cells[(r['split'],ep['event'],variant,r['edge'])][r['target']] += 1
    variants = [(e,'default',True) for e in EVENTS] + [('U-E2','default',False),
        ('U-E2','in_lane_pass',False),('U-E4','cyclist_follow',False),
        ('U-E4','cyclist_bypass',True),('U-E4','cyclist_bypass',False)]
    missing_edges = []
    for split in ('train','val','test'):
        for event,branch,ret in variants:
            template = template_for(event,branch)
            variant = branch + (('/return' if ret else '/no_return') if template=='bypass' else '')
            reachable_common = [e for e in COMMON if any(applicable(e,state,lon)
                for state in TEMPLATE_STATES[template] for lon in LONGITUDINAL)]
            for edge in (*event_edges(template,ret), *reachable_common):
                key = (split,event,variant,edge.key)
                if not all(edge_cells[key][a] for a in ('YES','NO')):
                    missing_edges.append('/'.join(key))
    return dict(ready=not missing and not missing_edges,missing_event_support=missing,
                label_slices=dict(Counter(r['slice'] for r in rows)),
                edge_support_scope='reviewed readiness only; catchup excluded',
                missing_transition_edges=missing_edges,
                missing_longitudinal_edges=[k for k in missing_edges if k.rsplit('/',1)[-1] in {e.key for e in COMMON}],
                cells={k:dict(rows=v['rows'],groups=len(v['groups']),answers=dict(v['answers'])) for k,v in cells.items()})


def load_dataset(path, *, require_ready=False):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    check_contract(manifest['contract'])
    if check_observation_contract(manifest.get('observation_contract')) != observation_contract(manifest['rgb_mode']):
        raise ValueError('dataset observation contract mismatch')
    if manifest['exposure_sha256'] != digest(sorted(groups())):
        raise ValueError('development exposure changed; rebuild splits')
    if file_sha(path/'review_queue.jsonl')!=manifest['review_queue_sha256']:
        raise ValueError('review queue hash mismatch')
    result = {}
    ownership = {}
    for split in ('train','val','test'):
        file = path/f'{split}.jsonl'
        if file_sha(file) != manifest['files'][file.name]:
            raise ValueError('dataset content hash mismatch')
        result[split] = read_rows(file)
        for r in result[split]:
            validate_observation(manifest['observation_contract'],r['observation'],len(r['images']))
            group = r['physical_group']
            if r['split'] != split or (group in ownership and ownership[group] != split):
                raise ValueError('physical route split leakage')
            ownership[group] = split
            if split != 'train' and group in groups():
                raise ValueError('development route in holdout')
    if require_ready and not coverage_report(sum(result.values(),[]))['ready']:
        raise ValueError('insufficient independent YES/NO support; see manifest coverage; no fallback split')
    return result,manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=ROOT.parents[1]/'lead_data')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--rgb-mode',type=int,choices=(2,4),default=4)
    a=p.parse_args()
    m=build(json.loads(a.annotations.read_text()),a.data_root,a.output_dir,rgb_mode=a.rgb_mode)
    print(json.dumps({k:m[k] for k in ('ready','counts','coverage')},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
