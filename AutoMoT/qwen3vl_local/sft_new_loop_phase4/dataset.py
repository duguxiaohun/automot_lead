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
from .route_calibration import interval_facts, label_condition, reviewed_band_labels, band_summary
from .controller import Episode
from .route_context import episode_edge, route_bound, validate_source_context, RECOVER_FOLLOW
from . import risk_review, visible_scope
from .maneuver_safety import GUARDED
from .admission import training_report
from .data_paths import data_path
from .identity import ROOT, contract, digest, file_sha, write_json, check_contract
from .route_prompts import prompt
from .input_identity import rgb_content_sha, model_input_key, validate_model_inputs
from .observation import observation_contract, check_observation_contract, validate_observation
from .taxonomy import EVENTS, get_edge, template_for, event_edges, COMMON, applicable, TEMPLATE_STATES, LONGITUDINAL

EPISODE_FIELDS = {'event','instance_id','branch','state','longitudinal','direction','return_direction',
                  'return_required','target_corridor','return_corridor','route_segments','segment_index','route_context'}


def groups():
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import development_route_groups
    reviewed = json.loads((ROOT/'rgb_review_20260929.json').read_text())
    boundaries = json.loads((ROOT/'rgb_boundary_review_v3.json').read_text())
    increment = json.loads((ROOT/'rgb_boundary_review_v4.json').read_text())
    reaudit = json.loads((ROOT/'reaudit_exposure_20260929.json').read_text())
    third = json.loads((ROOT/'third_audit_exposure_20260930.json').read_text())
    eleventh = json.loads((ROOT/'eleventh_audit_exposure_20260930.json').read_text())
    tenth = json.loads((ROOT/'tenth_audit_exposure_20260930.json').read_text())
    ninth = json.loads((ROOT/'ninth_audit_exposure_20260930.json').read_text())
    eighth = json.loads((ROOT/'eighth_audit_exposure_20260930.json').read_text())
    fifteenth = json.loads((ROOT/'fifteenth_audit_exposure_20260930.json').read_text())
    sixteenth = json.loads((ROOT/'sixteenth_audit_exposure_20260930.json').read_text())
    latest_rounds = json.loads((ROOT/'eighteenth_nineteenth_audit_exposure_20261001.json').read_text())
    seventeenth = json.loads((ROOT/'seventeenth_audit_exposure_20261001.json').read_text())
    fourteenth = json.loads((ROOT/'fourteenth_audit_exposure_20260930.json').read_text())
    latest = json.loads((ROOT/'twelfth_thirteenth_audit_exposure_20260930.json').read_text())
    twentieth = json.loads((ROOT/'twentieth_audit_exposure_20261001.json').read_text())
    twenty_second = json.loads((ROOT/'twenty_second_audit_exposure_20261002.json').read_text())
    twenty_third = json.loads((ROOT/'twenty_third_audit_exposure_20261002.json').read_text())
    twenty_first = json.loads((ROOT/'twenty_first_audit_exposure_20261001.json').read_text())
    return (set(development_route_groups()) | set(reviewed['train_only_groups'])
            | set(boundaries['train_only_groups']) | set(reaudit['train_only_groups'])
            | set(third['train_only_groups']) | set(increment['train_only_groups'])
            | set(eighth['train_only_groups']) | set(ninth['train_only_groups']) | set(tenth['train_only_groups']) | set(eleventh['train_only_groups'])
            | set(latest['train_only_groups']) | set(fourteenth['train_only_groups']) | set(fifteenth['train_only_groups'])
            | set(sixteenth['train_only_groups']) | set(seventeenth['train_only_groups']) | set(latest_rounds['train_only_groups'])
            | set(twentieth['train_only_groups']) | set(twenty_first['train_only_groups']) | set(twenty_second['train_only_groups']) | set(twenty_third['train_only_groups']) | set(json.loads((ROOT/'thirtieth_audit_exposure_20261005.json').read_text())['train_only_groups']) | set(json.loads((ROOT/'thirty_first_audit_exposure_20261005.json').read_text())['train_only_groups']) | set(json.loads((ROOT/'thirty_second_audit_exposure_20261005.json').read_text())['train_only_groups']) | set(json.loads((ROOT/'thirty_third_audit_exposure_20261006.json').read_text())['train_only_groups']) | set(json.loads((ROOT/'thirty_fourth_audit_exposure_20261006.json').read_text())['train_only_groups']) | set(json.loads((ROOT/'thirty_sixth_audit_exposure_20261006.json').read_text())['train_only_groups']) | set(json.loads((ROOT/'thirty_fifth_audit_exposure_20261006.json').read_text())['train_only_groups']))


HOLDOUT_PLAN = 'formal_holdout_plan_20260930.json'
PRODUCER_CHECK_PLAN = 'producer_manual_check_plan_v20_20261008.json'


def producer_check_reservations():
    plan=json.loads((ROOT/PRODUCER_CHECK_PLAN).read_text())
    if plan.get('policy')!='prospective_producer_manual_check_v1' or not plan.get('frozen_sources'):
        raise ValueError('invalid producer manual-check protocol')
    for name,sha in plan['frozen_sources'].items():
        if file_sha(ROOT/name)!=sha:raise ValueError('producer manual-check sources changed; freeze a new protocol')
    extension=plan.get('instance_only_extension')
    if extension:
        name=extension['path']
        if Path(name).name!=name or file_sha(ROOT/name)!=extension['sha256']:raise ValueError('independent pool extension changed')
    owners={}
    for row in plan['reservations']:
        group,split=row['physical_group'],row['split']
        if group in owners or split not in ('val','test'):raise ValueError('invalid producer-check reservation')
        owners[group]=split
    return owners


def holdout_reservations():
    plan = json.loads((ROOT/HOLDOUT_PLAN).read_text())
    frozen = plan.get('frozen_rubric_sha256', {})
    if set(frozen) != {'taxonomy.py', 'prompts.py', 'calibration.py', 'observation.py'}:
        raise ValueError('missing frozen evaluation rubric')
    for name, sha in frozen.items():
        if file_sha(ROOT/name) != sha:
            raise ValueError('frozen evaluation rubric changed; review evaluation protocol before reuse')
    owners = {}
    for row in plan['reservations']:
        group, split = row['physical_group'], row['split']
        if group in owners or split not in ('val', 'test'):
            raise ValueError('invalid/duplicate evaluation reservation')
        owners[group] = split
    extra=producer_check_reservations()
    # Retired abnormal reservations remain quarantined in their original split.
    for row in json.loads((ROOT/PRODUCER_CHECK_PLAN).read_text()).get('excluded_reservations',[]):
        group,split=row['physical_group'],row['split']
        if group in extra or split not in ('val','test'):raise ValueError('invalid retired reservation')
        extra[group]=split
    if set(extra)&set(owners):raise ValueError('producer check overlaps model evaluation reservations')
    owners.update(extra)
    return owners


def split_for(group, exposed, seed=20260929, *, reservations=None):
    # Prospective evaluation ownership survives changes to the sampling seed.
    # A later development exposure invalidates the reservation; never silently
    # turn these evaluation annotations into training supervision.
    owners = holdout_reservations() if reservations is None else reservations
    if group in owners:
        if group in exposed:
            raise ValueError('reserved evaluation group has development exposure; replace holdout plan')
        return owners[group]
    if group in exposed:
        return 'train'
    value = int(digest([seed,group])[:12],16) / 16**12
    return 'test' if value < .1 else 'val' if value < .2 else 'train'


def check_evaluation_protocol(record, reservations):
    if record['physical_group'] in producer_check_reservations():
        raise ValueError('producer manual-check routes cannot supply model selection labels')
    if record['physical_group'] in reservations:
        if route_bound(Episode(**record['episode'])) or record['edge'] in GUARDED | {'recover_follow'}:
            raise ValueError('route extension requires a new prospective evaluation protocol')
        if record.get('evaluation_annotation_protocol') != HOLDOUT_PLAN:
            raise ValueError('reserved evaluation annotation lacks label-only protocol')
    elif record.get('evaluation_annotation_protocol'):
        raise ValueError('evaluation protocol attached to unreserved physical group')


def source_route(record):
    """Require unambiguous source components before joining paths or splitting."""
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import physical_route_group
    for name in ('scenario', 'route_id'):
        value = record.get(name)
        if (not isinstance(value, str) or not value or value in ('.', '..')
                or '/' in value or '\\' in value or '\x00' in value):
            raise ValueError('invalid source route component')
    return physical_route_group(record['scenario'], record['route_id'])


def validate_source_identity(row, observation_spec):
    # Declared split/group and image hashes alone do not bind the images to
    # their reserved route. Check the canonical provenance for every input.
    if row.get('physical_group') != source_route(row):
        raise ValueError('source route physical_group mismatch')
    validate_observation(observation_spec, row['observation'], len(row['images']))
    expected = [f"{row['scenario']}/{row['route_id']}/rgb/{f:04d}.jpg"
                for f in row['observation']['history_frames']]
    if row['images'] != expected:
        raise ValueError('source route/history image path mismatch')
    if any(len(row.get(k, [])) != len(expected)
           for k in ('image_sha256', 'image_rgb_sha256')):
        raise ValueError('source image identity count mismatch')


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))


def build(annotations, data_root, output, *, rgb_mode=4, seed=20260929, candidate_pool=None, teacher_registry=None, production_index=None, teacher_selection=None):
    from qwen3vl_local.sft_new_loop_phase3.trajectory_action import _load_meta
    from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route
    if rgb_mode not in (2,4):
        raise ValueError('history mode must be 2 or 4')
    output, data_root = Path(output), Path(data_root).resolve()
    if output.exists():
        raise FileExistsError('use a fresh dataset directory')
    from .teacher_approval import validate_registry
    teacher_approval=validate_registry(teacher_registry) if teacher_registry is not None else None
    if production_index is not None and candidate_pool is None:raise ValueError('production ledger needs candidate inventory')
    exposed, rows, seen = groups(), [], {}
    reservations = holdout_reservations()
    rgb_cache = {}
    known_risks = risk_review.registry()
    if candidate_pool is not None:
        from .candidate_pool import validate
        validate(candidate_pool)
        if candidate_pool['seed']!=seed:raise ValueError('candidate split seed mismatch')
    offsets = (-6,-4,-2,0) if rgb_mode == 4 else (-4,0)
    for ann in annotations:
        group = source_route(ann)
        teacher_target=None
        if 'teacher_provenance' in ann or ann.get('label_basis') in ('approved_rule_teacher','weak_rule_teacher') or str(ann.get('evidence_id','')).startswith('teacher/'):
            from .teacher_data import validate_annotation as validate_teacher_annotation
            if teacher_approval is None:raise ValueError('teacher registry required')
            teacher_target=validate_teacher_annotation(ann,teacher_approval,data_root,rgb_mode)
        if ann.get('condition_provenance') is not None or str(ann.get('evidence_id','')).startswith('conditions/'):
            from .condition_builder import validate_annotation
            validate_annotation(ann,data_root)
        if (teacher_target is None and not ann.get('reviewer')) or not ann.get('evidence_id') or not ann.get('observation'):
            raise ValueError('annotation lacks reviewed evidence')
        if ann.get('label_basis') not in ('per_frame_conditions','reviewed_transition_band','approved_rule_teacher','weak_rule_teacher'):
            raise ValueError('recorded action timing is not a Phase4 permission label')
        if ann['label_basis']=='per_frame_conditions' and ann.get('slice','readiness')!='readiness':
            raise ValueError('legacy catchup requires reviewed_transition_band with per-frame conditions and conflict scope')
        ep_fields = ann['episode']
        if set(ep_fields) - EPISODE_FIELDS:
            raise ValueError('unknown episode annotation field')
        validate_source_context(ann)
        ep = Episode(**ep_fields)
        edge = episode_edge(ep,ann['edge'])
        run = data_path(data_root, f"{ann['scenario']}/{ann['route_id']}")
        if is_abnormal_lead_route(run,ann['scenario'])[0]:
            raise ValueError('invalid/abnormal source route')
        split = split_for(group,exposed,seed,reservations=reservations)
        check_evaluation_protocol(dict(ann,physical_group=group),reservations)
        band = ann.get('transition_band') if ann['label_basis']=='reviewed_transition_band' else None
        if ann['label_basis']=='reviewed_transition_band' and not band:
            raise ValueError('missing frame-by-frame transition review')
        start,end = (band['review_start'],band['review_end']) if band else (ann['start'],ann['end'])
        reviewed = reviewed_band_labels(ep,edge.key,band,context_valid=ann.get('context_valid')) if band else None
        admission = risk_review.annotation_review(ann,data_root,rgb_mode,known_risks)
        facts = interval_facts(ann.get('facts',{}),start,end,ann['evidence_id']) if teacher_target is None else None
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
            if teacher_target is not None:
                target=teacher_target
            elif reviewed is not None:
                target, label_slice = reviewed[frame]
            else:
                target = label_condition(ep,edge.key,frame,facts,context_valid=ann.get('context_valid'))
            key = digest([group,ann['route_id'],frame,ep_fields,edge.key])
            source_hashes = [file_sha(p) for p in paths]
            for path,sha in zip(paths,source_hashes):
                if sha not in rgb_cache:
                    from PIL import Image
                    with Image.open(path) as im:
                        rgb_cache[sha] = rgb_content_sha(im)
            row = dict(id=key,physical_group=group,scenario=ann['scenario'],route_id=ann['route_id'],
                       split=split,episode=ep_fields,edge=edge.key,observation=obs,
                       images=[str(p.relative_to(data_root)) for p in paths],
                       image_sha256=source_hashes,image_rgb_sha256=[rgb_cache[s] for s in source_hashes],target=target,
                       slice=label_slice if reviewed is not None else ann.get('slice','readiness'),evidence_id=ann['evidence_id'],
                       label_basis=ann['label_basis'],transition_band=band_summary(ep,edge.key,band) if band else None,
                       frame_review=band['frames'][str(frame)] if band else None,
                       prompt_sha256=digest(text))
            if group in reservations:
                row['evaluation_annotation_protocol'] = ann['evaluation_annotation_protocol']
            if teacher_target is not None:
                row.update(teacher_provenance=ann['teacher_provenance'],reference_kind='rule_teacher')
            if 'condition_provenance' in ann:
                row['condition_provenance']=ann['condition_provenance']
            if admission is not None:
                proof = risk_review.row_review(row,admission,known_risks)
                if proof is not None:
                    row['risk_review'] = proof
                    if proof['disposition'] != 'usable':
                        row['target'],row['slice'] = 'UNKNOWN','uncertain'
                target = row['target']
            if ann.get('visible_scope_review') is not None:
                proof = ann['visible_scope_review']
                row['visible_scope_review'] = {**proof, 'frames': {
                    str(f): proof.get('frames', {}).get(str(f)) for f in frames}}
            if not visible_scope.reviewed(row):
                row['target'], row['slice'] = 'UNKNOWN', 'uncertain'
                row['review_reason'] = 'visible_scope_reaudit_required'
            from .visual_review import row_supported
            if ann.get('visual_review') is not None:
                proof=ann['visual_review']
                row['visual_review']=dict(policy=proof.get('policy'),
                    evidence=proof.get('modes',{}).get(str(rgb_mode)),
                    reviewed_target=target,reviewed_phase=label_slice if reviewed is not None else ann.get('slice','readiness'))
            if not row_supported(row):
                row['target'],row['slice']='UNKNOWN','uncertain'
                row['review_reason']='actual_input_visual_evidence_unresolved'
            target = row['target']
            row['model_input_sha256'] = model_input_key(row)
            if key in seen:
                if seen[key]['target'] != target or seen[key]['slice'] != row['slice']:
                    raise ValueError('conflicting interval annotations')
                continue
            seen[key] = row
            rows.append(row)
    rows=list(seen.values())
    from .teacher_data import reconcile_weak_inputs
    rows,weak_input_exclusions=reconcile_weak_inputs(rows)
    production=None
    if candidate_pool is not None:
        from .candidate_pool import coverage
        production=coverage(candidate_pool,rows,production_index=production_index,approved_classes=sorted(teacher_approval['approved']) if teacher_approval else [])
    input_check = validate_model_inputs(rows)
    risk_check = risk_review.validate_rows(rows,known_risks)
    output.mkdir(parents=True)
    if candidate_pool is not None:write_json(output/'candidate_pool.json',candidate_pool)
    if teacher_registry is not None:write_json(output/'teacher_registry.json',teacher_registry)
    if production_index is not None:
        import shutil
        index=json.loads(Path(production_index).read_text());bundle=output/'production';bundle.mkdir()
        shutil.copyfile(production_index,bundle/'index.json')
        for r in index['routes']:shutil.copyfile(Path(production_index).parent/r['artifact'],bundle/r['artifact'])
    review_queue=[r for r in rows if r['target'] not in ('YES','NO')]
    dump_rows(output/'review_queue.jsonl',review_queue)
    rows=[r for r in rows if r['target'] in ('YES','NO')]
    for split in ('train','val','test'):
        dump_rows(output/f'{split}.jsonl',[r for r in rows if r['split']==split])
    coverage = coverage_report(rows)
    admission = training_report({s:[r for r in rows if r['split']==s]
                                 for s in ('train','val','test')},coverage)
    manifest = dict(contract=contract(),rgb_mode=rgb_mode,observation_contract=observation_contract(rgb_mode),seed=seed,annotations_sha256=digest(annotations),
                    candidate_pool_sha256=file_sha(output/'candidate_pool.json') if candidate_pool is not None else None,
                    production_coverage=production,teacher_selection=teacher_selection,
                    teacher_registry_sha256=file_sha(output/'teacher_registry.json') if teacher_registry is not None else None,
                    production_index_sha256=file_sha(output/'production/index.json') if production_index is not None else None,
                    exposure_sha256=digest(sorted(exposed)),counts=dict(Counter(r['split'] for r in rows)),
                    files={f'{s}.jsonl':file_sha(output/f'{s}.jsonl') for s in ('train','val','test')},
                    coverage=coverage,ready=admission['trainable'],ready_scope='training_data',complete_coverage=coverage['ready'],
                    trainable=admission['trainable'],training_admission=admission,model_answers=['YES','NO'],
                    review_queue_sha256=file_sha(output/'review_queue.jsonl'),review_queue_count=len(review_queue),
                    model_input_check=input_check,risk_review_check=risk_check,weak_input_exclusions=weak_input_exclusions,
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
    missing_edges, transition_support = [], {}
    for split in ('train','val','test'):
        for event,branch,ret in variants:
            template = template_for(event,branch)
            variant = branch + (('/return' if ret else '/no_return') if template=='bypass' else '')
            reachable_common = [e for e in (*COMMON, RECOVER_FOLLOW) if any(applicable(e,state,lon)
                for state in TEMPLATE_STATES[template] for lon in LONGITUDINAL)]
            for edge in (*event_edges(template,ret), *reachable_common):
                key = (split,event,variant,edge.key)
                transition_support['/'.join(key)]={a:edge_cells[key][a] for a in ('YES','NO')}
                if not all(edge_cells[key][a] for a in ('YES','NO')):
                    missing_edges.append('/'.join(key))
    segment_cells = defaultdict(Counter)
    for r in rows:
        ep = r['episode']
        segments = ep.get('route_segments',[])
        if ep['event']=='R-E3' and len(segments)>1 and r['slice']=='readiness':
            role = 'final' if ep.get('segment_index',0)==len(segments)-1 else 'intermediate'
            segment_cells[(r['split'],role,r['edge'])][r['target']] += 1
    missing_segments = [f'{split}/R-E3/{role}/{edge}'
        for split in ('train','val','test') for role in ('intermediate','final')
        for edge in ('enter','complete')
        if not all(segment_cells[(split,role,edge)][answer] for answer in ('YES','NO'))]
    roundabout_cells = defaultdict(Counter)
    for r in rows:
        if r['episode'].get('route_context',{}).get('kind') == 'roundabout' and r['slice']=='readiness':
            roundabout_cells[(r['split'],r['edge'])][r['target']] += 1
    roundabout_support = {f'{split}/R-E5/roundabout/{edge}':
        {answer:roundabout_cells[(split,edge)][answer] for answer in ('YES','NO')}
        for split in ('train','val','test') for edge in ('proceed','re_yield','complete')}
    missing_roundabouts = [k for k,v in roundabout_support.items() if not all(v.values())]
    return dict(ready=not missing and not missing_edges and not missing_segments and not missing_roundabouts,
                roundabout_support=roundabout_support,missing_roundabout_support=missing_roundabouts,
                missing_segmented_route_support=missing_segments,missing_event_support=missing,
                label_slices=dict(Counter(r['slice'] for r in rows)),
                edge_support_scope='reviewed readiness only; catchup excluded',
                transition_support=transition_support,
                missing_transition_edges=missing_edges,
                missing_longitudinal_edges=[k for k in missing_edges if k.rsplit('/',1)[-1] in {e.key for e in (*COMMON, RECOVER_FOLLOW)}],
                cells={k:dict(rows=v['rows'],groups=len(v['groups']),answers=dict(v['answers'])) for k,v in cells.items()})


def load_dataset(path, *, require_ready=False, require_trainable=False, require_complete_coverage=False):
    """Training readiness and optional exhaustive coverage use separate gates."""
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    check_contract(manifest['contract'])
    reservations, exposed = holdout_reservations(), groups()
    teacher_approval=None
    if manifest.get('teacher_registry_sha256') is not None:
        from .teacher_approval import validate_registry
        if file_sha(path/'teacher_registry.json')!=manifest['teacher_registry_sha256']:raise ValueError('teacher registry hash mismatch')
        teacher_approval=validate_registry(json.loads((path/'teacher_registry.json').read_text()))
    production_index=None
    if manifest.get('production_index_sha256') is not None:
        production_index=path/'production/index.json'
        if file_sha(production_index)!=manifest['production_index_sha256']:raise ValueError('production index hash mismatch')
    if check_observation_contract(manifest.get('observation_contract')) != observation_contract(manifest['rgb_mode']):
        raise ValueError('dataset observation contract mismatch')
    if manifest['exposure_sha256'] != digest(sorted(exposed)):
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
            if split != 'train' and group in exposed:
                raise ValueError('development route in holdout')
    all_rows = sum(result.values(),[]) + read_rows(path/'review_queue.jsonl')
    for r in all_rows:
        if ('teacher_provenance' in r or r.get('label_basis') in ('approved_rule_teacher','weak_rule_teacher')
                or r.get('reference_kind')=='rule_teacher' or str(r.get('evidence_id','')).startswith('teacher/')):
            from .teacher_data import validate_row as validate_teacher_row
            if teacher_approval is None:raise ValueError('teacher registry missing at load')
            validate_teacher_row(r,teacher_approval)
        validate_source_context(r)
        validate_source_identity(r,manifest['observation_contract'])
        from .visual_review import row_supported
        if not row_supported(r) and r['target']!='UNKNOWN':
            raise ValueError('unsupported actual-input visual evidence cannot be supervision')
        if 'condition_provenance' in r or str(r.get('evidence_id','')).startswith('conditions/'):
            from .condition_builder import validate_row
            if 'condition_provenance' not in r:
                raise ValueError('missing condition provenance')
            validate_row(r)
        expected = split_for(r['physical_group'],exposed,manifest['seed'],reservations=reservations)
        if r['split'] != expected:
            raise ValueError('physical route split differs from reserved/seed ownership')
        check_evaluation_protocol(r,reservations)
    checked = validate_model_inputs(all_rows)
    visible_scope.validate_rows(all_rows)
    if risk_review.validate_rows(all_rows,risk_review.registry()) != manifest.get('risk_review_check'):
        raise ValueError('risk admission summary mismatch')
    if checked != manifest.get('model_input_check'):
        raise ValueError('model input check summary mismatch')
    coverage = coverage_report(sum(result.values(),[]))
    if manifest.get('candidate_pool_sha256') is not None:
        from .candidate_pool import coverage as pool_coverage
        pool_path=path/'candidate_pool.json'
        if file_sha(pool_path)!=manifest['candidate_pool_sha256']:raise ValueError('candidate pool hash mismatch')
        if pool_coverage(json.loads(pool_path.read_text()),all_rows,production_index=production_index,approved_classes=sorted(teacher_approval['approved']) if teacher_approval else [])!=manifest.get('production_coverage'):
            raise ValueError('candidate production coverage mismatch')
    admission = training_report(result,coverage)
    if (coverage != manifest['coverage'] or admission != manifest.get('training_admission')
            or manifest.get('trainable') != admission['trainable']
            or manifest.get('complete_coverage') != coverage['ready']
            or manifest['ready'] != admission['trainable'] or manifest.get('ready_scope')!='training_data'):
        raise ValueError('dataset admission/coverage summary mismatch')
    if (require_trainable or require_ready) and not admission['trainable']:
        raise ValueError('training prerequisites failed: '+ '; '.join(admission['errors']))
    if require_complete_coverage and not coverage['ready']:
        raise ValueError('insufficient independent YES/NO support; see manifest coverage; no fallback split')
    return result,manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path)
    p.add_argument('--teacher-registry',type=Path)
    p.add_argument('--production-index',type=Path)
    p.add_argument('--max-teacher-questions-per-route',type=int,default=100,help='0 includes all eligible teacher questions; positive values cap each route')
    p.add_argument('--candidate-pool',type=Path,help='Frozen full RGB inventory created before annotation')
    p.add_argument('--condition-stream',type=Path,
                   help='Causal facts JSONL; shared rules generate all applicable questions')
    p.add_argument('--data-root',type=Path,default=ROOT.parents[1]/'lead_data')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--rgb-mode',type=int,choices=(2,4),default=4)
    a=p.parse_args()
    if not a.annotations and not a.condition_stream and not a.production_index:
        p.error('supply --annotations and/or --condition-stream')
    annotations=json.loads(a.annotations.read_text()) if a.annotations else []
    compilation=None
    if a.condition_stream:
        from .condition_builder import compile_stream
        compiled,compilation=compile_stream(a.condition_stream,a.data_root)
        annotations+=compiled
    pool=json.loads(a.candidate_pool.read_text()) if a.candidate_pool else None
    registry=json.loads(a.teacher_registry.read_text()) if a.teacher_registry else None
    if a.production_index:
        if pool is None or registry is None:p.error('--production-index needs --candidate-pool and --teacher-registry')
        from .teacher_data import compile_bundle
        compiled,teacher_report=compile_bundle(pool,a.production_index,registry,rgb_mode=a.rgb_mode,max_per_route=a.max_teacher_questions_per_route)
        annotations+=compiled
    m=build(annotations,a.data_root,a.output_dir,rgb_mode=a.rgb_mode,candidate_pool=pool,
            teacher_registry=registry,production_index=a.production_index,
            teacher_selection=dict(max_per_route=teacher_report['max_per_route'],route_budget_excluded=teacher_report['counts']['route_budget_excluded'],selection_policy=teacher_report['selection_policy']) if a.production_index else None)
    if a.production_index:write_json(a.output_dir/'teacher_compilation.json',teacher_report)
    if compilation:
        write_json(a.output_dir/'condition_compilation.json',compilation)
    print(json.dumps({k:m[k] for k in ('trainable','complete_coverage','counts','training_admission','coverage')},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
