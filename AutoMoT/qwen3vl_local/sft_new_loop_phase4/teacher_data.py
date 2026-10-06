"""Compile approved teacher questions without calling them human RGB reviews."""
from collections import Counter,defaultdict,deque
import json
from pathlib import Path
from .data_paths import data_path
from .identity import digest,file_sha
from .teacher_approval import validate_registry,target
from .teacher_replay import validate_question,accounting


def selection_key(question):
    """Stable causal task identity, independent of RGB mode and build hashes.

    Keep instance identity (including actor and start), state, branch, route
    target, edge and answer. Do not collapse different participants or phases.
    """
    return digest([question['scenario'],question['route_id'],question['physical_group'],
                   question['frame_id'],question['episode'],question['edge'],
                   question['slice'],question['rule_target']])


def label_basis(approval):
    return 'weak_rule_teacher' if approval.get('supervision_policy')=='weak_experiment_train_only' else 'approved_rule_teacher'


def validate_proof(proof,approval):
    if not isinstance(proof,dict) or proof.get('registry_sha256')!=approval['registry_sha256']:
        raise ValueError('teacher approval registry mismatch')
    q=validate_question(proof['question'],expected_teacher_sha=approval.get('teacher_sha256'))
    answer,reason=target(q,approval)
    if answer not in ('YES','NO') or proof.get('approval_sha256')!={**approval['approved'],**approval.get('weak_classes',{})}.get(q['rule_class']):
        raise ValueError('unapproved teacher supervision')
    return q


def validate_annotation(ann,approval,root,mode):
    q=validate_proof(ann.get('teacher_provenance'),approval)
    if (ann.get('label_basis')!=label_basis(approval) or ann.get('slice')!='readiness'
            or mode!=q['rgb_mode'] or ann['start']!=q['frame_id'] or ann['end']!=q['frame_id']
            or any(ann[k]!=q[k] for k in ('scenario','route_id','episode','edge','facts'))
            or ann.get('evidence_id')!='teacher/'+q['question_id'] or ann.get('context_valid') is not True
            or ann.get('transition_band') is not None):raise ValueError('teacher annotation detached from approved question')
    root=Path(root).resolve()
    control_sources=[s for r in q.get('control_evidence',{}).values() for o in r['observations'] for s in o['sources']]
    for source in q['causal_sources']+control_sources:
        path=data_path(root, source['path'])
        if file_sha(path)!=source['sha256']:raise ValueError('teacher causal source changed: '+source['path'])
    from qwen3vl_local.sft_new_loop_phase3.trajectory_action import _load_meta
    current=next(s for s in q['causal_sources'] if s['kind']=='metas' and s['frame_id']==q['frame_id'])
    if float(_load_meta(root/current['path'])['speed'])!=q['ego_speed']:
        raise ValueError('teacher ego motion differs from causal metadata')
    return q['rule_target']


def validate_row(row,approval):
    q=validate_proof(row.get('teacher_provenance'),approval)
    if (row.get('label_basis')!=label_basis(approval) or row.get('reference_kind')!='rule_teacher'
            or any(row[k]!=q[k] for k in ('scenario','route_id','episode','edge','physical_group','split','slice'))
            or row['observation']['frame_id']!=q['frame_id'] or row['observation']['history_frames']!=q['history_frames']
            or row['images']!=[s['path'] for s in q['input_sources']]
            or row['image_sha256']!=[s['sha256'] for s in q['input_sources']]
            or row['observation']['speed_mps']!=max(0.,q['ego_speed'])
            or row['target']!=q['rule_target'] or row['evidence_id']!='teacher/'+q['question_id']):
        raise ValueError('teacher row/input/target differs from approved proof')


def compile_bundle(pool,index_path,registry,*,rgb_mode,max_per_route=100):
    from .dataset import holdout_reservations
    from .risk_review import registry as risk_registry,teacher_window_check
    if rgb_mode not in (2,4) or type(max_per_route) is not int or max_per_route<0:raise ValueError('invalid teacher compilation budget')
    index_path=Path(index_path);coverage=accounting(pool,index_path);approval=validate_registry(registry)
    reserved=holdout_reservations();risks=risk_registry();counts=Counter();annotations=[]
    index=json.loads(index_path.read_text())
    for receipt in index['routes']:
        strata=defaultdict(list)
        with (index_path.parent/receipt['artifact']).open() as stream:
            next(stream)
            for line in stream:
                row=json.loads(line)
                for q in row.get('questions',[]):
                    if q['rgb_mode']!=rgb_mode:continue
                    counts['considered']+=1
                    answer,reason=target(q,approval)
                    if answer=='UNKNOWN':counts[reason]+=1;continue
                    if q['physical_group'] in reserved:counts['reserved_manual_evaluation']+=1;continue
                    if not teacher_window_check(q,risks.get((q['scenario'],q['route_id']),[]))['outside_registered_windows']:counts['known_risk_window_requires_separate_review']+=1;continue
                    strata[(q['episode']['event'],q['edge'],answer,q.get('ego_motion','unknown'))].append(q)
        from .teacher_review import observable as input_key
        observable=defaultdict(list)
        for values in strata.values():
            for q in values:
                key=input_key(q)
                observable[key].append(q)
        strata=defaultdict(list)
        for values in observable.values():
            if len({q['rule_target'] for q in values})>1:
                counts['conflicting_observable_questions_excluded']+=len(values);continue
            q=sorted(values,key=selection_key)[0]
            counts['duplicate_observable_questions_excluded']+=len(values)-1
            strata[(q['episode']['event'],q['edge'],q['rule_target'],q.get('ego_motion','unknown'))].append(q)
        # Round-robin event/edge/answer within each route before event-equal
        # training sampling; long waiting runs cannot consume the route budget.
        queues=[deque(sorted(v,key=lambda q:(not q.get('near_transition',False),selection_key(q)))) for _,v in sorted(strata.items())]
        selected=[]
        while queues and (max_per_route==0 or len(selected)<max_per_route):
            for queue in queues:
                if max_per_route and len(selected)==max_per_route:break
                if queue:selected.append(queue.popleft())
            queues=[q for q in queues if q]
        counts['route_budget_excluded']+=sum(map(len,queues))
        for q in selected:
            proof=dict(question=q,registry_sha256=approval['registry_sha256'],approval_sha256={**approval['approved'],**approval.get('weak_classes',{})}[q['rule_class']])
            annotations.append(dict(scenario=q['scenario'],route_id=q['route_id'],episode=q['episode'],edge=q['edge'],
                start=q['frame_id'],end=q['frame_id'],facts=q['facts'],context_valid=True,slice='readiness',
                label_basis=label_basis(approval),rule_teacher=q['rule_class'],observation='causal rule evidence; not human review',
                evidence_id='teacher/'+q['question_id'],frame_sha256={str(q['frame_id']):q['input_sources'][-1]['sha256']},
                teacher_provenance=proof))
    counts['compiled']=len(annotations)
    return annotations,dict(counts=dict(counts),approval=approval,production=coverage,rgb_mode=rgb_mode,
        max_per_route=max_per_route,automatic_catchup_limit=0,selection_policy='semantic_task_order_v1',
        evaluation_scope='teacher consistency only; manual reserved routes excluded')


def reconcile_weak_inputs(rows):
    """Preserve reviewed labels; quarantine conflicting weak observable inputs."""
    groups=defaultdict(list)
    for row in rows:groups[row['model_input_sha256']].append(row)
    dropped=set();counts=Counter()
    for values in groups.values():
        weak=[r for r in values if r.get('label_basis')=='weak_rule_teacher']
        reviewed=[r for r in values if r.get('label_basis')!='weak_rule_teacher']
        if not weak:continue
        if reviewed:
            for r in weak:
                dropped.add(r['id']);counts['manual_input_overlap']+=1
        elif len({r['target'] for r in weak})>1:
            for r in weak:dropped.add(r['id']);counts['conflicting_weak_input']+=1
    return [r for r in rows if r['id'] not in dropped],dict(counts)
