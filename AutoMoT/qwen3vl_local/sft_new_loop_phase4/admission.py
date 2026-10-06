"""Training prerequisites are distinct from exhaustive evaluation coverage."""
from collections import Counter
from .taxonomy import EVENTS

POLICY = 'phase4_training_admission_v3'


def review_capacity_report(data):
    """User's scale floor counts distinct physical routes, never frames/repeats.

    These numerical floors are necessary diagnostics, not proof of annotation
    accuracy. Independent assessment remains a separate requirement.
    """
    from .taxonomy import event_edges
    cells={}
    for event,(_,template,_) in EVENTS.items():
        keys={e.key for e in event_edges(template)} | {'recover_follow'}
        for edge in sorted(keys):
            rows=[r for r in data.get('train',[]) if r['episode']['event']==event
                  and r['edge']==edge and r['slice']=='readiness']
            cells[event+'/'+edge]={answer:len({r['physical_group'] for r in rows if r['target']==answer})
                                   for answer in ('YES','NO')}
    evaluation={split:{event:len({r['physical_group'] for r in data.get(split,[])
                                  if r['episode']['event']==event}) for event in EVENTS}
                for split in ('val','test')}
    missing_train=[dict(transition=k,answer=a,routes=v[a],required=20) for k,v in cells.items()
                   for a in ('YES','NO') if v[a]<20]
    missing_eval=[dict(split=s,event=e,routes=n,required=5) for s,events in evaluation.items()
                  for e,n in events.items() if n<5]
    return dict(policy='phase4_distinct_route_review_floor_v1',train_routes_per_transition_answer=cells,
        evaluation_routes=evaluation,missing_train=missing_train,missing_evaluation=missing_eval,
        numerical_floor_met=not missing_train and not missing_eval,
        independent_accuracy_certified=False,
        scope='event edges plus recover_follow; branch/state coverage remains separately required')


def training_report(data, coverage):
    errors, counts = [], {}
    owners = {}
    for split in ('train', 'val', 'test'):
        rows = data.get(split, [])
        answers = Counter(r['target'] for r in rows)
        counts[split] = dict(rows=len(rows), answers=dict(answers))
        if not rows:
            errors.append(f'{split}: empty split')
        elif set(answers) != {'YES', 'NO'}:
            errors.append(f'{split}: independent binary YES/NO support required')
        for row in rows:
            group = row['physical_group']
            if group in owners and owners[group] != split:
                errors.append(f'physical route leakage: {group}')
            owners[group] = split
    readiness = Counter(r['target'] for r in data.get('train', []) if r['slice']=='readiness')
    if not all(readiness[a] for a in ('YES', 'NO')):
        errors.append('train: readiness YES/NO required; catchup alone cannot teach permission')
    event_scope = {s: sorted({r['episode']['event'] for r in data.get(s, [])})
                   for s in ('train', 'val', 'test')}
    support = {}
    for row in data.get('train', []):
        key = row['episode']['event'] + '/' + row['edge']
        cell = support.setdefault(key, {p: {'YES': 0, 'NO': 0} for p in ('readiness', 'catchup')})
        if row['slice'] in cell:
            cell[row['slice']][row['target']] += 1
    # Expose exact current-state support: PROCEED/FOLLOW completion examples
    # cannot fill the recovery transition's training gap.
    recovery_support = {}
    for event in EVENTS:
        for edge in ('stable', 'recover_follow'):
            rows = [r for r in data.get('train', []) if r['episode']['event'] == event
                    and r['episode'].get('longitudinal') == 'RECOVER' and r['edge'] == edge
                    and r['slice'] == 'readiness']
            recovery_support[f'{event}/RECOVER/{edge}'] = {a: sum(r['target'] == a for r in rows) for a in ('YES','NO')}
    return dict(policy=POLICY, weak_supervision_rows=sum(r.get('label_basis')=='weak_rule_teacher' for r in data.get('train',[])), trainable=not errors, errors=sorted(set(errors)),
                event_scope=event_scope,
                physical_routes={s: len({r['physical_group'] for r in data.get(s, [])}) for s in event_scope},
                missing_evaluation_events={s: sorted(set(EVENTS)-set(event_scope[s])) for s in ('val','test')},
                train_readiness_catchup_support=support,
                train_recovery_state_support=recovery_support,
                catchup_dominated_transitions=[k for k,v in support.items() if v['catchup']['YES'] > v['readiness']['YES']],
                counts=counts, complete_coverage=coverage['ready'],
                coverage_role='diagnostic; incomplete cells do not block training',
                evaluation_scope='observed independent labelled subsets; not full-loop certification',
                train_missing_readiness_yes=[k for k,v in coverage['transition_support'].items()
                                             if k.startswith('train/') and not v['YES']],
                train_missing_readiness_no=[k for k,v in coverage['transition_support'].items()
                                            if k.startswith('train/') and not v['NO']],
                missing_event_cells=len(coverage['missing_event_support']),
                missing_transition_cells=len(coverage['missing_transition_edges']),
                missing_segment_cells=len(coverage['missing_segmented_route_support']))
