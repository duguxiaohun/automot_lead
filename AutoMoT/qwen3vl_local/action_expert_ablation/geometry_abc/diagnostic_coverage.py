"""Bounded, deterministic validation coverage and isolated-slot response reports."""
from collections import Counter,defaultdict
import math
import numpy as np

ROLES={'future':('targets',9),'history':('history',10)}


def labels(record,variant,role):
    return [t['token_id'] if t['valid'] else 9 for t in record['representations'][variant][ROLES[role][0]]]


def select(rows,variant,groups_per_cell=4):
    if groups_per_cell!=4:raise ValueError('fixed_condition_coverage_budget')
    cells=defaultdict(dict);counts=Counter();events=defaultdict(dict);lookup={}
    for row in rows:
        r=row.get('abc_record',row)
        if r['split']!='val':raise ValueError('condition_diagnostics_validation_only')
        if r['id'] in lookup:raise ValueError('duplicate_condition_candidate')
        lookup[r['id']]=row;group=r['physical_group']
        for role in ROLES:
            for h,c in enumerate(labels(r,variant,role)):
                if not 0<=c<ROLES[role][1]:raise ValueError('invalid_condition_coverage_label')
                key=f'{role}/{h}/{c}';counts[key]+=1
                cells[key][group]=min(r['id'],cells[key].get(group,r['id']))
        events[r['event']][group]=min(r['id'],events[r['event']].get(group,r['id']))
    chosen=set()
    # Up to four distinct groups per supported cell; no rare category can be
    # hidden by the event-first representatives. Each addition is deterministic.
    for groups in list(cells.values())+list(events.values()):
        chosen.update(sorted(groups.values())[:groups_per_cell])
    selected=[lookup[k] for k in sorted(chosen)];selected_cells=defaultdict(list)
    for row in selected:
        r=row.get('abc_record',row)
        for role in ROLES:
            for h,c in enumerate(labels(r,variant,role)):selected_cells[f'{role}/{h}/{c}'].append(r)
    report={}
    for role,(_,classes) in ROLES.items():
        for h in range(4):
            for c in range(classes):
                key=f'{role}/{h}/{c}';seen=selected_cells[key];available=len(cells[key]);actual=len({r['physical_group'] for r in seen})
                if actual<min(groups_per_cell,available):raise ValueError('supported_condition_cell_not_covered')
                report[key]=dict(support_records=counts[key],support_groups=available,selected_records=len(seen),
                    selected_groups=actual,status='N/A' if not available else 'covered',sparse_group_support=0<available<groups_per_cell)
    return selected,dict(schema='abc_condition_coverage_v2',variant=variant,groups_per_cell=groups_per_cell,
        candidate_records=len(rows),selected_ids=sorted(chosen),selected_records=len(selected),
        maximum_records=groups_per_cell*(76+len(events)),cells=report,
        selection='first stable ID per distinct physical group per supported role/horizon/region plus event',
        history_intervention_scope='decoder history slot only; Qwen observation text remains fixed',
        missing_history_9='UNKNOWN; replacement with region 0 is a separate missingness intervention')


def statistics(values):
    if any(not math.isfinite(x) or x<0 for x in values):raise ValueError('invalid_condition_response')
    if not values:return dict(cases=0,nonzero_cases=0,nonzero_fraction=None,p00=None,p25=None,p50=None,p75=None,p100=None)
    n=sum(x>1e-6 for x in values)
    return dict(cases=len(values),nonzero_cases=n,nonzero_fraction=n/len(values),
        **{f'p{q:02d}':float(np.quantile(values,q/100)) for q in (0,25,50,75,100)})


def detailed(cases,coverage):
    if len({c['id'] for c in cases})!=len(cases) or sorted(c['id'] for c in cases)!=coverage['selected_ids']:
        raise ValueError('condition_coverage_members_changed')
    cells=defaultdict(list);groups=defaultdict(set)
    for case in cases:
        for role in ROLES:
            ids=case['labels'] if role=='future' else case['history_labels']
            if len(ids)!=4 or len(case['slot_responses'][role])!=4:raise ValueError('four_slot_responses_required')
            for h,(c,value) in enumerate(zip(ids,case['slot_responses'][role])):
                key=f'{role}/{h}/{c}';cells[key].append(float(value));groups[key].add(case['physical_group'])
    reports={};reviews=[]
    if set(cells)-set(coverage['cells']):raise ValueError('unknown_condition_cell')
    for key,support in coverage['cells'].items():
        stats=statistics(cells[key])
        if stats['cases']!=support['selected_records'] or len(groups[key])!=support['selected_groups']:
            raise ValueError('condition_response_coverage_changed')
        status='N/A' if not support['support_records'] else ('observed_response' if stats['nonzero_fraction']==1 else 'response_review_required')
        if status=='response_review_required':reviews.append(key)
        reports[key]=dict(support,**stats,response_status=status)
    return dict(cells=reports,review_cells=reviews,
        interpretation='Zero isolated-slot responses require review, not proof of collapse; nonzero response is not accuracy.',
        unsupported_policy='N/A, never failure solely for no support; fewer than four physical groups is a limitation',
        role_gradient_claim=False)
