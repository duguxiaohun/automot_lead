"""Paired endpoint protection, including losses hidden by net gains."""
from collections import defaultdict


def indexed(cases):
    result = {}
    for c in cases:
        r = c['record']; key = r['id']
        if key in result: raise ValueError('duplicate_case')
        p = c['prediction']['token_ids']
        if len(p) != 4 or any(type(i) is not int or not 0 <= i < 9 for i in p):
            raise ValueError('invalid_prediction')
        result[key] = c
    if not result: raise ValueError('empty_cohort')
    return result


def scores(rows):
    result = {}
    for c in range(9):
        tp = sum(y == c and p == c for y,p in rows)
        actual = sum(y == c for y,p in rows); predicted = sum(p == c for y,p in rows)
        result[str(c)] = dict(support=actual, predicted=predicted,
            precision=tp/predicted if predicted else None,
            recall=tp/actual if actual else None,
            f1=2*tp/(actual+predicted) if actual+predicted else None)
    return result


def compare(before, after):
    a,b = indexed(before),indexed(after)
    if a.keys() != b.keys(): raise ValueError('paired_members_differ')
    losses=[]; gains=[]; old=defaultdict(list); new=defaultdict(list)
    exact={'before':0,'after':0,'complete_cases':0};stationary={'before_far_errors':0,'after_far_errors':0,'endpoints':0}
    for key,x in a.items():
        y=b[key]; r=x['record']
        # Includes current observation, input identity, target mask and original labels.
        if r != y['record']: raise ValueError('paired_record_or_truth_changed')
        if all(t['valid'] for t in r['targets']):
            exact['complete_cases']+=1
            for name,case in (('before',x),('after',y)):
                exact[name]+=all(t['token_id']==p for t,p in zip(r['targets'],case['prediction']['token_ids']))
        for h,t in enumerate(r['targets']):
            if not t['valid']: continue
            if type(t['token_id']) is not int or not 0 <= t['token_id'] < 9:
                raise ValueError('hard_target_required')
            gt=t['token_id']; p=x['prediction']['token_ids'][h]; q=y['prediction']['token_ids'][h]
            case=dict(id=key,event=r['event'],horizon_s=(h+1)*.5,truth=gt,before=p,after=q)
            if p==gt and q!=gt: losses.append(case)
            if p!=gt and q==gt: gains.append(case)
            if r['observation']['speed_mps']<=.2:
                stationary['endpoints']+=1
                stationary['before_far_errors']+=p%3==2 and gt%3!=2
                stationary['after_far_errors']+=q%3==2 and gt%3!=2
            for cell in ('all',f'event/{r["event"]}',f'horizon/{h}',f'event_horizon/{r["event"]}/{h}',
                         f'group/{r["physical_group"]}',f'event_class/{r["event"]}/{gt}'):
                old[cell].append((gt,p)); new[cell].append((gt,q))
    report={}; declines=[]
    for cell, rows in old.items():
        sa,sb=scores(rows),scores(new[cell])
        aa=sum(y==p for y,p in rows)/len(rows); ab=sum(y==p for y,p in new[cell])/len(rows)
        report[cell]=dict(n=len(rows),before_accuracy=aa,after_accuracy=ab,before=sa,after=sb)
        for c in sa:
            for metric in ('precision','recall','f1'):
                va,vb=sa[c][metric],sb[c][metric]
                # Removing the last false positive makes precision/F1 0 -> N/A,
                # not worse. Positive -> N/A remains a review, and supported
                # recall/F1 plus every old-correct loss are checked separately.
                if va is not None and ((vb is None and va>0) or
                                       (vb is not None and vb < va-1e-12)):
                    declines.append(dict(cell=cell,category=c,metric=metric,before=va,after=vb))
    groups=[v for k,v in report.items() if k.startswith('group/')]
    macro={side:sum(v[side+'_accuracy'] for v in groups)/len(groups) for side in ('before','after')}
    return dict(status='tradeoff_review_required' if losses or declines or macro['after']<macro['before']
                else 'observed_cohort_protection_passed', old_correct_losses=losses,new_correct=gains,
                metric_declines=declines,group_macro_accuracy=macro,strata=report,
                all_four_correct=exact,stationary_far_errors=stationary,
                unknown_scenes_guaranteed=False,independent_holdout_claim=False)
