"""Paired complete-cohort accounting; a repaired error never cancels a hidden regression."""
from collections import Counter,defaultdict
from .frozen.vocabulary import ACTIONS
from .frozen.context_taxonomy import CONTEXT_BY_ID

def indexed(rows):
    out={}
    for r in rows:
        if r['id'] in out:raise ValueError('duplicate_member')
        ids=r['token_ids']
        if len(ids)!=4 or any(type(x) is not int or not 0<=x<6 for x in ids):raise ValueError('invalid_four_predictions')
        out[r['id']]=r
    return out

def compare(records,original,candidate):
    left,right=indexed(original),indexed(candidate)
    if len({r['id'] for r in records})!=len(records):raise ValueError('duplicate_reference')
    if set(left)!=set(right) or set(left)!={r['id'] for r in records}:raise ValueError('paired_membership_mismatch')
    counts=defaultdict(Counter);losses=[];gains=[];cm=[[[0]*6 for _ in range(6)] for __ in range(2)];trans=defaultdict(Counter)
    for r in records:
        ps=[left[r['id']]['token_ids'],right[r['id']]['token_ids']]
        allowed=[i for i,a in enumerate(ACTIONS) if a in (*CONTEXT_BY_ID[r['event']].action_keys,'KEEP')]
        if any(p not in allowed for ids in ps for p in ids):raise ValueError('prediction_outside_event_domain')
        hist=r['observation']['history'][-1];h=hist['token_id'] if hist['valid'] else None;previous_truth=h
        for i,t in enumerate(r['targets']):
            if not t['valid']:previous_truth=None;continue
            q=t.get('raw') or t['target'];a,b=ps[0][i],ps[1][i];pure=t['target'].count(1.)==1;y=t['target'].index(1.) if pure else None
            hit_a,hit_b=q[a]>0,q[b]>0
            keys=['allowed/all','allowed/event/'+r['event']]
            if pure:keys+=['pure/all','pure/event/'+r['event'],'pure/class/'+ACTIONS[y],f'pure/horizon/{i}',
                'pure/event_class/'+r['event']+'/'+ACTIONS[y],'pure/group/'+r['physical_group'],
                'pure/history/'+('unknown' if h is None else 'matches' if h==y else 'conflicts'),
                'pure/speed/'+('stationary' if r['observation']['speed_mps']<.1 else 'moving')]
            else:keys+=['mixed/all']
            for k in keys:
                z=counts[k];z['n']+=1;z['old_correct']+=hit_a;z['new_correct']+=hit_b;z['loss']+=hit_a and not hit_b;z['gain']+=hit_b and not hit_a
            # Event restriction can hide raw vocabulary/format deterioration.
            # Report it only when both forwards actually supplied the diagnostic.
            diagnostics=[left[r['id']].get('prediction',{}),right[r['id']].get('prediction',{})]
            for field in ('format_valid','letter_format_valid','raw_class_ids','unrestricted_token_ids'):
                if not all(field in d for d in diagnostics):continue
                values=[d[field][i] for d in diagnostics]
                hits=(values if field.endswith('valid') else [0<=v<6 and q[v]>0 for v in values])
                z=counts['diagnostic/'+field];z['n']+=1;z['old_correct']+=bool(hits[0]);z['new_correct']+=bool(hits[1])
                z['loss']+=bool(hits[0]) and not hits[1];z['gain']+=bool(hits[1]) and not hits[0]
            detail=dict(id=r['id'],query=i,event=r['event'],group=r['physical_group'],truth=t['name'],original=a,candidate=b,pure=pure)
            if hit_a and not hit_b:losses.append(detail)
            if hit_b and not hit_a:gains.append(detail)
            if pure:
                cm[0][y][a]+=1;cm[1][y][b]+=1
                if previous_truth is not None:
                    k='history_to_future' if i==0 else 'adjacent_future';z=trans[k];changed=previous_truth!=y;z['true_change' if changed else 'true_stable']+=1
                    for j,label in enumerate(('old','new')):
                        prev=h if i==0 else ps[j][i-1];detected=prev!=ps[j][i]
                        z[label+('_detected' if changed else '_false_change')]+=detected
                        if changed:z[label+'_correct_change']+=prev==previous_truth and ps[j][i]==y
            previous_truth=y
    classes=[]
    for i,a in enumerate(ACTIONS):
        row=dict(action=a,support=sum(cm[0][i]))
        for j,label in enumerate(('old','new')):
            tp=cm[j][i][i];pred=sum(v[i] for v in cm[j]);n=row['support']
            row[label+'_recall']=tp/n if n else None;row[label+'_precision']=tp/pred if pred else None
            row[label+'_f1']=2*tp/(n+pred) if n+pred else None
        classes.append(row)
    declines=[k for k,v in counts.items() if v['new_correct']<v['old_correct']]
    for r in classes:
        for field in ('recall','precision','f1'):
            a,b=r['old_'+field],r['new_'+field]
            if a is not None and (b is None or b<a):declines.append(r['action']+'/'+field)
    summary={}
    for label in ('old','new'):
        supported=sum(bool(r['support']) for r in classes)
        summary[label]={field:sum(r[label+'_'+field] or 0 for r in classes if r['support'])/supported if supported else None for field in ('recall','f1')}
        group=[v for k,v in counts.items() if k.startswith('pure/group/')]
        summary[label]['group_macro_accuracy']=sum(v[label+'_correct']/v['n'] for v in group)/len(group) if group else None
    return dict(status='tradeoff_review_required' if losses or declines else 'no_detected_regression',
        counts=dict(counts),classes=classes,summary=summary,transitions=dict(trans),lost_correct=losses,repaired=gains,
        declining_slices=declines,automatic_training_admission=False,
        independent_holdout_claim=False,warning='Repeated development review; no causal attribution without input interventions; no precision claim for missing support.')
