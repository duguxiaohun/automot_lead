"""Offline paired prediction review for a frozen base validation cohort.

This is a conservative regression screen, not training admission or inference
provenance validation. Never loads a model, changes labels, or selects test cases.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ACTIONS=('DECELERATE','STOP','RESUME','LANE_CHANGE_LEFT','LANE_CHANGE_RIGHT','KEEP')

def read_predictions(path):
    """Accept replay's JSON array and the existing JSONL interchange format."""
    text=Path(path).read_text(encoding='utf-8')
    rows=json.loads(text) if text.lstrip().startswith('[') else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows,list) or any(not isinstance(r,dict) for r in rows):
        raise ValueError('predictions must be a JSON array or JSONL objects')
    return rows

def indexed(rows):
    out={}
    for r in rows:
        k=tuple(r['key'])
        if len(k)!=5 or k in out:raise ValueError('invalid or duplicate case key')
        out[k]=r
    return out

def score(rows, pred):
    n=len(rows); correct=sum(pred[tuple(r['key'])]==r['gt'] for r in rows)
    actions={}
    for a in ACTIONS:
        support=sum(r['gt']==a for r in rows)
        predicted=sum(pred[tuple(r['key'])]==a for r in rows)
        tp=sum(r['gt']==a and pred[tuple(r['key'])]==a for r in rows)
        actions[a]=dict(support=support,correct=tp,predicted=predicted,
                       recall=tp/support if support else None,
                       precision=tp/predicted if predicted else 0.,
                       f1=2*tp/(support+predicted) if support+predicted else None)
    def macro(field):
        values=[v[field] for v in actions.values() if v['support'] and v[field] is not None]
        return sum(values)/len(values) if values else None
    return dict(n=n,correct=correct,accuracy=correct/n if n else None,
                format_valid=sum(pred[tuple(r['key'])] in ACTIONS for r in rows),
                macro_recall=macro('recall'),macro_f1=macro('f1'),actions=actions)

def compare(reference, predictions, reference_sha256):
    # The cohort is the entire saved validation set, not a user-selected error subset.
    rows=[r for r in reference if r['key'][0]=='val']
    if not rows:raise ValueError('empty validation cohort')
    ref=indexed(rows); new=indexed(predictions)
    if set(ref)!=set(new):raise ValueError('candidate must contain exactly every validation key')
    if any(r.get('reference_sha256')!=reference_sha256 for r in predictions):
        raise ValueError('reference SHA mismatch')
    for r in rows:
        if r['gt'] not in ACTIONS:raise ValueError('invalid frozen ground truth')
        if r.get('correct') != (r['pred']==r['gt']):raise ValueError('inconsistent reference correctness')
    for k,r in new.items():
        if 'gt' in r and r['gt']!=ref[k]['gt']:raise ValueError('candidate attempted label change')
        if not isinstance(r['raw_output'],str):raise ValueError('prediction must be text')
    old={k:r['pred'] for k,r in ref.items()};pred={k:r['raw_output'].strip() for k,r in new.items()}
    buckets=defaultdict(list)
    for r in rows:
        buckets['ALL'].append(r)
        buckets['action/'+r['gt']].append(r)
        buckets['event/'+r['key'][4]].append(r)
        buckets['event_action/'+r['key'][4]+'/'+r['gt']].append(r)
        buckets['route_run/'+r['key'][1]+'/'+r['key'][2]].append(r)
        for f in r.get('features',[]):buckets['semantic/'+f].append(r)
    metrics={};regressions=[]
    for name,subset in buckets.items():
        before=score(subset,old);after=score(subset,pred)
        fixed=[r['key'] for r in subset if old[tuple(r['key'])]!=r['gt'] and pred[tuple(r['key'])]==r['gt']]
        lost=[r['key'] for r in subset if old[tuple(r['key'])]==r['gt'] and pred[tuple(r['key'])]!=r['gt']]
        metrics[name]=dict(before=before,after=after,delta_correct=after['correct']-before['correct'],fixed=fixed,lost=lost)
        if after['correct']<before['correct']:regressions.append(name)
        # A false-positive shift can hurt one event without changing any correct
        # answers, and be cancelled by the opposite shift in another event.
        if name.startswith('event/'):
            for action in ACTIONS:
                for field in ('precision','f1'):
                    a=before['actions'][action][field];b=after['actions'][action][field]
                    if a is not None and b is not None and b<a:
                        regressions.append(name+'/action_metric/'+action+'/'+field)
            for field in ('macro_recall','macro_f1','format_valid'):
                if after[field]<before[field]:regressions.append(name+'/'+field)
    before=metrics['ALL']['before'];after=metrics['ALL']['after']
    for action in ACTIONS:
        for field in ('precision', 'f1'):
            a=before['actions'][action][field];b=after['actions'][action][field]
            if a is not None and b is not None and b<a:
                regressions.append('action_metric/'+action+'/'+field)
    for name in ['macro_recall','macro_f1','format_valid']:
        if after[name]<before[name]:regressions.append(name)
    macro={}
    for prefix in ['event/','route_run/']:
        selected=[v for k,v in metrics.items() if k.startswith(prefix)]
        macro[prefix.rstrip('/')]=dict(before=sum(v['before']['accuracy'] for v in selected)/len(selected),after=sum(v['after']['accuracy'] for v in selected)/len(selected),groups=len(selected))
    for name,values in macro.items():
        if values['after']<values['before']:regressions.append('group_macro/'+name)
    losses=metrics['ALL']['lost']
    return dict(status='regression_detected' if regressions else ('tradeoff_review_required' if losses else 'no_regression_observed'),
                review_required=bool(regressions or losses),
                previously_correct_lost=losses,
                loss_slices=[name for name,value in metrics.items() if value['lost']],
                reference_sha256=reference_sha256,scope='entire frozen validation cohort only',
                regression_slices=regressions,group_macro=macro,metrics=metrics,
                promotion_authorized=False,
                limitations=['Prediction scoring only; SHA binding does not attest actual inference inputs.',
                             'Any observed slice decline triggers review; small correlated groups do not imply statistical significance.',
                             'Validation is development-exposed; unchanged labels and paired holdout evaluation are still required.',
                             'A route run is not necessarily an independent physical scenario group.'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();data=args.reference.read_bytes();h=hashlib.sha256(data).hexdigest()
    result=compare(json.loads(data),read_predictions(args.predictions),h)
    # Exclusive output creation preserves prior candidate reviews.
    with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(result['status'])
    if result['review_required']:raise SystemExit(2)

if __name__=='__main__':main()
