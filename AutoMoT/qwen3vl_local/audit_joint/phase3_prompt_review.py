"""Explicit inference-only prompt intervention; frozen checkpoint sources stay intact."""
from contextlib import contextmanager
from collections import defaultdict
import copy
import inspect
from pathlib import Path

from . import phase3_behavior_diagnostics as diag

VARIANT = 'native_clarified_v1'
CLARIFICATION = (
    'Predict the recorded ego behavior, not a recommended safe action. '
    'The anchor event fixes the allowed action domain only; it does not imply that '
    'the current obstacle, traffic condition or waiting state persists at future origins. '
    'Reassess each future origin separately: a lane crossing completed at or before '
    'that origin is not a new upcoming crossing, and a past STOP or RESUME does not '
    'force the same answer at later origins. '
    'Use the stated native speed and crossing rules without replacing them with '
    'a preferred maneuver or a fixed event-specific action sequence. '
)


def contract(objective):
    recipe = dict(variant=VARIANT,original_system=objective.INSTRUCTION,
                  original_message_builder=inspect.getsource(objective.messages),
                  prepend_system=CLARIFICATION)
    return dict(variant=VARIANT,prompt_sha256=diag.digest(recipe),recipe=recipe,
                training_performed=False,inference_only=True,
                labels_changed=False,history_changed=False,image_changed=False,
                checkpoint_prompt_is_original=True,
                efficacy='unverified until paired validation results are received')


@contextmanager
def prompt_scope(objective):
    """Restore the original function even on packing/inference failure."""
    original=objective.messages
    def messages(*args,**kwargs):
        result=original(*args,**kwargs)
        # Copy only the modified message, retaining image objects by identity.
        return [dict(result[0],content=CLARIFICATION+result[0]['content']),*result[1:]]
    objective.messages=messages
    try:
        yield
    finally:
        objective.messages=original


def predict(bundle, records, ids, *, predictor=None):
    from .phase3_single_frame import objective,runtime
    if predictor is None:predictor=runtime.predict
    work=[records[i] for i in sorted(ids)]
    before=copy.deepcopy(work)
    with prompt_scope(objective):
        cases,_=predictor(bundle,work)
    if work!=before:raise ValueError('Prompt inference mutated admitted records')
    if [c['id'] for c in cases]!=[r['id'] for r in work]:
        raise ValueError('Prompt prediction membership/order changed')
    for c,r in zip(cases,work):
        if (c['windows']!=r['windows'] or c['history']!=r['history'] or
                c['history_input']!=r['history'] or c['image_mode']!='actual'):
            raise ValueError('Prompt intervention changed labels/history/image mode')
    return cases


def merge(plans,shards):
    from .phase3_matched_image_eval import receiver_shard
    if not shards or sorted(s['rank'] for s in shards)!=list(range(len(shards))):
        raise ValueError('Missing/duplicate prompt ranks')
    result={}
    for shard in shards:
        if shard['world']!=len(shards) or shard['variant']!=VARIANT:
            raise ValueError('Prompt shard world/variant changed')
        cases=shard['cases'];one={c['id']:c for c in cases}
        if len(one)!=len(cases) or set(one)!=receiver_shard(plans,shard['rank'],len(shards)):
            raise ValueError('Missing/duplicate/misplaced prompt cases')
        if result.keys() & one.keys():raise ValueError('Overlapping prompt shards')
        result.update(one)
    return result


def transition_scores(reference, alternative, metadata):
    """Paired target transitions, not claims about executed maneuver timing.

    Mixed windows keep their entire acceptable set. No UNKNOWN or missing
    interval is silently forward-filled. History comparisons explicitly name
    the latest known slot; adjacent comparisons require two pure valid targets.
    """
    diag.paired_scores(reference, alternative, metadata)  # shared integrity checks
    buckets = defaultdict(list)
    excluded = dict(invalid_future=0, no_known_history=0, adjacent_mixed_or_missing=0)
    for a,b in zip(reference,alternative):
        known = [(i,x) for i,x in enumerate(a['history']) if x in diag.ACTIONS]
        for pos,(w,p,q) in enumerate(zip(a['windows'],a['predictions'],b['predictions'])):
            if not w['valid']:
                excluded['invalid_future'] += 1
                continue
            target = w['allowed_answers']
            item = dict(id=a['id'],group=metadata[a['id']]['group'],event=a['event'],position=pos+1,
                        reference_hit=p in target,alternative_hit=q in target)
            if known:
                slot,source = known[-1]
                key = source+'->'+(target[0] if len(target)==1 else '{'+','.join(target)+'}')
                buckets[('history_to_future',key)].append(dict(item,history_slot=slot+1,
                    reference_repeats_history=p==source,alternative_repeats_history=q==source))
            else: excluded['no_known_history'] += 1
            if pos:
                prev = a['windows'][pos-1]
                if prev['valid'] and len(prev['allowed_answers'])==len(target)==1:
                    key = prev['allowed_answers'][0]+'->'+target[0]
                    buckets[('adjacent_future_targets',key)].append(item)
                else: excluded['adjacent_mixed_or_missing'] += 1
    def counts(rows):
        n=len(rows);x=sum(r['reference_hit'] for r in rows);y=sum(r['alternative_hit'] for r in rows)
        return dict(windows=n,anchors=len({r['id'] for r in rows}),physical_groups=len({r['group'] for r in rows}),
            reference_hits=x,alternative_hits=y,reference_accuracy=x/n,alternative_accuracy=y/n,
            reference_only=sum(r['reference_hit'] and not r['alternative_hit'] for r in rows),
            alternative_only=sum(r['alternative_hit'] and not r['reference_hit'] for r in rows),
            delta=(x-y)/n)
    result=dict(history_to_future={},adjacent_future_targets={},excluded=excluded,
        definition='History uses latest known slot, reported explicitly; mixed targets stay sets. Adjacent future targets require pure valid neighbors. Windows within an anchor/route are not independent.',
        delta_sign='positive means original is better')
    for (view,key),rows in sorted(buckets.items()):
        score=counts(rows)
        for field,name in [('position','positions'),('event','events'),('group','routes')]:
            score[name]={str(v):counts([r for r in rows if r[field]==v]) for v in sorted({r[field] for r in rows})}
        if view=='history_to_future':
            score['history_slots']={str(v):sum(r['history_slot']==v for r in rows) for v in sorted({r['history_slot'] for r in rows})}
            score['reference_repeats_history']=sum(r['reference_repeats_history'] for r in rows)
            score['alternative_repeats_history']=sum(r['alternative_repeats_history'] for r in rows)
        result[view][key]=score
    return result


def save(job,plans,output,cases,intervention):
    from . import phase3_base_trial as trial
    old=trial.read(Path(job['output'])/'single_frame_val_cases.json')
    old={c['id']:c for c in old}
    expected={p['receiver'] for plan in plans for p in plan['pairs']}
    if set(cases)!=expected:raise ValueError('Prompt result coverage changed')
    common=set.intersection(*({p['receiver'] for p in plan['pairs']} for plan in plans))
    metadata={r['id']:r for r in plans[0]['metadata']}
    result=dict(intervention=intervention,reference='saved original-prompt actual-image validation',
                alternative=VARIANT,reference_inferred_again=False,
                delta_sign='positive delta means ORIGINAL is better; negative means clarified is better',
                predictions_changed=sum(cases[i]['predictions']!=old[i]['predictions'] for i in expected))
    for name,ids in [('union',expected),('primary_common',common)]:
        keys=sorted(ids)
        a=[old[i] for i in keys];b=[cases[i] for i in keys]
        if any(x['history_input']!=y['history_input'] or y['image_mode']!='actual' for x,y in zip(a,b)):
            raise ValueError('Prompt comparison input changed')
        result[name]=dict(anchors=len(keys),**diag.paired_scores(a,b,metadata))
        result[name]['transitions']=transition_scores(a,b,metadata)
        result[name]['events']={e:diag.paired_scores([x for x in a if x['event']==e],
            [x for x in b if x['event']==e],metadata)['subsets'] for e in sorted({x['event'] for x in a})}
    trial.write(output/'prompt_cases.json',[cases[i] for i in sorted(cases)])
    trial.write(output/'metrics.json',result)
    return result
