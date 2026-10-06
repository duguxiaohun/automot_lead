"""Full-pool epochs with exact event equality and explicitly unbounded repetition."""
from collections import Counter,defaultdict
import math
import random
from .identity import digest
from .sampling import CapacityError,frame_key

POLICY='full_pool_event_equal_v1'


def validate_dataset_scope(manifest,pairing=None):
    if manifest.get('production_index_sha256'):
        selection=manifest.get('teacher_selection') or {}
        if selection.get('max_per_route')!=0 or selection.get('route_budget_excluded')!=0:
            raise ValueError('full training requires an uncapped teacher build: --max-teacher-questions-per-route 0')
        if not (manifest.get('production_coverage') or {}).get('complete_causal_production'):
            raise ValueError('full training requires complete eligible-route production')
    if pairing is not None and any(n!=pairing['pairs'] for n in pairing['original_train_counts'].values()):
        raise ValueError('pairing would discard valid training rows; full training requires complete pairing or no paired-with')


def plan(rows,*,epoch,seed=20260929,budget=None,world_size=1,expected_events=None,history=None):
    if (not rows or type(epoch) is not int or epoch<0 or type(world_size) is not int or world_size<1):
        raise ValueError('invalid full-pool sampling inputs')
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('duplicate full-pool question ID')
    pools=defaultdict(list)
    for i,r in enumerate(rows):
        if r['target'] not in ('YES','NO'):raise ValueError('full training requires binary supervision')
        pools[r['episode']['event']].append(i)
    events=sorted(pools)
    if expected_events is not None and set(events)!=set(expected_events):raise ValueError('full-pool event mismatch')
    unit=math.lcm(len(events),world_size)
    minimum=math.ceil(max(map(len,pools.values()))*len(events)/unit)*unit
    if budget is not None and (type(budget) is not int or budget<minimum or budget%unit):
        raise CapacityError(dict(policy=POLICY,requested=budget,minimum_full_pool_budget=minimum,
                                 divisible_by=unit,reason='every question at least once and exact event equality required'))
    presentations=minimum if budget is None else budget;quota=presentations//len(events)
    binding=dict(policy=POLICY,pool_sha256=digest(sorted(
        [r['id'],r['episode']['event'],r['target'],r.get('model_input_sha256'),frame_key(r)] for r in rows)),
        seed=seed,world_size=world_size,budget=budget,per_event_quota=quota)
    if history is None:
        history=dict(**binding,next_epoch=0,counts={i:0 for i in ids},total_presentations=0)
        for prior in range(epoch):
            _,audit=plan(rows,epoch=prior,seed=seed,budget=budget,world_size=world_size,
                         expected_events=expected_events,history=history)
            history=audit['next_history']
    if (any(history.get(k)!=v for k,v in binding.items()) or history.get('next_epoch')!=epoch
            or set(history.get('counts',{}))!=set(ids)):
        raise ValueError('full-pool resume history/config mismatch')
    counts=history['counts']
    if (any(type(n) is not int or n<epoch for n in counts.values())
            or sum(counts.values())!=epoch*presentations
            or history.get('total_presentations')!=epoch*presentations
            or any(sum(counts[ids[i]] for i in pools[e])!=epoch*quota for e in events)):
        raise ValueError('invalid full-pool exposure history')
    result=[]
    for e in events:
        ordered=sorted(pools[e],key=lambda i:(counts[ids[i]],digest([seed,epoch,ids[i]])))
        cycles,remainder=divmod(quota,len(ordered))
        result.extend(ordered*cycles+ordered[:remainder])
    random.Random(seed+epoch).shuffle(result)
    repeats=Counter(result);frames=Counter(frame_key(rows[i]) for i in result)
    updated={ident:counts[ident]+repeats[i] for i,ident in enumerate(ids)}
    return result,dict(policy=POLICY,epoch=epoch,seed=seed,world_size=world_size,
        presentations=presentations,per_event_quota=quota,event_counts={e:quota for e in events},
        unique_questions=len(repeats),unique_frames=len(frames),max_frame_repeat=max(frames.values()),
        max_question_repeat=max(repeats.values()),minimum_full_pool_budget=minimum,
        coverage_scope='every admitted training question each epoch; exact event presentation equality',
        repetition_policy='explicit full mode; small-pool question/frame repeat caps do not apply',
        event_repetition={e:dict(pool=len(pools[e]),presentations=quota,
            min_question_repeat=min(repeats[i] for i in pools[e]),max_question_repeat=max(repeats[i] for i in pools[e])) for e in events},
        cumulative_unique_questions=len(rows),cumulative_missing_questions=0,
        event_coverage={e:dict(pool=len(pools[e]),seen=len(pools[e])) for e in events},
        next_history=dict(**binding,next_epoch=epoch+1,counts=updated,total_presentations=(epoch+1)*presentations))
