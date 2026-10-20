"""Retain exact native event/action anchor plan; audit new target distribution separately."""
from collections import Counter, defaultdict
import random
from qwen3vl_local.audit_joint.phase3_base_dense import baseline
from qwen3vl_local.audit_joint.phase3_base_dense.data import Pool, ACTIONS
from .dataset import identity
from qwen3vl_local.sft_new_loop_phase3 import train


def plan(rows, records, args, *, epoch, world_size=4, history=None, strict=True):
    candidates = [r for r in rows if r.split=='train' and Pool.action(r) is not None]
    work, report = baseline.independent_work(candidates,args,epoch=epoch,world_size=world_size,history=history)
    native_ids = report['ordered_ids']
    quotas = Counter((x.row.context_id,Pool.action(x.row)) for x in work)
    buckets = defaultdict(list)
    for row in candidates:
        rec = records.get(identity(row))
        if rec and any(w['valid'] for w in rec['windows']):
            buckets[row.context_id,Pool.action(row)].append(train._make_item(row,seed=report['seed'],action_output_mode='choice'))
    rng = random.Random(f'single_frame_native_quotas:{report["seed"]}')
    selected = []
    for key, count in sorted(quotas.items()):
        if buckets[key]:
            selected.extend(train.route_diverse_sample(buckets[key],target=count,rng=rng))
    rng.shuffle(selected)
    work = selected
    report.update(native_anchor_plan_sha256=baseline.digest(native_ids),
                  ordered_ids=[identity(x.row) for x in work],
                  selected_cases=len(work),unique_cases=len({identity(x.row) for x in work}),
                  event_counts=dict(Counter(x.row.context_id for x in work)),
                  action_counts=dict(Counter(Pool.action(x.row) for x in work)),
                  event_action_counts=dict(Counter(x.row.context_id+'/'+Pool.action(x.row) for x in work)),
                  eligibility='at least one complete future pair; reselect within frozen native event/action quotas')
    raw, weighted, windows, repeats, point_repeats = Counter(), Counter(), Counter(), Counter(), Counter()
    missing, empty = [], []
    support_before = {(r.context_id,Pool.action(r)) for r in candidates}
    support_after = set()
    for item in work:
        key = identity(item.row)
        if key not in records:
            missing.append(key)
            continue
        r = records[key]
        valid = [w for w in r['windows'] if w['valid']]
        if not valid:
            empty.append(key)
            continue
        support_after.add((r['event'],r['anchor_action']))
        for i, w in enumerate(r['windows']):
            if not w['valid']:
                continue
            windows[f'{r["event"]}/{i}'] += 1
            repeats[r['event'],tuple(w['source_ids'])] += 1
            for source in w['source_ids']:point_repeats[r['event'],source] += 1
            for a,q,qp in zip(ACTIONS,w['raw'],w['target']):
                raw[r['event']+'/'+a] += q/len(valid)
                weighted[r['event']+'/'+a] += qp/len(valid)
    errors = []
    if Counter((x.row.context_id,Pool.action(x.row)) for x in work)!=quotas:
        errors.append('Native event/action quotas cannot be filled from eligible anchors')
    if missing: errors.append(f'{len(missing)} selected anchors absent from dataset')
    if empty: errors.append(f'{len(empty)} selected anchor presentations have no complete future window')
    lost = sorted(support_before-support_after)
    if lost: errors.append('Action support lost after target admission: '+str(lost))
    report.update(raw_future_loss_mass=dict(raw), preferred_future_loss_mass=dict(weighted),
        planned_query_exposure=dict(windows), planned_supervised_windows=sum(windows.values()),
        physical_window_exposure_histogram=dict(Counter(repeats.values())),
        distinct_physical_window_pairs=len(repeats), zero_label_presentations=len(empty),
        physical_origin_exposure_histogram=dict(Counter(point_repeats.values())),
        distinct_physical_origins=len(point_repeats),
        lost_anchor_action_support=lost, errors=errors,
        loss_normalization='mean over complete windows per anchor, then mean across ranks',
        future_action_distribution_not_guaranteed_equal=True,
        overlap_policy='native independent sampling retained; repeats counted, not independent evidence',
        labels_or_quotas_reallocated=False, training_anchor_members_may_change=True)
    if strict and errors:
        raise ValueError('Single-frame training admission: '+'; '.join(errors))
    return work, report


def evaluation_coverage(records):
    totals, valid, unique = Counter(), Counter(), defaultdict(set)
    for r in records:
        for i,w in enumerate(r['windows']):
            cell = f'{r["event"]}/{i}'
            totals[cell] += 1
            valid[cell] += w['valid']
            if w['valid']: unique[cell].add(tuple(w['source_ids']))
    return dict(requested=dict(totals),valid=dict(valid),
        coverage={k:valid[k]/n for k,n in totals.items()},
        distinct_targets={k:len(unique[k]) for k in totals},
        empty_cells=[k for k in totals if not valid[k]],
        interpretation='Coverage only; no predictions and no historical accuracy comparison')
