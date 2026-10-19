"""Independent anchors, retained action support, and audited query loss mass."""
from collections import Counter, defaultdict
import math
import random
from . import baseline
from .data import Pool
from qwen3vl_local.sft_new_loop_phase3 import train
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity
from qwen3vl_local.sft_new_loop_phase3.sampling import route_diverse_sample


def options(job):
    from qwen3vl_local.sft_new_loop_phase3 import trajectory_action as rule
    dt=rule.FRAME_DT_SECONDS
    horizons=[rule.IMMEDIATE_HORIZON_FRAMES,rule.LONGITUDINAL_HORIZON_FRAMES,rule.LATERAL_HORIZON_FRAMES]
    step=job.get('forecast_step_seconds','auto')
    step_frames=max(horizons) if step=='auto' else float(step)/dt
    span_frames=float(job.get('forecast_span_seconds',6))/dt
    if (not math.isfinite(step_frames) or not math.isfinite(span_frames)
            or step_frames<1 or step_frames!=int(step_frames) or span_frames!=int(span_frames)
            or span_frames<step_frames or span_frames%step_frames or span_frames/step_frames>7):
        raise ValueError('Forecast span/step must align to 0.25s, span divisible by step, 2..8 query positions')
    coverage=float(job.get('min_future_coverage',.8))
    if not .5<=coverage<=1:
        raise ValueError('min-future-coverage must be between 0.5 and 1')
    supervision=job.get('supervision','all')
    if supervision not in ('all','single'):
        raise ValueError('supervision must be all or single')
    return dict(offset_frames=list(range(0,int(span_frames)+1,int(step_frames))),
        step_source='maximum_native_rule_horizon' if step=='auto' else 'explicit',
        native_horizon_frames=horizons,frame_dt_seconds=dt,
        semantics='native upcoming-window labels at fixed anchors, not execution-duration tokens',
        overlapping_native_windows=step_frames<max(horizons),min_future_coverage=coverage,
        supervision=supervision)


def active_indices(record,supervision):
    if supervision=='all':return list(range(len(record['actions'])))
    return [0 if record['mode']=='future' else len(record['actions'])-1]


def make_plan(pool,args,job,epoch,world_size=4,history=None):
    opts=options(job)
    original=[r for r in pool.anchor_rows if r.split=='train' and Pool.action(r) is not None]
    reference,ref_audit=baseline.independent_work(original,args,epoch=epoch,world_size=world_size,history=history)
    quotas=Counter((x.row.context_id,Pool.action(x.row)) for x in reference)
    eligible,coverage=pool.eligible(job['mode'],'train')
    groups=defaultdict(list)
    for r in eligible:
        groups[(r.context_id,Pool.action(r))].append(r)
    supported={(r.context_id,Pool.action(r)) for r in original}
    lost=sorted(f'{e}/{a}' for e,a in supported if not groups[e,a])
    if lost:raise ValueError('Dense filtering removed originally supported actions: '+', '.join(lost))
    seed=int(args.seed)+epoch*1000003
    rng=random.Random(f'{seed}:independent_dense_anchors')
    work=[]
    for cell,n in sorted(quotas.items()):
        items=[train._make_item(r,seed=seed,action_output_mode='choice') for r in groups[cell]]
        rng.shuffle(items)
        work.extend(route_diverse_sample(items,target=n,rng=rng))
    rng.shuffle(work)
    if len(work)%world_size:raise ValueError('Incomplete independent global batch')
    records={};counts=Counter();available=Counter();missing=Counter();query_support=defaultdict(set);raw_queries=Counter();supervisable=Counter()
    for x in work:
        rec=pool.record(x.row,job['mode']);records[rec['id']]=rec
        active=set(active_indices(rec,opts['supervision']))
        for i,a in enumerate(rec['actions']):
            raw_queries[f'{x.row.context_id}/{i}/{a or "MISSING"}']+=1
            available[x.row.context_id,i]+=1
            if a is not None and quotas.get((x.row.context_id,a),0)>0:supervisable[x.row.context_id,i]+=1
            if a is None:missing[x.row.context_id,i]+=1
            elif i in active:
                counts[x.row.context_id,a]+=1
                query_support[x.row.context_id,a].add(rec['source_ids'][i])
    # Gate each event/time position, including a single-query control, so its
    # anchor/input population remains exactly paired with the dense objective.
    if job['mode']=='future':
        bad={f'{e}/{i}':1-missing[e,i]/n for (e,i),n in available.items()
             if i and 1-missing[e,i]/n<opts['min_future_coverage']}
        if bad:raise ValueError('Future supervision coverage below per-event/per-position threshold: '+str(bad))
    effective_bad={f'{e}/{i}':supervisable[e,i]/n for (e,i),n in available.items()
                   if supervisable[e,i]/n<opts['min_future_coverage']}
    if effective_bad:
        raise ValueError('Matched-action effective query coverage below per-event/per-position threshold: '+str(effective_bad))
    # Both arms use the SAME original anchor action quotas. Never recompute
    # target mass from the queries enabled by this arm. Auxiliary-only actions
    # have zero mass in BOTH arms and are explicitly reported, not redistributed.
    query_quotas=dict(quotas);weights={}
    for cell,n in query_quotas.items():
        if n and not counts[cell]:raise ValueError('Matched action mass has no active query support')
        weights[cell]=n/counts[cell] if counts[cell] else 0.
    planned_queries=Counter();mass=Counter()
    for rec in records.values():
        active=set(active_indices(rec,opts['supervision']))
        rec['query_weights']=[weights.get((rec['event'],a),0.) if i in active and a is not None else 0.
                              for i,a in enumerate(rec['actions'])]
    for x in work:
        rec=records[baseline.digest(case_identity(x))]
        for i,(a,w) in enumerate(zip(rec['actions'],rec['query_weights'])):
            if w>0:planned_queries[f'{rec["event"]}/{i}/{a}']+=1;mass[f'{rec["event"]}/{a}']+=w
    expected_mass={f'{e}/{a}':n for (e,a),n in query_quotas.items()}
    if any(not math.isclose(mass[k],v,rel_tol=1e-10,abs_tol=1e-8) for k,v in expected_mass.items()):
        raise AssertionError('Query loss mass does not satisfy event/action quotas')
    audit=dict(ref_audit,ordered_ids=[baseline.digest(case_identity(x)) for x in work],
        coverage=coverage,options=opts,
        anchor_action_quotas={f'{e}/{a}':n for (e,a),n in quotas.items()},
        planned_query_exposure=dict(planned_queries),planned_query_loss_mass=expected_mass,
        planned_available_query_labels=dict(raw_queries),
        query_weight_by_event_action={f'{e}/{a}':w for (e,a),w in weights.items()},
        action_mass_policy='frozen_original_anchor_quotas_shared_by_single_and_all',
        zero_mass_query_actions=sorted(f'{e}/{a}' for e,a in counts if not query_quotas.get((e,a),0)),
        unique_active_query_targets={f'{e}/{a}':len(ids) for (e,a),ids in query_support.items()},
        planned_supervised_queries=sum(planned_queries.values()),
        anchor_unique=len(records),unique_cases=len(records),phase='plan_not_consumption',
        loss_normalization='sum of quota-weighted query CE per anchor; average across ranks')
    audit['input_plan_sha256']=baseline.digest([dict(id=baseline.digest(case_identity(x)),
        images=records[baseline.digest(case_identity(x))]['images'],
        premises=records[baseline.digest(case_identity(x))]['premises'],
        offsets=records[baseline.digest(case_identity(x))]['offsets_seconds']) for x in work])
    audit['future_label_coverage']={f'{e}/{i}':1-missing[e,i]/n for (e,i),n in available.items()}
    audit['effective_query_coverage']={f'{e}/{i}':supervisable[e,i]/n for (e,i),n in available.items()}
    return work,audit,records
