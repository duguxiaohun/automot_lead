"""Frozen independent anchors; normalize repeated physical future targets per event."""
from collections import Counter, defaultdict
from pathlib import Path
import math
import os
from . import baseline
from .data import Pool
from qwen3vl_local.audit_joint import phase3_base_trial as trial
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity


def admission(available,valid,identities,opts,threshold=None):
    """Explicit sparse-target contract; distinct frames are NOT independent trials."""
    coverage={k:valid[k]/n for k,n in available.items()}
    density={k.split('/')[0]:sum(v for cell,v in valid.items()
             if cell.startswith(k.split('/')[0]+'/') and cell!=k)/n
             for k,n in available.items() if k.endswith('/0')}
    unique={k:len(identities[k]) for k in available if not k.endswith('/0')}
    if opts['trajectory_coverage_policy']=='strict':
        minimum=opts['min_future_coverage'] if threshold is None else threshold
        bad={k:f'coverage={v:g} < {minimum:g}' for k,v in coverage.items() if v<minimum}
    else:
        bad={k:'current native anchor must remain labeled' for k,v in coverage.items() if k.endswith('/0') and v!=1}
        bad.update({k:f'distinct physical targets={v} < 2' for k,v in unique.items() if v<2})
        bad.update({e+'/mean_future_queries':f'{v:g} < 1' for e,v in density.items() if v<1})
    return dict(policy=opts['trajectory_coverage_policy'],coverage=coverage,
        distinct_physical_targets=unique,mean_future_queries_per_anchor_by_event=density,
        masked_thresholds=dict(mean_future_queries_per_anchor=1,distinct_targets_per_event_position=2),
        caveat='Distinct timepoints on a route remain correlated; no statistical independence claim',bad=bad)


def make_plan(pool,args,job,epoch,world_size,history,opts):
    original=[r for r in pool.anchor_rows if r.split=='train' and Pool.action(r) is not None]
    work,audit=baseline.independent_work(original,args,epoch=epoch,world_size=world_size,history=history)
    records={};available=Counter();valid=Counter();reasons=Counter();exposure=Counter();events=Counter();quotas=Counter();identities=defaultdict(set)
    for item in work:
        rec=pool.record(item.row,'future');records[rec['id']]=rec
        events[rec['event']]+=1;quotas[rec['event'],rec['actions'][0]]+=1
        for i,(action,source) in enumerate(zip(rec['actions'],rec['source_ids'])):
            cell=f'{rec["event"]}/{i}';available[cell]+=1
            if action is not None:
                valid[cell]+=1;identities[cell].add(source)
                if i:exposure[rec['event'],source]+=1
            else:reasons[cell+'/'+rec['target_reasons'][i]]+=1
    result=admission(available,valid,identities,opts)
    coverage=result['coverage']
    report=dict(epoch=epoch,**result,missing_reasons=dict(reasons),
        anchor_selection='unchanged_independent_base_plan')
    if int(os.environ.get('RANK','0'))==0:
        trial.write(Path(job['output'])/f'trajectory_train_coverage_epoch{epoch}.json',report)
    if result['bad']:
        raise ValueError('Trajectory intent coverage admission failed; see trajectory_train_coverage report: '+str(result['bad']))
    unique=Counter(e for e,_ in exposure)
    for rec in records.values():
        # Half the event loss mass retains the original anchor action quotas.
        # Half is shared uniformly among distinct physical future targets in this
        # event, divided by their total exposure (including repeated anchors).
        rec['query_weights']=[.5]+[
            events[rec['event']]/(2*unique[rec['event']]*exposure[rec['event'],s]) if a is not None else 0.
            for a,s in zip(rec['actions'][1:],rec['source_ids'][1:])]
    counts=Counter();mass=Counter();target_mass=Counter()
    for item in work:
        rec=records[baseline.digest(case_identity(item))]
        for i,(a,w,s) in enumerate(zip(rec['actions'],rec['query_weights'],rec['source_ids'])):
            if w:
                counts[f'{rec["event"]}/{i}/{a}']+=1;mass[f'{rec["event"]}/{a}']+=w
                if i:target_mass[rec['event'],s]+=w
    event_mass=Counter()
    for key,value in mass.items():event_mass[key.split('/')[0]]+=value
    if any(not math.isclose(event_mass[e],n,rel_tol=1e-10) for e,n in events.items()):
        raise AssertionError('Trajectory loss event mass changed')
    audit.update(options=opts,coverage=report,future_label_coverage=coverage,effective_query_coverage=coverage,
        anchor_action_quotas={f'{e}/{a}':n for (e,a),n in quotas.items()},
        planned_query_exposure=dict(counts),planned_query_loss_mass=dict(mass),
        planned_supervised_queries=sum(counts.values()),anchor_unique=len(records),unique_cases=len(records),
        action_mass_policy='half_native_anchor_quotas_half_uniform_unique_physical_future_targets_per_event',
        action_mass_control='anchor-domain future actions can change total action mass; not a density-only ablation',
        future_physical_exposure_histogram=dict(Counter(exposure.values())),
        future_unique_targets_by_event=dict(unique),
        future_physical_target_loss_mass_sha256=baseline.digest(sorted((e,s,v) for (e,s),v in target_mass.items())),
        phase='plan_not_consumption',loss_normalization='one unit per anchor averaged across ranks',
        input_plan_sha256=baseline.digest([dict(id=baseline.digest(case_identity(x)),
            images=records[baseline.digest(case_identity(x))]['images'],
            premises=records[baseline.digest(case_identity(x))]['premises'],
            offsets=records[baseline.digest(case_identity(x))]['offsets_seconds']) for x in work]))
    return work,audit,records
