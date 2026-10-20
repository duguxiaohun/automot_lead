"""Frozen evaluation anchors; matched readout is primary, four-RGB transfer secondary."""
from collections import Counter, defaultdict
from . import baseline, planning
from .data import Pool
from qwen3vl_local.sft_new_loop_phase3 import train
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity
from qwen3vl_local.audit_joint import phase3_base_trial as trial


def selected(pool,split,kind='generation'):
    ref=trial.reference()
    spec=ref['test_plan'] if split=='test' else ref['validation'][kind]
    work=[train._make_item(r,seed=spec['seed'],action_output_mode='choice') for r in pool.anchor_rows if r.split==split]
    work=[x for x in work if train.choice_rejection_reason(x.spec) is None]
    result,_=trial.select_reference(work,ref,seed=spec['seed'],limit=0 if split=='test' else spec['per_event'],split=split)
    return result


def split_plan(job,pool,split):
    from .runtime import common_record
    from pathlib import Path
    reasons=Counter();identities=defaultdict(set)
    work=selected(pool,split);records={};support=Counter();totals=Counter();ids=[];matched=[]
    count=len(planning.options(job)['offset_frames']) if job['mode']=='future' else 5
    primary=0 if job['mode']=='future' else 4
    for item in work:
        identity=baseline.digest(case_identity(item));ids.append(identity);event=item.row.context_id
        totals[event]+=1
        common=common_record(item.row,job['mode'])
        records[f'eval/{split}/transfer/{identity}']=dict(common,usage='evaluation')
        rec=pool.record(item.row,job['mode'])
        if rec is None:continue
        records[f'eval/{split}/matched/{identity}']=dict(rec,usage='evaluation')
        for i,action in enumerate(rec['actions']):
            if action is not None:
                support[event,i]+=1;identities[f'{event}/{i}'].add(rec['source_ids'][i])
            elif rec.get("target_reasons"):reasons[f"{event}/{i}/"+rec["target_reasons"][i]]+=1
        if rec['actions'][primary] is not None:matched.append(identity)
    threshold=job.get('min_eval_coverage',.8)
    coverage={f'{event}/{i}':support[event,i]/n for event,n in totals.items() for i in range(count)}
    bad={key:value for key,value in coverage.items() if value<threshold}
    if job.get('forecast_protocol')=='trajectory':
        from .trajectory_plan import admission
        admitted=admission({f'{e}/{i}':n for e,n in totals.items() for i in range(count)},
            Counter({f'{e}/{i}':n for (e,i),n in support.items()}),identities,planning.options(job),threshold)
        bad=admitted['bad']
        trial.write(Path(job['output'])/f'trajectory_eval_coverage_{split}.json',dict(admitted,missing_reasons=dict(reasons)))
    if bad:raise ValueError(f'{split} fixed-anchor evaluation coverage below per-event/query threshold: {bad}')
    report=dict(split=split,ordered_anchor_ids=ids,primary_query=primary,matched_anchor_ids=matched,
        matched_membership_sha256=baseline.digest(sorted(matched)),coverage=coverage,
        min_eval_coverage=threshold,support={f'{e}/{i}':n for (e,i),n in support.items()},
        event_anchor_counts=dict(totals),primary='same input protocol and anchor readout as single/all training',
        secondary='four-image single-query transfer; not the matched dense-supervision comparison')
    if job.get('forecast_protocol')=='trajectory':
        report['trajectory_admission']=admitted
        report['min_eval_coverage_applies']=job.get('trajectory_coverage_policy','masked')=='strict'
    report['input_plan_sha256']=baseline.digest(records)
    return report,records


def build(job,pool):
    from .runtime import common_record
    reports={};records={}
    # Both splits are admitted before any training, including a quick run. No
    # test predictions or scores are computed by this coverage/shape admission.
    for split in ('val','test'):
        reports[split],items=split_plan(job,pool,split);records.update(items)
    loss_ids=[]
    for item in selected(pool,'val','loss'):
        identity=baseline.digest(case_identity(item));loss_ids.append(identity)
        records[f'eval/val/loss/{identity}']=dict(common_record(item.row,job['mode']),usage='evaluation')
    reports['native_loss_anchor_ids']=loss_ids
    reports['status']='coverage_admitted_no_predictions'
    return reports,records


def score_matched(cases,primary_query):
    supported=[r for r in cases if r['targets'][primary_query] is not None]
    correct=sum(r['predictions'][primary_query]['action']==r['targets'][primary_query] for r in supported)
    constrained=sum(r['predictions'][primary_query]['constrained_action']==r['targets'][primary_query] for r in supported)
    if not supported:raise ValueError('No supported matched evaluation anchors')
    return dict(total=len(supported),correct=correct,exact=correct/len(supported),
        constrained_correct=constrained,constrained_exact=constrained/len(supported),query=primary_query)
