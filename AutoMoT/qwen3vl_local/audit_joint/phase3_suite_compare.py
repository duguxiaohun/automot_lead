"""Offline action-level comparison; no model inference or test-based selection."""
from collections import Counter
import json
from pathlib import Path
from . import phase3_base_trial as trial


def score(targets,predictions,ids):
    correct=0;events={};actions={};confusion=Counter()
    for identity in ids:
        target=targets[identity];prediction=predictions.get(identity);ok=prediction==target['action']
        correct+=ok;confusion[target['action']+'/'+(prediction or 'MISSING_OR_INVALID')]+=1
        for name,table in ((target['event'],events),(target['action'],actions)):
            c=table.setdefault(name,dict(total=0,correct=0));c['total']+=1;c['correct']+=ok
    for cell in [*events.values(),*actions.values()]:cell['accuracy']=cell['correct']/cell['total']
    return dict(total=len(ids),correct=correct,accuracy=correct/len(ids) if ids else None,
        by_event=events,by_action=actions,confusion=dict(confusion))


def summarize(targets,predictions):
    universe=set(targets)
    if any(not set(p)<=universe for p in predictions.values()):raise ValueError('Prediction outside frozen anchors')
    common=universe.intersection(*(set(p) for p in predictions.values()))
    return dict(fixed_anchor_total=len(universe),common_supported_total=len(common),
        arms={arm:dict(predicted_anchor_count=len(p),coverage=len(p)/len(universe),
            fixed_anchors_missing_counted_incorrect=score(targets,p,sorted(universe)),
            common_supported_anchors=score(targets,p,sorted(common))) for arm,p in predictions.items()},
        common_anchor_ids=sorted(common),
        metric='unconstrained current-action accuracy; raw action phrases for base, A-F token readouts for A/B',
        caveat='architecture/prompt/output and eligible training pool differ; not a pure density ablation')


def compare(jobs):
    from .phase3_base_dense import data,baseline,evaluation
    from qwen3vl_local.sft_new_loop_phase3 import train
    from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity
    first=jobs['base'];rows=[]
    for split in ('val','test'):rows.extend(train._read_rows(Path(first['index']),split,data_root=Path(first['data_root'])))
    pool=data.Pool(rows);report={}
    for split in ('val','test'):
        targets={};by_observation={}
        for item in evaluation.selected(pool,split):
            identity=baseline.digest(case_identity(item));r=item.row
            targets[identity]=dict(action=data.Pool.action(r),event=r.context_id)
            key=(r.scenario,r.route_id,r.frame_id,r.context_id,r.prompt_road_structure)
            if key in by_observation:raise ValueError('Ambiguous fixed observation')
            by_observation[key]=identity
        predictions={}
        for arm,job in jobs.items():
            root=Path(job['output']);p={}
            if arm=='base':
                paths=([root/'train/final_generation_val_cases.jsonl'] if split=='val'
                       else sorted((root/'test').glob('cases*.jsonl')))
                if not paths:raise ValueError('Missing base case records')
                for path in paths:
                    for line in path.read_text().splitlines():
                        r=json.loads(line)
                        if split=='val' and r['step']!=7680:continue
                        road=r['prompt_spec']['road_structure']
                        key=(r['scenario'],r['route_id'],r['frame_id'],r['context_id'],road)
                        if key not in by_observation:raise ValueError('Base record outside frozen observations')
                        identity=by_observation[key]
                        if identity in p:raise ValueError('Duplicate base prediction')
                        pred=r['raw_output'].strip();pred=pred if pred in data.ACTIONS else None
                        if bool(r['all_ok'])!=(pred==targets[identity]['action']):raise ValueError('Base saved score mismatch')
                        p[identity]=pred
                if set(p)!=set(targets):raise ValueError('Missing fixed base outputs')
            else:
                query=0 if job['mode']=='future' else 4
                for r in trial.read(root/f'dense_{split}_cases.json'):
                    identity=r['id']
                    if identity not in targets or identity in p:raise ValueError('Invalid dense identity')
                    if r['targets'][query]!=targets[identity]['action']:raise ValueError('Dense anchor label changed')
                    p[identity]=r['predictions'][query]['action']
            predictions[arm]=p
        report[split]=summarize(targets,predictions)
    return dict(status='three_methods_scored_no_selection',results=report,
        historical_best=dict(mode='4rgb_choice',val_correct=242,val_total=320,test_correct=274,test_total=380),
        historical_models_rerun=False,epochs_per_method=3,optimizer_updates_per_method=7680,
        initialization='same_qwen_no_trained_adapter',quality_conclusion='review outcomes; no automatic winner or claimed recovery')
