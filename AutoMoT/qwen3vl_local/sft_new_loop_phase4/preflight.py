"""逐层预检：数据/支持量/因果输入/采样/离线模型；缺项不伪造ready。"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='qwen3vl_local.sft_new_loop_phase4'

import argparse
import json
from collections import Counter
from pathlib import Path
from .dataset import load_dataset,coverage_report
from .identity import ROOT
from .sampling import plan,CapacityError
from .model import load_images
from .controller import Episode
from .route_prompts import prompt


def sampling_coverage(rows, *, epochs=7, world_size=1, policy='event_equal', **kwargs):
    """Invalid diagnostic configurations must not reject a different valid run."""
    from .sampling import coverage_schedule
    try:
        if policy!='legacy_ring':
            return coverage_schedule(rows,epochs=epochs,world_size=world_size,policy=policy,**kwargs)
        counts=Counter();timeline=[]
        for epoch in range(epochs):
            indices,_=plan(rows,epoch=epoch,world_size=world_size,policy=policy,**kwargs)
            counts.update(rows[i]['id'] for i in indices)
            timeline.append(dict(epoch=epoch,seen=len(counts),missing=len(rows)-len(counts)))
        return dict(feasible=True,epochs=epochs,world_size=world_size,pool=len(rows),timeline=timeline,
                    complete=len(counts)==len(rows),missing_ids=sorted(r['id'] for r in rows if r['id'] not in counts))
    except ValueError as ex:
        return dict(feasible=False,diagnostic=getattr(ex,'diagnostic',dict(error=str(ex))),timeline=[],complete=False)


def inspect(dataset,model_dir=None,data_root=None,*,sampling_policy=None,epoch_samples=None,max_question_repeat=4,cap=8,seed=20260929,world_size=1,epochs=7,paired_with=None,accumulation=8):
    data,m=load_dataset(dataset)
    pairing=None
    if paired_with is not None:
        from .paired_training import load_view
        data,pairing=load_view(data,m,paired_with)
    report=dict(data_ready=m['trainable'],training_admission=m['training_admission'],
                complete_coverage=m['complete_coverage'],coverage=m['coverage'],
                scope='training_host' if model_dir else 'data_only',
                risk_review=m['risk_review_check'],sampling_feasible=False)
    if pairing is not None:
        report['training_admission']=pairing['training_admission']
        report['coverage']=coverage_report(sum(data.values(),[]))
        report['complete_coverage']=report['coverage']['ready']
    report['paired_training']=pairing
    report['production_coverage']=m.get('production_coverage')
    from .branch_support import report as branch_report
    report['branch_support']=branch_report(data)
    report['stationary_readiness_support']=stationary_readiness_support(data['train'])
    from .admission import review_capacity_report
    report['review_capacity']=review_capacity_report(data)
    report['weak_supervision_rows']=sum(r.get('label_basis')=='weak_rule_teacher' for rows in data.values() for r in rows)
    report['formal_data_ready']=bool(not report['weak_supervision_rows'] and report['complete_coverage'] and m.get('production_coverage',{})
                                    and m['production_coverage']['complete_causal_production']
                                    and report['review_capacity']['numerical_floor_met']
                                    and bool(m['production_coverage'].get('approved_rule_classes')))
    report['approved_rule_classes']=(m.get('production_coverage') or {}).get('approved_rule_classes',[])
    if sampling_policy in ('phase3_balanced','full_event_equal'):
        from .full_sampling import validate_dataset_scope
        validate_dataset_scope(m,pairing)
    if data['train']:
        policy=sampling_policy or ('event_weighted' if report['weak_supervision_rows'] else 'event_equal')
        try:
            indices,report['sampling']=plan(data['train'],epoch=0,policy=policy,budget=epoch_samples,max_question_repeat=max_question_repeat,cap=cap,seed=seed,world_size=world_size)
            report['sampling_repetition']=sampling_repetition(data['train'],indices)
            import math
            report['training_workload']=dict(questions_per_epoch=len(indices),microbatches_per_rank=len(indices)//world_size,optimizer_steps_per_epoch=math.ceil(len(indices)/(world_size*accumulation)),epochs=epochs,total_presentations=len(indices)*epochs)
            report['sampling'].pop('next_history',None)
            report['sampling_feasible']=True
        except ValueError as ex:
            report['sampling_feasible']=False;report['sampling']=getattr(ex,'diagnostic',dict(error=str(ex)))
        report['default_seven_epoch_coverage']={str(world):sampling_coverage(data['train'],epochs=7,world_size=world,policy=policy,budget=epoch_samples,max_question_repeat=max_question_repeat,cap=cap,seed=seed)
                                              for world in (1,4)}
        report['requested_sampling_config']=dict(policy=policy,epoch_samples=epoch_samples,max_question_repeat=max_question_repeat,cap=cap,seed=seed,world_size=world_size,epochs=epochs)
        report['requested_epoch_coverage']=sampling_coverage(data['train'],epochs=epochs,world_size=world_size,policy=policy,budget=epoch_samples,max_question_repeat=max_question_repeat,cap=cap,seed=seed)
        report['sampling_feasible'] = report['sampling_feasible'] and report['requested_epoch_coverage']['feasible']
        for schedule in report['default_seven_epoch_coverage'].values():
            seen=schedule['timeline'][-1]['seen'] if schedule['timeline'] else None
            schedule['pool_fraction_seen']=seen/len(data['train']) if seen is not None else None
            schedule['epoch_meaning']=('full admitted question pool plus event-equal oversampling' if policy=='full_event_equal' else 'a constrained sampling budget, not a full traversal of the question pool')
    if data_root:
        for rows in data.values():
            for row in rows:
                prompt(Episode(**row['episode']),row['edge'],row['observation'])
                for im in load_images(row,data_root):
                    im.close()
        report['rgb_verified']=sum(map(len,data.values()))
    if model_dir:
        from qwen3vl_local.qwen35.preflight import check
        report['model']=check(Path(model_dir),action=False)
    report['model_checked']=model_dir is not None
    report['rgb_checked']=data_root is not None
    # None means not checked here, not a missing server asset. A server launch
    # must still check both actual RGB and its local base model.
    report['ready']=(report['data_ready'] and report['sampling_feasible'] and bool(report['model']['ready'])
                     if model_dir is not None and data_root is not None else None)
    return report


def stationary_readiness_support(rows,minimum=1):
    """Nonblocking coverage diagnostic: moving/catchup YES cannot fill a gap."""
    from .taxonomy import EVENTS,event_edges,COMMON
    from .weighted_sampling import motion_stratum
    if type(minimum) is not int or minimum<1:raise ValueError('invalid stationary YES floor')
    cells={}
    for event,(_,template,_) in EVENTS.items():
        for edge in (*event_edges(template),*COMMON):
            if edge.key not in ('proceed','depart','enter','return','release'):continue
            selected=[r for r in rows if r['episode']['event']==event and r['edge']==edge.key and r['target']=='YES']
            ready=[r for r in selected if r.get('slice','readiness')=='readiness']
            stationary=[r for r in ready if motion_stratum(r)=='stationary']
            cells[event+'/'+edge.key]=dict(stationary_yes=len(stationary),
                stationary_routes=len({r['physical_group'] for r in stationary}),
                manual_stationary_yes=sum(r.get('label_basis') not in ('weak_rule_teacher','approved_rule_teacher') for r in stationary),
                moving_yes=sum(motion_stratum(r)=='moving' for r in ready),
                unknown_motion_yes=sum(motion_stratum(r)=='unknown' for r in ready),
                catchup_yes=sum(r.get('slice')=='catchup' for r in selected))
    gaps={k:minimum-v['stationary_yes'] for k,v in cells.items() if v['stationary_yes']<minimum}
    return dict(minimum_stationary_yes=minimum,cells=cells,missing_cells=gaps,blocking=False,
        warning='missing stationary readiness YES; moving/catchup samples do not demonstrate release from a stop' if gaps else None)


def sampling_repetition(rows,indices):
    """Expose the cost of event equality without silently changing quotas."""
    counts=Counter(indices);events=sorted({r['episode']['event'] for r in rows})
    return {event:dict(pool_questions=len(pool),presentations=sum(counts[i] for i in pool),
        distinct_questions=sum(counts[i]>0 for i in pool),
        repeated_presentations=sum(max(0,counts[i]-1) for i in pool),
        max_question_repeat=max((counts[i] for i in pool),default=0),
        presentations_per_pool_question=sum(counts[i] for i in pool)/len(pool))
        for event in events for pool in [[i for i,r in enumerate(rows) if r['episode']['event']==event]]}


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--paired-with',type=Path)
    p.add_argument('--model-dir',type=Path)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--require-ready',action='store_true')
    p.add_argument('--require-trainable',action='store_true')
    p.add_argument('--require-complete-coverage',action='store_true')
    p.add_argument('--sampling-policy',choices=('phase3_balanced','full_event_equal','event_equal','event_edge_answer','event_weighted','legacy_ring'))
    p.add_argument('--epoch-samples',type=int)
    p.add_argument('--max-question-repeat',type=int,default=4)
    p.add_argument('--cap',type=int,default=8)
    p.add_argument('--seed',type=int,default=20260929)
    p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--epochs',type=int,default=7)
    p.add_argument('--accumulation',type=int,default=8)
    a=p.parse_args()
    if min(a.cap,a.world_size,a.epochs,a.max_question_repeat,a.accumulation)<1:p.error('sampling parameters must be positive')
    if a.require_ready and (a.model_dir is None or a.data_root is None):
        p.error('--require-ready is a training-host check; supply --model-dir and --data-root')
    report=inspect(a.dataset,a.model_dir,a.data_root,sampling_policy=a.sampling_policy,epoch_samples=a.epoch_samples,max_question_repeat=a.max_question_repeat,cap=a.cap,seed=a.seed,world_size=a.world_size,epochs=a.epochs,paired_with=a.paired_with,accumulation=a.accumulation)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if ((a.require_ready and report['ready'] is not True)
            or (a.require_trainable and (not report['data_ready'] or not report['sampling_feasible']))
            or (a.require_complete_coverage and not report['complete_coverage'])):
        raise SystemExit(2)

if __name__=='__main__':
    main()
