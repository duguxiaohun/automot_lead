"""Exact branch/edge support. Screening, catchup and teacher agreement are separate."""
from collections import defaultdict
from .taxonomy import EVENTS,transitions
from .route_context import RECOVER_FOLLOW
from .weighted_sampling import motion_stratum


def branches():
    for event in EVENTS:
        if event=='U-E2':
            yield event,'default',True
            yield event,'default',False
            yield event,'in_lane_pass',True
        elif event=='U-E4':
            yield event,'default',True
            yield event,'cyclist_follow',True
            yield event,'cyclist_bypass',True
            yield event,'cyclist_bypass',False
        else:yield event,'default',True


def branch_key(ep):
    # return_required affects only the bypass template, not ordinary yielding.
    from .taxonomy import template_for
    event=ep['event'];branch=ep.get('branch','default')
    returning=ep.get('return_required',True) if template_for(event,branch)=='bypass' else True
    return event,branch,returning


def report(data):
    buckets=defaultdict(list)
    for split in ('train','val','test'):
        for r in data.get(split,[]):buckets[(split,*branch_key(r['episode']),r['edge'])].append(r)
    result={};missing=[]
    for event,branch,returning in branches():
        prefix=f'{event}/{branch}/'+('return' if returning else 'no_return')
        edges=(*transitions(event,branch,returning),RECOVER_FOLLOW)
        supported={}
        for edge in edges:
            cells={}
            for split in ('train','val','test'):
                rows=buckets[(split,event,branch,returning,edge.key)]
                cells[split]={phase:{answer:dict(questions=len(rs),physical_routes=len({r['physical_group'] for r in rs}),
                    stationary_questions=sum(motion_stratum(r)=='stationary' for r in rs))
                    for answer in ('YES','NO') for rs in [[r for r in rows if r['slice']==phase and r['target']==answer]]}
                    for phase in ('readiness','catchup')}
            supported[edge.key]=cells
            for answer in ('YES','NO'):
                if not cells['train']['readiness'][answer]['questions']:
                    missing.append(dict(event=event,branch=branch,return_required=returning,edge=edge.key,
                        answer=answer,scope='readiness',reason='no_supervision',priority=1 if edge.key in ('depart','enter','return','proceed','complete') else 2))
        result[prefix]=dict(edges=supported,training_questions=sum(len(v) for k,v in buckets.items() if k[:4]==('train',event,branch,returning)))
    evaluation={split:{event:dict(questions=len(rs),physical_routes=len({r['physical_group'] for r in rs}))
        for event in EVENTS for rs in [[r for r in data.get(split,[]) if r['episode']['event']==event]]} for split in ('val','test')}
    return dict(policy='phase4_branch_edge_support_v1',branches=result,missing_training_cells=missing,
        evaluation=evaluation,missing_evaluation_events={s:[e for e,v in cells.items() if not v['questions']] for s,cells in evaluation.items()},
        task_plan=sorted(missing,key=lambda r:(r['priority'],r['event'],r['branch'],r['return_required'],r['edge'],r['answer'])),
        blocking=False,scope='annotation task inventory, not labels; all zero branches explicit; catchup cannot fill readiness; quantity is not independent accuracy')
