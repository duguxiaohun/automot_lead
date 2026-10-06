"""无需模型的流程demo；合成条件不是RGB效果验证。"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='qwen3vl_local.sft_new_loop_phase4'

import argparse
import json
from .controller import Episode
from .route_prompts import prompt
from .maneuver_safety import binding, GUARDED


def synthetic_clearances(ep, frame, edges):
    """Demo-only planner assertions; never use these in a real runtime adapter."""
    return [dict(**binding(ep, key, frame), source='planner', evidence_id=f'synthetic-demo/{frame}/{key}',
                 clear=True, rear_side_coverage_confirmed=True) for key in edges if key in GUARDED]


def run(event='U-E2',branch='default',return_required=True,direction='LEFT'):
    e=Episode(event,'demo',branch=branch,longitudinal='HOLD',direction=direction,
              return_direction='RIGHT' if direction=='LEFT' else 'LEFT',return_required=return_required,
              target_corridor='established bypass/route corridor',return_corridor='original route corridor')
    chain=['depart','pass','return','complete'] if (event=='U-E2' and branch=='default') or branch=='cyclist_bypass' else ['enter','complete'] if event in ('R-E2','R-E3') else ['proceed','complete']
    if not return_required and 'return' in chain:
        chain.remove('return')
    trace=[]
    frame=10
    for key in chain:
        if key=='complete' and e.longitudinal=='RECOVER':
            answers={edge.key:('YES' if edge.key=='stable' else 'NO') for edge in e.questions()}
            question=prompt(e,'stable',dict(frame_id=frame,history_frames=[frame-4,frame],speed_mps=0.))
            trace.append(dict(frame=frame,condition='stable',question=question,answer='YES',prior=e.advance(frame,answers,execution_committed=True)))
            frame+=1
        answers={edge.key:'NO' for edge in e.questions()}
        answers[key]='YES'
        question=prompt(e,key,dict(frame_id=frame,history_frames=[frame-4,frame],speed_mps=0.))
        prior=e.advance(frame,answers,execution_committed=True,
                        maneuver_clearances=synthetic_clearances(e,frame,[key]))
        trace.append(dict(frame=frame,condition=key,question=question,answer='YES',prior=prior))
        frame+=1
    return trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--event',default='U-E2')
    p.add_argument('--branch',default='default')
    p.add_argument('--no-return',action='store_true')
    p.add_argument('--direction',choices=('LEFT','RIGHT','FORWARD'),default='LEFT')
    p.add_argument('--segmented-exit',action='store_true',help='two adjacent entries with a fresh gap decision between them')
    p.add_argument('--roundabout',action='store_true',help='target-bound exit with renewed yielding and STOP after exit')
    a=p.parse_args()
    print(json.dumps(roundabout() if a.roundabout else segmented_exit() if a.segmented_exit else run(a.event,a.branch,not a.no_return,a.direction),ensure_ascii=False,indent=2))


def segmented_exit():
    from .runtime import Phase4Loop
    loop=Phase4Loop(rgb_mode=2)
    segments=[dict(segment_id='middle',source_corridor='left lane',target_corridor='middle lane',direction='RIGHT',adjacent=True),
              dict(segment_id='right',source_corridor='middle lane',target_corridor='right lane',direction='RIGHT',adjacent=True)]
    ep=Episode('R-E3','exit-demo',direction='RIGHT',target_corridor='middle lane',route_segments=segments)
    loop.establish(ep,verified=True)
    trace=[]
    # Synthetic conditions/executor evidence, not model predictions or RGB labels.
    for frame,yes in enumerate((('enter',),('complete','hold'),(),('enter',),('complete','stable')),10):
        obs=dict(frame_id=frame,history_frames=[frame-4,frame],speed_mps=4.)
        questions=[]
        def predictor(active,key,observation,images):
            answer='YES' if key in yes else 'NO'
            questions.append(dict(segment_id=active.segment_id,question=prompt(active,key,observation),answer=answer))
            return answer
        receipts=[]
        if 'enter' in yes:
            receipts=[dict(instance_id=ep.instance_id,segment_id=ep.segment_id,source='executor',
                           evidence_id=f'synthetic-{frame}',successor_observed=True,
                           decision_frame=frame,observed_frame=frame,started_frame=frame,edges=['enter'])]
        trace.append(dict(frame=frame,questions=questions,
                          result=loop.tick(obs,[None]*2,predictor,execution_receipts=receipts,
                                           maneuver_clearances=synthetic_clearances(ep,frame,yes))))
    return trace

def roundabout():
    from .runtime import Phase4Loop
    from .route_context import route_identity
    loop=Phase4Loop(rgb_mode=2)
    ep=Episode('R-E5','roundabout-demo',longitudinal='HOLD',target_corridor='exit beside the red building',
        route_context=dict(kind='roundabout',completion_boundary='past the exit mouth and circulation conflict',
                           navigation_evidence_id='synthetic-navigation'))
    loop.establish(ep,verified=True)
    trace=[]
    # Synthetic only: entry, renewed conflict, fresh permission, exit while stopping, recovery.
    for frame,yes in enumerate((('proceed',),('re_yield','hold'),('proceed',),('complete','hold'),('release',),('stable',)),10):
        obs=dict(frame_id=frame,history_frames=[frame-4,frame],speed_mps=0.)
        questions=[]
        def predictor(active,key,observation,images):
            answer='YES' if key in yes else 'NO'
            questions.append(dict(question=prompt(active,key,observation),answer=answer))
            return answer
        starts=[k for k in yes if k in ('proceed','release')]
        receipts=[]
        if starts:
            receipts=[dict(instance_id=ep.instance_id,route_context_id=route_identity(ep),source='executor',
                evidence_id=f'synthetic-{frame}',successor_observed=True,decision_frame=frame,
                observed_frame=frame,started_frame=frame,edges=starts)]
        trace.append(dict(frame=frame,questions=questions,result=loop.tick(obs,[None]*2,predictor,execution_receipts=receipts)))
    return trace


if __name__=='__main__':
    main()
