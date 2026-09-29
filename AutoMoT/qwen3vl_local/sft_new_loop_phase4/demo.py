"""无需模型的流程demo；合成条件不是RGB效果验证。"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='qwen3vl_local.sft_new_loop_phase4'

import argparse
import json
from .controller import Episode
from .prompts import prompt


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
        prior=e.advance(frame,answers,execution_committed=True)
        trace.append(dict(frame=frame,condition=key,question=question,answer='YES',prior=prior))
        frame+=1
    return trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--event',default='U-E2')
    p.add_argument('--branch',default='default')
    p.add_argument('--no-return',action='store_true')
    p.add_argument('--direction',choices=('LEFT','RIGHT','FORWARD'),default='LEFT')
    a=p.parse_args()
    print(json.dumps(run(a.event,a.branch,not a.no_return,a.direction),ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
