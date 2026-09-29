"""完整逐题自由生成与预测状态的序列回放，分开报告而不混称闭环仿真。"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='qwen3vl_local.sft_new_loop_phase4'

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from .controller import Episode
from .dataset import load_dataset,dump_rows
from .identity import ROOT,write_json,digest
from .model import generate,load_for_inference


def metrics(results):
    confusion=Counter((r['target'],r['prediction']) for r in results)
    groups=defaultdict(list)
    for r in results:
        groups[r['event']+'/'+r['edge']+'/'+r['slice']].append(r)
    accuracy=sum(r['target']==r['prediction'] for r in results)/len(results) if results else None
    no_count=sum(r['target']=='NO' for r in results)
    ready_count=sum(r['target']=='YES' and r['slice']=='readiness' for r in results)
    catchup_yes=[r for r in results if r['target']=='YES' and r['slice']=='catchup']
    return dict(count=len(results),accuracy=accuracy,
        catchup_recall=sum(r['prediction']=='YES' for r in catchup_yes)/len(catchup_yes) if catchup_yes else None,
        premature_yes_rate=sum(r['target']=='NO' and r['prediction']=='YES' for r in results)/no_count if no_count else None,
        readiness_recall=sum(r['target']=='YES' and r['prediction']=='YES' and r['slice']=='readiness' for r in results)/ready_count if ready_count else None,
        malformed=sum(r['prediction']=='MALFORMED' for r in results),
        confusion={f'{a}->{b}':n for (a,b),n in sorted(confusion.items())},
        groups={k:dict(count=len(v),accuracy=sum(x['target']==x['prediction'] for x in v)/len(v)) for k,v in groups.items()})


def evaluate_rows(bundle,rows,data_root,max_length=8192):
    results=[]
    for i,r in enumerate(rows):
        answer,raw=generate(bundle,r,data_root,max_length)
        results.append(dict(id=r['id'],target=r['target'],prediction=answer,raw=raw,
                            event=r['episode']['event'],edge=r['edge'],slice=r['slice']))
        if i%20==0:
            print(f'[generate] {i+1}/{len(rows)}',flush=True)
    return metrics(results),results


def replay(initial, observations, predictor):
    """Separate answer, controller acceptance, execution evidence and final completion.

    Fixed observations cannot simulate counterfactual vehicle motion. Open readiness
    episodes without an execution receipt remain right-censored, including repeated YES.
    Unknown truth, omitted frames or unqueried edges end the provably continuous
    readiness segment; the next reviewed YES starts a separate measurement.
    """
    from .controller import STARTS
    ep=Episode(**initial)
    steps=[]
    early=unknown=covered=missed_answers=missed_execution=0
    active={}
    delays={k:[] for k in ('answer','accepted','execution')}
    censored=[]
    for obs in observations:
        if ep.finished or ep.needs_recheck:
            break
        frame=obs['frame_id']
        if frame<=ep.last_frame:
            raise ValueError('nonchronological replay')
        qs=ep.questions()
        if ep.needs_recheck:
            break
        queried={e.key for e in qs}
        for key in list(active):
            reason = ('observation_gap' if steps and frame>steps[-1]['frame_id']+1 else
                      'transition_not_queried' if key not in queried else None)
            if reason:
                censored.append(dict(edge=key,end_frame=frame,reason=reason,**active.pop(key)))
        before=ep.to_dict()
        answers={e.key:predictor(ep,e.key,obs) for e in qs}
        for e in qs:
            key=e.key
            truth=obs.get('truth',{}).get(key)
            if truth not in ('YES','NO'):
                unknown+=1
                if key in active:
                    censored.append(dict(edge=key,end_frame=frame,reason='truth_unknown',**active.pop(key)))
                continue
            covered+=1
            early+=int(truth=='NO' and answers[key]=='YES')
            missed_answers+=int(truth=='YES' and answers[key]!='YES')
            if truth=='NO':
                if key in active:
                    censored.append(dict(edge=key,end_frame=frame,reason='condition_closed',**active.pop(key)))
                continue
            record=active.setdefault(key,dict(ready_since=frame,answer_frame=None,accepted_frame=None))
            record['last_ready_frame']=frame
            if answers[key]=='YES' and record['answer_frame'] is None:
                record['answer_frame']=frame
                delays['answer'].append(frame-record['ready_since'])
        prior=ep.advance(frame,answers,context_valid=obs.get('context_valid',True),
                         execution_committed=obs.get('execution_committed',False),
                         progress_fault=obs.get('progress_fault',False))
        if obs.get('execution_receipt') is not None:
            ep.confirm_execution(obs['execution_receipt'])
            prior=ep.prior()
        record=ep.history[-1] if ep.history and ep.history[-1].get('frame_id')==frame else {}
        accepted=set(record.get('accepted',[]))
        confirmed={key for key in accepted if key not in STARTS or record.get('committed')}
        for key in list(active):
            r=active[key]
            if key in accepted and r['accepted_frame'] is None:
                r['accepted_frame']=frame
                delays['accepted'].append(frame-r['ready_since'])
            if key in confirmed:
                delays['execution'].append(frame-r['ready_since'])
                del active[key]
        missed_execution+=sum(obs.get('truth',{}).get(e.key)=='YES' and e.key not in confirmed for e in qs)
        steps.append(dict(frame_id=frame,before=before,answers=answers,accepted=sorted(accepted),
                          execution_confirmed=sorted(confirmed),prior=prior,after=ep.to_dict()))
    last=steps[-1]['frame_id'] if steps else initial.get('last_frame',-1)
    censored.extend(dict(edge=key,end_frame=last,reason='sequence_end_or_recheck',**r) for key,r in active.items())
    return dict(completed=ep.finished,event_stage_completed=ep.state=='DONE',
                completion_frame=last if ep.finished else None,needs_recheck=ep.needs_recheck,
                covered_questions=covered,uncovered_questions=unknown,premature_transitions=early,
                missed_ready=missed_answers,missed_answers=missed_answers,missed_execution=missed_execution,
                answer_delay_frames=delays['answer'],accepted_delay_frames=delays['accepted'],
                execution_delay_frames=delays['execution'],delay_frames=delays['execution'],
                unconfirmed_transitions=censored,incomplete=not ep.finished,
                delay_policy='phase4_contiguous_reviewed_readiness_v1',
                stalled=ep.needs_recheck,wait_reason=ep.wait_reason,steps=steps,
                scope='predicted-state offline replay; execution requires external evidence; not vehicle simulation')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,default=ROOT.parents[1]/'checkpoints/Qwen3.5-4B')
    p.add_argument('--data-root',type=Path,default=ROOT.parents[1]/'lead_data')
    p.add_argument('--split',choices=('train','val','test'),default='test')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    data,m=load_dataset(a.dataset,require_ready=True)
    saved=json.loads((a.adapter/'phase4_contract.json').read_text())
    if saved['dataset']!=digest(m):
        raise ValueError('adapter/dataset mismatch')
    if a.output_dir.exists():
        raise FileExistsError('use a fresh evaluation directory')
    import torch
    bundle=load_for_inference(a.model_dir,a.adapter,torch.device(a.device))
    report,rows=evaluate_rows(bundle,data[a.split],a.data_root)
    a.output_dir.mkdir(parents=True)
    dump_rows(a.output_dir/'cases.jsonl',rows)
    write_json(a.output_dir/'metrics.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
