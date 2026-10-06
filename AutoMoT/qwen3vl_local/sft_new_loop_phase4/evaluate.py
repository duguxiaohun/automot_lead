"""完整逐题自由生成与预测状态的序列回放，分开报告而不混称闭环仿真。"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='qwen3vl_local.sft_new_loop_phase4'

import argparse
from collections import Counter,defaultdict
import json
from .maneuver_safety import GUARDED
from .visible_scope import POLICY as VISIBLE_POLICY
from pathlib import Path
from .controller import Episode
from .route_context import route_identity
from .dataset import load_dataset,dump_rows
from .identity import ROOT,write_json,digest
from .model import generate,load_for_inference
from .taxonomy import EVENTS


def _reference_metrics(results):
    confusion=Counter((r['target'],r['prediction']) for r in results)
    groups=defaultdict(list)
    for r in results:
        groups[r['event']+'/'+r['edge']+'/'+r['slice']].append(r)
    accuracy=sum(r['target']==r['prediction'] for r in results)/len(results) if results else None
    no_count=sum(r['target']=='NO' for r in results)
    ready_count=sum(r['target']=='YES' and r['slice']=='readiness' for r in results)
    catchup_yes=[r for r in results if r['target']=='YES' and r['slice']=='catchup']
    events={event:[r for r in results if r['event']==event] for event in EVENTS}
    event_scores={e:sum(r['target']==r['prediction'] for r in rs)/len(rs) if rs else None
                  for e,rs in events.items()}
    observed=[v for v in event_scores.values() if v is not None]
    missing=[e for e,v in event_scores.items() if v is None]
    return dict(count=len(results),accuracy=accuracy,
        event_accuracy=event_scores,event_counts={e:len(rs) for e,rs in events.items()},
        missing_events=missing,event_macro_accuracy=sum(observed)/len(EVENTS) if not missing else None,
        event_macro_accuracy_observed=sum(observed)/len(observed) if observed else None,
        selection_metric='event_macro_accuracy_observed_v1',
        catchup_recall=sum(r['prediction']=='YES' for r in catchup_yes)/len(catchup_yes) if catchup_yes else None,
        premature_yes_rate=sum(r['target']=='NO' and r['prediction']=='YES' for r in results)/no_count if no_count else None,
        readiness_recall=sum(r['target']=='YES' and r['prediction']=='YES' and r['slice']=='readiness' for r in results)/ready_count if ready_count else None,
        malformed=sum(r['prediction']=='MALFORMED' for r in results),
        confusion={f'{a}->{b}':n for (a,b),n in sorted(confusion.items())},
        groups={k:dict(count=len(v),accuracy=sum(x['target']==x['prediction'] for x in v)/len(v)) for k,v in groups.items()})


def metrics(results):
    teacher=[r for r in results if r.get('reference_kind')=='rule_teacher']
    reviewed=[r for r in results if r.get('reference_kind')!='rule_teacher']
    out=_reference_metrics(reviewed)
    if teacher:
        measured=_reference_metrics(teacher)
        out['teacher_consistency']=dict(count=measured['count'],agreement=measured['accuracy'],
            event_agreement=measured['event_accuracy'],
            groups={k:dict(count=v['count'],agreement=v['accuracy']) for k,v in measured['groups'].items()},
            scope='agreement with teacher rules (approval not implied), not independent human accuracy')
        out['reference_scope']='top-level accuracy/selection uses reviewed references only'
        out['total_count']=len(results)
    return out


def evaluate_rows(bundle,rows,data_root,max_length=8192):
    from .paired_eval import case_identity
    results=[]
    for i,r in enumerate(rows):
        answer,raw=generate(bundle,r,data_root,max_length)
        results.append(dict(id=r['id'],target=r['target'],prediction=answer,raw=raw,
                            event=r['episode']['event'],edge=r['edge'],slice=r['slice'],reference_kind=r.get('reference_kind','reviewed_rgb'),
                            paired_identity=case_identity(r)))
        if i%20==0:
            print(f'[generate] {i+1}/{len(rows)}',flush=True)
    return metrics(results),results


def replay(initial, observations, predictor, *, safety_adapter=None):
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
            reason = ('route_segment_changed' if active[key].get('segment_id') != ep.segment_id else
                      'observation_gap' if steps and frame>steps[-1]['frame_id']+1 else
                      'transition_not_queried' if key not in queried else None)
            if reason:
                censored.append(dict(edge=key,end_frame=frame,reason=reason,**active.pop(key)))
        truth_bound = ((not ep.route_segments or obs.get('segment_id')==ep.segment_id)
                       and (not ep.route_context or obs.get('route_context_id')==route_identity(ep)))
        before=ep.to_dict()
        answers={e.key:predictor(ep,e.key,obs) for e in qs}
        for e in qs:
            key=e.key
            visible_truth_bound = key not in GUARDED or obs.get('visible_truth_scope') == VISIBLE_POLICY
            truth=(obs.get('truth',{}).get(key) if truth_bound and visible_truth_bound else None)
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
            record=active.setdefault(key,dict(ready_since=frame,answer_frame=None,accepted_frame=None,**({"segment_id":ep.segment_id} if ep.route_segments else {})))
            record['last_ready_frame']=frame
            if answers[key]=='YES' and record['answer_frame'] is None:
                record['answer_frame']=frame
                delays['answer'].append(frame-record['ready_since'])
        clearances=obs.get('maneuver_clearances',())
        safety_audit=[]
        if safety_adapter is not None:
            if clearances:
                raise ValueError('choose computed safety requests or explicit clearances, not both')
            clearances,safety_audit=safety_adapter(ep,obs)
        elif obs.get('maneuver_safety_requests'):
            raise ValueError('offline safety requests require an explicit safety adapter')
        prior=ep.advance(frame,answers,context_valid=obs.get('context_valid',True),
                         execution_committed=obs.get('execution_committed',False),
                         progress_fault=obs.get('progress_fault',False),
                         maneuver_clearances=clearances)
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
        missed_execution+=sum(truth_bound and (e.key not in GUARDED or obs.get('visible_truth_scope')==VISIBLE_POLICY)
                              and obs.get('truth',{}).get(e.key)=='YES' and e.key not in confirmed for e in qs)
        steps.append(dict(frame_id=frame,before=before,answers=answers,accepted=sorted(accepted),
                          execution_confirmed=sorted(confirmed),prior=prior,after=ep.to_dict(),safety_audit=safety_audit))
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
                answer_conflict_frames=[s['frame_id'] for s in steps if s['prior']['wait_reason'] in
                    ('contradictory_same_axis_answers','contradictory_restriction_and_progress')],
                safety_missing_frames=[s['frame_id'] for s in steps if s['prior']['wait_reason'].startswith('maneuver_safety_unconfirmed:')],
                safety_denied_frames=[s['frame_id'] for s in steps if s['after']['history'] and
                                      s['after']['history'][-1].get('frame_id')==s['frame_id'] and
                                      s['after']['history'][-1].get('safety_denied')],
                stalled=ep.needs_recheck,wait_reason=ep.wait_reason,steps=steps,
                scope='predicted-state offline replay; execution requires external evidence; not vehicle simulation')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--teacher-references',type=Path,help='separate held-out teacher consistency pack; never used for selection')
    p.add_argument('--model-dir',type=Path,default=ROOT.parents[1]/'checkpoints/Qwen3.5-4B')
    p.add_argument('--data-root',type=Path,default=ROOT.parents[1]/'lead_data')
    p.add_argument('--split',choices=('train','val','test'),default='test')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--require-complete-coverage',action='store_true')
    a=p.parse_args()
    data,m=load_dataset(a.dataset,require_trainable=True,
                        require_complete_coverage=a.require_complete_coverage)
    saved=json.loads((a.adapter/'phase4_contract.json').read_text())
    if saved['dataset']!=digest(m):
        raise ValueError('adapter/dataset mismatch')
    if a.output_dir.exists():
        raise FileExistsError('use a fresh evaluation directory')
    import torch
    bundle=load_for_inference(a.model_dir,a.adapter,torch.device(a.device))
    evaluation_rows=list(data[a.split])
    if a.teacher_references:
        from .teacher_evaluation import load
        evaluation_rows+=load(a.teacher_references,split=a.split,rgb_mode=m['observation_contract']['rgb_mode'])
    report,rows=evaluate_rows(bundle,evaluation_rows,a.data_root)
    report.update(split=a.split,dataset=digest(m),expected_count=len(evaluation_rows),
                  coverage_complete=m['complete_coverage'],coverage=m['coverage'],
                  evaluation_scope=m['training_admission']['evaluation_scope'])
    a.output_dir.mkdir(parents=True)
    dump_rows(a.output_dir/'cases.jsonl',rows)
    write_json(a.output_dir/'metrics.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
