"""Frozen teacher samples -> answer-blind RGB packets -> complete review import.

Keep the private snapshot separate from the public reviewer directory. This
module does not invent human answers, appoint reviewers, or approve a rule.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from .data_paths import data_path
from .identity import ROOT,digest,file_sha,write_json
from . import teacher_rules as rules
from .teacher_replay import accounting,validate_question
from .controller import Episode
from .route_prompts import state_pair

POLICY='teacher_blind_review_v1'


def observable(q):
    # No paths or frame numbers: identical JPEG content/state pairs cannot be
    # counted twice just because they were copied to another source location.
    return digest([[s['sha256'] for s in q['input_sources']],state_pair(Episode(**q['episode']),q['edge'])])


def seal(value):
    return dict(value,sha256=digest(value))


def checked(value):
    if value.get('sha256')!=digest({k:v for k,v in value.items() if k!='sha256'}):raise ValueError('review artifact digest mismatch')
    return value


def card(q):
    source,destination=state_pair(Episode(**q['episode']),q['edge'])
    # No answer, facts, class name, source scenario or participant-role hints.
    return dict(id=digest(['blind',q['question_id']]),rgb_mode=q['rgb_mode'],
        current_state=source,candidate_next_state=destination,
        images=[dict(file=f"{digest(['blind',q['question_id']])}_{i}.jpg",sha256=s['sha256']) for i,s in enumerate(q['input_sources'])])


def public_packet(snapshot):
    cards=sorted((card(q) for q in snapshot['questions']),key=lambda c:c['id'])
    return seal(dict(policy=POLICY,purpose=snapshot['purpose'],cards=cards,
        instruction='Use only these causal images. Verify event identity and current state as well as the transition. Unresolved evidence is UNKNOWN. No future images or teacher answers.'))


def validate_snapshot(snapshot):
    checked(snapshot)
    if snapshot.get('policy')!=POLICY or snapshot.get('teacher')!=rules.identity() or snapshot.get('purpose') not in ('development','independent'):
        raise ValueError('stale/invalid teacher review snapshot')
    seen=set();keys=set();sha=digest(snapshot['teacher'])
    for q in snapshot['questions']:
        validate_question(q,expected_teacher_sha=sha)
        if q['question_id'] in seen or observable(q) in keys:raise ValueError('duplicate observable review question')
        seen.add(q['question_id']);keys.add(observable(q))
    if snapshot['public_sha256']!=public_packet(snapshot)['sha256']:raise ValueError('public packet differs from frozen sample')
    return snapshot


def select(pool,index_path,*,purpose='development',per_class=200,per_route=10,seed=20261004,rgb_modes=(2,4)):
    from .teacher_approval import budget_requirements,best_case_feasibility
    from .dataset import producer_check_reservations,source_route,groups,split_for,holdout_reservations
    if purpose not in ('development','independent') or type(per_class) is not int or type(per_route) is not int or min(per_class,per_route)<1:
        raise ValueError('invalid review scope/budget')
    if not rgb_modes or set(rgb_modes)-{2,4}:raise ValueError('invalid review RGB modes')
    from .candidate_pool import validate
    validate(pool);index_path=Path(index_path);coverage=accounting(pool,index_path)
    index=json.loads(index_path.read_text());reserved=producer_check_reservations();exposed=groups();owners=holdout_reservations()
    pools=defaultdict(list);offered=Counter();population=Counter();exclusions=Counter()
    # Bounded per-class/answer/physical-route queue, not an all-frames list.
    for receipt in index['routes']:
        allowed=receipt['split']=='train' if purpose=='development' else receipt['physical_group'] in reserved
        if not allowed:exclusions['outside_review_scope_frames']+=receipt['frames'];continue
        with (index_path.parent/receipt['artifact']).open() as stream:
            next(stream)
            for line in stream:
                for q in json.loads(line).get('questions',[]):
                    if q['rgb_mode'] not in rgb_modes:continue
                    group=source_route(q)
                    if group!=q['physical_group'] or split_for(group,exposed,reservations=owners)!=q['split']:raise ValueError('review source ownership mismatch')
                    cls=q['rule_class'];population[cls]+=1
                    stratum=(cls,q['rule_target'],group);offered[stratum]+=1
                    rank=(not q.get('near_transition',False),digest([seed,q['question_id']]))
                    bucket=pools[stratum];bucket.append((rank,q));bucket.sort(key=lambda x:x[0])
                    del bucket[per_route:]
    selected=[];summaries={};used_inputs=set()
    for cls in sorted(population):
        options=[v for s,items in pools.items() if s[0]==cls for v in items]
        used_routes=Counter();used_answers=Counter();used_context=Counter();chosen=[]
        while options and len(chosen)<per_class:
            def rank(item):
                initial,q=item;town=q['route_id'].split('_')[0]
                return (used_routes[q['physical_group']],used_answers[q['rule_target']],
                        used_context[(q['scenario'],town)],initial)
            item=min(options,key=rank);options.remove(item);q=item[1];key=observable(q)
            if used_routes[q['physical_group']]>=per_route:continue
            if key in used_inputs:exclusions['duplicate_observable_question']+=1;continue
            used_inputs.add(key);chosen.append(q);used_routes[q['physical_group']]+=1
            used_answers[q['rule_target']]+=1;used_context[(q['scenario'],q['route_id'].split('_')[0])]+=1
        selected.extend(chosen)
        summaries[cls]=dict(population_questions=population[cls],sample_questions=len(chosen),sample_routes=len(used_routes),
            answers=dict(used_answers),transition_questions=sum(q.get('near_transition',False) for q in chosen),
            unselected_questions=population[cls]-len(chosen),
            sample_route_deficit=max(0,20-len(used_routes)),judgment_deficit=max(0,100-len(chosen)),
            answer_routes={a:len({q['physical_group'] for q in chosen if q['rule_target']==a}) for a in ('YES','NO')},
            approval_budget=budget_requirements(),best_case_feasibility=best_case_feasibility(chosen))
    body=dict(policy=POLICY,teacher=rules.identity(),purpose=purpose,candidate_pool_sha256=pool['sha256'],
        production_index_sha256=coverage['index_sha256'],seed=seed,per_class=per_class,per_route=per_route,
        rgb_modes=list(rgb_modes),sampling='deterministic route/scenario/Town/answer diversity and causal transition oversampling; not a probability sample',
        questions=selected,statistics=dict(classes=summaries,exclusions=dict(exclusions)),supervision_approved=False)
    body['public_sha256']=public_packet(body)['sha256']
    return seal(body)


def render(snapshot,data_root,output):
    validate_snapshot(snapshot);output=Path(output);root=Path(data_root).resolve()
    if output.exists():raise FileExistsError('use a new public review directory')
    # Validate all sources before writing a partially misleading review packet.
    for q in snapshot['questions']:
        for s in q['input_sources']:
            path=data_path(root, s['path'])
            if file_sha(path)!=s['sha256']:raise ValueError('review RGB source changed')
    output.mkdir(parents=True)
    import shutil
    public=public_packet(snapshot)
    for q in snapshot['questions']:
        for s,dest in zip(q['input_sources'],card(q)['images'],strict=True):shutil.copyfile(root/s['path'],output/dest['file'])
    write_json(output/'packet.json',public)
    write_json(output/'decisions.json',dict(policy=POLICY,public_sha256=public['sha256'],reviewer='',
        independence_attested=False,blind_to_rule_answers=False,frozen_before_review=False,
        decisions=[dict(card_id=c['id'],reference=None,event_identity_confirmed=None,state_confirmed=None,
            evidence='',current_input_resolves_answer=None) for c in public['cards']]))
    # Static local HTML. No network, uploads, source geometry or teacher answers.
    from html import escape
    sections=[]
    for c in public['cards']:
        images=''.join(f'<img src="{escape(x["file"])}" alt="causal RGB {i+1}" />' for i,x in enumerate(c['images']))
        sections.append(f'<article><h2>{escape(c["id"])}</h2><p>Current: {escape(c["current_state"])}</p><p>Next: {escape(c["candidate_next_state"])}</p>{images}</article>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Blind RGB review</title><style>body{font:16px sans-serif;max-width:1200px;margin:auto}img{width:100%;display:block}article{margin:3em 0}h2{font-size:14px;overflow-wrap:anywhere}</style>'+''.join(sections))
    return dict(cards=len(public['cards']),public_sha256=public['sha256'],output=str(output))


def binding(questions,purpose,public_sha256,production_index_sha256):
    """Frozen ordered sample; removing difficult judgments breaks its identity."""
    return seal(dict(policy=POLICY,purpose=purpose,teacher_sha256=digest(rules.identity()),
        question_ids=[q['question_id'] for q in questions],public_sha256=public_sha256,
        production_index_sha256=production_index_sha256))


def validate_binding(reference,purpose):
    plan=checked(reference.get('sample_plan',{}))
    samples=reference['samples']
    if (plan.get('policy')!=POLICY or plan.get('purpose')!=purpose or plan.get('teacher_sha256')!=digest(rules.identity())
            or not plan.get('public_sha256') or not plan.get('production_index_sha256')
            or plan.get('question_ids')!=[s['question_id'] for s in samples]):
        raise ValueError('review does not cover its complete frozen sample')
    expected=public_packet(dict(purpose=purpose,questions=[s['question'] for s in samples]))['sha256']
    if plan['public_sha256']!=expected:raise ValueError('review public inputs detached from sample')
    for sample in samples:
        if any(type(sample.get(k)) is not bool for k in ('event_identity_confirmed','state_confirmed','current_input_resolves_answer')):
            raise ValueError('review lacks identity/state/current-input checks')
        if sample['reference']!='UNKNOWN' and not all(sample[k] for k in ('event_identity_confirmed','state_confirmed','current_input_resolves_answer')):
            raise ValueError('unresolved event/state/input cannot support a binary reference')
        if sample.get('near_transition')!=sample['question'].get('near_transition',False):raise ValueError('transition stratum changed during review')
    return plan


def import_review(snapshot,review):
    validate_snapshot(snapshot);public=public_packet(snapshot)
    if (review.get('policy')!=POLICY or review.get('public_sha256')!=public['sha256']
            or not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip()
            or review.get('blind_to_rule_answers') is not True or review.get('frozen_before_review') is not True):
        raise ValueError('missing reviewer/blind review/frozen packet binding')
    if snapshot['purpose']=='independent' and review.get('independence_attested') is not True:
        raise ValueError('independent reviewer attestation required')
    decisions={}
    for d in review['decisions']:
        if d['card_id'] in decisions:raise ValueError('duplicate review decision')
        decisions[d['card_id']]=d
    if set(decisions)!={c['id'] for c in public['cards']}:raise ValueError('review must explicitly cover every selected card')
    strata=defaultdict(list)
    for q in snapshot['questions']:
        decision=decisions[card(q)['id']]
        if decision.get('reference') not in ('YES','NO','UNKNOWN') or not isinstance(decision.get('evidence'),str) or not decision['evidence'].strip():
            raise ValueError('unfilled review reference/evidence')
        sample={k:q[k] for k in ('question_id','rule_class','rule_target','input_sha256','scenario','route_id','physical_group')}
        sample.update(question=q,teacher_record_sha256=digest(q),near_transition=q.get('near_transition',False),
            reference=decision['reference'],rgb_review_evidence_id=digest(decision),evidence=decision['evidence'],
            **{k:decision.get(k) for k in ('event_identity_confirmed','state_confirmed','current_input_resolves_answer')})
        strata[q['rule_class']].append(sample)
    references={}
    for cls,samples in strata.items():
        qs=[s['question'] for s in samples]
        ref=dict(source='development_rgb_review' if snapshot['purpose']=='development' else 'independent_rgb_review',
            reviewer=review['reviewer'],independence_attested=review.get('independence_attested',False),
            blind_to_rule_answers=True,frozen_before_review=True,samples=samples,
            sample_plan=binding(qs,snapshot['purpose'],public_packet(dict(purpose=snapshot['purpose'],questions=qs))['sha256'],snapshot['production_index_sha256']))
        validate_binding(ref,snapshot['purpose']);references[cls]=ref
    return dict(policy=POLICY,teacher=snapshot['teacher'],snapshot_sha256=snapshot['sha256'],
        review_sha256=digest(review),references=references,supervision_approved=False)


def assemble_registry(development,independent,authors):
    from .teacher_approval import POLICY as APPROVAL_POLICY,CRITERIA,validate_registry
    if not authors or any(not isinstance(a,str) or not a.strip() for a in authors):raise ValueError('rule authors must be recorded')
    for imported,purpose in ((development,'development'),(independent,'independent')):
        if imported.get('policy')!=POLICY or imported.get('teacher')!=rules.identity():raise ValueError('stale imported review')
        for ref in imported['references'].values():validate_binding(ref,purpose)
    shared=set(development['references'])&set(independent['references'])
    registry=dict(policy=APPROVAL_POLICY,criteria=CRITERIA,teacher=rules.identity(),rules=[
        dict(rule_class=cls,teacher=rules.identity(),rule_authors=authors,
             development=development['references'][cls],independent=independent['references'][cls])
        for cls in sorted(shared)])
    result=validate_registry(registry)
    return registry,dict(approval=result,unpaired_development=sorted(set(development['references'])-shared),
                         unpaired_independent=sorted(set(independent['references'])-shared))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    build=sub.add_parser('prepare');build.add_argument('--candidate-pool',type=Path,required=True);build.add_argument('--production-index',type=Path,required=True)
    build.add_argument('--purpose',choices=('development','independent'),required=True);build.add_argument('--private-snapshot',type=Path,required=True)
    build.add_argument('--public-dir',type=Path,required=True);build.add_argument('--data-root',type=Path,default=ROOT.parents[1]/'lead_data')
    build.add_argument('--per-class',type=int,default=200);build.add_argument('--per-route',type=int,default=10)
    imp=sub.add_parser('import');imp.add_argument('--private-snapshot',type=Path,required=True);imp.add_argument('--decisions',type=Path,required=True);imp.add_argument('--output',type=Path,required=True)
    assemble=sub.add_parser('assemble');assemble.add_argument('--development',type=Path,required=True)
    assemble.add_argument('--independent',type=Path,required=True);assemble.add_argument('--rule-author',action='append',required=True)
    assemble.add_argument('--output',type=Path,required=True)
    build.add_argument('--rgb-modes',type=int,choices=(2,4),nargs='+',default=[2,4])
    a=p.parse_args()
    if a.command=='prepare':
        if a.private_snapshot.exists() or a.public_dir.exists():raise FileExistsError('use new private/public review paths')
        if a.private_snapshot.resolve().is_relative_to(a.public_dir.resolve()):raise ValueError('private answers cannot be saved in public reviewer directory')
        snapshot=select(json.loads(a.candidate_pool.read_text()),a.production_index,purpose=a.purpose,per_class=a.per_class,per_route=a.per_route,rgb_modes=a.rgb_modes)
        write_json(a.private_snapshot,snapshot);result=render(snapshot,a.data_root,a.public_dir)
    elif a.command=='assemble':
        if a.output.exists() or a.output.with_suffix('.report.json').exists():raise FileExistsError('use a new registry/report path')
        registry,result=assemble_registry(json.loads(a.development.read_text()),json.loads(a.independent.read_text()),a.rule_author)
        write_json(a.output,registry);write_json(a.output.with_suffix('.report.json'),result)
    else:
        if a.output.exists():raise FileExistsError('use a new imported review path')
        result=import_review(json.loads(a.private_snapshot.read_text()),json.loads(a.decisions.read_text()));write_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('references','teacher')},ensure_ascii=False))

if __name__=='__main__':main()
