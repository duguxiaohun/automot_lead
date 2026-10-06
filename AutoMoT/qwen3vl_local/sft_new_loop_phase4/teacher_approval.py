"""Per-rule approval from frozen development and independent RGB references.

Registry content is an auditable reviewer attestation, not proof that a human
actually looked. This module never fills references or self-approves a rule.
"""
import math
from .identity import digest
from .teacher_rules import identity

POLICY='causal_teacher_class_approval_v2'
WEAK_POLICY='causal_teacher_weak_experiment_v1'

CRITERIA=dict(min_judgments=100,min_routes=20,min_yes_precision_lower=.95,
              min_no_precision_lower=.90,min_transition_judgments=30)


def wilson_lower(correct,total):
    if type(total) is not int or type(correct) is not int or not 0<=correct<=total:raise ValueError('invalid precision counts')
    if not total:return 0.
    z=1.959963984540054;p=correct/total
    return (p+z*z/(2*total)-z*math.sqrt(p*(1-p)/total+z*z/(4*total*total)))/(1+z*z/total)


def budget_requirements():
    """Best-case lower bounds; errors and route diversity can require more."""
    out={}
    for answer,key in [('YES','min_yes_precision_lower'),('NO','min_no_precision_lower')]:
        n=1
        while wilson_lower(n,n)<CRITERIA[key]:n+=1
        out[answer]=dict(min_all_correct_judgments=n,min_distinct_routes=CRITERIA['min_routes'])
    return dict(answers=out,min_total_judgments=CRITERIA['min_judgments'],
                note='Necessary best-case counts, not a promise of approval. UNKNOWN human judgments count as errors.')


def report(samples):
    out={}
    for answer in ('YES','NO'):
        rows=[s for s in samples if s['rule_target']==answer]
        right=sum(s['reference']==answer for s in rows)
        out[answer]=dict(physical_routes=len({s['physical_group'] for s in rows}),judgments=len(rows),correct=right,precision=right/len(rows) if rows else None,
                         precision_lower_95=wilson_lower(right,len(rows)))
    # Reference UNKNOWN is not removed from precision denominators.
    return dict(answers=out,judgments=len(samples),physical_routes=len({s['physical_group'] for s in samples}),
        abstention_rate=sum(s['rule_target']=='UNKNOWN' for s in samples)/len(samples) if samples else None,
        transition_judgments=sum(s['near_transition'] for s in samples),
        sampling_scope='audited sample; population abstention belongs to production ledger')


def weak_registry(rule_classes,experiment):
    if not isinstance(experiment,str) or not experiment.strip():raise ValueError('name the weak-supervision experiment')
    from .teacher_rules import rule_class
    from .taxonomy import EVENTS,event_edges,COMMON
    from .route_context import RECOVER_FOLLOW
    allowed={rule_class(event,edge.key,mode) for event in ('U-E1','U-E4')
             for edge in (*event_edges(EVENTS[event][1]),*COMMON,RECOVER_FOLLOW) for mode in (2,4)}
    if not rule_classes or set(rule_classes)-allowed:raise ValueError('unsupported weak teacher classes')
    return dict(policy=WEAK_POLICY,teacher=identity(),experiment=experiment,rule_classes=sorted(set(rule_classes)),
                train_only=True,independently_approved=False)


def best_case_feasibility(samples):
    """Upper bound assuming every binary teacher judgment is correct."""
    from .teacher_review import observable
    unique={observable(q):q for q in samples};qs=list(unique.values());budget=budget_requirements();deficits={}
    for answer in ('YES','NO'):
        rows=[q for q in qs if q['rule_target']==answer];b=budget['answers'][answer]
        deficits[answer]=dict(judgments=max(0,b['min_all_correct_judgments']-len(rows)),
            routes=max(0,b['min_distinct_routes']-len({q['physical_group'] for q in rows})))
    transitions=max(0,CRITERIA['min_transition_judgments']-sum(q.get('near_transition',False) for q in qs))
    return dict(potentially_passable=not transitions and len(qs)>=CRITERIA['min_judgments'] and not any(v for d in deficits.values() for v in d.values()),
                answer_deficits=deficits,transition_deficit=transitions,judgment_deficit=max(0,CRITERIA['min_judgments']-len(qs)),
                assumption='all binary judgments correct; feasibility is not approval')


def validate_registry(registry):
    if registry.get('policy')==WEAK_POLICY:
        expected=weak_registry(registry.get('rule_classes'),registry.get('experiment'))
        if registry!=expected:raise ValueError('invalid weak experimental registry')
        return dict(approved={},decisions={},weak_classes={c:digest([WEAK_POLICY,c,registry]) for c in registry['rule_classes']},
                    registry_sha256=digest(registry),teacher_sha256=digest(registry['teacher']),supervision_policy='weak_experiment_train_only')
    if registry.get('policy')!=POLICY or registry.get('teacher')!=identity():raise ValueError('stale or invalid teacher registry')
    if registry.get('criteria')!=CRITERIA:raise ValueError('approval criteria must match frozen policy')
    # Validate actual frozen physical ownership, not arbitrary split strings.
    from .teacher_replay import validate_question
    from .dataset import groups,holdout_reservations,source_route,split_for,producer_check_reservations
    exposed=groups();reserved=holdout_reservations();independent=producer_check_reservations()
    approved={};decisions={};seen=set()
    for record in registry.get('rules',[]):
        cls=record['rule_class']
        if cls in seen:raise ValueError('duplicate rule approval')
        seen.add(cls)
        if record.get('teacher')!=registry['teacher']:raise ValueError('rule source mismatch')
        calibration=record['development'];audit=record['independent']
        if (calibration.get('source')!='development_rgb_review' or not calibration.get('samples')
                or audit.get('source')!='independent_rgb_review' or not audit.get('reviewer')
                or audit['reviewer'] in record.get('rule_authors',[]) or not record.get('rule_authors')
                or audit.get('independence_attested') is not True or audit.get('blind_to_rule_answers') is not True
                or audit.get('frozen_before_review') is not True):
            raise ValueError('independent review and prior development calibration are required')
        from .teacher_review import validate_binding,observable
        validate_binding(calibration,'development');validate_binding(audit,'independent')
        identifiers=set();inputs=set();dev_routes=set();audit_routes=set()
        for kind,reference in [('development',calibration),('independent',audit)]:
            for sample in reference['samples']:
                key=sample['question_id']
                if key in identifiers:raise ValueError('duplicate/cross-calibration reference')
                identifiers.add(key)
                if (sample.get('rule_class')!=cls or sample.get('rule_target') not in ('YES','NO','UNKNOWN')
                        or sample.get('reference') not in ('YES','NO','UNKNOWN')
                        or type(sample.get('near_transition')) is not bool
                        or not sample.get('input_sha256') or not sample.get('teacher_record_sha256')
                        or not sample.get('rgb_review_evidence_id')):raise ValueError('invalid independent observation')
                question=validate_question(sample['question'],expected_teacher_sha=digest(registry['teacher']))
                if (sample['teacher_record_sha256']!=digest(question)
                        or any(sample[k]!=question[k] for k in ('question_id','rule_class','rule_target','input_sha256','scenario','route_id','physical_group'))):
                    raise ValueError('review is detached from frozen teacher question/input')
                from .route_prompts import state_pair
                from .controller import Episode
                input_key=observable(question)
                if input_key in inputs:raise ValueError('duplicate observable question in approval sample')
                inputs.add(input_key)
                group=source_route(sample)
                if group!=sample['physical_group']:raise ValueError('reference physical identity mismatch')
                split=split_for(group,exposed,reservations=reserved)
                if question['split']!=split:raise ValueError('review question split mismatch')
                if kind=='development':
                    if split!='train':raise ValueError('calibration consumes non-training route')
                    dev_routes.add(group)
                else:
                    if group not in independent or split not in ('val','test'):raise ValueError('unreserved independent reference')
                    audit_routes.add(group)
        if dev_routes&audit_routes:raise ValueError('calibration/independent route leakage')
        measured=report(audit['samples']);passed=(measured['judgments']>=CRITERIA['min_judgments']
            and measured['physical_routes']>=CRITERIA['min_routes']
            and all(measured['answers'][answer]['physical_routes']>=CRITERIA['min_routes'] for answer in ('YES','NO'))
            and measured['transition_judgments']>=CRITERIA['min_transition_judgments']
            and measured['answers']['YES']['precision_lower_95']>=CRITERIA['min_yes_precision_lower']
            and measured['answers']['NO']['precision_lower_95']>=CRITERIA['min_no_precision_lower'])
        decisions[cls]=dict(approved=passed,measured=measured,record_sha256=digest(record))
        if passed:approved[cls]=digest(record)
    return dict(approved=approved,decisions=decisions,registry_sha256=digest(registry),teacher_sha256=digest(registry['teacher']))


def target(question,approval):
    if question['rule_target'] not in ('YES','NO'):return 'UNKNOWN','rule_abstained'
    if approval.get('supervision_policy')=='weak_experiment_train_only':
        if question['split']!='train':return 'UNKNOWN','weak_teacher_train_only'
        if question['edge']=='restrict':return 'UNKNOWN','restriction_seed_tautology'
        if (question['episode']['event']=='U-E1' and question['edge']=='proceed'
                and question['rule_target']=='YES' and question.get('ego_motion')=='moving'):
            return 'UNKNOWN','moving_proceed_requires_visual_catchup'
        if question['rule_class'] in approval['weak_classes']:return question['rule_target'],'weak_rule_experiment'
    if question['rule_class'] not in approval['approved']:return 'UNKNOWN','rule_class_not_approved'
    return question['rule_target'],'approved_rule_class'


def main():
    import argparse,json
    from pathlib import Path
    from .identity import write_json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path)
    p.add_argument('--weak-experiment')
    p.add_argument('--rule-class',action='append')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('use a new approval report')
    if args.weak_experiment:
        if args.registry:p.error('choose registry validation or a weak experiment')
        result=weak_registry(args.rule_class,args.weak_experiment)
    else:
        if not args.registry:p.error('--registry required')
        result=validate_registry(json.loads(args.registry.read_text()))
    write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
