"""CPU admission for the short-view, automatic-reference first-epoch baseline.

Uses the same filtered pools, selection and sampling code as the trainer.
This is neither server readiness nor joint G0/independent label certification.
"""
import argparse
import json
from pathlib import Path
import shutil

from .io import digest, write_json
from .training_scope import dataset_bindings, phase4_support, phase4_schedule
from .training_budget import plan as disk_plan

COVERAGE_POLICIES = ('strict', 'existing_data_development')
DEVELOPMENT_GAPS = frozenset({'U-E5/proceed/YES'})
ADMITTED = ('admitted_rule_baseline', 'admitted_existing_data_development')
CRITICAL_EDGES = (('U-E2','return'), ('U-E3','proceed'), ('U-E5','proceed'),
                  ('U-E6','proceed'), ('U-E7','complete'), ('R-E5','proceed'))


def coverage_decision(missing, policy):
    """Separate this known capability gap from data validity; never fill a cell."""
    if policy not in COVERAGE_POLICIES:
        raise ValueError('unknown coverage policy')
    allowed = DEVELOPMENT_GAPS if policy == 'existing_data_development' else frozenset()
    return dict(blocking_missing_support=[k for k in missing if k not in allowed],
                unsupported_capabilities=[k for k in missing if k in allowed])


def budget_check(request, space_path):
    if request.get('modes') != 2 or request.get('retained_checkpoints_per_mode') != 1:
        raise ValueError('first-epoch budget must cover two modes and one checkpoint per mode')
    if request.get('reserve_bytes', 0) < 20 * 1024**3:
        raise ValueError('first-epoch budget requires at least 20 GiB reserve')
    parent=Path(space_path).absolute()
    while not parent.exists():
        parent=parent.parent
    current_free=shutil.disk_usage(parent).free
    # Previously reported capacity is not a guarantee at launch time.
    request=dict(request,free_bytes=min(request['free_bytes'],current_free))
    result=disk_plan(request)
    result.update(space_path=str(parent.resolve()),current_free_bytes=current_free)
    return result


def critical_support(rows):
    """Automatic train/readiness evidence only; other edges cannot fill a cell."""
    from .training_scope import original_basis
    rows=[r for r in rows if r['split']=='train' and original_basis(r) in ('weak_rule_teacher','approved_rule_teacher')]
    critical={}
    for event,edge in CRITICAL_EDGES:
        for answer in ('YES','NO'):
            matches=[r for r in rows if r['episode']['event']==event and r['edge']==edge
                     and r['slice']=='readiness' and r['target']==answer]
            critical[f'{event}/{edge}/{answer}']=dict(rows=len(matches),physical_groups=len({r['physical_group'] for r in matches}))
    return dict(critical_support=critical,missing_critical_support=[k for k,v in critical.items() if not v['rows']])


def measure(data, evaluation, *, batch_size=None, anchor_stride=2):
    support=phase4_support(data)
    critical=critical_support(data['train'])
    if batch_size is None:
        schedule=phase4_schedule(data['train'],1)
    else:
        from qwen3vl_local.sft_new_loop_phase4.rgb_short.batch_sampling import plan
        from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
        worlds={}
        for world in (1,4):
            indices,audit=plan(data['train'],epoch=0,world_size=world,batch_size=batch_size,
                               anchor_stride=anchor_stride,expected_events=EVENTS)
            audit.pop('next_history')
            audit['events']={e:dict(presentations=n) for e,n in audit['event_counts'].items()}
            worlds[str(world)]=[audit]
        schedule=dict(worlds=worlds,world1_world4_order_equal=worlds['1'][0]['ordered_ids_sha256']==worlds['4'][0]['ordered_ids_sha256'],
            policy='contiguous global batches; membership/event quotas fixed, rank layout may change order')
    ids={s:sorted(r['id'] for r in data[s]) for s in ('train','val','test')}
    for world in ('1','4'):
        if any(v['presentations']!=1024 for v in schedule['worlds'][world][0]['events'].values()):
            raise ValueError('first epoch requires 1024 presentations per event')
    if batch_size is None and not schedule['world1_world4_order_equal']:
        raise ValueError('world1/world4 sampling order differs')
    compact_evaluation={k:v for k,v in evaluation.items() if k not in ('train_ids','plans','exclusions')}
    compact_evaluation['exclusion_count']=len(evaluation['exclusions'])
    compact_evaluation['plans']={split:{k:v for k,v in record.items() if k!='selected_ids'}
                                 for split,record in evaluation['plans'].items()}
    return dict(support=support,**critical,
                schedule=schedule,selected_counts={s:len(v) for s,v in ids.items()},
                selected_ids_sha256=digest(ids),train_ids_sha256=digest(ids['train']),evaluation=compact_evaluation)


def check(dataset, references, *, budget=None, space_path=None, coverage_policy='strict', data_root=None, batch_size=1):
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.data import load_dataset
    from qwen3vl_local.sft_new_loop_phase4.full_sampling import validate_dataset_scope
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.evaluation_policy import prepare,pack_binding
    report=dict(status='blocked',selection_mode='rule_consistency',epochs=1,modes={},errors=[],
                coverage_policy=coverage_policy,unsupported_capabilities=[],
                independent_accuracy_certified=False,joint_g0_certified=False,
                all_critical_capabilities_supported=False)
    try:
        coverage_decision([], coverage_policy)
        before=dataset_bindings(dataset)
        ref_before=None
        try:
            ref_before=pack_binding(references)
        except (ValueError,OSError,KeyError,TypeError) as ex:
            report['reference_error']=f'{type(ex).__name__}: {ex}'
            report['errors'].append('references: '+report['reference_error'])
        for mode in (2,4):
            try:
                raw,manifest=load_dataset(dataset,rgb_mode=mode,require_trainable=False)
                validate_dataset_scope(manifest)
                # Preserve both modes' source diagnostics even if one fails admission.
                source_critical=critical_support(raw['train'])
                report['modes'][str(mode)]=dict(source_support=phase4_support(raw),
                    source_automatic_support=dict(**source_critical,
                        scope='native loaded automatic training rows; not final evaluation or training admission'))
                source_decision=coverage_decision(source_critical['missing_critical_support'],coverage_policy)
                report['modes'][str(mode)]['source_coverage_decision']=source_decision
                report['unsupported_capabilities']=sorted(set(report['unsupported_capabilities']) |
                                                         set(source_decision['unsupported_capabilities']))
                if source_decision['blocking_missing_support']:
                    report['errors'].append(f'RGB{mode} source missing automatic critical support: '+
                                            str(source_decision['blocking_missing_support']))
                if ref_before is None:
                    report['modes'][str(mode)]['error']='automatic reference inputs unavailable; selection not run'
                    continue
                data,evaluation=prepare(raw,mode='rule_consistency',references=references,
                                        rgb_mode=mode,short=True,seed=20260929,data_root=data_root or Path(__file__).resolve().parents[2]/'lead_data')
                result=measure(data,evaluation,batch_size=batch_size,anchor_stride=2 if mode==2 else 1)
                report['modes'][str(mode)].update(result)
                decision=coverage_decision(result['missing_critical_support'],coverage_policy)
                report['modes'][str(mode)]['coverage_decision']=decision
                report['unsupported_capabilities']=sorted(set(report['unsupported_capabilities']) |
                                                         set(decision['unsupported_capabilities']))
                if decision['blocking_missing_support']:
                    report['errors'].append(f'RGB{mode} missing critical support: '+str(decision['blocking_missing_support']))
            except (ValueError,OSError,KeyError,TypeError) as ex:
                report['modes'].setdefault(str(mode),{})['error']=f'{type(ex).__name__}: {ex}'
                report['errors'].append(f'RGB{mode}: {type(ex).__name__}: {ex}')
        if all('selected_ids_sha256' in report['modes'][str(mode)] for mode in (2,4)):
            two,four=report['modes']['2'],report['modes']['4']
            if two['train_ids_sha256']!=four['train_ids_sha256']:
                raise ValueError('RGB2/RGB4 training membership differs')
            # Evaluation remains event-equal and label-blind, with membership
            # chosen using each mode's actual anchor stride.
        if before!=dataset_bindings(dataset) or (ref_before is not None and ref_before!=pack_binding(references)):
            raise ValueError('baseline inputs changed during admission')
        report['input_bindings']=dict(dataset=before,references=ref_before)
    except (ValueError,OSError,KeyError,TypeError) as ex:
        report['errors'].append(f'{type(ex).__name__}: {ex}')
    try:
        if budget is None:
            raise ValueError('missing first-epoch budget')
        report['budget']=budget_check(budget,space_path)
        if report['budget']['sufficient'] is not True:
            report['errors'].append('first-epoch budget incomplete or insufficient')
    except (ValueError,OSError,KeyError,TypeError) as ex:
        report['errors'].append(f'budget: {ex}')
    if not report['errors']:
        report['status']=ADMITTED[coverage_policy=='existing_data_development']
        report['all_critical_capabilities_supported']=all(
            not m['missing_critical_support'] for m in report['modes'].values())
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--teacher-references',type=Path,required=True)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--budget',type=Path)
    p.add_argument('--space-path',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--coverage-policy',choices=COVERAGE_POLICIES,default='strict')
    a=p.parse_args()
    from .training_scope import check_output
    check_output(a.output,a.dataset)
    if a.output.exists() or a.output.resolve().is_relative_to(a.teacher_references.resolve()):
        raise ValueError('use a fresh report outside reference inputs')
    budget=None;budget_error=None
    if a.budget:
        try:
            budget=json.loads(a.budget.read_text())
            if not isinstance(budget,dict):
                raise ValueError('budget request must be an object')
        except (OSError,ValueError) as ex:
            budget=None;budget_error=f'budget input {a.budget}: {type(ex).__name__}: {ex}'
    report=check(a.dataset,a.teacher_references,budget=budget,space_path=a.space_path,coverage_policy=a.coverage_policy,data_root=a.data_root,batch_size=a.batch_size)
    if budget_error:
        report['errors'].append(budget_error)
    write_json(a.output,report)
    print(json.dumps({k:report[k] for k in ('status','coverage_policy','unsupported_capabilities','errors')},ensure_ascii=False))
    return 0 if report['status'] in ADMITTED else 2


if __name__=='__main__':raise SystemExit(main())
