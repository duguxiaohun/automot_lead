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
    priorities=[('U-E2','return'),('U-E3','proceed'),('U-E5','proceed'),
                ('U-E6','proceed'),('U-E7','complete')]
    critical={}
    for event,edge in priorities:
        for answer in ('YES','NO'):
            matches=[r for r in rows if r['episode']['event']==event and r['edge']==edge
                     and r['slice']=='readiness' and r['target']==answer]
            critical[f'{event}/{edge}/{answer}']=dict(rows=len(matches),physical_groups=len({r['physical_group'] for r in matches}))
    return dict(critical_support=critical,missing_critical_support=[k for k,v in critical.items() if not v['rows']])


def measure(data, evaluation):
    support=phase4_support(data)
    critical=critical_support(data['train'])
    schedule=phase4_schedule(data['train'],1)
    ids={s:[r['id'] for r in data[s]] for s in ('train','val','test')}
    for world in ('1','4'):
        if any(v['presentations']!=1024 for v in schedule['worlds'][world][0]['events'].values()):
            raise ValueError('first epoch requires 1024 presentations per event')
    if not schedule['world1_world4_order_equal']:
        raise ValueError('world1/world4 sampling order differs')
    compact_evaluation={k:v for k,v in evaluation.items() if k not in ('train_ids','plans','exclusions')}
    compact_evaluation['exclusion_count']=len(evaluation['exclusions'])
    compact_evaluation['plans']={split:{k:v for k,v in record.items() if k!='selected_ids'}
                                 for split,record in evaluation['plans'].items()}
    return dict(support=support,**critical,
                schedule=schedule,selected_counts={s:len(v) for s,v in ids.items()},
                selected_ids_sha256=digest(ids),evaluation=compact_evaluation)


def check(dataset, references, *, budget=None, space_path=None):
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.data import load_dataset
    from qwen3vl_local.sft_new_loop_phase4.full_sampling import validate_dataset_scope
    from qwen3vl_local.sft_new_loop_phase4.evaluation_policy import prepare,pack_binding
    report=dict(status='blocked',selection_mode='rule_consistency',epochs=1,modes={},errors=[],
                independent_accuracy_certified=False,joint_g0_certified=False)
    try:
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
                if source_critical['missing_critical_support']:
                    report['errors'].append(f'RGB{mode} source missing automatic critical support: '+
                                            str(source_critical['missing_critical_support']))
                if ref_before is None:
                    report['modes'][str(mode)]['error']='automatic reference inputs unavailable; selection not run'
                    continue
                data,evaluation=prepare(raw,mode='rule_consistency',references=references,
                                        rgb_mode=mode,short=True,seed=20260929)
                result=measure(data,evaluation)
                report['modes'][str(mode)].update(result)
                if result['missing_critical_support']:
                    report['errors'].append(f'RGB{mode} missing critical support: '+str(result['missing_critical_support']))
            except (ValueError,OSError,KeyError,TypeError) as ex:
                report['modes'].setdefault(str(mode),{})['error']=f'{type(ex).__name__}: {ex}'
                report['errors'].append(f'RGB{mode}: {type(ex).__name__}: {ex}')
        if all('selected_ids_sha256' in report['modes'][str(mode)] for mode in (2,4)):
            two,four=report['modes']['2'],report['modes']['4']
            if two['selected_ids_sha256']!=four['selected_ids_sha256'] or (
                two['schedule']['worlds']['1'][0]['ordered_ids_sha256'] !=
                four['schedule']['worlds']['1'][0]['ordered_ids_sha256']):
                raise ValueError('RGB2/RGB4 selected membership or sampling order differs')
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
    if not report['errors']:report['status']='admitted_rule_baseline'
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--teacher-references',type=Path,required=True)
    p.add_argument('--budget',type=Path)
    p.add_argument('--space-path',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
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
    report=check(a.dataset,a.teacher_references,budget=budget,space_path=a.space_path)
    if budget_error:
        report['errors'].append(budget_error)
    write_json(a.output,report)
    print(json.dumps(dict(status=report['status'],errors=report['errors']),ensure_ascii=False))
    return 0 if report['status']=='admitted_rule_baseline' else 2


if __name__=='__main__':raise SystemExit(main())
