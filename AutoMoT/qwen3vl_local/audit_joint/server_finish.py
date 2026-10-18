"""Finish an existing current-contract pilot into automatic rule data, then stop before GPU."""
import argparse
import json
from pathlib import Path
import time

from .io import digest, file_sha, write_json

DECODED_LIMIT = 128_000_000
ADMISSION_LIMIT = 64_000_000


def raw_training_support(index):
    """Necessary pre-compilation gate, not certification of the filtered pool.

    Keep these cells aligned with rule_baseline.measure (regression checked).
    A zero native readiness count cannot become a valid automatic training row.
    """
    from qwen3vl_local.sft_new_loop_phase4.teacher_rules import rule_class
    from .rule_baseline import CRITICAL_EDGES
    critical = {}
    for event, edge in CRITICAL_EDGES:
        for answer in ('YES', 'NO'):
            key = rule_class(event, edge, 4, 'readiness') + '/' + answer
            receipts = [r for r in index['routes'] if r['split'] == 'train' and r['targets'].get(key, 0) > 0]
            critical[f'{event}/{edge}/{answer}'] = dict(
                rows=sum(r['targets'][key] for r in receipts),
                physical_groups=len({r['physical_group'] for r in receipts}))
    missing = [k for k, v in critical.items() if not v['rows']]
    return dict(status='missing_raw_support' if missing else 'raw_support_present',
                critical_support=critical, missing_critical_support=missing,
                scope='necessary raw train/readiness/RGB4 support only; final filtered admission still required')


def build(pilot, root, output, model_dir, workers=4, *, coverage_policy='strict', reuse_complete_replay=False):
    from qwen3vl_local.sft_new_loop_phase4 import candidate_pool, teacher_replay, dataset, teacher_data, teacher_evaluation
    from qwen3vl_local.sft_new_loop_phase4.full_pipeline import default_registry
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.data import create
    from .first_epoch_budget import profile, request
    from .rule_baseline import check, coverage_decision
    result = dict(status='building',stage='inputs',stages={},errors=[],training_started=False,
                  coverage_policy=coverage_policy,unsupported_capabilities=[])
    started = time.monotonic()
    def save():
        result['elapsed_seconds'] = time.monotonic()-started
        pending=output/'build_result.pending'
        pending.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        pending.replace(output/'build_result.json')
    def stage(name):
        result['stage'] = name; save()
        print(json.dumps(dict(stage=name,elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
    try:
        coverage_decision([],coverage_policy)
        pilot = Path(pilot).absolute(); output = Path(output).absolute()
        if pilot.resolve() == output.resolve() or output.resolve().is_relative_to(pilot.resolve()):
            raise ValueError('build output must be fresh and outside pilot inputs')
        if pilot.stat().st_dev != output.stat().st_dev:
            raise ValueError('pilot replay and output must share the guarded filesystem')
        pool = candidate_pool.validate(json.loads((pilot/'candidates.json').read_text()))
        result['candidate_pool_sha256'] = file_sha(pilot/'candidates.json')
        from .server_context import context
        result['server_context'] = context(pilot=pilot,output=output,data_root=root,model_dir=model_dir)
        result['reuse_complete_replay'] = reuse_complete_replay
        # Budget topology first, so a missing model config cannot waste a full build.
        stage('checkpoint_budget_profile')
        budget_profile = profile(model_dir)
        write_json(output/'model_budget_profile.json',budget_profile)
        stage('complete_replay')
        if reuse_complete_replay:
            from .replay_recovery import recover
            replay = recover(pool,pilot/'production')
        else:
            replay = teacher_replay.produce(
                pool,root,pilot/'production',workers=workers,compression='gzip')
        ambiguous=replay.get('history_ambiguous_question_ids',[])
        result['stages']['complete_replay'] = {k:v for k,v in replay.items() if k!='history_ambiguous_question_ids'}
        result['stages']['complete_replay'].update(history_ambiguous_count=len(ambiguous),
                                                  history_ambiguous_ids_sha256=digest(ambiguous))
        if not replay['complete_causal_production']:
            raise ValueError('incomplete replay; do not publish a partial shared collection')
        stage('raw_training_support')
        index_path = pilot/'production/index.json'
        if file_sha(index_path) != replay['index_sha256']:
            raise ValueError('replay index changed before raw support check')
        raw_support = raw_training_support(json.loads(index_path.read_text()))
        if file_sha(index_path) != replay['index_sha256']:
            raise ValueError('replay index changed during raw support check')
        raw_support['index_sha256'] = replay['index_sha256']
        decision=coverage_decision(raw_support['missing_critical_support'],coverage_policy)
        raw_support.update(coverage_policy=coverage_policy,**decision)
        write_json(output/'raw_training_support.json', raw_support)
        result['stages']['raw_training_support'] = raw_support
        result['unsupported_capabilities']=decision['unsupported_capabilities']
        if decision['blocking_missing_support']:
            result.update(status='blocked', stage='finished_cpu', compilation_skipped=True,
                blocking_stage='raw_training_support',
                errors=['raw training replay missing critical support: ' + str(decision['blocking_missing_support'])])
            save()
            return result
        stage('rgb4_rule_dataset')
        registry = default_registry()
        anns, selection = teacher_data.compile_bundle(pool,pilot/'production/index.json',registry,rgb_mode=4,max_per_route=0)
        manifest = dataset.build(anns,root,output/'data4',rgb_mode=4,candidate_pool=pool,
            teacher_registry=registry,production_index=pilot/'production/index.json',
            teacher_selection={k:selection[k] for k in ('max_per_route','selection_policy')} |
                              dict(route_budget_excluded=selection['counts']['route_budget_excluded']))
        del anns
        result['stages']['rgb4_rule_dataset'] = dict(counts=manifest['counts'],manifest_sha256=file_sha(output/'data4/manifest.json'))
        stage('automatic_holdout')
        refs = teacher_evaluation.export(pool,pilot/'production/index.json',root,output/'references4',rgb_mode=4)
        result['stages']['automatic_holdout'] = refs['files']
        stage('student_view')
        view = create(output/'data4',output/'view',data_root=root)
        result['stages']['student_view'] = dict(counts=view['counts'],observations=view['observation_contracts'])
        stage('first_epoch_budget')
        stored = {name:file_sha(output/name) for name in ('data4/manifest.json','references4/references.json','view/student_view.json')}
        budget = request(budget_profile,output,stored_evidence=stored)
        write_json(output/'first_epoch_budget.json',budget)
        stage('automatic_admission')
        admission = check(output/'view',output/'references4',budget=budget,space_path=output,
                          coverage_policy=coverage_policy,data_root=root)
        write_json(output/'automatic_admission.json',admission)
        result.update(status=admission['status'],stage='finished_cpu',errors=admission['errors'],
                      unsupported_capabilities=admission['unsupported_capabilities'])
    except Exception as exc:
        result.update(status='blocked',errors=[f'{type(exc).__name__}: {exc}'])
    save()
    return result


def package(output, exit_code, *, destination=None):
    from .rule_baseline import ADMITTED
    from .server_pilot import package as make_package
    from .training_budget import plan
    output = Path(output)
    target = Path(destination) if destination is not None else output/'handoff'
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'preserve existing handoff; choose a fresh destination: {target}')
    if destination is not None and output.resolve().is_relative_to(target.resolve()):
        raise ValueError('handoff destination must not contain the original output')
    names = ('build_storage.json','source_check.json','build_result.json','model_budget_profile.json','raw_training_support.json','first_epoch_budget.json','automatic_admission.json')
    members, diagnostics = {}, []
    # JSON can compress well below 30MB; bound decoded memory separately.
    # Keep the final ZIP's existing 30MB hard limit in make_package.
    remaining = DECODED_LIMIT - 2_000_000

    def failure(path, exc):
        record = dict(path=str(path), error=f'{type(exc).__name__}: {exc}'[:2000])
        try:
            record.update(bytes=path.stat().st_size, sha256=file_sha(path))
            with path.open('rb') as stream:
                record['preview'] = stream.read(1024).decode(errors='replace')
        except (OSError, RuntimeError) as read_error:
            record['read_error'] = str(read_error)[:1000]
        diagnostics.append(record)

    def add(name, value):
        nonlocal remaining
        size = len(json.dumps(value, ensure_ascii=False, indent=2).encode()) + 1
        if size > remaining:
            raise ValueError('diagnostic member exceeds remaining package budget')
        remaining -= size
        members[name] = value

    def read(path):
        if not path.exists():
            return None
        try:
            limit = ADMISSION_LIMIT if path.name == 'automatic_admission.json' else 10_000_000
            if path.stat().st_size > limit:
                raise ValueError(f'report exceeds {limit} byte diagnostic input limit')
            before = file_sha(path)
            with path.open('rb') as stream:
                raw = stream.read(limit + 1)
            if len(raw) > limit:
                raise ValueError(f'report exceeds {limit} byte diagnostic input limit')
            import hashlib
            if hashlib.sha256(raw).hexdigest() != before or file_sha(path) != before:
                raise ValueError('report changed during collection')
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError('expected a JSON object')
            return data
        except (OSError, ValueError, RuntimeError, RecursionError) as exc:
            failure(path, exc)
            return None

    for name in names:
        value = read(output/name)
        if value is not None:
            try:
                add(name, value)
            except (ValueError, RecursionError) as exc:
                failure(output/name, exc)
    result = members.get('build_result.json',dict(status='interrupted_before_report'))
    # Keep the original build report byte semantics: a completed admission
    # rejection (exit 2) is not an interrupted build. Record execution outcome
    # separately, including failures after a report claimed success.
    if exit_code == 0 and result.get('status') in ADMITTED:
        status = result['status']
    elif exit_code == 2 and result.get('stage') == 'finished_cpu' and result.get('status') == 'blocked':
        status = 'admission_blocked'
    else:
        status = 'blocked_or_interrupted'
    completion=dict(status=status,command_exit_code=exit_code,
                    build_result_exists=(output/'build_result.json').is_file())
    if completion['build_result_exists']:
        completion.update(build_status=result.get('status'),build_stage=result.get('stage'),
                          coverage_policy=result.get('coverage_policy','strict'),
                          unsupported_capabilities=result.get('unsupported_capabilities',[]))
    for key, value in list(completion.items()):
        if len(json.dumps(value, ensure_ascii=False, indent=2).encode()) > 16_000:
            completion[key] = dict(omitted=True, canonical_sha256=digest(value))
            failure(output/'build_result.json', ValueError(f'oversized completion field: {key}'))
    if 'first_epoch_budget.json' in members:
        try:
            add('budget_plan.json', plan(members['first_epoch_budget.json']))
        except (ValueError, TypeError, AttributeError, RecursionError) as exc:
            failure(output/'first_epoch_budget.json', exc)
    for folder, filename in (('data4','manifest.json'),('references4','references.json'),('view','student_view.json')):
        p = output/folder/filename
        data = read(p)
        if data is not None:
            try:
                add(f'{folder}_summary.json', dict(path=str(p),sha256=file_sha(p),
                    declared={k:data[k] for k in ('counts','files','production_reference_sha256','production_index_sha256') if k in data}))
            except (OSError, ValueError, RecursionError) as exc:
                failure(p, exc)
    log = output/'build.log'
    if log.exists():
        try:
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-100_000))
                members['log_tail.json'] = dict(sha256=file_sha(log),tail=stream.read().decode(errors='replace'))
        except OSError as exc:
            failure(log, exc)
    if diagnostics:
        completion.update(execution_status=completion['status'],status='blocked_or_interrupted', diagnostics_complete=False,
                          diagnostic_errors=len(diagnostics))
        members['collection_errors.json'] = dict(errors=diagnostics,
            scope='Incomplete diagnostics; ZIP integrity does not grant data admission')
    members['finish_receipt.json']=completion
    from .server_context import context
    members['server_context.json']=context(output=output)
    members['server_context.json']['collector_sha256']=file_sha(Path(__file__))
    target.mkdir(parents=True,exist_ok=False)
    receipt=make_package(target,members,uncompressed_limit=DECODED_LIMIT)
    (target/'pilot_report.zip').rename(target/'fullflow_report.zip')
    (target/'pilot_package.json').rename(target/'fullflow_package.json')
    # Repack never overwrites the original completion or its old handoff.
    write_json((target if destination is not None else output)/'finish_receipt.json',completion)
    return receipt


def main():
    from .rule_baseline import COVERAGE_POLICIES, ADMITTED
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('build','prepare','package'))
    p.add_argument('--pilot',type=Path)
    p.add_argument('--reuse-complete-replay',action='store_true',
                   help='build only: validate complete existing replay; never recompute routes')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('lead_data'))
    p.add_argument('--model-dir',type=Path,default=Path('checkpoints/Qwen3.5-4B'))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--exit-code',type=int,default=0)
    p.add_argument('--handoff-output',type=Path,help='package only: fresh destination; preserve original receipt/handoff')
    p.add_argument('--coverage-policy',choices=COVERAGE_POLICIES,default='strict')
    a=p.parse_args()
    if a.reuse_complete_replay and a.command!='build':
        p.error('--reuse-complete-replay requires build --pilot')
    if a.command=='package':
        print(json.dumps(package(a.output,a.exit_code,destination=a.handoff_output)));return 0
    if a.handoff_output:p.error('--handoff-output is only valid with package')
    if (a.output/'build_result.json').exists() or any((a.output/n).exists() for n in ('data4','references4','view')):
        p.error('use a fresh output; partial outputs are preserved')
    if a.command=='prepare':
        if a.pilot:p.error('prepare creates a fresh inventory of existing raw data; do not pass --pilot')
        from qwen3vl_local.sft_new_loop_phase4 import candidate_pool
        a.pilot=a.output/'inputs'
        try:
            # Fail before even scanning the existing dataset when assets are absent.
            from .local_paths import check_inputs
            local = check_inputs(a.data_root, a.model_dir, a.output, require_fresh_output=False)
            if local['errors']:
                raise ValueError('; '.join(local['errors']))
            a.pilot.mkdir(exist_ok=False)
            candidate_pool.scan(a.data_root,a.pilot/'candidates.json')
        except Exception as ex:
            write_json(a.output/'build_result.json',dict(status='blocked',stage='candidate_inventory',
                coverage_policy=a.coverage_policy,training_started=False,
                errors=[f'{type(ex).__name__}: {ex}']))
            return 2
    elif not a.pilot:p.error('--pilot required')
    r=build(a.pilot,a.data_root,a.output,a.model_dir,a.workers,coverage_policy=a.coverage_policy,reuse_complete_replay=a.reuse_complete_replay)
    print(json.dumps(dict(status=r['status'],stage=r['stage'],errors=r['errors']),ensure_ascii=False))
    return 0 if r['status'] in ADMITTED else 2


if __name__=='__main__':raise SystemExit(main())
