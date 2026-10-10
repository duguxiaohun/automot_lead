"""Read-only support audit and inference on actually sampled training examples.

Kept outside the frozen student/producer contracts so existing adapters remain
loadable. This diagnoses fitting and distribution gaps, never grants new labels.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
import re
from pathlib import Path
import time

from .io import file_sha, write_rows
from .handoff import verify_package
from ..sft_new_loop_phase4.identity import digest, write_json
from ..sft_new_loop_phase4.paired_eval import load as load_cases, case_identity, validate as validate_case
from ..sft_new_loop_phase4.phase3_sampling import motion
from ..sft_new_loop_phase4.controller import Episode
from ..sft_new_loop_phase4.rgb_short.contract import check_contract
from ..sft_new_loop_phase4.rgb_short.data import load_dataset
from ..sft_new_loop_phase4.rgb_short.diagnostics import binary_diagnostics, sample_binding
from ..sft_new_loop_phase4.rgb_short.model import load_for_inference, generate
from ..sft_new_loop_phase4.rgb_short.train import adapter_identity
from ..sft_new_loop_phase4.evaluate import _reference_metrics
from ..sft_new_loop_phase4.taxonomy import EVENTS

FIT_FILES = ('fit_selection.jsonl', 'cases.jsonl', 'fit_metrics.json')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def natural(value):
    if type(value) is not int or value < 0:
        raise ValueError('invalid nonnegative count')
    return value


def file_binding(path):
    path = Path(path)
    resolved = str(path.resolve())
    sha = file_sha(path)
    if str(path.resolve()) != resolved:
        raise ValueError('logical file changed while hashing')
    return dict(resolved_path=resolved, sha256=sha)


def validation_metrics(cases):
    result = {}
    for name, teacher in (('transferred_reviewed_reference', False), ('teacher_consistency', True)):
        chosen = [r for r in cases if (r.get('original_reference_kind', r['reference_kind']) == 'rule_teacher') == teacher]
        measured = _reference_metrics(chosen)
        result[name] = dict(count=measured['count'], agreement=measured['accuracy'],
            event_macro_agreement_observed=measured['event_macro_accuracy_observed'],
            event_agreement=measured['event_accuracy'], missing_events=measured['missing_events'],
            malformed=measured['malformed'], confusion=measured['confusion'])
    result['selection_agreement'] = result['transferred_reviewed_reference']['event_macro_agreement_observed']
    return result


def coarse(row):
    return (row['episode']['event'], row['edge'], row['slice'], row['target'])


def state_cell(row):
    # Coarse support is not evidence of a matching state/branch/motion distribution.
    ep = Episode(**row['episode'])
    return (*coarse(row), ep.state, ep.longitudinal, ep.branch, ep.return_required, motion(row))


def label_basis(row):
    return row.get('student_reference', {}).get('original_label_basis', row['label_basis'])


def evidence(run_root, epoch):
    if type(epoch) is not int or epoch < 0:
        raise ValueError('epoch must be a nonnegative integer')
    root = Path(run_root)
    prefix = f'epoch_{epoch:03d}'
    run_binding = file_binding(root / 'run.json')
    run = read(root / 'run.json')
    check_contract(run['source'])
    world = natural(run['world_size'])
    if world < 1:
        raise ValueError('empty world')
    names = ['run.json', 'student_view.json', 'data_admission.json', prefix+'_sampling.json',
             prefix+'_samples.jsonl', prefix+'_cases.jsonl', prefix+'_validation.json',
             prefix+'_runtime.json', *[f'{prefix}_rank{r}_steps.jsonl' for r in range(world)]]
    bindings = {n: file_binding(root / n) for n in names}
    if bindings['run.json'] != run_binding:
        raise ValueError('run changed during evidence audit')
    hashes = {n: b['sha256'] for n,b in bindings.items()}
    view = read(root / 'student_view.json')
    if (view['source'] != run['source'] or
            view['selected_ids'] != run['training']['paired_training']['selected_ids'] or
            view['observation_contracts'][str(run['observation_contract']['rgb_mode'])] != run['observation_contract']):
        raise ValueError('run/view identity mismatch')
    samples = rows(root / (prefix+'_samples.jsonl'))
    sampling = read(root / (prefix+'_sampling.json'))
    if not samples or sampling['epoch'] != epoch or sampling['world_size'] != world:
        raise ValueError('sampling epoch/world mismatch')
    if sampling['presentations'] != len(samples) or len(samples) % world:
        raise ValueError('incomplete presentations')
    if sampling['ordered_samples'] != sample_binding(samples, world):
        raise ValueError('sample order binding mismatch')
    identities = {}
    for position, item in enumerate(samples):
        if (item['position'], item['rank'], item['local_index']) != (position, position % world, position // world):
            raise ValueError('sample position/rank mismatch')
        if item['target'] not in ('YES', 'NO'):
            raise ValueError('nonbinary sample')
        identity = (item['target'], item['original_label_basis'])
        if identities.setdefault(item['id'], identity) != identity:
            raise ValueError('same sampled ID has conflicting target or provenance')
    strata_targets = Counter()
    for key, count in sampling['strata'].items():
        _, _, answer, _, _ = key.split('/')
        strata_targets[answer] += natural(count)
    if strata_targets != Counter(item['target'] for item in samples):
        raise ValueError('sample targets differ from strata summary')
    runtime = read(root / (prefix+'_runtime.json'))
    if runtime['validation_status'] != 'complete' or sorted(r['rank'] for r in runtime['ranks']) != list(range(world)):
        raise ValueError('runtime incomplete')
    for rank in range(world):
        actual = rows(root / f'{prefix}_rank{rank}_steps.jsonl')
        expected = samples[rank::world]
        if len(actual) != len(expected):
            raise ValueError('missing executed steps')
        updates = 0
        accumulation = run['training']['accumulation']
        if type(accumulation) is not int or accumulation < 1:
            raise ValueError('invalid accumulation')
        for index, (step, item) in enumerate(zip(actual, expected)):
            if step['epoch'] != epoch or any(step[k] != item[k] for k in ('id', 'position', 'rank', 'local_index')):
                raise ValueError('executed step differs from selected sample')
            update = (index+1) % accumulation == 0 or index+1 == len(actual)
            updates += update
            if step['optimizer_update'] is not update or step['optimizer_steps'] != updates or not math.isfinite(step['loss']):
                raise ValueError('executed update/loss invalid')
        summary = next(r for r in runtime['ranks'] if r['rank'] == rank)
        if summary['optimizer_steps'] != updates or summary['microbatches'] != len(actual):
            raise ValueError('runtime does not match executed steps')
    cases, _ = load_cases(root / (prefix+'_cases.jsonl'))
    validation = read(root / (prefix+'_validation.json'))
    if (validation['dataset'] != run['dataset'] or validation['epoch'] != epoch or validation['split'] != 'val'
            or len(cases) != view['counts']['val'] or validation['count'] != len(cases)
            or validation['expected_count'] != len(cases) or digest(list(cases)) != view['selected_ids']['val']):
        raise ValueError('validation membership mismatch')
    if any(r['paired_identity']['fields']['split'] != 'val' for r in cases.values()):
        raise ValueError('non-val reference')
    if any(validation.get(k) != v for k,v in validation_metrics(list(cases.values())).items()):
        raise ValueError('validation metrics differ from cases')
    admission = read(root / 'data_admission.json')['training']
    if {s:v['rows'] for s,v in admission['counts'].items()} != view['counts']:
        raise ValueError('pool counts differ from view')
    if bindings != {n:file_binding(root/n) for n in names}:
        raise ValueError('run changed during evidence audit')
    return dict(run=run, view=view, samples=samples, sampling=sampling, cases=cases,
                admission=admission, hashes=hashes, bindings=bindings)


def support_report(ev):
    pool = ev['admission']['train_readiness_catchup_support']
    # These keys are exhaustive aggregates of train rows under the bound source.
    pool_total = sum(natural(n) for phases in pool.values() for answers in phases.values() for n in answers.values())
    if pool_total != ev['view']['counts']['train']:
        raise ValueError('incomplete full-pool support aggregates')
    sampled = Counter()
    for key, n in ev['sampling']['strata'].items():
        event, edge, answer, phase, _motion = key.split('/')
        if event not in EVENTS or answer not in ('YES','NO') or phase not in ('readiness','catchup') or _motion not in ('moving','stationary','unknown'):
            raise ValueError('invalid sampled support category')
        sampled[event, edge, phase, answer] += natural(n)
    if sum(sampled.values()) != len(ev['samples']):
        raise ValueError('incomplete sampled support aggregates')
    for (event,edge,phase,answer),number in sampled.items():
        if number and not pool.get(event+'/'+edge, {}).get(phase, {}).get(answer, 0):
            raise ValueError('sampled category absent from pool')
    grouped = defaultdict(list)
    for case in ev['cases'].values():
        grouped[coarse(case['paired_identity']['fields'])].append(case)
    cells = []
    for (event, edge, phase, answer), cases in sorted(grouped.items()):
        available = natural(pool.get(event+'/'+edge, {}).get(phase, {}).get(answer, 0))
        exposures = sampled[event, edge, phase, answer]
        if not available and exposures:
            raise ValueError('sampled category absent from pool')
        status = 'missing_pool_support' if not available else 'pool_present_not_sampled' if not exposures else 'sampled_fit_unknown'
        cells.append(dict(event=event, edge=edge, phase=phase, answer=answer, pool_rows=available,
                          sampled_presentations=exposures, val_rows=len(cases),
                          val_correct=sum(r['target']==r['prediction'] for r in cases),
                          val_physical_groups=len({r['paired_identity']['fields']['physical_group'] for r in cases}),
                          status=status, training_fit='not_run'))
    return dict(cells=cells, val_rows=len(ev['cases']),
                val_without_pool_support=sum(c['val_rows'] for c in cells if c['status']=='missing_pool_support'),
                sources_sha256=ev['hashes'], scope='aggregate category support, not state/instance matching or causal explanation of errors')


def select_training(data, ev, event, edge, phase='readiness'):
    train = {r['id']:r for r in data['train']}
    if len(train) != len(data['train']):
        raise ValueError('duplicate train IDs')
    if {s:digest([r['id'] for r in rs]) for s,rs in data.items()} != ev['view']['selected_ids']:
        raise ValueError('live dataset membership differs from run')
    if {r['id']:case_identity(r) for r in data['val']} != {k:r['paired_identity'] for k,r in ev['cases'].items()}:
        raise ValueError('live validation identities differ')
    counts = Counter(); strata = Counter()
    for sample in ev['samples']:
        row = train.get(sample['id'])
        if row is None or row['target'] != sample['target'] or label_basis(row) != sample['original_label_basis']:
            raise ValueError('sample is not the bound train row')
        counts[row['id']] += 1
        strata['/'.join((row['episode']['event'],row['edge'],row['target'],row['slice'],motion(row)))] += 1
    if dict(strata) != ev['sampling']['strata']:
        raise ValueError('live sampled strata differ')
    pool = Counter(coarse(r) for r in data['train'])
    recorded = ev['admission']['train_readiness_catchup_support']
    expected = Counter({(e.split('/')[0],e.split('/')[1],p,a):n
                       for e,ps in recorded.items() for p,answers in ps.items() for a,n in answers.items() if n})
    if pool != expected:
        raise ValueError('live full-pool support differs')
    selected = [train[key] for key in sorted(counts) if train[key]['episode']['event']==event and train[key]['edge']==edge
                and (phase=='all' or train[key]['slice']==phase)]
    return selected, counts


def distribution_report(data, counts, event, edge, phase='readiness'):
    match = lambda r:r['episode']['event']==event and r['edge']==edge and (phase=='all' or r['slice']==phase)
    pool = Counter(state_cell(r) for r in data['train'] if match(r))
    sampled = Counter()
    for r in data['train']:
        if match(r):
            sampled[state_cell(r)] += counts[r['id']]
    val = Counter(state_cell(r) for r in data['val'] if match(r))
    return [dict(cell=list(cell), pool_rows=pool[cell], sampled_presentations=sampled[cell], val_rows=number)
            for cell,number in sorted(val.items())]


def fit_summary(cases):
    groups = defaultdict(list)
    for case in cases:
        fields = case['paired_identity']['fields']
        key = (*state_cell(fields), case['original_label_basis'])
        groups[key].append(case)
    weighted = []
    for case in cases:
        weighted.extend([case]*case['sampled_presentations'])
    def metrics(items):
        result=binary_diagnostics(items)
        result['reference_agreement']=sum(r['target']==r['prediction'] for r in items)/len(items) if items else None
        return result
    return dict(unique_questions=metrics(cases),
                exposure_weighted=metrics(weighted),
                strata=[dict(cell=list(k), metrics=metrics(v),
                             physical_groups=len({r['paired_identity']['fields']['physical_group'] for r in v}))
                        for k,v in sorted(groups.items())],
                cell_fields=['event','edge','phase','answer','state','longitudinal','branch','return_required','motion','original_label_basis'],
                scope='in-sample fit diagnostic; repeated exposures are weights, not independent examples; not accuracy certification')


def verify_fit(output):
    """Recompute a completed fit report from its exact selected set; no GPU needed."""
    output = Path(output)
    report = read(output/'fit_report.json')
    if (report.get('schema') != 'joint_training_fit_v1' or report.get('command') != 'fit'
            or report.get('status') != 'complete' or report.get('training_fit') != 'measured'
            or report.get('checkpoint_assets_verified') is not True):
        raise ValueError('fit inference is not complete and measured')
    if report.get('output_files_sha256') != {name:file_sha(output/name) for name in FIT_FILES}:
        raise ValueError('fit output bytes changed')
    selection = rows(output/'fit_selection.jsonl')
    expected = {r['id']:r for r in selection}
    actual, _ = load_cases(output/'cases.jsonl')
    if (not expected or len(expected) != len(selection) or expected.keys() != actual.keys()
            or len(actual) != report['selection_count'] or len(actual) != report['completed_count']):
        raise ValueError('fit selection/case set differs or is incomplete')
    for key,item in expected.items():
        case = actual[key]
        if (case['paired_identity'] != item['paired_identity'] or
                case['paired_identity']['fields']['split'] != 'train' or
                case['sampled_presentations'] != item['sampled_presentations'] or
                case['original_label_basis'] != item['original_label_basis'] or
                list(state_cell(case['paired_identity']['fields'])) != item['cell'] or
                natural(item['sampled_presentations']) < 1):
            raise ValueError('fit selected identity or exposure changed')
    if read(output/'fit_metrics.json') != fit_summary(list(actual.values())):
        raise ValueError('fit metrics differ from cases')
    return dict(status='verified', unique_questions=len(actual),
                scope='complete local result identities and metrics; does not reload model, certify labels or independently prove GPU execution')


def fit(a, report, stage):
    stage('run_evidence')
    ev = evidence(a.run, a.epoch)
    report['sources_sha256'] = ev['hashes']
    report['source_bindings'] = ev['bindings']
    report['support'] = support_report(ev)
    stage('live_dataset')
    view_path = a.dataset/'student_view.json'
    view_binding = file_binding(view_path)
    data, manifest = load_dataset(a.dataset, rgb_mode=ev['run']['observation_contract']['rgb_mode'])
    if file_binding(view_path) != view_binding:
        raise ValueError('student view changed while loading dataset')
    if digest(manifest) != ev['run']['dataset']:
        raise ValueError('dataset contract differs')
    selected, counts = select_training(data, ev, a.event, a.edge, a.phase)
    report['selection_count'] = len(selected)
    report['selected_answers'] = dict(Counter(r['target'] for r in selected))
    report['state_cell_fields'] = ['event','edge','phase','answer','state','longitudinal','branch','return_required','motion']
    report['state_support'] = distribution_report(data, counts, a.event, a.edge, a.phase)
    write_rows(a.output/'fit_selection.jsonl', [dict(id=r['id'], sampled_presentations=counts[r['id']],
        cell=list(state_cell(r)), original_label_basis=label_basis(r), paired_identity=case_identity(r)) for r in selected])
    # Release the full 330k-row pool before model loading; only selected rows survive.
    del data, manifest
    if not selected:
        report.update(status='no_sampled_support', stage='complete', training_fit='not_run')
        return
    if a.prepare_only:
        report.update(status='complete', stage='complete', training_fit='not_run')
        return
    stage('checkpoint_assets')
    prefix = f'epoch_{a.epoch:03d}'
    folder = a.run/prefix
    receipt_path = a.run/(prefix+'_checkpoint.json')
    receipt_binding = file_binding(receipt_path)
    receipt = read(receipt_path)
    adapter_bindings = {name:file_binding(folder/name) for name in adapter_identity(folder)}
    if (receipt['epoch'] != a.epoch or receipt['adapter'] != prefix or
            read(folder/'phase4_contract.json') != ev['run'] or receipt['files'] != adapter_identity(folder)):
        raise ValueError('checkpoint assets/contract differ')
    if file_binding(receipt_path) != receipt_binding:
        raise ValueError('checkpoint receipt changed during audit')
    report['checkpoint_receipt_sha256'] = receipt_binding['sha256']
    report['checkpoint_files_sha256'] = receipt['files']
    report['view_binding'] = view_binding
    report['checkpoint_assets_verified'] = True
    # Inference-only: optimizer state is not loaded or required.
    stage('device')
    device = a.device
    if device == 'cuda':
        from ..sft_new_loop_phase4.devices import select_gpus
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = select_gpus(1)
        device = 'cuda:0'
    stage('model_load')
    bundle = load_for_inference(a.model_dir, folder, device)
    stage('inference')
    cases = []
    with (a.output/'cases.jsonl').open('x') as stream:
        for row in selected:
            answer, raw = generate(bundle, row, a.data_root, ev['run']['training']['max_length'])
            case = dict(id=row['id'], target=row['target'], prediction=answer, raw=raw,
                event=row['episode']['event'],edge=row['edge'],slice=row['slice'],
                reference_kind=row.get('reference_kind','reviewed_rgb'),original_label_basis=label_basis(row),
                sampled_presentations=counts[row['id']],paired_identity=case_identity(row))
            validate_case(case)
            stream.write(json.dumps(case,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            cases.append(case)
            report['completed_count'] = len(cases)
            report['training_fit'] = 'partial'
            if len(cases)==1 or len(cases)%20==0 or len(cases)==len(selected):
                print(f'[train-fit] completed={len(cases)}/{len(selected)}',flush=True)
            stage('inference')
    # Guard against concurrent changes to the evidence used for selecting samples.
    if (ev['bindings'] != {n:file_binding(a.run/n) for n in ev['bindings']}
            or receipt['files'] != adapter_identity(folder)
            or adapter_bindings != {n:file_binding(folder/n) for n in adapter_bindings}
            or file_binding(receipt_path) != receipt_binding or file_binding(view_path) != view_binding):
        raise ValueError('source changed during fitting diagnostic')
    write_json(a.output/'fit_metrics.json', fit_summary(cases))
    report.update(status='complete', stage='complete', training_fit='measured',
                  cases_sha256=file_sha(a.output/'cases.jsonl'),
                  output_files_sha256={name:file_sha(a.output/name) for name in FIT_FILES})


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    sub = p.add_subparsers(dest='command',required=True)
    verify = sub.add_parser('verify-fit',allow_abbrev=False)
    verify.add_argument('output',type=Path)
    support = sub.add_parser('support',allow_abbrev=False)
    support.add_argument('--package',type=Path,required=True,help='Verified extracted handoff directory')
    support.add_argument('--output',type=Path,required=True)
    support.add_argument('--epoch',type=int,default=0)
    fitp = sub.add_parser('fit',allow_abbrev=False)
    for name in ('run','dataset','data-root','model-dir','output'):
        fitp.add_argument('--'+name,type=Path,required=True)
    fitp.add_argument('--epoch',type=int,default=0)
    fitp.add_argument('--event',default='U-E1')
    fitp.add_argument('--edge',default='complete')
    fitp.add_argument('--phase',choices=('readiness','catchup','all'),default='readiness')
    fitp.add_argument('--prepare-only',action='store_true',help='Verify selection and state support without loading model')
    fitp.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    a = p.parse_args()
    if a.command == 'verify-fit':
        try:
            print(json.dumps(verify_fit(a.output),ensure_ascii=False))
            return 0
        except Exception as error:
            print(json.dumps(dict(status='failed',error=str(error)),ensure_ascii=False))
            return 2
    # Separate output only, never write inside an immutable package/run/dataset.
    for root in ([a.package] if a.command=='support' else [a.run,a.dataset,a.data_root,a.model_dir]):
        if a.output.resolve().is_relative_to(root.resolve()):
            p.error('output must be outside input directories')
    a.output.mkdir(parents=True,exist_ok=False)
    from ..sft_new_loop_phase4.rgb_short.contract import contract
    report = dict(dependency_source=contract(), helper_sha256={
        n:file_sha(Path(__file__).parent/n) for n in ('io.py','handoff.py')},schema='joint_training_fit_v1',status='in_progress',stage='initializing',
                  tool_sha256=file_sha(Path(__file__)),command=a.command,epoch=a.epoch,
                  request={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                  training_fit='not_run',completed_count=0)
    start = time.perf_counter()
    def stage(name):
        report.update(stage=name,elapsed_seconds=time.perf_counter()-start)
        write_json(a.output/'fit_report.json',report)
    stage('initializing')
    try:
        if a.command=='support':
            stage('package_verification')
            report['package']=verify_package(a.package)
            inventory=read(a.package/'handoff_manifest.json')['results']
            results={}
            for name in inventory:
                if not re.fullmatch(r'[A-Za-z0-9_-]+',name):
                    raise ValueError('unsafe result label')
                if not (a.package/'results'/name/'student_view.json').is_file():
                    continue
                results[name]=support_report(evidence(a.package/'results'/name,a.epoch))
            if not results:
                raise ValueError('package has no student-view runs')
            write_json(a.output/'support_report.json',dict(runs=results,scope='development support audit; no new labels or fitting inference'))
            report.update(status='complete',stage='complete')
        else:
            fit(a,report,stage)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',error=dict(type=type(error).__name__,message=str(error)))
        stage(report['stage'])
        print(json.dumps(dict(status=report['status'],stage=report['stage'],
                              error=report['error'],report=str(a.output/'fit_report.json')),ensure_ascii=False))
        return 130 if isinstance(error,KeyboardInterrupt) else 2
    stage(report['stage'])
    print(json.dumps(dict(status=report['status'],training_fit=report['training_fit'],
                          selected=report.get('selection_count'),completed=report['completed_count'],
                          report=str(a.output/'fit_report.json')),ensure_ascii=False))
    return 0 if report['status']=='complete' else 2


if __name__=='__main__':
    raise SystemExit(main())
