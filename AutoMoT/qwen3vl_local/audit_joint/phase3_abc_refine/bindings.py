"""Content-bound admissions; cached success never substitutes for current inputs."""
import json
from pathlib import Path
from ..abc_delivery import sha
from . import runtime, regression


def model_identity(directory):
    from ..phase3_suite import model_snapshot
    return {k:v['sha256'] for k,v in model_snapshot(Path(directory)).items()}


def adapter_identity(checkpoint,model,variant):
    from ..geometry_abc_v2 import regions
    checkpoint=Path(checkpoint);receipt=json.loads((checkpoint/'receipt.json').read_text())
    if receipt['binding']['model']!=model:raise ValueError('baseline_adapter_model_changed')
    # Baseline adapters always use the frozen old contract.
    if receipt['binding']['contract']!=runtime.BASE_CONTRACT(variant):
        raise ValueError('baseline_adapter_variant_changed')
    value=sha(checkpoint/'state.pt')
    if receipt['state_sha256']!=value:raise ValueError('baseline_adapter_weights_changed')
    return dict(path=str(checkpoint.resolve()),state_sha256=value,receipt_sha256=sha(checkpoint/'receipt.json'))


def verify_processor(path,dataset_path,model):
    value=json.loads(Path(path).read_text())
    if (not value.get('full_pool') or not value.get('no_truncation') or value.get('code')!=runtime.identity()
        or value.get('dataset_sha256')!=sha(dataset_path) or value.get('model')!=model):
        raise ValueError('stale_full_pool_processor_check')
    if set(value.get('extrema',{}))!=set('abc'):raise ValueError('split_processor_extrema_required')
    return value


def verify_probe(path,dataset_path,model,variant,processor_path):
    path=Path(path);value=json.loads(path.read_text())
    expected=dict(dataset_sha256=sha(dataset_path),code=runtime.identity(),model=model,variant=variant,
                  contract=runtime.contract(variant),processor_sha256=sha(processor_path))
    if any(value.get(k)!=v for k,v in expected.items()) or not value.get('checkpoint_reload_verified'):
        raise ValueError('stale_probe_model_variant_or_processor')
    checkpoint=path.parent/'checkpoint'
    receipt=json.loads((checkpoint/'receipt.json').read_text())
    if (value['checkpoint_sha256']!=sha(checkpoint/'state.pt') or receipt['state_sha256']!=value['checkpoint_sha256']
        or value['receipt_sha256']!=sha(checkpoint/'receipt.json') or receipt['binding']!=value['binding']
        or any(receipt['binding'].get(k)!=expected[k] for k in ('code','model','contract','dataset_sha256'))):
        raise ValueError('probe_checkpoint_changed')
    from .probe_inputs import selection
    from ..geometry_abc_v2 import dataset
    processor=verify_processor(processor_path,dataset_path,model)
    with runtime.activated():artifact=dataset.load(dataset_path,variant=variant,check_sources=False)
    expected_selection=selection(artifact['records'],processor,variant)
    if value.get('input_selection')!=expected_selection:raise ValueError('probe_input_selection_changed')
    if set(value.get('rank_receipts',{}))!={f'rank{i}.json' for i in range(4)}:raise ValueError('probe_rank_receipts_missing')
    holdout=[]
    for rank in range(4):
        p=path.parent/f'rank{rank}.json'
        if sha(p)!=value['rank_receipts'][p.name]:raise ValueError('probe_rank_receipt_changed')
        proof=json.loads(p.read_text())
        if proof['backward']!=expected_selection['backward'][rank]:raise ValueError('probe_backward_split_changed')
        for item in proof['holdout_forward_only']:
            if item['updates']!=0:raise ValueError('holdout_probe_updated')
            holdout.append({k:v for k,v in item.items() if k!='updates'})
    if sorted(holdout,key=lambda x:x['id'])!=sorted(expected_selection['holdout_forward_only'],key=lambda x:x['id']):
        raise ValueError('holdout_forward_coverage_changed')
    return value


def verify_replay(path,baseline_dataset,artifact,model,variant,adapter):
    from ..geometry_abc_v2 import planning
    path=Path(path);value=json.loads(path.read_text())
    expected=dict(source=runtime.identity(),baseline_dataset_sha256=sha(baseline_dataset),model=model,
                  variant=variant,adapter_sha256=adapter['state_sha256'],adapter_receipt_sha256=adapter['receipt_sha256'],
                  full_validation=True,processor_checked=True,fresh_KV=True)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('paired_replay_model_or_inputs_changed')
    original=json.loads(path.with_name('original_cases.json').read_text())
    candidate=json.loads(path.with_name('candidate_cases.json').read_text())
    actual=regression.compare(original,candidate)
    if any(value.get(k)!=v for k,v in actual.items()):raise ValueError('paired_replay_report_changed')
    if actual['status']!='observed_cohort_protection_passed':raise ValueError('candidate_needs_verified_paired_validation')
    expected_rows={r['id']:r for r in planning.evaluation(artifact['records'],'val')}
    observed=regression.indexed(original)
    if observed.keys()!=expected_rows.keys() or any(observed[k]['record']!=r for k,r in expected_rows.items()):
        raise ValueError('paired_replay_validation_incomplete')
    return original,candidate,value


def file_proof(path):
    return dict(path=str(Path(path).resolve()),sha256=sha(path))


def verify_file(proof):
    path=Path(proof['path'])
    if sha(path)!=proof['sha256']:raise ValueError('frozen_admission_evidence_changed')
    return path
