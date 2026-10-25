"""Content-bound admissions; cached success never substitutes for current inputs."""
import json
from pathlib import Path
from ..abc_delivery import sha
from . import runtime, regression


def model_identity(directory):
    from ..phase3_suite import model_snapshot
    directory=Path(directory)
    if any(directory.rglob('adapter_config.json')) or any(directory.rglob('adapter_model.*')):
        raise ValueError('original_base_model_required_not_adapter_directory')
    return {k:v['sha256'] for k,v in model_snapshot(Path(directory)).items()}


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


def file_proof(path):
    return dict(path=str(Path(path).resolve()),sha256=sha(path))


def verify_file(proof):
    path=Path(proof['path'])
    if sha(path)!=proof['sha256']:raise ValueError('frozen_admission_evidence_changed')
    return path
