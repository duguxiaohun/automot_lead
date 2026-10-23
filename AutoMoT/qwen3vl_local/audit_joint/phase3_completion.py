"""CPU verification of saved single/abc completion; no model load or inference."""
from collections import Counter
import importlib
import json
from pathlib import Path

from . import phase3_comparison_policy as policy
from .geometry_v1.dataset import digest
from .geometry_v1.sources import file_sha


def read(path):
    return json.loads(Path(path).read_text())


def verify_study(root, run, package, variant):
    completion = read(run/'training_complete.json')
    binding = completion['binding']
    if (completion.get('total_steps') != 9600 or completion.get('selected_checkpoint') != 'epoch_3'
            or completion.get('selection') != 'final_completed_epoch'
            or binding.get('comparison_policy') != policy.contract()):
        raise ValueError('completed_run_comparison_contract_mismatch')
    dataset = importlib.import_module(f'.{package}.dataset', __package__)
    regions = importlib.import_module(f'.{package}.regions', __package__)
    metrics = importlib.import_module(f'.{package}.metrics', __package__)
    ready = read(root/'ready_1280.json')
    if (ready.get('status') != 'ready' or ready.get('optimizer_steps') != [3200]*3
            or binding.get('dataset_sha256') != ready['dataset_sha256']
            or binding.get('contract') != regions.contract(variant)
            or binding.get('code') != dataset.code_identity()
            or binding.get('admission_sha256') != ready['admission_sha256']
            or binding.get('shared_plan_sha256') != ready['plan_sha256']):
        raise ValueError('completed_run_data_or_source_binding_mismatch')
    for name, sha in (('admission', ready['admission_sha256']), ('plan', ready['plan_sha256'])):
        saved = read(run/(name+'.json'))
        if (saved != read(ready[name]) or saved.get('sha256') != sha
                or digest({k: v for k, v in saved.items() if k != 'sha256'}) != sha):
            raise ValueError('completed_run_'+name+'_mismatch')
    projection = read(run/'plan.json')['plans']
    if len(projection) != 3:
        raise ValueError('completed_run_epoch_count_mismatch')
    exposure = Counter()
    hashes = {}
    for epoch, plan in enumerate(projection, 1):
        folder = run/f'epoch_{epoch}'
        path = folder/'receipt.json'
        receipt = read(path)
        if (completion['checkpoint_receipts'].get(folder.name) != file_sha(path)
                or receipt['binding'] != binding or receipt['epoch'] != epoch
                or receipt['step'] != epoch*3200 or receipt['status'] != 'epoch_complete'
                or receipt['state_sha256'] != file_sha(folder/'state.pt')):
            raise ValueError('completed_run_checkpoint_mismatch')
        actual = read(run/f'epoch_{epoch}.json')
        exposure.update(plan['ordered_ids'])
        if (plan['selected'] != 12800 or plan['optimizer_steps'] != 3200
                or len(plan['ordered_ids']) != 12800 or actual['plan'] != plan
                or actual['result']['status'] != 'epoch_complete'
                or actual['result']['steps'] != 3200
                or actual['result']['actual_exposure'] != dict(exposure)):
            raise ValueError('completed_run_consumption_mismatch')
        hashes[f'epoch_{epoch}/receipt.json'] = file_sha(path)
        hashes[f'epoch_{epoch}/state.pt'] = receipt['state_sha256']
        hashes[f'epoch_{epoch}.json'] = file_sha(run/f'epoch_{epoch}.json')
    # Bind exact test records, including labels, to the admitted dataset. This
    # deliberately skips raw-source/model scans: it is saved-result verification.
    artifact = dataset.load(ready['dataset'], variant=variant, check_sources=False,
                            expected_sha256=ready['dataset_sha256'])
    expected = sorted((r for r in artifact['records'] if r['split'] == 'test'), key=lambda r: r['id'])
    cases = read(run/'test/cases.json')
    if not expected or [c['record'] for c in cases] != expected:
        raise ValueError('completed_run_test_members_mismatch')
    if digest(metrics.evaluate(cases)) != digest(read(run/'test/metrics.json')):
        raise ValueError('completed_run_test_metrics_mismatch')
    for name in ('training_complete.json', 'plan.json', 'admission.json', 'test/cases.json', 'test/metrics.json'):
        hashes[name] = file_sha(run/name)
    return dict(verified_files=hashes, dataset_sha256=ready['dataset_sha256'],
                test_cases=len(cases), saved_results_verified=True, inference_rerun=False)
