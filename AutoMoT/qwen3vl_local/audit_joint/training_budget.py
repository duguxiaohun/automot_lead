"""Explicit incremental peak disk budget; missing bounds never mean zero.

Existing files (including the base model and retained old experiments) are already
charged against free_bytes. All component values describe NEW bytes on that same
filesystem. Bounds and their evidence are supplied by the operator, not inferred
from a small-route compression ratio.
"""
import argparse
import json
from pathlib import Path
from .io import write_json

COMPONENTS = ('shared_replay', 'compiled_datasets', 'automatic_holdout', 'phase3_index',
              'adapters_per_checkpoint', 'optimizer_per_checkpoint', 'logs_and_reports',
              'temporary_peak')


def plan(request):
    for key in ('free_bytes', 'reserve_bytes', 'modes', 'retained_checkpoints_per_mode'):
        value = request.get(key)
        if type(value) is not int or value < (1 if key in ('modes', 'retained_checkpoints_per_mode') else 0):
            raise ValueError(f'invalid {key}')
    bounds, missing = {}, []
    for name in COMPONENTS:
        record = request.get('components', {}).get(name, {})
        value = record.get('upper_bound_bytes')
        if value is None or not record.get('evidence'):
            missing.append(name)
        elif type(value) is not int or value < 0:
            raise ValueError(f'invalid bound: {name}')
        else:
            bounds[name] = value
    count = request['modes'] * request['retained_checkpoints_per_mode']
    subtotal = sum(v * (count if k.endswith('_per_checkpoint') else 1) for k, v in bounds.items())
    required = None if missing else subtotal + request['reserve_bytes']
    return dict(schema='training_peak_budget_v1', request=request, missing_bounds=missing,
                checkpoint_count=count, known_new_bytes=subtotal, required_free_bytes=required,
                sufficient=None if required is None else request['free_bytes'] >= required,
                status='incomplete' if missing else 'fits_declared_bounds' if request['free_bytes'] >= required else 'insufficient',
                scope='conditional on declared upper bounds and one filesystem; not measured full-pool capacity',
                retention='budget every retained epoch for both modes; temporary_peak includes concurrent writes and checkpoint staging')


def main():
    p=argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    if a.request.resolve()==a.output.resolve():
        raise ValueError('output must not overwrite request')
    report=plan(json.loads(a.request.read_text()))
    write_json(a.output, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['sufficient'] is True else 2


if __name__=='__main__':
    raise SystemExit(main())
