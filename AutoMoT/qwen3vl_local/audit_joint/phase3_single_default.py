"""Pinned previous single v3 selection. No training, inference or implicit promotion."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

APP = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_suffix('.json')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def selection(app=APP):
    selected = json.loads(CONFIG.read_text())
    expected = dict(schema='phase3_single_default_v1', method='single',
        package='phase3_single_v3', variant='six',
        run='checkpoints/phase3_single_v3/runs/six_1280', checkpoint='epoch_3',
        contract_version='current_reference_phase3_single_v3', optimizer_updates=9600,
        refinement_status='paused_by_user', automatic_promotion=False)
    if any(selected.get(k) != v for k, v in expected.items()):
        raise ValueError('single_default_selection_mismatch')
    required = {'training_complete.json', 'epoch_3/receipt.json', 'test/cases.json',
                'test/metrics.json', 'plan.json', 'admission.json'}
    if not required.issubset(selected['evidence_sha256']):
        raise ValueError('single_default_evidence_incomplete')
    for name in selected['evidence_sha256']:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
            raise ValueError('single_default_unsafe_evidence_path')
    run = Path(app) / selected['run']
    checkpoint = run / selected['checkpoint']
    return dict(selected, run_path=str(run), checkpoint_path=str(checkpoint),
                state_present=(checkpoint / 'state.pt').is_file(), weights_verified=False)


def check_evidence(selected):
    run = Path(selected['run_path'])
    for name, digest in selected['evidence_sha256'].items():
        if sha(run / name) != digest:
            raise ValueError(f'single_default_run_evidence_mismatch: {name}')
    receipt = json.loads((run / 'epoch_3/receipt.json').read_text())
    if receipt['state_sha256'] != selected['state_sha256']:
        raise ValueError('single_default_checkpoint_identity_mismatch')


def verified_checkpoint(app=APP):
    selected = selection(app)
    check_evidence(selected)
    checkpoint = Path(selected['checkpoint_path'])
    if sha(checkpoint / 'state.pt') != selected['state_sha256']:
        raise ValueError('single_default_weights_mismatch')
    # Native verification covers sources, ready, plan, all epoch weights,
    # exposure, exact test membership and recomputed metrics. No GPU inference.
    from .phase3_compare import verify_completion
    with contextlib.redirect_stdout(sys.stderr):
        verify_completion(Path(app), 'single')
    if selection(app) != selected:
        raise ValueError('single_default_selection_changed')
    check_evidence(selected)
    if sha(checkpoint / 'state.pt') != selected['state_sha256']:
        raise ValueError('single_default_weights_changed')
    return checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('operation', choices=('status', 'checkpoint'), nargs='?', default='status')
    args = parser.parse_args(argv)
    if args.operation == 'status':
        print(json.dumps(selection(), ensure_ascii=False, indent=2))
    else:
        print(verified_checkpoint())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
