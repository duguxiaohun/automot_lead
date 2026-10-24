"""Read the pinned base selection; never select a run by latest or modification time.

status works without weights. adapter verifies the selected historical package and
the existing native completion contract before printing its adapter directory.
This module does not select, import or modify single's labels or prompts.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

APP = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_suffix('.json')
REQUIRED = {
    'launch_workflow.json', 'experiment/job.json', 'experiment/preflight.json',
    'experiment/train/train_run_manifest.json', 'experiment/train/final_generation.json',
    'experiment/train/final_generation_val_cases.jsonl', 'experiment/test/metrics.json',
    *(f'experiment/test/cases_rank{rank}.jsonl' for rank in range(4)),
}


def selection(app=APP):
    selected = json.loads(CONFIG.read_text())
    if selected['schema'] != 'phase3_base_default_v1' or selected['method'] != 'base':
        raise ValueError('base_default_requires_base_selection')
    # Keep the established canonical launcher and this selection in agreement.
    if selected['run'] != 'checkpoints/phase3_compare_base/run_base_bounded_v3':
        raise ValueError('base_default_canonical_run_mismatch')
    if selected['adapter'] != 'experiment/train/final' or selected['archive'] != 'handoff/phase3_report.zip':
        raise ValueError('base_default_artifact_path_mismatch')
    expected = dict(prompt_version='original', prompt_variant='v23_rgb_stage_candidate_20261006',
                    history_rgb_mode='4rgb', action_output_mode='choice',
                    refinement_status='paused_by_user', automatic_promotion=False,
                    single_policy='independent_branch_unchanged')
    if any(selected.get(k) != v for k, v in expected.items()):
        raise ValueError('base_default_semantics_mismatch')
    run = Path(app) / selected['run']
    return dict(selected, run_path=str(run), adapter_path=str(run / selected['adapter']),
                archive_path=str(run / selected['archive']),
                adapter_present=(run / selected['adapter']).is_dir(),
                archive_present=(run / selected['archive']).is_file(),
                weights_verified=False)


def archive_evidence(selected):
    # Parse the exact bytes whose digest was checked, not a second path read.
    raw = Path(selected['archive_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != selected['archive_sha256']:
        raise ValueError('base_default_archive_mismatch: expected previous bounded_v3 package')
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or not REQUIRED.issubset(names):
            raise ValueError('base_default_archive_members_mismatch')
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
                raise ValueError('base_default_unsafe_archive_path')
        # Logs are truncated by the existing exporter; compare complete JSON
        # evidence only. Never extract the archive or overwrite a run.
        return {name: hashlib.sha256(bundle.read(name)).hexdigest() for name in names
                if PurePosixPath(name).suffix in ('.json', '.jsonl')}


def check_evidence(selected, evidence):
    run = Path(selected['run_path']).resolve()
    for name, digest in evidence.items():
        path = run / name
        if not path.resolve().is_relative_to(run):
            raise ValueError('base_default_evidence_outside_run')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'base_default_run_evidence_mismatch: {name}')
    manifest = json.loads((run / 'experiment/train/train_run_manifest.json').read_text())
    for key in ('prompt_variant', 'history_rgb_mode', 'action_output_mode', 'production_prompt_sha256'):
        if manifest.get(key) != selected[key]:
            raise ValueError(f'base_default_training_semantics_mismatch: {key}')


def verified_adapter(app=APP):
    selected = selection(app)
    evidence = archive_evidence(selected)
    check_evidence(selected, evidence)
    adapter = Path(selected['adapter_path'])
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)
    config = json.loads((adapter / 'sft_new_loop_phase3_adapter_config.json').read_text())
    for key in ('prompt_variant', 'history_rgb_mode', 'action_output_mode'):
        if config.get(key) != selected[key]:
            raise ValueError(f'base_default_adapter_semantics_mismatch: {key}')
    # Reuse the full native contract/consumption/adapter-SHA check. This does not
    # train, infer, promote quality, or write an outer completion receipt.
    from .phase3_compare import verify_completion
    with contextlib.redirect_stdout(sys.stderr):
        verify_completion(Path(app), 'base')
    # Refuse a changed report/run while the native verifier was executing.
    if archive_evidence(selected) != evidence:
        raise ValueError('base_default_archive_changed_during_verification')
    check_evidence(selected, evidence)
    return adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('operation', choices=('status', 'adapter'), nargs='?', default='status')
    args = parser.parse_args(argv)
    if args.operation == 'status':
        print(json.dumps(selection(), ensure_ascii=False, indent=2))
    else:
        print(verified_adapter())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
