import argparse
import json
from pathlib import Path

from .baseline import run, verify_bundle
from .config import make_config
from .io import write_json
from .handoff import pack, verify_package


def main():
    parser = argparse.ArgumentParser(description='Local G0 baseline capture and cross-task isolation audit', allow_abbrev=False)
    subs = parser.add_subparsers(dest='command', required=True)
    prepare = subs.add_parser('prepare', allow_abbrev=False)
    prepare.add_argument('--project-root', type=Path,
                         help='Default: Git root containing this module; supports nested and flat application layouts')
    prepare.add_argument('--data-root', type=Path, help='Default: lead_data under the resolved application root')
    prepare.add_argument('--phase3-prompt-variant', required=True,
                         choices=['baseline', 'v23_rgb_stage_candidate_20261006'])
    for name in ('phase3-data-root', 'action-data-root', 'candidate-data-root'):
        prepare.add_argument('--' + name, type=Path, help='Actual native data root; defaults to --data-root')
    for name in ('phase3-index', 'phase3-adapter', 'phase4-data2', 'phase4-data4',
                 'action-data', 'action-effective-index', 'model-dir', 'candidate-val'):
        prepare.add_argument('--' + name, type=Path)
    prepare.add_argument('--verify-images', action='store_true', help='Use native Phase4 source and pixel identity checks')
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--storage-mode', choices=['references', 'full'], default='references')
    capture = subs.add_parser('run', allow_abbrev=False)
    capture.add_argument('--config', type=Path, required=True)
    capture.add_argument('--output', type=Path, required=True)
    capture.add_argument('--archive', type=Path, help='Default: sibling OUTPUT.audit.zip; always <=30,000,000 bytes')
    capture.add_argument('--storage-mode', choices=['references', 'full'])
    capture.add_argument('--min-free-gib', type=float)
    space = subs.add_parser('storage-plan', allow_abbrev=False)
    space.add_argument('--config', type=Path, required=True)
    space.add_argument('--output', type=Path, required=True)
    space.add_argument('--storage-mode', choices=['references', 'full'])
    space.add_argument('--min-free-gib', type=float)
    originals = subs.add_parser('verify-inputs', allow_abbrev=False)
    originals.add_argument('capture', type=Path)
    usage = subs.add_parser('storage-inventory', allow_abbrev=False)
    usage.add_argument('--root', type=Path, required=True)
    usage.add_argument('--core-dir', type=Path)
    package = subs.add_parser('pack', allow_abbrev=False)
    package.add_argument('--capture', type=Path, required=True)
    package.add_argument('--output', type=Path)
    package.add_argument('--result-dir', action='append', default=[], metavar='NAME=PATH',
                         help='Optional completed run/eval directory; repeat for multiple explicitly selected runs')
    verify_zip = subs.add_parser('verify-package', allow_abbrev=False)
    verify_zip.add_argument('archive', type=Path, help='Original ZIP or extracted handoff directory')
    recovery = subs.add_parser('recovery-check', allow_abbrev=False,
                               help='Read-only Phase4 replay reuse and disk lower-bound check')
    recovery.add_argument('--data-dir', type=Path, required=True)
    recovery.add_argument('--data-root', type=Path, required=True)
    recovery.add_argument('--annotations', type=Path,
                          default=Path(__file__).resolve().parents[1] / 'sft_new_loop_phase4/reviewed_state_pairs_v9.json')
    recovery.add_argument('--output', type=Path, required=True)
    verify = subs.add_parser('verify', allow_abbrev=False)
    verify.add_argument('output', type=Path)
    action = subs.add_parser('export-action-pool', allow_abbrev=False)
    action.add_argument('--argv-json', type=Path, required=True, help='Exact Action dataset/runtime CLI flags as a JSON string array')
    action.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'recovery-check':
        from .recovery import check_recovery
        if args.output.resolve().is_relative_to(args.data_dir.resolve()):
            parser.error('--output must be outside the immutable Phase4 data directory')
        if args.output.exists():
            parser.error('--output already exists; preserve the previous report')
        report = check_recovery(args.data_dir, args.data_root, args.annotations)
        write_json(args.output, report)
        print(json.dumps(dict(status=report['status'], output=str(args.output),
                              replay_reusable=report['replay_reusable_under_current_request'],
                              training_ready=False, storage=report['storage']), ensure_ascii=False))
        return 0 if report['status'] == 'diagnosed' else 2
    if args.command == 'prepare':
        write_json(args.output, dict(make_config(args), storage_mode=args.storage_mode))
        print(json.dumps(dict(request=str(args.output), status='prepared'), ensure_ascii=False))
        return 0
    if args.command == 'verify-inputs':
        from .storage import verify_inputs
        report = verify_inputs(args.capture)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report['status'] == 'verified' else 2
    if args.command == 'storage-inventory':
        from .storage import inventory
        print(json.dumps(inventory(args.root, args.core_dir), ensure_ascii=False, indent=2))
        return 0
    if args.command == 'verify':
        print(json.dumps(verify_bundle(args.output), ensure_ascii=False))
        return 0
    if args.command == 'verify-package':
        print(json.dumps(verify_package(args.archive), ensure_ascii=False))
        return 0
    if args.command == 'pack':
        results = []
        for value in args.result_dir:
            label, separator, path = value.partition('=')
            if not separator or not path:
                parser.error('--result-dir requires NAME=PATH')
            results.append((label, path))
        print(json.dumps(pack(args.capture, args.output, results), ensure_ascii=False))
        return 0
    if args.command == 'export-action-pool':
        from .action_pool import export_pool
        print(json.dumps(export_pool(json.loads(args.argv_json.read_text()), args.output), ensure_ascii=False))
        return 0
    config = json.loads(args.config.read_text())
    if config.get('schema') != 'joint_audit_request_v1':
        raise ValueError('unknown request schema')
    import math
    minimum = args.min_free_gib if args.min_free_gib is not None else config.get('min_free_bytes', 2 * 1024 ** 3) / 1024 ** 3
    if not math.isfinite(minimum) or minimum < 0:
        parser.error('--min-free-gib must be finite and nonnegative')
    config['min_free_bytes'] = int(minimum * 1024 ** 3)
    if args.storage_mode is not None:
        config['storage_mode'] = args.storage_mode
    if args.command == 'storage-plan':
        from .storage import plan
        report = plan(config, args.output, config.get('storage_mode', 'references'), config['min_free_bytes'])
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report['sufficient'] else 2
    manifest, splits = run(config, args.output)
    handoff = pack(args.output, args.archive)
    print(json.dumps(dict(output=str(args.output), source_files=manifest['source_file_count'],
                          handoff=handoff,
                          baseline_ready=manifest['reproducible_baseline_ready'], blockers=manifest['blockers'],
                          split_status=splits['status'], conflict_pairs=len(splits['conflicts'])), ensure_ascii=False))
    # A report was produced, but G0 is not passed. Do not let shell pipelines
    # interpret successful report writing as permission to launch experiments.
    return 2 if manifest['blockers'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
