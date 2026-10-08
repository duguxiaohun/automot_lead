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
    prepare.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[3])
    prepare.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[2] / 'lead_data')
    prepare.add_argument('--phase3-prompt-variant', required=True,
                         choices=['baseline', 'v23_rgb_stage_candidate_20261006'])
    for name in ('phase3-index', 'phase3-adapter', 'phase4-data2', 'phase4-data4',
                 'action-data', 'action-effective-index', 'model-dir', 'candidate-val'):
        prepare.add_argument('--' + name, type=Path)
    prepare.add_argument('--verify-images', action='store_true', help='Use native Phase4 source and pixel identity checks')
    prepare.add_argument('--output', type=Path, required=True)
    capture = subs.add_parser('run', allow_abbrev=False)
    capture.add_argument('--config', type=Path, required=True)
    capture.add_argument('--output', type=Path, required=True)
    capture.add_argument('--archive', type=Path, help='Default: sibling OUTPUT.audit.zip; always <=30,000,000 bytes')
    package = subs.add_parser('pack', allow_abbrev=False)
    package.add_argument('--capture', type=Path, required=True)
    package.add_argument('--output', type=Path)
    package.add_argument('--result-dir', action='append', default=[], metavar='NAME=PATH',
                         help='Optional completed run/eval directory; repeat for multiple explicitly selected runs')
    verify_zip = subs.add_parser('verify-package', allow_abbrev=False)
    verify_zip.add_argument('archive', type=Path)
    verify = subs.add_parser('verify', allow_abbrev=False)
    verify.add_argument('output', type=Path)
    action = subs.add_parser('export-action-pool', allow_abbrev=False)
    action.add_argument('--argv-json', type=Path, required=True, help='Exact Action dataset/runtime CLI flags as a JSON string array')
    action.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        write_json(args.output, make_config(args))
        print(json.dumps(dict(request=str(args.output), status='prepared'), ensure_ascii=False))
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
