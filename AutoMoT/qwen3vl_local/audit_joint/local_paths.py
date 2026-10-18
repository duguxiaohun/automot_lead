"""Machine-neutral local input checks; no host-to-path mappings or discovery.

Callers provide this machine's paths. An optional server label is metadata,
never a switch choosing another machine's data, cache, checkpoint, or GPU.
"""
import argparse
import json
from pathlib import Path

from .server_context import context


def check_inputs(data_root, model_dir, output, *, require_fresh_output=True):
    paths = {name: Path(value).absolute() for name, value in
             (('data_root', data_root), ('model_dir', model_dir), ('output', output))}
    errors = []
    if not paths['data_root'].is_dir():
        errors.append('DATA_ROOT missing on this machine: '+str(paths['data_root']))
    if not (paths['model_dir']/'config.json').is_file():
        errors.append('MODEL_DIR/config.json missing on this machine: '+str(paths['model_dir']))
    dest = paths['output']
    if require_fresh_output and (dest.exists() or dest.is_symlink()):
        errors.append('OUTPUT_DIR already exists; preserve it and choose a fresh path: '+str(dest))
    for name in ('data_root', 'model_dir'):
        if dest.resolve().is_relative_to(paths[name].resolve()):
            errors.append('OUTPUT_DIR must be outside '+name)
    parent = dest.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir():
        errors.append('OUTPUT_DIR ancestor is not a directory: '+str(parent))
    return dict(status='local_paths_present' if not errors else 'blocked', errors=errors,
        server_context=context(**paths),
        scope='Path existence and output separation only; no label, model-weight, storage or training admission.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check_inputs(args.data_root, args.model_dir, args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if not result['errors'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
