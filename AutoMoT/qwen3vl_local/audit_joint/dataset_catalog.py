#!/usr/bin/env python3
"""Local content-addressed dataset references, provenance and conservative reuse.

Registration never copies data or rewrites an existing manifest. A current link
is a convenience only: training/recovery always records the resolved version.
No cross-contract teacher migration or implicit cross-server path substitution.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

POLICY = 'local_dataset_catalog_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_binding(path):
    path = Path(path).absolute()
    resolved = str(path.resolve(strict=True))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
        after = os.fstat(stream.fileno())
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if fields(before) != fields(after) or fields(after) != fields(path.stat()) or resolved != str(path.resolve(strict=True)):
        raise ValueError('dataset changed while hashing: ' + str(path))
    return dict(path=str(path), resolved=resolved, sha256=h.hexdigest(), bytes=after.st_size)


def write(path, value):
    temp = path.with_name(path.name+'.pending')
    with temp.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def provenance(root):
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(root), *args], text=True,
                stderr=subprocess.DEVNULL, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = git('status', '--porcelain', '--untracked-files=no', '--', 'qwen3vl_local', 'AutoMoT/qwen3vl_local')
    return dict(hostname=socket.gethostname(), server_label=os.environ.get('AUTOMOT_SERVER', 'unspecified'),
        registered_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), root=str(Path(root).resolve()),
        registration_git_commit=git('rev-parse', 'HEAD'), registration_source_dirty=None if status is None else bool(status),
        scope='registration environment, not a claim about historical generation Git or machine')


def register(index, catalog, phase, *, producer_sources, root, origin='native_current_build'):
    if phase not in ('phase3', 'phase4'):
        raise ValueError('unknown dataset phase')
    if not isinstance(producer_sources, dict) or not producer_sources:
        raise ValueError('explicit original producer source identities are required')
    index = Path(index).resolve(strict=True)
    manifest = index.with_name('manifest.json')
    inputs = dict(index=file_binding(index), manifest=file_binding(manifest))
    native = json.loads(manifest.read_text())
    if phase == 'phase3' and native.get('source_scope') != 'collection_results':
        raise ValueError('only formal collection_results datasets may be registered')
    required_paths = {str(index.parent)}
    if phase == 'phase4':
        # Binding just train.jsonl would miss holdout and external replay changes.
        expected = dict(native['files'])
        if set(expected) != {'train.jsonl', 'val.jsonl', 'test.jsonl'}:
            raise ValueError('Phase4 requires all three native splits')
        expected['review_queue.jsonl'] = native['review_queue_sha256']
        for field, name in (('candidate_pool_sha256', 'candidate_pool.json'),
                            ('teacher_registry_sha256', 'teacher_registry.json'),
                            ('production_index_sha256', 'production/index.json'),
                            ('production_reference_sha256', 'production_reference.json')):
            if native.get(field): expected[name] = native[field]
        for name, expected_sha in expected.items():
            path = index.parent/name
            if not path.resolve().is_relative_to(index.parent):
                raise ValueError('native dataset member escapes dataset')
            value = file_binding(path)
            if value['sha256'] != expected_sha:
                raise ValueError('native dataset member SHA differs: '+name)
            inputs[name] = value
        if native.get('production_reference_sha256'):
            reference = json.loads((index.parent/'production_reference.json').read_text())
            if reference['schema'] != 'shared_production_reference_v1':
                raise ValueError('unknown replay reference')
            for key, ref in [('production_index', reference['index']), *reference['routes'].items()]:
                value = file_binding(ref['logical_path'])
                if value['resolved'] != ref['resolved_path'] or value['sha256'] != ref['sha256']:
                    raise ValueError('external replay reference changed: '+key)
                inputs['external/'+key] = value
                required_paths.add(str(Path(value['resolved']).parent))
    # Native identity/teacher/split information is preserved verbatim as metadata,
    # never replaced with a new source SHA to make an old dataset appear current.
    identity = dict(phase=phase, index_sha256=inputs['index']['sha256'], manifest_sha256=inputs['manifest']['sha256'])
    version = digest(identity)
    directory = Path(catalog).resolve()/phase
    if directory.is_relative_to(index.parent):
        raise ValueError('catalog must be outside immutable dataset directory')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.catalog.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = directory/'current'
        if current.exists() and not current.is_symlink():
            raise ValueError('current is a real directory/file; refusing to replace it')
        versions = directory/'versions'; versions.mkdir(exist_ok=True)
        path = versions/(version+'.json')
        record = dict(policy=POLICY, version=version, identity=identity, inputs=inputs,
            producer_sources=producer_sources, origin=origin, registration=provenance(root),
            native_versions={k:v for k,v in native.items() if 'version' in k or k in ('teacher', 'source_scope')},
            generation_provenance=native.get('provenance'),
            generation_provenance_unknown='provenance' not in native,
            required_paths=sorted(required_paths), data_copied=False,
            admission_scope='reference registration only; native dataset/model admission remains required')
        if path.exists():
            previous = json.loads(path.read_text())
            if (previous['identity'] != identity or previous['producer_sources'] != producer_sources or
                    previous['inputs'] != inputs):
                raise ValueError('registered version location/source binding differs; inspect explicitly')
            record = previous
        else:
            write(path, record)
        if {k:file_binding(v['path']) for k,v in inputs.items()} != inputs:
            raise ValueError('dataset changed during registration')
        pending = directory/'current.pending'
        pending.symlink_to(index.parent, target_is_directory=True)
        os.replace(pending, current)
        write(directory/'current.json', dict(version=version, record=str(path), index=str(index),
                                            required_paths=record['required_paths']))
    return dict(version=version, index=str(index), record=str(path), current=str(current),
                index_sha256=identity['index_sha256'], data_copied=False)


def resolve(catalog, phase, *, producer_sources=None):
    directory = Path(catalog)/phase
    current = json.loads((directory/'current.json').read_text())
    record = json.loads(Path(current['record']).read_text())
    if record['policy'] != POLICY or digest(record['identity']) != current['version']:
        raise ValueError('catalog identity changed')
    if any(record['identity'][name+'_sha256'] != record['inputs'][name]['sha256'] for name in ('index', 'manifest')):
        raise ValueError('catalog identity/input binding differs')
    if producer_sources is not None and producer_sources != record['producer_sources']:
        raise ValueError('producer dependencies changed; registered dataset is not reusable')
    for binding in record['inputs'].values():
        if file_binding(binding['path']) != binding:
            raise ValueError('registered input changed')
    if (directory/'current').resolve(strict=True) != Path(record['inputs']['index']['resolved']).parent:
        raise ValueError('current link changed independently of version pointer')
    return record


def incremental_plan(previous, requested):
    """Plan independently declared shards; no label/source identity rewriting.

    A shard can be a route or an event only when the producer supplies its full
    causal dependency contract. Missing dependencies invalidate everything.
    Current native P4 has a global teacher contract: any teacher change therefore
    invalidates all its shards, including events whose local rule did not change.
    """
    required = ('schema', 'shared_sources', 'split_identity', 'input_identity', 'shards')
    if any(k not in requested or k not in previous for k in required):
        raise ValueError('incomplete incremental dependency contract')
    if not requested['shared_sources'] or not requested['input_identity'] or not requested['split_identity']:
        raise ValueError('empty shared/input/split identity')
    changed = [k for k in required[:-1] if previous[k] != requested[k]]
    reused, rebuild = [], []
    for name, spec in sorted(requested['shards'].items()):
        if not spec.get('sources') or not spec.get('input_sha256'):
            raise ValueError('shard lacks causal dependencies: '+name)
        (reused if not changed and previous['shards'].get(name) == spec else rebuild).append(name)
    return dict(reuse=reused, rebuild=rebuild, removed=sorted(previous['shards'].keys()-requested['shards'].keys()),
                global_invalidations=changed, labels_rewritten=False, scope='dependency plan only; native admission still required')


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('register')
    a.add_argument('--index', type=Path, required=True)
    a.add_argument('--catalog', type=Path, required=True)
    a.add_argument('--phase', choices=('phase3', 'phase4'), required=True)
    a.add_argument('--producer-sources', type=Path, required=True, help='Explicit original source SHA map JSON')
    a.add_argument('--root', type=Path, required=True)
    a = sub.add_parser('resolve')
    a.add_argument('--catalog', type=Path, required=True)
    a.add_argument('--phase', choices=('phase3', 'phase4'), required=True)
    a = sub.add_parser('plan')
    a.add_argument('--previous', type=Path, required=True)
    a.add_argument('--requested', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'register':
        value = register(args.index, args.catalog, args.phase,
                         producer_sources=json.loads(args.producer_sources.read_text()), root=args.root)
    elif args.command == 'resolve':
        value = resolve(args.catalog, args.phase)
    else:
        value = incremental_plan(json.loads(args.previous.read_text()), json.loads(args.requested.read_text()))
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
