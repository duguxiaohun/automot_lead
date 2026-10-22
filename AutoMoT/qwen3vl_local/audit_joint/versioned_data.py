"""Machine-local, immutable generation cache. No Git, network or training.

Human versions describe semantics; full dependency identities decide reuse.
Builders write at their final path (native manifests may contain absolute paths).
Failed builds remain visible and are never treated as completed datasets.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time

SCHEMA = 'automot_generation_cache_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def bind(path):
    path = Path(path).absolute()
    resolved = str(path.resolve(strict=True))
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        h = hashlib.sha256()
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
        after = os.fstat(stream.fileno())
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if stamp(before) != stamp(after) or stamp(after) != stamp(path.stat()) or resolved != str(path.resolve(strict=True)):
        raise ValueError('File changed while hashing: ' + str(path))
    return dict(path=str(path), resolved=resolved, sha256=h.hexdigest(), bytes=after.st_size)


def bindings(paths):
    return {str(Path(p).absolute()): bind(p) for p in sorted(set(map(Path, paths)))}


def write(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    with pending.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(pending, path)


@contextmanager
def lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another dataset preparation owns: ' + str(path)) from exc
        yield


def safe_name(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('Invalid cache family/version: ' + str(value))
    return value


def outputs(payload):
    if payload.is_symlink() or not payload.is_dir():
        raise ValueError('Generation payload must be a real directory')
    result = {}
    # Only cache-owned files; links could silently traverse another dataset.
    for directory, dirs, files in os.walk(payload, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError('Cache payload must not contain links: ' + str(path))
        for name in files:
            path = Path(directory) / name
            result[str(path.relative_to(payload))] = bind(path)
    if not result:
        raise ValueError('Empty generation output')
    return result


def ensure(root, family, version, contract, build, validate, *, protected=()):
    """contract() binds all generation dependencies; validate retains native gates.

    The caller must declare every causal dependency (including split and full
    route history). It must not use this API to cache training/evaluation jobs.
    runtime/prompt dependencies should be a separate stage from raw labels.
    """
    root = Path(root).absolute()
    family, version = safe_name(family), safe_name(version)
    normalized = lambda: json.loads(json.dumps(contract(), allow_nan=False))
    requested = normalized()
    if not isinstance(requested, dict) or not requested:
        raise ValueError('Nonempty explicit generation dependency contract required')
    identity = dict(schema=SCHEMA, family=family, semantic_version=version, dependencies=requested)
    key = digest(identity)
    directory = root / family / version / key
    if any(p.is_symlink() for p in (root/family,root/family/version,directory)):
        raise ValueError('Cache family/version must not redirect through a symlink')
    for p in protected:
        p = Path(p).resolve()
        if directory.resolve().is_relative_to(p) or p.is_relative_to(directory.resolve()):
            raise ValueError('Generation cache overlaps protected input: ' + str(p))
    with lock(root / '.locks' / (family + '-' + key + '.lock')):
        if directory.is_symlink():
            raise ValueError('Version directory cannot be a symlink')
        payload = directory / 'data'
        receipt_path = directory / 'complete.json'
        if directory.exists():
            if not receipt_path.is_file():
                raise ValueError('Incomplete generation preserved; inspect failure.json: ' + str(directory))
            receipt = json.loads(receipt_path.read_text())
            if receipt['identity'] != identity or receipt['key'] != key:
                raise ValueError('Generation identity mismatch')
            if outputs(payload) != receipt['files']:
                raise ValueError('Completed cache output changed: ' + str(directory))
            validate(payload)
            if outputs(payload) != receipt['files']:
                raise ValueError('Native validator mutated cached output')
            action = 'reuse'
        else:
            directory.mkdir(parents=True)
            try:
                build(payload)  # native builders often require an absent output
                validate(payload)
                files = outputs(payload)
                if normalized() != requested:
                    raise ValueError('Generation dependencies changed during build')
                receipt = dict(key=key, identity=identity, files=files, hostname=socket.gethostname(),
                               created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                               generation_completed=True, training_performed=False)
                write(receipt_path, receipt)
            except BaseException as exc:
                write(directory / 'failure.json', dict(error=f'{type(exc).__name__}: {exc}',
                      generation_completed=False, training_performed=False))
                raise
            action = 'build'
        if normalized() != requested:
            raise ValueError('Generation dependencies changed during reuse')
        return dict(action=action, version=version, key=key, directory=str(payload),
                    receipt=str(receipt_path), training_performed=False)


def discover(root, filename, *, depth=4):
    """Bounded local discovery, no mtime/latest preference or symlink traversal."""
    root = Path(root).absolute(); found = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        rel = Path(directory).relative_to(root)
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and
                         d not in ('runtime', 'training_states', 'final', '__pycache__', 'node_modules') and
                         not (Path(directory)/d).is_symlink()) if len(rel.parts) < depth else []
        if filename in files:
            found.append(Path(directory)/filename)
    return sorted(found)


def unique_compatible(candidates, inspect):
    """inspect returns a content identity or None for a demonstrably other version.

    Malformed/corrupt requested-version assets raise, rather than trigger rebuild.
    Two compatible but different datasets require an explicit choice.
    """
    accepted = []
    for path in sorted(set(map(Path, candidates))):
        identity = inspect(path)
        if identity is not None:
            accepted.append((path, identity))
    if len({digest(identity) for _, identity in accepted}) > 1:
        raise ValueError('Multiple different compatible local datasets; select an explicit path: ' +
                         ', '.join(str(p) for p, _ in accepted))
    return accepted[0][0] if accepted else None
