"""Lossless, bounded local audit delivery shared by the two ABC experiments."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

LIMIT = 30_000_000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


@contextmanager
def lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.pipeline.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def pack(output, required, *, optional=None, summary=None, limit=LIMIT):
    """Never trim core records. Failed/repeated publication preserves previous ZIP."""
    output = Path(output).resolve()
    if limit <= 0 or limit > LIMIT: raise ValueError('invalid_package_limit')
    sources = {**required, **(optional or {})}
    if len(sources) != len(required) + len(optional or {}): raise ValueError('duplicate_archive_member')
    for name, source in sources.items():
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or name == 'manifest.json':
            raise ValueError('unsafe_or_reserved_archive_member')
        source = Path(source).resolve(strict=True)
        if not source.is_file() or source == output: raise ValueError('invalid_package_source')
        if source.suffix in ('.pt', '.pth', '.safetensors', '.laz', '.pkl'):
            raise ValueError('weights_and_raw_sensors_not_audit_members')
    output.parent.mkdir(parents=True, exist_ok=True)
    with lock(output.parent):
        fd, name = tempfile.mkstemp(prefix='.audit-', suffix='.zip', dir=output.parent)
        os.close(fd)
        pending = Path(name)
        selected = dict(required)
        omissions = []
        def build():
            entries = {n: dict(sha256=sha(p), bytes=Path(p).stat().st_size,
                                required=n in required) for n,p in sorted(selected.items())}
            manifest = dict(schema='abc_lossless_audit_v1', core_complete=True,
                            members=entries, omitted_optional=omissions, summary=summary or {})
            with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
                for n,p in sorted(selected.items()): z.write(p,n)
                z.writestr('manifest.json', json.dumps(manifest, separators=(',', ':'), allow_nan=False))
            if any(sha(selected[n]) != v['sha256'] for n,v in entries.items()):
                raise ValueError('package_input_changed')
            return manifest
        try:
            manifest = build()
            if pending.stat().st_size > limit: raise ValueError('core_package_exceeds_limit_no_cases_dropped')
            for n,p in sorted((optional or {}).items()):
                selected[n] = p
                build()
                # Reserve space for the omission manifest, not just compressed members.
                if pending.stat().st_size > limit - 16384:
                    del selected[n]
                    omissions.append(dict(name=n, sha256=sha(p), reason='optional_size_budget'))
            manifest = build()
            if pending.stat().st_size > limit: raise ValueError('package_exceeds_limit')
            with zipfile.ZipFile(pending) as z:
                if z.testzip() is not None: raise ValueError('package_crc_failure')
                for n,v in manifest['members'].items():
                    if hashlib.sha256(z.read(n)).hexdigest() != v['sha256']:
                        raise ValueError('package_member_hash_failure')
            pending.replace(output)
            receipt = dict(directory=str(output.parent), package=str(output), bytes=output.stat().st_size,
                           sha256=sha(output), core_complete=True, members=len(manifest['members']))
            write(output.with_suffix('.receipt.json'), receipt)
            print(json.dumps(receipt, ensure_ascii=False), flush=True)
            return receipt
        finally:
            pending.unlink(missing_ok=True)
