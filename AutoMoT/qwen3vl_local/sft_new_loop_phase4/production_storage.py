"""Immutable replay references and streaming, lossless route compression."""
from contextlib import contextmanager
import gzip
import fcntl
import io
import json
from pathlib import Path
import shutil

from .identity import file_sha, write_json


@contextmanager
def production_lock(directory):
    """Serialize builders and reference publication; keep the lock inode."""
    with (Path(directory) / '.production.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as ex:
            raise ValueError('production builder/publication is active') from ex
        yield


def publication(index):
    """An installed reference freezes the exact index, including partial pools."""
    path = Path(index).parent / 'published.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record != dict(policy='immutable_production_v1', index_name=Path(index).name,
                      index_sha256=file_sha(index)):
        raise ValueError('published production index changed')
    return record


def open_route(path):
    path = Path(path)
    return gzip.open(path, 'rt', encoding='utf-8') if path.name.endswith('.gz') else path.open(encoding='utf-8')


@contextmanager
def write_route(path, *, compressed=False):
    # Exclusive temporary file; no uncompressed intermediate or whole-route buffer.
    with Path(path).open('xb') as raw:
        if compressed:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=6, mtime=0) as packed:
                with io.TextIOWrapper(packed, encoding='utf-8', newline='\n') as stream:
                    yield stream
        else:
            with io.TextIOWrapper(raw, encoding='utf-8', newline='\n') as stream:
                yield stream


def binding(path):
    path = Path(path).absolute()
    resolved = str(path.resolve(strict=True))
    sha = file_sha(path)
    if str(path.resolve(strict=True)) != resolved:
        raise ValueError('production source changed during capture')
    return dict(logical_path=str(path), resolved_path=resolved, sha256=sha)


def check_binding(record):
    path = Path(record['logical_path'])
    if not path.is_absolute() or binding(path) != record:
        raise ValueError('production source path/content changed')
    return path


def capture(index):
    index = Path(index).absolute()
    source = binding(index)
    receipts = json.loads(index.read_text())['routes']
    files = {}
    for row in receipts:
        name = row['artifact']
        if Path(name).name != name or name in files or name == 'index.json':
            raise ValueError('invalid/duplicate production artifact')
        files[name] = binding(index.parent / name)
        if files[name]['sha256'] != row['sha256']:
            raise ValueError('production receipt changed during capture')
    check_binding(source)
    return dict(schema='shared_production_reference_v1', index=source, routes=files)


def install(index, output, *, mode='references'):
    if mode not in ('references', 'copy'):
        raise ValueError('unknown production storage mode')
    # The same lock is held across receipt capture, publication and installation.
    # A failed installation can leave a sealed source; never unseal it silently.
    with production_lock(Path(index).parent):
        return _install(index, output, mode=mode)


def _install(index, output, *, mode):
    publication(index)
    record = capture(index)
    output = Path(output)
    bundle = output / 'production'
    bundle.mkdir()
    shutil.copyfile(index, bundle / 'index.json')
    if file_sha(bundle / 'index.json') != record['index']['sha256']:
        raise ValueError('production index changed during installation')
    if mode == 'copy':
        for name, item in record['routes'].items():
            shutil.copyfile(check_binding(item), bundle / name)
            if file_sha(bundle / name) != item['sha256']:
                raise ValueError('production artifact changed during copy')
        return None
    if publication(index) is None:
        write_json(Path(index).parent / 'published.json',
                   dict(policy='immutable_production_v1', index_name=Path(index).name,
                        index_sha256=record['index']['sha256']))
    write_json(output / 'production_reference.json', record)
    return file_sha(output / 'production_reference.json')


def resolve(output, manifest):
    output = Path(output)
    snapshot = output / 'production/index.json'
    if file_sha(snapshot) != manifest['production_index_sha256']:
        raise ValueError('production index hash mismatch')
    sha = manifest.get('production_reference_sha256')
    if sha is None:
        return snapshot
    descriptor = output / 'production_reference.json'
    if file_sha(descriptor) != sha:
        raise ValueError('production reference hash mismatch')
    record = json.loads(descriptor.read_text())
    if record.get('schema') != 'shared_production_reference_v1':
        raise ValueError('unknown production reference schema')
    index = check_binding(record['index'])
    if record['index']['sha256'] != manifest['production_index_sha256']:
        raise ValueError('production source/snapshot mismatch')
    # Recapture also binds each artifact to the index directory and receipt SHA.
    if capture(index) != record:
        raise ValueError('production route set/path/content changed')
    return index
