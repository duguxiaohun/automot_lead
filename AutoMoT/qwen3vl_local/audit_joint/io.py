"""Content-addressed local snapshots; never rewrite an existing evidence bundle."""
import hashlib
import json
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    path = Path(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    after = path.stat()
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if signature(before) != signature(after):
        raise RuntimeError(f'file changed while hashing: {path}')
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_rows(path, rows):
    with Path(path).open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')


def snapshot_file(path, root):
    """Snapshot bytes, including uncommitted/untracked source. Verify the copy."""
    path, root = Path(path), Path(root)
    sha = file_sha(path)
    dest = root / 'blobs' / sha
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with path.open('rb') as src, dest.open('xb') as out:
            for block in iter(lambda: src.read(8 * 1024 * 1024), b''):
                out.write(block)
    if file_sha(dest) != sha or file_sha(path) != sha:
        raise RuntimeError(f'source changed during snapshot: {path}')
    return dict(sha256=sha, bytes=dest.stat().st_size,
                blob=str(dest.relative_to(root)), resolved_path=str(path.resolve()))


def source_files(roots):
    """Explicit project roots only; generated outputs and weights are not sources."""
    suffixes = {'.py', '.sh', '.json', '.jsonl', '.md', '.txt', '.toml', '.yaml', '.yml', '.jinja'}
    excluded = {'__pycache__', '.git', '.pytest_cache', 'probe_output', 'checkpoints',
                'node_modules', 'data', 'logs', 'runs', 'collection_output'}
    result = set()
    for root in map(Path, roots):
        if not root.exists():
            raise FileNotFoundError(root)
        if root.is_file():
            result.add(root.absolute())
            continue
        for parent, dirs, files in os.walk(root, followlinks=False):
            # Directory links require an explicit root; do not silently omit dependencies.
            links = [d for d in dirs if d not in excluded and (Path(parent) / d).is_symlink()]
            if links:
                raise ValueError(f'explicit source root required for directory symlinks: {parent}: {links}')
            dirs[:] = sorted(d for d in dirs if d not in excluded)
            for name in sorted(files):
                path = Path(parent) / name
                if path.suffix in suffixes:
                    result.add(path.absolute())
    return sorted(result)
