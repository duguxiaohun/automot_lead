"""Content identity for the exact local safetensors base and its input assets."""
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError(f'Model asset changed while hashing: {path}')
    return digest.hexdigest()


def base_asset_hashes(root):
    root = Path(root).expanduser().resolve()
    index = root / 'model.safetensors.index.json'
    if (root / 'model.safetensors').is_file():
        weights = ['model.safetensors']
    elif index.is_file():
        weights = sorted(set(json.loads(index.read_text())['weight_map'].values()))
        if not weights:
            raise ValueError('Empty base weight index')
    else:
        raise FileNotFoundError(f'Missing local safetensors base weights: {root}')
    files = {p.name: p for p in root.iterdir()
             if p.is_file() and p.suffix in {'.json', '.jinja', '.txt', '.model', '.tiktoken'}}
    for name in weights:
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f'Missing or external base weight shard: {name}')
        files[name] = path
    return {name: sha256_file(path) for name, path in sorted(files.items())}


def require_weight_identity(assets):
    if not assets or not any(name.endswith('.safetensors') for name in assets):
        raise ValueError('Missing base weight SHA256 identity; load the base with LocalModel before saving/loading adapters')
