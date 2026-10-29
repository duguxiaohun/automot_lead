"""Restore the pinned history from local evidence; no network or model loading."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import time

from . import history
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof

ARCHIVE = 'phase12_history_20261030.tar.gz'
MEMBER = 'checkpoints/phase12_history_20261030/contract.json'
MAX_BYTES = 64 * 1024 * 1024


def publish(content, target):
    """Validate before publishing; an existing file is never replaced."""
    expected = proof.read(history.RELEASE)['contract_sha256']
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('Reconstructed historical contract differs from the pinned release')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.history-', dir=target.parent) as temporary:
        pending = Path(temporary)/'contract.json'
        pending.write_bytes(content)
        history.load(pending)
        try:
            os.link(pending, target)
        except FileExistsError:
            history.load(target)  # A concurrent preparer may have published it.
    return history.load(target)


def read_member(tar, member):
    if not member.isfile() or member.size > MAX_BYTES:
        raise ValueError('Historical archive member is not a bounded regular file')
    with tar.extractfile(member) as stream:
        return stream.read(MAX_BYTES + 1)


def bundle_source(path, descriptor):
    """Read only bound JSON evidence, never extract a bundle wholesale."""
    with tarfile.open(path, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(set(names)) != len(names):
            raise ValueError('Duplicate historical archive member')
        candidates = [m for m in members if m.name.endswith('/lora_production/evaluation_inputs.json')
                      and descriptor['source_name'] in PurePosixPath(m.name).parts]
        if len(candidates) != 1:
            raise ValueError('Missing/ambiguous historical inputs in '+str(path))
        base = PurePosixPath(candidates[0].name).parent
        wanted = ['evaluation_inputs.json', 'evaluation_complete.json', *descriptor['case_shards']]
        with tempfile.TemporaryDirectory(prefix='phase12-history-') as temporary:
            root = Path(temporary)/descriptor['source_name']/'lora_production'
            root.mkdir(parents=True)
            for name in wanted:
                if PurePosixPath(name).name != name or name in ('.', '..'):
                    raise ValueError('Unsafe historical shard name')
                (root/name).write_bytes(read_member(tar, tar.getmember(str(base/name))))
            return checked_source(root/'evaluation_inputs.json', descriptor)


def checked_source(path, descriptor):
    source = history.import_inputs(path, primary=descriptor['primary'])
    # Native eval directories and packaged directories have different names;
    # their byte-bound input/completion/shards must still be exactly the release.
    source['source_name'] = descriptor['source_name']
    if any(source[k] != value for k, value in descriptor.items()):
        raise ValueError('Historical source differs from release: '+str(path))
    return source


def discover(root, *, timeout=120):
    """Bounded metadata-only walk; checkpoints itself may be a shared symlink."""
    started = last = time.monotonic()
    count = 0
    for directory, dirs, files in os.walk(root.resolve(), followlinks=False):
        count += 1
        now = time.monotonic()
        if now - started > timeout or count > 100000:
            raise ValueError('Historical discovery limit reached; use --history-contract with a verified local file')
        if now - last >= 10:
            print(f'[history] scanning local metadata: {count} directories', flush=True)
            last = now
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in
                         ('code', 'app', 'lead_data', 'node_modules', '__pycache__')
                         and 'runtime' not in d and not d.startswith(('checkpoint-', 'Qwen')))
        for name in sorted(files):
            if name == 'evaluation_inputs.json' or name.endswith('_audit_bundle.tar.gz'):
                yield Path(directory)/name


def ensure(target, app):
    target, app = Path(target), Path(app)
    if target.exists() or target.is_symlink():
        return history.load(target)  # A wrong existing contract is an error, not a fallback.
    release = proof.read(history.RELEASE)
    for archive in (app/ARCHIVE, app/'checkpoints'/ARCHIVE, target.parent/ARCHIVE):
        if archive.is_file():
            with tarfile.open(archive, 'r:gz') as tar:
                matching = [m for m in tar.getmembers() if m.name == MEMBER]
                if len(matching) != 1:
                    raise ValueError('Missing/ambiguous released contract in '+str(archive))
                content = read_member(tar, matching[0])
            result = publish(content, target)
            print('[history] restored verified local archive: '+str(target), flush=True)
            return result
    descriptors = release.get('sources', [])
    if not descriptors:
        raise ValueError('Missing historical contract and source descriptors: '+str(target))
    wanted = {s['inputs_sha256']: s for s in descriptors}
    by_name = {s['source_name']+'.tar.gz': s for s in descriptors}
    found, archives, rejected = {}, [], []
    def try_source(path, descriptor, importer):
        try:
            found[descriptor['inputs_sha256']] = importer(path, descriptor)
        except (ValueError, KeyError, TypeError, OSError, tarfile.TarError) as exc:
            # Diagnostic packages may carry inputs alone, and interrupted runs
            # may coexist with a complete audited copy. Neither is authoritative.
            rejected.append(f'{path}: {type(exc).__name__}: {exc}')
            print('[history] unusable copy: '+rejected[-1], flush=True)
            return
        print('[history] verified '+descriptor['source_name'], flush=True)
    print('[history] contract absent; checking existing local historical evaluations', flush=True)
    for path in discover(app/'checkpoints'):
        if path.name == 'evaluation_inputs.json':
            if path.stat().st_size > MAX_BYTES:
                continue
            digest = proof.sha(path)
            if digest in wanted and digest not in found:
                try_source(path, wanted[digest], checked_source)
        elif path.name in by_name:
            archives.append(path)
        if len(found) == len(wanted):
            break
    for path in archives:
        descriptor = by_name[path.name]
        digest = descriptor['inputs_sha256']
        if digest not in found:
            try_source(path, descriptor, bundle_source)
    missing = [s['source_name'] for s in descriptors if s['inputs_sha256'] not in found]
    if missing:
        raise ValueError('Historical evidence incomplete; no run/training created. Missing:\n  '
                         +'\n  '.join(missing)+'\nUse existing local historical bundles or the released '
                         +ARCHIVE+'; do not re-sample or load old LoRA weights. '
                         +f'Unusable copies: {len(rejected)} (see messages above).')
    sources = [found[s['inputs_sha256']] for s in descriptors]
    from qwen3vl_local.sft_new_loop_phase1.training_policy import physical_group
    groups = sorted({physical_group(tuple(r['key'])) for s in sources for r in s['records']})
    obj = dict(schema=history.SCHEMA, sources=sources, holdout_groups=groups,
               scope='Only explicitly imported verified historical tests; not all possible historical experiments')
    result = publish((json.dumps(obj, ensure_ascii=False, separators=(',', ':'))+'\n').encode(), target)
    print('[history] rebuilt exact released contract from local evidence: '+str(target), flush=True)
    return result
