"""Bounded decoded-record memory over the original, fully hashed JSONL files.

Offsets and per-line hashes are built from the same bytes as the admitted split
hash. Views retain only indices; every cache miss verifies bytes before decoding.
No derived dataset, changed ordering, or weakened source check is introduced.
"""
from array import array
from collections import OrderedDict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

from qwen3vl_local.audit_joint.abc_delivery import sha
from .contracts import source_identity


class Store:
    cache_limit = 8

    def __init__(self, path, expected, split, data_root):
        self.path = str(Path(path).resolve())
        self.split = split
        self.data_root = str(data_root)
        self.offsets = array('Q', [0])
        self.hashes = bytearray()
        self.ids = []
        self.lookup = {}
        self.cache = OrderedDict()
        whole = hashlib.sha256()
        with open(self.path, 'rb') as f:
            for raw in f:
                whole.update(raw)
                r = json.loads(raw)
                if r['split'] != split:
                    raise ValueError('indexed_split_member_changed')
                if r['id'] in self.lookup:
                    raise ValueError('duplicate_probe_members')
                self.lookup[r['id']] = len(self.ids)
                self.ids.append(r['id'])
                self.offsets.append(self.offsets[-1] + len(raw))
                self.hashes.extend(hashlib.sha256(raw).digest())
        if whole.hexdigest() != expected:
            raise ValueError('ABC_split_changed')

    def sample(self, index):
        if index not in self.cache:
            with open(self.path, 'rb') as f:
                f.seek(self.offsets[index])
                raw = f.read(self.offsets[index + 1] - self.offsets[index])
            if hashlib.sha256(raw).digest() != self.hashes[32*index:32*(index+1)]:
                raise ValueError('indexed_record_changed: ' + self.path)
            from .runtime import sample
            self.cache[index] = sample(json.loads(raw), self.data_root)
            if len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getstate__(self):
        # Spawned sensor workers receive offsets, never decoded pool contents.
        # ID lookup is only used by the parent sampler; worker views already
        # contain integer indices, including repetitions and rank slicing.
        return dict(vars(self), cache=OrderedDict(), ids=[], lookup={})


class Row(Mapping):
    __slots__ = ('store', 'index')

    def __init__(self, store, index):
        self.store, self.index = store, index

    def __getitem__(self, key):
        return self.store.sample(self.index)[key]

    def __iter__(self):
        return iter(self.store.sample(self.index))

    def __len__(self):
        return len(self.store.sample(self.index))


class Rows(Sequence):
    def __init__(self, store, indices=None):
        self.store = store
        self.indices = range(len(store.ids)) if indices is None else indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Rows(self.store, self.indices[index])
        return Row(self.store, self.indices[index])

    def ordered(self, ids):
        return Rows(self.store, array('Q', (self.store.lookup[k] for k in ids)))

    def probe_records(self):
        return (dict(id=self.store.ids[i], split=self.store.split) for i in self.indices)


def load(directory, *, data_root, manifest_sha256):
    from .consumption import checked_bytes
    from qwen3vl_local.audit_joint import label_quarantine
    directory = Path(directory)
    manifest = json.loads(checked_bytes(directory/'manifest.json', manifest_sha256))
    if manifest['code'] != source_identity(data_only=True):
        raise ValueError('ABC_data_code_changed')
    if manifest['ledger_sha256'] != sha(label_quarantine.LEDGER):
        raise ValueError('ABC_quarantine_changed')
    if manifest['scoped_smoke_only'] or not manifest['decoded']:
        raise ValueError('smoke_data_not_formally_ready')
    rows = {split: Rows(Store(directory/(split+'.jsonl'), value, split, data_root))
            for split, value in manifest['splits'].items()}
    for folder, value in manifest['routes'].items():
        if sha(directory/'routes'/folder/'receipt.json') != value:
            raise ValueError('ABC_route_receipt_changed')
    return rows, manifest


def checkpoint(path):
    """Map native ZIP tensor storage; unused optimizer tensors stay unpaged.

    No eager fallback: this run's checkpoints use native torch.save ZIP format.
    Mapping is private; callers must release the dictionary after copying states.
    """
    import torch
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
