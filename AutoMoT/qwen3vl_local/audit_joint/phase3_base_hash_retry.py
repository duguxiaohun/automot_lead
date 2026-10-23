"""Bounded recovery for ctime-only changes in the base input-binding manifest.

The frozen generic hasher stays strict. Recovery requires two fresh strict reads
of the same file, both matching the existing receipt, with no further stat drift.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

FIELDS = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
POLICY = 'input_bindings_ctime_only_two_strict_reads_v1'


def signature(path):
    value = Path(path).stat()
    return {key: getattr(value, key) for key in FIELDS}


def verified_sha(path, expected, strict_sha):
    path = Path(path)
    try:
        actual = strict_sha(path)
    except RuntimeError as exc:
        prefix = f'file changed while hashing: {path}; '
        if not str(exc).startswith(prefix):
            raise
        try:
            detail = json.loads(str(exc)[len(prefix):])
            before, after, named = (detail[key] for key in
                ('descriptor_before', 'descriptor_after', 'path_after'))
            eligible = (set(before) == set(after) == set(named) == set(FIELDS)
                and before['st_ctime_ns'] != after['st_ctime_ns']
                and all(before[k] == after[k] for k in FIELDS if k != 'st_ctime_ns')
                and after == named and detail['bytes_read'] == after['st_size'])
        except (KeyError, TypeError, ValueError):
            raise exc
        if not eligible:
            raise
        print(f'[base hash recovery] ctime-only drift; checking two fresh reads: {path}; '
              f'rank={os.environ.get("RANK", "parent")}', file=sys.stderr, flush=True)
        for attempt in (1, 2):
            if signature(path) != after:
                raise RuntimeError(f'base hash recovery: metadata changed before read {attempt}: {path}') from exc
            actual = strict_sha(path)  # No retry of a second unstable read.
            if signature(path) != after:
                raise RuntimeError(f'base hash recovery: metadata changed after read {attempt}: {path}') from exc
            if actual != expected:
                raise ValueError(f'Common input binding receipt changed: {path}') from exc
        print(f'[base hash recovery] two stable reads match original receipt: {path}',
              file=sys.stderr, flush=True)
    if actual != expected:
        raise ValueError(f'Common input binding receipt changed: {path}')
    return actual


@contextmanager
def activate(common, trial):
    """Scope recovery to common.validate's one receipt-bound manifest only."""
    previous = common.validate
    if getattr(previous, '_phase3_bound_hash_retry', False):
        yield
        return

    def validate(root, data):
        root = Path(root)
        receipt = trial.read(root/'common_receipt.json')
        expected = receipt.get('input_bindings_sha256')
        if not expected:
            return previous(root, data)
        target = (root/'input_bindings.json').absolute()
        strict_sha = trial.file_sha

        def bound_sha(path):
            if Path(path).absolute() == target:
                return verified_sha(path, expected, strict_sha)
            return strict_sha(path)

        trial.file_sha = bound_sha
        try:
            return previous(root, data)
        finally:
            trial.file_sha = strict_sha

    validate._phase3_bound_hash_retry = True
    common.validate = validate
    try:
        yield
    finally:
        common.validate = previous
