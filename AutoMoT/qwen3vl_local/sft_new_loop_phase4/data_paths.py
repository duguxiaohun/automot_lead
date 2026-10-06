"""Logical paths in a trusted, possibly symlinked LEAD dataset.

The configured dataset tree may link scenarios, routes or individual assets to
other disks. Preserve that logical namespace in provenance; resolving children
and requiring their physical locations beneath the root rejects valid mounts.
Reject traversal/absolute references before joining. Callers still validate
route/frame identity and content hashes; this is not a sandbox for untrusted
filesystem owners (the dataset also contains trusted pickle inputs).
"""
from pathlib import Path


def data_path(root, relative):
    value = str(relative)
    if (not value or '\\' in value or '\x00' in value
            or any(part in ('', '.', '..') for part in value.split('/'))
            or Path(value).is_absolute()):
        raise ValueError(f'invalid relative dataset path: {value!r}')
    return Path(root).resolve() / value
