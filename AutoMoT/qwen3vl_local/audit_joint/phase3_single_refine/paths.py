"""Dataset-relative names may traverse project-managed data mount symlinks."""
from pathlib import Path


def image_path(root, name):
    # The verified native dataset authorizes the member, and callers retain
    # content hashes. A mounted scene need not physically live under lead_data.
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('invalid_dataset_image_path')
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or relative == Path('.'):
        raise ValueError('invalid_dataset_image_path')
    logical = Path(root) / relative
    resolved = logical.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError('dataset_image_not_file')
    # Keep the logical path so repeated hash reads see any changed link target.
    return logical
