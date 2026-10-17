import hashlib
import os
from pathlib import Path

import pytest
from qwen3vl_local.audit_joint.io import file_sha


def test_open_refreshes_metadata_before_hash_snapshot(tmp_path, monkeypatch):
    """A refresh at open is outside the read transaction, not an in-read write."""
    path=tmp_path/'receipt.json';path.write_bytes(b'unchanged content')
    original=Path.open
    def opening(self,*args,**kwargs):
        if self==path:
            s=self.stat();os.utime(self,ns=(s.st_atime_ns,s.st_mtime_ns+1_000_000))
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',opening)
    assert file_sha(path)==hashlib.sha256(b'unchanged content').hexdigest()


@pytest.mark.parametrize('change',['overwrite','replace','symlink','metadata'])
def test_changes_during_open_read_are_rejected(tmp_path,monkeypatch,change):
    path=tmp_path/'receipt.json';path.write_bytes(b'old content')
    target=path
    if change=='symlink':
        target=tmp_path/'old';path.rename(target);path.symlink_to(target)
    original=Path.open
    class ChangingReader:
        def __init__(self,stream):self.stream=stream;self.changed=False
        def __enter__(self):return self
        def __exit__(self,*a):self.stream.close()
        def fileno(self):return self.stream.fileno()
        def read(self,*a):
            data=self.stream.read(*a)
            if not self.changed:
                self.changed=True
                if change=='overwrite':
                    with original(path,'wb') as f:f.write(b'new content with changed length')
                elif change=='metadata':
                    s=target.stat();os.utime(target,ns=(s.st_atime_ns,s.st_mtime_ns+1_000_000))
                else:
                    other=tmp_path/'new'
                    with original(other,'wb') as f:f.write(b'new content')
                    if change=='replace':other.replace(path)
                    else:path.unlink();path.symlink_to(other)
            return data
    def opening(self,*a,**kw):
        stream=original(self,*a,**kw)
        return ChangingReader(stream) if self==path and a==('rb',) else stream
    monkeypatch.setattr(Path,'open',opening)
    with pytest.raises(RuntimeError,match='file changed while hashing') as exc:file_sha(path)
    assert 'descriptor_before' in str(exc.value) and 'path_after' in str(exc.value)
