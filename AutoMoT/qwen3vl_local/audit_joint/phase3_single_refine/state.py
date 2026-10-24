"""Atomic stage state and a nonblocking per-experiment lock."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,ensure_ascii=False,allow_nan=False,separators=(',',':'))
            stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        Path(name).unlink(missing_ok=True)

@contextmanager
def lock(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.single.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('another_single_pipeline_is_running') from None
        yield
