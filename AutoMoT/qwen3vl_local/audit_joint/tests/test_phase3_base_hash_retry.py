import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from qwen3vl_local.audit_joint import io, phase3_base_hash_retry as guard


def fixture(tmp_path):
    path = tmp_path/'input_bindings.json'
    path.write_bytes(b'original content')
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def change_on_read(monkeypatch, path, change):
    original = Path.open
    calls = []
    class Reader:
        def __init__(self, stream, attempt):
            self.stream, self.attempt, self.done = stream, attempt, False
        def __enter__(self): return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def fileno(self): return self.stream.fileno()
        def read(self, *args):
            data = self.stream.read(*args)
            if not self.done:
                self.done = True
                # Allow a filesystem clock tick before the real mutation.
                time.sleep(.02)
                change(self.attempt, original)
            return data
    def opening(self, *args, **kwargs):
        stream = original(self, *args, **kwargs)
        if self == path and args == ('rb',):
            calls.append(1)
            return Reader(stream, len(calls))
        return stream
    monkeypatch.setattr(Path, 'open', opening)
    return calls


def test_stable_read_needs_no_retry(tmp_path):
    path, expected = fixture(tmp_path)
    calls = []
    def strict(p): calls.append(p); return io.file_sha(p)
    assert guard.verified_sha(path, expected, strict) == expected
    assert len(calls) == 1


def test_real_ctime_only_change_requires_two_fresh_reads(tmp_path, monkeypatch, capsys):
    path, expected = fixture(tmp_path)
    before = path.stat()
    def change(attempt, original):
        if attempt == 1: path.chmod(0o600)
    calls = change_on_read(monkeypatch, path, change)
    assert guard.verified_sha(path, expected, io.file_sha) == expected
    assert len(calls) == 3
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert path.stat().st_ctime_ns != before.st_ctime_ns
    assert 'two stable reads match original receipt' in capsys.readouterr().err


@pytest.mark.parametrize('damage', ['content_restored_mtime', 'mtime', 'replace', 'symlink', 'truncate', 'continuous_ctime', 'second_read_write'])
def test_recovery_rejects_mutation_and_never_loops(tmp_path, monkeypatch, damage):
    path, expected = fixture(tmp_path)
    if damage == 'symlink':
        old = tmp_path/'old'; path.rename(old); path.symlink_to(old)
    def change(attempt, original):
        if attempt == 1:
            s = path.stat()
            if damage in ('continuous_ctime', 'second_read_write'): path.chmod(0o600)
            elif damage == 'content_restored_mtime':
                with original(path, 'wb') as out: out.write(b'changed! content')
                os.utime(path, ns=(s.st_atime_ns, s.st_mtime_ns))
            elif damage == 'mtime': os.utime(path, ns=(s.st_atime_ns, s.st_mtime_ns+1000000))
            elif damage == 'truncate':
                with original(path, 'wb') as out: out.write(b'x')
            else:
                other = tmp_path/'new'
                with original(other, 'wb') as out: out.write(b'original content')
                if damage == 'replace': other.replace(path)
                else: path.unlink(); path.symlink_to(other)
        elif attempt == 2:
            if damage == 'continuous_ctime': path.chmod(0o640)
            elif damage == 'second_read_write':
                with original(path, 'wb') as out: out.write(b'changed! content')
    calls = change_on_read(monkeypatch, path, change)
    with pytest.raises((RuntimeError, ValueError)):
        guard.verified_sha(path, expected, io.file_sha)
    assert len(calls) <= 2


def test_stable_wrong_content_rejected(tmp_path):
    path, expected = fixture(tmp_path)
    path.write_bytes(b'other')
    with pytest.raises(ValueError, match='receipt changed'):
        guard.verified_sha(path, expected, io.file_sha)


@pytest.mark.parametrize('error', [OSError('storage error'), RuntimeError('other failure'), RuntimeError('file changed while hashing: x; {}')])
def test_unrelated_errors_not_retried(tmp_path, error):
    path, expected = fixture(tmp_path)
    calls = []
    def fail(p): calls.append(p); raise error
    with pytest.raises(type(error)): guard.verified_sha(path, expected, fail)
    assert len(calls) == 1


def test_scope_restores_and_never_recovers_other_files(tmp_path, monkeypatch):
    path, expected = fixture(tmp_path)
    (tmp_path/'common_receipt.json').write_text(json.dumps(dict(input_bindings_sha256=expected)))
    other = tmp_path/'index'; other.write_bytes(b'index')
    trial = SimpleNamespace(read=lambda p: json.loads(p.read_text()), file_sha=io.file_sha)
    common = SimpleNamespace(validate=lambda root, data: (trial.file_sha(root/'input_bindings.json'), trial.file_sha(other)))
    original = common.validate
    seen = []
    def verify(p, sha, strict): seen.append(p); return strict(p)
    monkeypatch.setattr(guard, 'verified_sha', verify)
    with guard.activate(common, trial):
        with guard.activate(common, trial): common.validate(tmp_path, None)
    assert seen == [path]
    assert common.validate is original and trial.file_sha is io.file_sha
    monkeypatch.setattr(guard, 'verified_sha', lambda *a: (_ for _ in ()).throw(ValueError('bad')))
    with pytest.raises(ValueError), guard.activate(common, trial): common.validate(tmp_path, None)
    assert common.validate is original and trial.file_sha is io.file_sha


def test_bounded_worker_installs_guard(tmp_path, monkeypatch):
    from qwen3vl_local.audit_joint import phase3_base_bounded as bounded
    before = bounded.common.validate
    monkeypatch.setattr(bounded.trial, 'verified_sources', lambda: 'fixture')
    with bounded.activate():
        assert bounded.common.validate._phase3_bound_hash_retry
        contract = bounded.contract()
        assert contract['input_binding_hash_policy'] == guard.POLICY
        assert 'phase3_base_hash_retry.py' in contract['sources']
    assert bounded.common.validate is before


def test_real_compatibility_subprocess_recovers_manifest_only(tmp_path):
    import subprocess
    import sys
    from qwen3vl_local.audit_joint import phase3_native_runtime as native
    runtime = native.prepare(tmp_path/'cache')
    script = r'''
import hashlib,json,os,sys,time
from pathlib import Path
from qwen3vl_local.audit_joint import phase3_base_bounded as base
from qwen3vl_local.audit_joint import phase3_common_data as common
from qwen3vl_local.sft_new_loop_phase3 import preflight
root=Path(sys.argv[1]);root.mkdir();(root/'native').mkdir()
data=root/'raw';data.mkdir()
for name in ('input_bindings.json','native/frame_index.jsonl','native/manifest.json'):
    (root/name).write_text('{}')
sha=base.trial.file_sha
release=base.trial.verified_sources()
(root/'reference.json').write_text(json.dumps(dict(index_sha256=sha(root/'native/frame_index.jsonl'),prompt_variant='fixture')))
receipt=dict(schema='phase3_common_dataset_v1',data_root=str(data),release_sha256=release,
    input_bindings_sha256=sha(root/'input_bindings.json'),files={name:sha(root/name) for name in
        ('native/frame_index.jsonl','native/manifest.json','reference.json')})
(root/'common_receipt.json').write_text(json.dumps(receipt))
# The index is synthetic; all receipt hashing and the validator are real.
preflight.check_index=lambda *a,**kw: None
target=root/'input_bindings.json';original=Path.open;calls=[]
class Reader:
    def __init__(self,s):self.s=s;self.done=False
    def __enter__(self):return self
    def __exit__(self,*a):return self.s.__exit__(*a)
    def fileno(self):return self.s.fileno()
    def read(self,*a):
        b=self.s.read(*a)
        if not self.done:
            self.done=True;time.sleep(.02);target.chmod(0o600)
        return b
def opening(p,*a,**kw):
    s=original(p,*a,**kw)
    if p==target and a==('rb',):
        calls.append(1)
        if len(calls)==1:return Reader(s)
    return s
Path.open=opening
before=common.validate
with base.activate():
    assert common.validate(root,data)==receipt
assert len(calls)==3
assert common.validate is before and base.trial.file_sha is sha
assert base.trial.verified_sources()==release
print('real native validator recovered with original SHA and restored scope')
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(tmp_path/'pool')],
        cwd=runtime, env=dict(os.environ, PYTHONPATH=str(runtime), PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'two stable reads match original receipt' in result.stderr
    assert 'restored scope' in result.stdout
