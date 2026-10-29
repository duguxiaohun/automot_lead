import os
from pathlib import Path
import time

import pytest
from qwen3vl_local.sft_new_loop_phase12 import history
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof


def fixture(tmp_path):
    path=tmp_path/'contract.json'
    proof.write(path,dict(schema=history.SCHEMA,sources=[],holdout_groups=[]))
    proof.write(history.RELEASE,dict(contract_sha256=proof.sha(path)))
    return path


def link_change(path):
    # Some filesystems coalesce metadata timestamps within a clock tick.
    time.sleep(.02)
    alias=path.with_name('temporary-link')
    os.link(path,alias);alias.unlink()


def intercept_read(monkeypatch,path,callback):
    original=Path.open
    calls=[]
    class Reader:
        def __init__(self,handle):self.handle=handle
        def __enter__(self):self.handle.__enter__();return self
        def __exit__(self,*args):return self.handle.__exit__(*args)
        def fileno(self):return self.handle.fileno()
        def read(self,*args):
            content=self.handle.read(*args)
            calls.append(1);callback(len(calls))
            return content
    def opening(self,*args,**kwargs):
        handle=original(self,*args,**kwargs)
        return Reader(handle) if self==path and args and args[0]=='rb' else handle
    monkeypatch.setattr(Path,'open',opening)
    return calls


def test_real_link_unlink_during_hash_is_recovered_only_for_pinned_contract(tmp_path,monkeypatch,capsys):
    path=fixture(tmp_path)
    calls=intercept_read(monkeypatch,path,lambda n:link_change(path) if n==1 else None)
    with pytest.raises(ValueError,match='File changed while hashing'):proof.sha(path)
    calls.clear()
    result=history.load(path)
    assert result['schema']==history.SCHEMA and len(calls)==3
    assert 'two stable reads' in capsys.readouterr().err


@pytest.mark.parametrize('change',['mtime','replacement','ongoing_ctime','content'])
def test_actual_mutations_fail_without_unbounded_retries(tmp_path,monkeypatch,change):
    path=fixture(tmp_path)
    def mutate(n):
        if change=='ongoing_ctime':link_change(path)
        elif n==1:
            if change=='mtime':
                s=path.stat();os.utime(path,ns=(s.st_atime_ns,s.st_mtime_ns+1000000000))
            elif change=='replacement':
                other=path.with_name('replacement');other.write_text(path.read_text());os.replace(other,path)
            else:path.write_text(path.read_text()+' ')
    calls=intercept_read(monkeypatch,path,mutate)
    with pytest.raises(ValueError):history.load(path)
    assert len(calls)<=2


@pytest.mark.parametrize('change',['ctime','content'])
def test_second_confirmation_must_agree_with_first(tmp_path,monkeypatch,change):
    path=fixture(tmp_path);original=history._read_contract_once;calls=[]
    def reading(p):
        calls.append(1)
        if len(calls)==3:
            if change=='ctime':link_change(path)
            else:path.write_text(path.read_text()+' ')
        data,states=original(p)
        if len(calls)==1:
            # First drift is deterministic; later mutations above are real.
            changed=list(states[-1]);changed[-1]+=1
            states=(*states[:-1],tuple(changed))
        return data,states
    monkeypatch.setattr(history,'_read_contract_once',reading)
    with pytest.raises(ValueError):history.load(path)
    assert len(calls)==3


def test_contract_parser_uses_the_verified_bytes_not_a_second_unbound_read(tmp_path,monkeypatch):
    path=fixture(tmp_path);original=history._contract_bytes
    def read_and_change(p,expected):
        data=original(p,expected)
        p.write_text('{}')
        return data
    monkeypatch.setattr(history,'_contract_bytes',read_and_change)
    assert history.load(path)['schema']==history.SCHEMA
    with pytest.raises(ValueError,match='content'):history.load(path)


def test_missing_and_persistent_wrong_sha_never_accepted(tmp_path):
    path=fixture(tmp_path);path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='content'):history.load(path)
    path.unlink()
    with pytest.raises(ValueError,match='Missing historical contract'):history.load(path)
