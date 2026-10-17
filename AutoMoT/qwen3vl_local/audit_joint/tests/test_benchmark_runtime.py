import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from qwen3vl_local.audit_joint import batch_benchmark as b
from qwen3vl_local.audit_joint import benchmark_runtime as rt


def test_worker_diagnostics_do_not_schedule_asynchronous_stack_walk(monkeypatch, capsys):
    calls=[]
    monkeypatch.setattr(rt.faulthandler, 'enable', lambda: calls.append('fatal_handler'))
    monkeypatch.setattr(rt.faulthandler, 'cancel_dump_traceback_later', lambda: calls.append('cancel_timer'))
    monkeypatch.setattr(rt.faulthandler, 'dump_traceback_later',
                        lambda *a, **kw: pytest.fail('do not walk a running worker stack periodically'))
    rt.trace_waits()
    assert calls == ['cancel_timer', 'fatal_handler']
    assert 'periodic_stack_dump=off' in capsys.readouterr().out


def test_parent_heartbeat_does_not_interrupt_worker(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rt, 'HEARTBEAT_SECONDS', .05)
    payload="import time; print('work START', flush=True); time.sleep(.3); print('work DONE', flush=True)"
    code=rt.run_owned([sys.executable, '-c', payload], tmp_path/'log', os.environ, 5)
    assert code == 0
    output=capsys.readouterr().out
    assert '[worker monitor]' in output and 'not stage completion' in output
    assert 'returncode=0' in output
    assert (tmp_path/'log').read_text() == 'work START\nwork DONE\n'


def test_real_cpu_worker_keeps_fatal_handler_and_single_thread(tmp_path):
    payload='''
import faulthandler, json, threading
from qwen3vl_local.audit_joint.benchmark_runtime import trace_waits
trace_waits()
assert faulthandler.is_enabled()
assert threading.active_count() == 1
for _ in range(1000):
    assert json.loads(json.dumps({'a': list(range(100))}))['a'][-1] == 99
print('CPU worker completed', flush=True)
'''
    assert rt.run_owned([sys.executable, '-c', payload], tmp_path/'log', os.environ, 5) == 0
    assert 'CPU worker completed' in (tmp_path/'log').read_text()

def records():
    return [dict(id=f'{e}/{f}',event=str(e),sequence=f'route{e}',frame=f,
                 action='STOP' if f<24 else 'KEEP',split='train',group=f'route{e}')
            for e in range(10) for f in range(40)]



def args(tmp_path):
    return SimpleNamespace(modes=[2],batches=[1],gpus='0,1,2,3',seed=7,
                           samples_per_event=32,update_samples=32,output=tmp_path/'prepared.json')


@pytest.mark.parametrize('mode',[2,4])
def test_prepared_keeps_exact_schedule_for_all_batches_without_reloading_pool(tmp_path,monkeypatch,mode):
    source=records();a=args(tmp_path);a.modes=[mode];calls=[]
    monkeypatch.setattr(b,'load_pool',lambda *args: calls.append(1) or (source,source,{'observation_contract':{}}))
    rt.prepare_phase4(a)
    assert calls==[1]
    a.prepared_samples=a.output;a.prepared_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest()
    monkeypatch.setattr(b,'load_pool',lambda *args: pytest.fail('workers must not load full pool'))
    for batch in (1,2,4,8):
        a.batches=[batch]
        items,compact,manifest,updates,actual=rt.read_prepared(a)
        original,expected=b.schedule(source,per_event=32,update_samples=32,world=4,batch=batch,mode=mode,seed=7)
        assert actual==expected
        assert [[compact[i]['id'] for i in row] for row in updates]==[[source[i]['id'] for i in row] for row in original]
        assert len(items)<=320
    a.seed=8
    with pytest.raises(ValueError,match='configuration'):rt.read_prepared(a)
    a.seed=7;a.output.write_text('{}')
    with pytest.raises(ValueError,match='digest'):rt.read_prepared(a)


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def test_timeout_cleans_worker_in_independent_session(tmp_path):
    childfile=tmp_path/'pid'
    code='import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); open('+repr(str(childfile))+',"w").write(str(os.getpid())); time.sleep(90)'
    parent='import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",'+repr(code)+'],start_new_session=True); time.sleep(90)'
    started=time.monotonic()
    rc=rt.run_owned([sys.executable,'-c',parent],tmp_path/'log',os.environ,1)
    assert rc==124 and time.monotonic()-started<5
    pid=int(childfile.read_text())
    for _ in range(50):
        if not alive(pid):break
        time.sleep(.02)
    assert not alive(pid)


@pytest.mark.parametrize('signum',[signal.SIGINT,signal.SIGTERM])
def test_signal_cleans_descendants(tmp_path,signum):
    childfile=tmp_path/'pid'
    payload='import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); open('+repr(str(childfile))+',"w").write(str(os.getpid())); time.sleep(90)'
    code='from qwen3vl_local.audit_joint.benchmark_runtime import run_owned; import sys,os; sys.exit(run_owned([sys.executable,"-c",'+repr(payload)+'],'+repr(str(tmp_path/'log'))+',os.environ,60))'
    proc=subprocess.Popen([sys.executable,'-c',code])
    try:
        deadline=time.monotonic()+5
        while not childfile.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert childfile.exists()
        pid=int(childfile.read_text());proc.send_signal(signum)
        assert proc.wait(timeout=5)==128+signum
        assert not alive(pid)
    finally:
        if proc.poll() is None:proc.kill();proc.wait()


def test_preparation_timeout_never_starts_gpu_or_next_mode(tmp_path,monkeypatch):
    (tmp_path/'student_view.json').write_text('{}');(tmp_path/'config.json').write_text('{}')
    commands=[]
    monkeypatch.setattr(b,'run_owned',lambda cmd,log,env,timeout: commands.append((cmd,env,timeout)) or 124)
    monkeypatch.setattr(sys,'argv',['bench','--phase','4','--dataset',str(tmp_path),
        '--data-root',str(tmp_path),'--model-dir',str(tmp_path),'--output',str(tmp_path/'out'),
        '--modes','2','4','--prepare-only','--prepare-timeout','120'])
    with pytest.raises(SystemExit) as ex:b.main()
    assert ex.value.code==2 and len(commands)==1
    assert commands[0][1]['CUDA_VISIBLE_DEVICES']=='' and commands[0][2]==120
    assert '--prepare-worker' in commands[0][0]
    report=json.loads((tmp_path/'out/summary.json').read_text())
    assert report['status']=='preparation_failed' and report['trials']==[]


def test_default_stops_after_first_trial_error(tmp_path,monkeypatch):
    (tmp_path/'index').write_text('{}');(tmp_path/'manifest.json').write_text('{}');(tmp_path/'config.json').write_text('{}')
    calls=[]
    monkeypatch.setattr(b,'run_owned',lambda *args: calls.append(1) or 124)
    monkeypatch.setattr(sys,'argv',['bench','--phase','3','--index',str(tmp_path/'index'),
        '--data-root',str(tmp_path),'--model-dir',str(tmp_path),'--output',str(tmp_path/'out'),
        '--modes','2','4','--batches','1','2','4'])
    with pytest.raises(SystemExit):b.main()
    assert len(calls)==1


def test_parent_prepares_once_then_reuses_snapshot_for_all_trials(tmp_path,monkeypatch):
    (tmp_path/'student_view.json').write_text('{}');(tmp_path/'config.json').write_text('{}')
    commands=[]
    def run(cmd,log,env,timeout):
        commands.append(cmd)
        if '--prepare-worker' in cmd:
            Path(cmd[cmd.index('--output')+1]).write_text('{"selected":true}')
        return 0
    monkeypatch.setattr(b,'run_owned',run)
    monkeypatch.setattr(b,'collect',lambda *args:dict(status='measured',returncode=0))
    monkeypatch.setattr(b,'summarize',lambda trials,repeats:dict(count=len(trials)))
    monkeypatch.setattr(sys,'argv',['bench','--phase','4','--dataset',str(tmp_path),
        '--data-root',str(tmp_path),'--model-dir',str(tmp_path),'--output',str(tmp_path/'out'),
        '--modes','2','--batches','1','2','4'])
    b.main()
    assert len(commands)==7 and '--prepare-worker' in commands[0]
    digests={c[c.index('--prepared-sha256')+1] for c in commands[1:]}
    assert len(digests)==1
    assert all('--prepared-samples' in c for c in commands[1:])
