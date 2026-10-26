import json
from pathlib import Path
import pytest
from qwen3vl_local.action_prior import host_memory,launch


@pytest.mark.parametrize('version',[1,2])
def test_cgroup_mount_root_and_membership_paths(tmp_path,version):
    proc=tmp_path/'proc';cg=tmp_path/'cg';(proc/'self').mkdir(parents=True)
    (proc/'self/status').write_text('VmRSS:\t123 kB\nVmHWM:\t150 kB\n')
    (proc/'meminfo').write_text('MemAvailable: 300 kB\n')
    (proc/'self/cgroup').write_text('0::/pod/container\n' if version==2 else '8:memory:/pod/container\n')
    base=cg if version==2 else cg/'memory';base.mkdir(parents=True)
    leaf=base/'pod/container';leaf.mkdir(parents=True)
    name='memory.events' if version==2 else 'memory.oom_control'
    (base/name).write_text('oom_kill 1\n');(leaf/name).write_text('oom_kill 2\n')
    result=host_memory.snapshot(proc,cg)
    assert result['process']['VmRSS_bytes']==123*1024
    assert result['cgroup'][str(base/name)]=='oom_kill 1\n'
    assert result['cgroup'][str(leaf/name)]=='oom_kill 2\n'


def test_unavailable_proc_is_reported(tmp_path):
    result=host_memory.snapshot(tmp_path/'missing',tmp_path/'missing')
    assert result['process']=={} and 'unavailable' in result['meminfo']


def test_parent_records_after_failed_child(tmp_path,monkeypatch):
    def failed(*args):raise RuntimeError('child killed')
    monkeypatch.setattr(launch,'run_logged',failed)
    with pytest.raises(RuntimeError,match='child killed'):
        host_memory.run_logged(['child'],tmp_path/'logs'/'a_probe.log')
    for suffix in ('before','after'):
        evidence=json.loads((tmp_path/'memory'/f'a_probe.{suffix}.json').read_text())
        assert 'cgroup' in evidence and evidence['pid']>0
