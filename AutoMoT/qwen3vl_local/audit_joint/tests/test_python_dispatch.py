import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from qwen3vl_local.audit_joint.python_dispatch import arguments,environment


def test_real_torchrun_parser_keeps_application_run_argument():
    from torch.distributed.run import get_args_parser
    argv=['-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','some.module','--run','a b']
    with pytest.raises(SystemExit):get_args_parser().parse_args(argv[2:])
    parsed=get_args_parser().parse_args(arguments(argv)[2:])
    assert parsed.training_script=='some.module'
    assert parsed.training_script_args==['--run','a b'] and parsed.module
    assert arguments(arguments(argv))==arguments(argv)


def test_normal_python_args_and_exit_code_preserved():
    env=dict(os.environ,**environment())
    done=subprocess.run([env['PYTHON'],'-c','import sys; print(sys.executable); raise SystemExit(7)'],
                        env=env,text=True,capture_output=True)
    assert done.returncode==7 and done.stdout.strip()==sys.executable


def test_real_cpu_torchrun_workers_receive_run_argument(tmp_path):
    (tmp_path/'dispatch_fixture.py').write_text('''import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--run',required=True);a=p.parse_args()
Path(a.run).mkdir(exist_ok=True)
(Path(a.run)/(os.environ['RANK']+'.json')).write_text(json.dumps(dict(world=os.environ['WORLD_SIZE'],run=a.run)))
''')
    env=dict(os.environ,**environment());env['PYTHONPATH']=str(tmp_path)+os.pathsep+str(Path.cwd())
    out=tmp_path/'output with spaces'
    done=subprocess.run([env['PYTHON'],'-m','torch.distributed.run','--standalone',
        '--nproc_per_node=2','-m','dispatch_fixture','--run',str(out)],env=env,
        text=True,capture_output=True,timeout=45)
    assert done.returncode==0,done.stderr
    assert [json.loads((out/f'{r}.json').read_text()) for r in (0,1)]==[
        dict(world='2',run=str(out)),dict(world='2',run=str(out))]
