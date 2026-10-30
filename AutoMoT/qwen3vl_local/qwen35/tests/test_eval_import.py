"""CPU regression for the reported import-time CUDA_VISIBLE_DEVICES mutation."""
import os
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.parametrize('module_name',['qwen3vl_local.sft_new_loop_phase2.eval','qwen3vl_local.goalgen.eval'])
@pytest.mark.parametrize('rank',[None,'0','1'])
@pytest.mark.parametrize('order',['torch_first','eval_first'])
def test_import_eval_does_not_select_gpus_or_change_cpu_environment(rank,order,module_name):
    script=r'''
import os,sys,subprocess
sys.argv=['cpu_test']
calls=[]
original=subprocess.check_output
def fake(command,*a,**kw):
    if command[0]=='nvidia-smi':
        calls.append(command)
        return '0, 0, 0\n'
    return original(command,*a,**kw)
subprocess.check_output=fake
before={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','GPU_IDS','RANK','LOCAL_RANK','WORLD_SIZE','GOALGEN_LOCAL_CUDA_INDEX')}
if sys.version and ORDER=='torch_first':
    import torch
    assert not torch.cuda.is_available()
import importlib
module=importlib.import_module(MODULE)
assert not calls, calls
assert before=={k:os.environ.get(k) for k in before}
from qwen3vl_local.sft_new_loop_phase1.run_integrity import environment
record=environment()
assert record['gpu_count']==0 and record['gpu_names']==[], record
'''.replace('ORDER',repr(order)).replace('MODULE',repr(module_name))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    for key in ('GPU_IDS','RANK','LOCAL_RANK','WORLD_SIZE'):env.pop(key,None)
    if rank is not None:env.update(RANK=rank,LOCAL_RANK=rank,WORLD_SIZE='2')
    result=subprocess.run([sys.executable,'-c',script],env=env,capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('module_name',['qwen3vl_local.sft_new_loop_phase2.eval','qwen3vl_local.goalgen.eval'])
def test_cli_gpu_selection_still_runs_before_torch_and_honors_native_policy(monkeypatch,module_name):
    # Execute the actual script prefix only, stopping before torch import. This
    # verifies CLI behavior without initializing CUDA or running an evaluator.
    import importlib
    module=importlib.import_module(module_name)
    monkeypatch.setenv("GOALGEN_LOCAL_CUDA_INDEX", "unchanged")
    source=Path(module.__file__).read_text().split('\nimport torch',1)[0]
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:'3, 0, 0\n')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.delenv('GPU_IDS',raising=False)
    monkeypatch.setenv('WORLD_SIZE','1')
    if 'sft_new_loop_phase2' in module_name:
        monkeypatch.setattr(sys,'argv',['eval','--device','cpu'])
        exec(compile(source,module.__file__,'exec'),dict(__name__='__main__',__file__=module.__file__))
        assert os.environ['CUDA_VISIBLE_DEVICES']==''
    monkeypatch.setattr(sys,'argv',['eval'])
    exec(compile(source,module.__file__,'exec'),dict(__name__='__main__',__file__=module.__file__))
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
    monkeypatch.setenv('GPU_IDS','2')
    exec(compile(source,module.__file__,'exec'),dict(__name__='__main__',__file__=module.__file__))
    assert os.environ['CUDA_VISIBLE_DEVICES']=='2'
