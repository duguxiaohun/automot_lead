"""Explicit recovery-only adapter read-device supplement to an unchanged snapshot."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def install(adapter_module, trace_dir):
    original = adapter_module.LocalPeftModel.from_pretrained.__func__
    def load(cls, model, path, *args, **kwargs):
        kwargs['torch_device'] = 'cpu'
        record = dict(pid=os.getpid(), local_rank=os.environ.get('LOCAL_RANK'),
                      adapter=str(path), torch_device='cpu', policy='adapter_read_cpu_then_native_model_to_device')
        dest=Path(trace_dir)/f'adapter_read_{os.getpid()}.jsonl'
        with dest.open('a') as f:f.write(json.dumps(record)+'\n')
        print('[adapter-load] '+json.dumps(record),flush=True)
        return original(cls,model,path,*args,**kwargs)
    adapter_module.LocalPeftModel.from_pretrained = classmethod(load)


def main():
    target=Path(sys.argv[1]).resolve()
    tree=Path(os.environ['PHASE12_RECOVERY_TREE']).resolve()
    if target not in [tree/f'qwen3vl_local/sft_new_loop_phase2/{name}.py' for name in ('train','eval')]:
        raise ValueError('Recovery only wraps native Phase2 train/eval entrypoints')
    sys.path.insert(0,str(tree))
    from qwen3vl_local.qwen35 import adapters
    expected=os.environ['PHASE12_RECOVERY_ADAPTER_SHA']
    if hashlib.sha256(Path(adapters.__file__).read_bytes()).hexdigest()!=expected:
        raise ValueError('Recovery adapter source changed')
    install(adapters,os.environ['PHASE12_RECOVERY_TRACE'])
    sys.argv=sys.argv[1:]
    runpy.run_path(str(target),run_name='__main__')


if __name__=='__main__':main()
