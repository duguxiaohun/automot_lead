"""Check the selected interpreter before importing the workflow or creating outputs."""
import importlib
import shlex
import sys


def check(operation):
    packages = {'numpy':'numpy', 'PIL':'Pillow'}
    if operation in ('all','check','probe','train','evaluate','package'):
        packages.update(torch='torch', transformers='transformers', peft='peft',
                        safetensors='safetensors', accelerate='accelerate', torchvision='torchvision')
    errors=[];missing=[]
    for module,package in packages.items():
        try:importlib.import_module(module)
        except Exception as exc:
            errors.append(f'{module}: {type(exc).__name__}: {exc}')
            if isinstance(exc,ModuleNotFoundError) and exc.name==module:missing.append(package)
    if errors:
        message='Selected Python: '+sys.executable+'\nEnvironment check failed before preparation:\n'+'\n'.join(errors)
        if missing:message+='\nInstall missing packages in this interpreter: '+shlex.join([sys.executable,'-m','pip','install',*missing])
        raise RuntimeError(message)


def main():
    operation=sys.argv[1] if len(sys.argv)>1 else '--help'
    print('[single environment] PYTHON='+sys.executable,flush=True)
    check(operation)
    # Direct script execution is intentional: this check itself needs only stdlib.
    from qwen3vl_local.audit_joint.phase3_single_refine.pipeline import main as pipeline_main
    pipeline_main()


if __name__=='__main__':main()
