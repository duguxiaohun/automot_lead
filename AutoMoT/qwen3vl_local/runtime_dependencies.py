"""Stdlib-first environment checks for public launchers; no package installation."""
import importlib
import shlex
import sys


def check(profile,operation):
    modules={'numpy':'numpy','PIL':'Pillow'}
    if profile in ('abc_refine','trajectory'):
        modules.update(torch='torch',transformers='transformers',peft='peft',
                       safetensors='safetensors',accelerate='accelerate',torchvision='torchvision')
    if profile=='abc_refine' and operation in ('prepare','all'):modules['ijson']='ijson'
    if profile=='trajectory' and operation in ('prepare','all'):modules['carla']='carla'
    errors=[];missing=[]
    for name,package in modules.items():
        try:importlib.import_module(name)
        except Exception as exc:
            errors.append(f'{name}: {type(exc).__name__}: {exc}')
            if isinstance(exc,ModuleNotFoundError) and exc.name==name:missing.append(package)
    if errors:
        message='Environment check failed before starting work. Python: '+sys.executable+'\n'+'\n'.join(errors)
        pip=[name for name in missing if name!='carla']
        if pip:message+='\n'+shlex.join([sys.executable,'-m','pip','install',*pip])
        if 'carla' in missing:message+='\nInstall the Python-compatible CARLA wheel from the project CARLA_0915/PythonAPI/carla/dist directory.'
        raise RuntimeError(message)


if __name__=='__main__':
    print('[runtime] PYTHON='+sys.executable,flush=True)
    check(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else '--help')
