"""Reuse the synced original Phase3 source snapshot for data and new training.

Current LeadMoT geometry options must not be copied into a frozen old release.
No Git or source-hash bypass; user asset paths are explicitly preserved.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import json

from . import versioned_data as cache

APP=Path(__file__).resolve().parents[2]
OVERLAY=('phase3_common_data.py','phase3_common_runtime.py','phase3_launch.py','phase3_launch.sh',
         'phase3_launch_worker.py','phase3_data_ready.py','phase3_prompt_runtime.py','phase3_prompt_review.py',
         'versioned_data.py','data_prepare.py','phase3_native_runtime.py',
         'phase3_base_coverage.py','phase3_base_coverage_sampling.py')


def prepare(checkpoints, *, app=APP):
    from .phase3_matched_image_launch import snapshot
    package=app/'qwen3vl_local/audit_joint/phase3_behavior_diagnostic_v1'
    manifest=json.loads((package/'runtime_sources.json').read_text())
    def contract():
        paths=[package/'runtime_sources.json']
        for name,value in manifest['files'].items():
            paths.append((app if value['source']=='project' else package/'compat_sources')/name)
        paths.extend(app/'qwen3vl_local/audit_joint'/name for name in OVERLAY)
        paths.extend(app/'qwen3vl_local/audit_joint'/name for name in (
            'phase3_matched_image_launch.py','phase3_matched_image_eval.py','phase3_behavior_diagnostics.py'))
        return dict(sources=cache.bindings(paths),native_release=manifest['files'][
            'qwen3vl_local/audit_joint/phase3_base_trial_sources.json']['sha256'])
    def build(output):
        snapshot(output,dict(release_sha256=manifest['files'][
            'qwen3vl_local/audit_joint/phase3_base_trial_sources.json']['sha256']),app=app)
        for name in OVERLAY:
            shutil.copyfile(app/'qwen3vl_local/audit_joint'/name,output/'qwen3vl_local/audit_joint'/name)
    def validate(output):
        env=dict(os.environ,PYTHONPATH=str(output),PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([sys.executable,'-B','-c',
            'from qwen3vl_local.audit_joint.phase3_base_trial import verified_sources; verified_sources()'],
            cwd=output,env=env,check=True)
    result=cache.ensure(Path(checkpoints)/'generation_cache','phase3_sources','native_v2',contract,build,validate)
    return Path(result['directory'])


def needed():
    if os.environ.get('AUTOMOT_PHASE3_VERIFIED_RUNTIME')=='1':return False
    from . import phase3_base_trial as trial
    try:trial.verified_sources()
    except ValueError:return True
    return False


def execute(module, arguments, *, checkpoints=None):
    if module not in ('phase3_launch','data_prepare'):raise ValueError('Unsupported compatibility entry')
    args=list(arguments)
    if module=='phase3_launch':
        defaults={'--data-root':APP/'lead_data','--model-dir':APP/'checkpoints/Qwen3.5-4B',
                  '--common-data-dir':APP/'checkpoints/phase3_common_v1','--collection-dir':APP/'keyframe_filter/collection_output'}
        paths={'--index','--dataset','--resume','--output',*defaults}
    else:
        defaults={'--data-root':APP/'lead_data','--model-dir':APP/'checkpoints/Qwen3.5-4B',
                  '--checkpoints':checkpoints or APP/'checkpoints','--collection':APP/'keyframe_filter/collection_output'}
        paths={'--existing','--receipt','--maps','--recipe',*defaults}
    present=set()
    for i,value in enumerate(args):
        key,sep,inline=value.partition('=')
        if key not in paths:continue
        present.add(key)
        if sep:args[i]=key+'='+str(Path(inline).absolute())
        else:args[i+1]=str(Path(args[i+1]).absolute())
    for flag,path in defaults.items():
        if flag not in present:args.extend([flag,str(Path(path).absolute())])
    runtime=prepare(Path(checkpoints or APP/'checkpoints'))
    env=dict(os.environ,PYTHONPATH=str(runtime),PYTHONDONTWRITEBYTECODE='1',AUTOMOT_PHASE3_VERIFIED_RUNTIME='1')
    print('[phase3 runtime] verified synchronized compatibility sources: '+str(runtime),flush=True)
    return subprocess.call([sys.executable,'-B','-m','qwen3vl_local.audit_joint.'+module,*args],cwd=runtime,env=env)
