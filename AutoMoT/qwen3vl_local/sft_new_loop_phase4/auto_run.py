"""Parameter-free host launcher; delegates all data/label/sampling checks unchanged.

Only model preparation may download official model assets. Training stays offline.
This launcher does not produce labels or replace the source-bound pipeline contract.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .identity import ROOT, digest, file_sha, write_json
from .full_pipeline import request_identity

MODEL_REPO = 'Qwen/Qwen3.5-4B'
MODEL_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
MODEL_FILES = ['config.json','tokenizer.json','tokenizer_config.json','preprocessor_config.json',
               'video_preprocessor_config.json','chat_template.jinja','model.safetensors.index.json',
               '*.safetensors','vocab.json','merges.txt','LICENSE']


def choose_data_dir(root, request):
    """Reuse a matching complete pair first, then resume, else select a new cache."""
    checkpoints=Path(root)/'checkpoints'
    candidates=sorted(checkpoints.glob('phase4_*/pipeline_ready.json'),reverse=True)
    candidates+=sorted(checkpoints.glob('phase4_auto_full/*/pipeline_ready.json'),reverse=True)
    candidates+=sorted(checkpoints.glob('phase4_*/build_request.json'),reverse=True)
    candidates+=sorted(checkpoints.glob('phase4_auto_full/*/build_request.json'),reverse=True)
    for path in candidates:
        try:
            record=json.loads(path.read_text())
            saved=record['request'] if path.name=='pipeline_ready.json' else record
        except (OSError,ValueError,KeyError):continue
        if saved==request:return path.parent
    # No old manifest is patched when sources or inputs change.
    return checkpoints/'phase4_auto_full'/digest(request)


@contextmanager
def model_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def model_ready(path):
    from qwen3vl_local.qwen35.preflight import check
    try:return check(path)['ready']
    except (OSError,ValueError,KeyError):return False


def download_model(target):
    """Fresh subprocess so the Hub's offline constant matches preparation mode."""
    from huggingface_hub import snapshot_download
    # Reuse cached files without network where possible; copy dereferenced blobs
    # so the local backend's shard containment checks remain effective.
    try:
        cached=Path(snapshot_download(MODEL_REPO,revision=MODEL_REVISION,local_files_only=True,token=False))
    except OSError:cached=None
    if cached is not None:
        import fnmatch
        target.mkdir(parents=True,exist_ok=True)
        for source in cached.iterdir():
            if source.is_file() and any(fnmatch.fnmatch(source.name,p) for p in MODEL_FILES):
                shutil.copyfile(source,target/source.name)
        if model_ready(target):return
    print(f'[Phase4] Downloading {MODEL_REPO}@{MODEL_REVISION} model assets.',flush=True)
    snapshot_download(MODEL_REPO,revision=MODEL_REVISION,local_dir=target,
                      allow_patterns=MODEL_FILES,token=False,max_workers=4)
    if not model_ready(target):raise ValueError('downloaded model failed local Qwen3.5 checks')
    write_json(target/'phase4_download.json',dict(repo=MODEL_REPO,revision=MODEL_REVISION))


def resolve_model(root, explicit=None):
    if explicit:
        path=Path(explicit).expanduser().resolve()
        if not model_ready(path):raise ValueError(f'explicit MODEL_DIR is incomplete/incompatible: {path}')
        return path
    default=Path(root)/'checkpoints/Qwen3.5-4B'
    if model_ready(default):return default
    target=Path(root)/'checkpoints/base_models/Qwen3.5-4B'/MODEL_REVISION
    with model_lock(target.parent/'.download.lock'):
        if not model_ready(target):
            env=dict(os.environ,HF_HUB_OFFLINE='0',TRANSFORMERS_OFFLINE='0',
                     HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
            subprocess.run([sys.executable,'-m','qwen3vl_local.sft_new_loop_phase4.auto_run',
                            '--download-model',str(target)],cwd=root,env=env,check=True)
        if not model_ready(target):raise ValueError('local model preparation did not complete')
    return target


def configuration(root, environ):
    root=Path(root).resolve()
    data=Path(environ.get('DATA_ROOT',root/'lead_data')).expanduser().resolve()
    if not data.is_dir():raise ValueError(f'original driving data not found: {data}')
    request=request_identity(data,ROOT/'reviewed_state_pairs_v9.json')
    chosen=choose_data_dir(root,request)
    return dict(data_root=str(data),data_dir=str(chosen),request=request)


def execute(config, root, *, environ=None, runner=subprocess.run, model_resolver=resolve_model):
    """Both runs use the same immutable data pair and separate output folders."""
    env=dict(os.environ if environ is None else environ)
    model=model_resolver(root,env.get('MODEL_DIR'))
    output=Path(root)/'checkpoints/sft_new_loop_phase4_pipeline'/f'all_{time.time_ns()}'
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'automatic_run.json',dict(**config,model_dir=str(model),rgb_order=[4,2],
        launcher_sources={n:file_sha(ROOT/n) for n in ('run.sh','auto_run.py')},
        sampling_policy='full_event_equal',epochs=7,scope='full admitted weak supervision'))
    # Inherited options from another experiment must not silently truncate/skip
    # this no-argument run or make the two experiments overwrite one another.
    for key in ('DATASET','PAIRED_WITH','PRODUCTION_INDEX','TEACHER_REGISTRY','CANDIDATE_POOL',
                'CONDITION_STREAM','ANNOTATIONS','MAX_TEACHER_QUESTIONS_PER_ROUTE','RGB_MODE',
                'MODE','TRAIN_MODE','SKIP_BUILD','SKIP_TRAIN','SKIP_EVAL','RUN_ROOT','OUTPUT_DIR',
                'PIPELINE_ROOT','HISTORY_RGB_MODE','ACTION_OUTPUT_MODE','EPOCHS','ACCUMULATION'):
        env.pop(key,None)
    env.update(PYTHON=sys.executable,DATA_ROOT=config['data_root'],DATA_DIR=config['data_dir'],
               MODEL_DIR=str(model),TRAIN_MODE='ddp',ACTION_OUTPUT_MODE='binary',EPOCHS='7',ACCUMULATION='8',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    print('[Phase4] Automatic full-pool RGB4 then RGB2; exact event presentation 1:1.',flush=True)
    print(f'[Phase4] Results: {output}',flush=True)
    for mode in (4,2):
        run_env=dict(env,HISTORY_RGB_MODE='4rgb' if mode==4 else '2rgb_endpoints',
                     SKIP_BUILD='0' if mode==4 else '1',PIPELINE_ROOT=str(output/f'rgb{mode}'))
        runner(['bash',str(ROOT/'run_full_pipeline.sh')],cwd=root,env=run_env,check=True)
    print(f'[Phase4] Both experiments completed: {output}',flush=True)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--plan-only',action='store_true',help='developer check; no download/build/training')
    p.add_argument('--download-model',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args();root=ROOT.parents[1]
    try:
        if a.download_model:download_model(a.download_model);return
        config=configuration(root,os.environ)
        if a.plan_only:
            print(json.dumps({k:v for k,v in config.items() if k!='request'},indent=2));return
        execute(config,root)
    except (OSError,ValueError,subprocess.CalledProcessError) as ex:
        p.exit(2,f'[Phase4] {ex}\n')


if __name__=='__main__':main()
