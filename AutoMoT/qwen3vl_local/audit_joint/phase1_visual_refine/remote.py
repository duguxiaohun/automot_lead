"""Run and package explicit prompt-only experiments against existing local runs."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

APP = Path(__file__).resolve().parents[3]
MODES = ('2rgb_endpoints', '4rgb')
DEFAULT_PIPELINE = 'checkpoints/sft_new_loop_phase1_pipeline/judgment_release_20261025_2rgb_endpoints_4rgb'


def asset_path(value, app=APP):
    path = Path(value)
    if '..' in path.parts: raise ValueError('Asset traversal is not allowed')
    if path.is_absolute():
        if 'checkpoints' not in path.parts: raise ValueError('Use a current-machine asset path')
        path = Path(*path.parts[path.parts.index('checkpoints'):])
    return app/path


def discover(pipeline, mode, app=APP):
    inputs = Path(pipeline)/mode/'eval_review/lora_production/evaluation_inputs.json'
    if not inputs.is_file(): raise FileNotFoundError(f'Missing completed {mode} evaluation: {inputs}')
    value = json.loads(inputs.read_text())
    if value.get('phase') != 1 or value.get('history_rgb_mode') != mode:
        raise ValueError('Phase/RGB mode mismatch')
    adapter = asset_path(value['arguments']['adapter_dir'], app)
    if not (adapter/'phase12_adapter_identity.json').is_file():
        raise FileNotFoundError(f'Original local adapter is required: {adapter}; no cross-machine weight fallback')
    return inputs, adapter


def choose_gpu():
    pinned = os.environ.get('GPU_IDS','').strip()
    if pinned:
        ids = [x.strip() for x in pinned.split(',')]
        if len(ids)!=1 or not ids[0].isdigit(): raise ValueError('This paired replay needs one GPU; use GPU_IDS=0')
        return ids[0]
    output = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'],text=True)
    rows = [tuple(int(x.strip()) for x in line.split(',')) for line in output.splitlines() if line.strip()]
    if not rows: raise RuntimeError('No visible GPU reported by nvidia-smi')
    return str(min(rows,key=lambda r:(r[1],r[2],r[0]))[0])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def package(output, inputs):
    from .experiment import verify_source_identity
    output=Path(output);completion=json.loads((output/'completion.json').read_text())
    if completion.get('status')!='completed': raise ValueError('Cannot package incomplete inference')
    verify_source_identity(completion['source_sha256'])
    for name,digest in completion['outputs'].items():
        if Path(name).name!=name or sha(output/name)!=digest: raise ValueError('Replay output binding mismatch')
    if sha(output/'comparison.json')!=completion['comparison_sha256'] or sha(inputs)!=completion['frozen_input_sha256']:
        raise ValueError('Comparison/input binding mismatch')
    archive=output.with_name(output.name+'_audit_bundle.tar.gz')
    with tarfile.open(archive,'x:gz') as tar:
        for name in ('baseline.jsonl','candidate.jsonl','comparison.json','completion.json','preparation.json'):
            tar.add(output/name,arcname=name,recursive=False)
        tar.add(inputs,arcname='evaluation_inputs.json',recursive=False)
        for path in sorted(Path(__file__).parent.glob('*')):
            if path.is_file() and path.suffix in ('.py','.sh','.md'):
                tar.add(path,arcname='code/'+path.name,recursive=False)
    if archive.stat().st_size>30*1024*1024: raise ValueError(f'Audit bundle exceeds 30 MiB: {archive}; retained in full')
    return archive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('all',*MODES),default='all',nargs='?')
    p.add_argument('--pipeline-root',default=DEFAULT_PIPELINE)
    p.add_argument('--model-dir',default='checkpoints/Qwen3.5-4B')
    p.add_argument('--data-root',default='lead_data')
    p.add_argument('--output-root',default='checkpoints/phase1_visual_refine_remote')
    args=p.parse_args();modes=MODES if args.mode=='all' else (args.mode,)
    # Validate every selected mode before loading a model; do not start a costly partial run.
    jobs=[(mode,*discover(args.pipeline_root,mode)) for mode in modes]
    if not Path(args.model_dir,'model.safetensors.index.json').is_file():
        raise FileNotFoundError(f'Missing local sharded Qwen3.5 base: {args.model_dir}')
    if not Path(args.data_root).is_dir(): raise FileNotFoundError(args.data_root)
    gpu=choose_gpu();os.environ['GPU_IDS']=gpu;os.environ['CUDA_VISIBLE_DEVICES']=gpu
    print(f'[gpu] physical GPU {gpu}; one process, fresh KV per arm',flush=True)
    # Import the backend only after selecting the GPU; use the activated interpreter.
    from .experiment import replay
    root=Path(args.output_root)/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S')+f'_{os.getpid()}')
    root.mkdir(parents=True,exist_ok=False);review=False
    for mode,inputs,adapter in jobs:
        print(f'[START] {mode}: {inputs}; adapter={adapter}',flush=True)
        report=replay(inputs,args.data_root,args.model_dir,adapter,root/mode)
        archive=package(root/mode,inputs)
        print(f'[BUNDLE] {archive}',flush=True)
        if report['status']=='review_required':
            review=True
            print(f'[REVIEW_REQUIRED] {mode}: old-correct losses or metric decline; saved evidence, continuing other mode',file=sys.stderr,flush=True)
    status='review_required' if review else 'no_observed_regression'
    (root/'status.json').write_text(json.dumps({'status':status,'modes':list(modes),'training_started':False})+'\n')
    print(f'[{"REVIEW_REQUIRED" if review else "DONE"}] {root}',flush=True)
    if review:raise SystemExit(2)


if __name__=='__main__':main()
