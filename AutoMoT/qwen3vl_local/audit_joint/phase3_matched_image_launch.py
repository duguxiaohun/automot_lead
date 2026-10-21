"""Four independent GPU workers using a SHA-verified runtime from synced whitelist files."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from . import phase3_behavior_diagnostics as diag
from . import phase3_matched_image_eval as evaluation

APP = Path(__file__).resolve().parents[2]
ROOTS = ('qwen3vl_local', 'keyframe_filter', 'lead_video_tools')
OVERLAY = ('phase3_behavior_diagnostics.py', 'phase3_matched_image_eval.py')


def inspect_existing(output):
    """Read prior evidence only. A stored status is not a live-process check."""
    output=Path(output).absolute()
    report=dict(output=str(output),exists=output.exists(),inspection_only=True,
                live_process_status='not_checked',files_modified=False)
    if not output.is_dir():
        report['reason']='Output is absent or is not a directory'
        return report
    receipt=output/'receipt.json'
    try:
        report['receipt']=json.loads(receipt.read_text())
    except (OSError,ValueError) as exc:
        report['receipt_error']=str(exc)
    report['metrics_exists']=(output/'metrics.json').is_file()
    report['log_tails']={}
    for path in [output/'preflight.log',*sorted(output.glob('rank_*.log'))]:
        if path.is_file():
            with path.open('rb') as stream:
                stream.seek(0,2);stream.seek(max(0,stream.tell()-6000))
                report['log_tails'][path.name]=stream.read().decode('utf-8',errors='replace')
    report['next_step']='Inspect the prior receipt/logs before retrying; preserve this directory. No automatic rerun or overwrite.'
    return report


class ExistingOutputError(ValueError):
    pass


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def gpu_ids(value):
    if value:
        if not re.fullmatch(r'[0-9]+(,[0-9]+)*', value):raise ValueError('GPU_IDS must contain numeric GPU IDs')
        ids = value.split(',')
        if len({int(x) for x in ids}) != len(ids):raise ValueError('Duplicate GPU IDs')
        return ids
    rows = list(csv.reader(subprocess.check_output([
        'nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()))
    parsed=[tuple(int(v.strip()) for v in r) for r in rows]
    if len(parsed)<4:raise ValueError('Default requires four GPUs; explicitly set GPU_IDS for fewer cards')
    return [str(r[0]) for r in sorted(parsed,key=lambda r:(r[1],r[2],r[0]))[:4]]


def snapshot(destination, job, *, app=APP):
    """Build from synchronized whitelist files only; Git is never consulted."""
    app=Path(app).resolve()
    package=app/'qwen3vl_local/audit_joint/phase3_behavior_diagnostic_v1'
    manifest_path=package/'runtime_sources.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('schema')!='phase3_synced_runtime_v1':raise ValueError('Unsupported synced runtime manifest')
    files=manifest['files'];bound=[]
    for name,entry in files.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or path.parts[0] not in ROOTS:
            raise ValueError('Unsafe runtime source path: '+name)
        if entry['source']=='project':source=app/path
        elif entry['source']=='compat_sources':source=package/'compat_sources'/path
        else:raise ValueError('Unknown runtime source kind')
        if not source.is_file() or diag.file_sha(source)!=entry['sha256']:
            raise ValueError('Synced runtime file missing or changed: '+str(source)+
                '; synchronize the published whitelist with bash ./sync_automot.sh')
        bound.append((path,source,entry['sha256']))
    release_name='qwen3vl_local/audit_joint/phase3_base_trial_sources.json'
    if files[release_name]['sha256']!=job['release_sha256']:
        raise ValueError('Job does not match the supported v2 runtime release; no hash bypass')
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    for path,source,expected in bound:
        target=destination/path;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        if diag.file_sha(target)!=expected:raise ValueError('Runtime source changed while copying: '+str(source))
    runtime=destination
    release=runtime/release_name
    spec=json.loads(release.read_text())
    for name,sha in spec['files'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or diag.file_sha(runtime/path)!=sha:
            raise ValueError('Published source mismatch: '+name)
    copied={}
    for name in OVERLAY:
        source=app/'qwen3vl_local/audit_joint'/name
        target=runtime/'qwen3vl_local/audit_joint'/name
        shutil.copyfile(source,target);copied[name]=diag.file_sha(target)
    return runtime,dict(source='synchronized_whitelist',remote_git_required=False,
        reference_commit=manifest['reference_commit'],runtime_manifest_sha256=diag.file_sha(manifest_path),
        release_sha256=diag.file_sha(release),verified_runtime_sources=len(files),
        verified_native_sources=len(spec['files']),diagnostic_sources=copied,
        current_checkout_modified=False,source_hash_bypass=False)


def processes(commands, cwd, envs, logs):
    """Stop siblings on first failure; keep independent per-rank logs."""
    children=[];streams=[];last=time.monotonic()
    try:
        for command,env,log in zip(commands,envs,logs):
            stream=log.open('x');streams.append(stream)
            children.append(subprocess.Popen(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT))
        while True:
            codes=[p.poll() for p in children]
            if any(c is not None and c!=0 for c in codes):
                failed=next(i for i,c in enumerate(codes) if c is not None and c!=0)
                raise RuntimeError(f'Worker {failed} exited {codes[failed]}; see {logs[failed]}')
            if all(c==0 for c in codes):break
            if time.monotonic()-last>=30:
                print(f'[diagnostic] completed={sum(c==0 for c in codes)}/{len(codes)}; logs: {logs[0].parent}',flush=True)
                last=time.monotonic()
            time.sleep(.2)
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for stream in streams:stream.close()


def launch(job_path, output, ids='', preflight_only=False):
    if Path(output).exists() or Path(output).is_symlink():
        raise ExistingOutputError(json.dumps(inspect_existing(output),ensure_ascii=False,indent=2))
    job_path=Path(job_path).resolve(strict=True);output=Path(output).resolve()
    job=json.loads(job_path.read_text())
    for key in ('output','dataset','data_root','model_dir','index'):
        if not Path(job[key]).is_absolute():raise ValueError('Original job must retain absolute asset paths: '+key)
    protected=[Path(job[k]).resolve() for k in ('output','data_root','model_dir')]
    protected.extend((Path(job['dataset']).resolve().parent,Path(job['index']).resolve().parent))
    protected.extend(APP/name for name in ROOTS)
    # Application root can contain the new run, but the run cannot contain source.
    if output==APP or APP.is_relative_to(output):raise ValueError('Output overlaps application sources')
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Output overlaps original experiment or assets')
    if output.exists() or output.is_symlink():raise ValueError('Use a new diagnostic output directory')
    selected=[] if preflight_only else gpu_ids(ids)
    plans_dir=Path(__file__).with_name('phase3_behavior_diagnostic_v1')
    plans=[json.loads((plans_dir/f'matched_images_seed_{seed}.json').read_text()) for seed in (11,29,47)]
    for plan in plans:diag.validate_plan(plan)
    if len(selected)>len({p['receiver'] for plan in plans for p in plan['pairs']}):raise ValueError('Too many inference workers')
    output.mkdir(parents=True,exist_ok=False)
    note=dict(status='preparing_runtime',gpu_ids=selected,training_performed=False,inference_performed=False)
    write(output/'receipt.json',note)
    try:
        runtime,provenance=snapshot(output/'runtime',job)
        note['runtime']=provenance;write(output/'receipt.json',note)
        # Copy the admitted plans as well, so another sync cannot change a running experiment.
        local_plans=output/'plans';local_plans.mkdir()
        paths=[]
        for plan in plans:
            p=local_plans/f'seed_{plan["seed"]}.json';write(p,plan);paths.append(str(p))
        env=dict(os.environ,PYTHONPATH=str(runtime),PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='')
        for key in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):env.pop(key,None)
        command=[sys.executable,'-m','qwen3vl_local.audit_joint.phase3_matched_image_eval',
            '--job',str(job_path),'--plan',*paths]
        print('[diagnostic] CPU preflight using synchronized original v2 sources (no Git)',flush=True)
        processes([command+['--preflight-only','--output',str(output/'preflight')]],runtime,[env],[output/'preflight.log'])
        if preflight_only:
            note['status']='preflight_completed';write(output/'receipt.json',note);return note
        print(f'[diagnostic] independent inference workers on GPUs {",".join(selected)}',flush=True)
        commands=[];envs=[];logs=[]
        for rank,gpu in enumerate(selected):
            commands.append(command+['--rank',str(rank),'--world',str(len(selected)),
                '--output',str(output/f'rank_{rank}'),'--device','cuda:0'])
            envs.append(dict(env,CUDA_VISIBLE_DEVICES=gpu,GPU_IDS=gpu));logs.append(output/f'rank_{rank}.log')
        processes(commands,runtime,envs,logs)
        shards=[json.loads((output/f'rank_{r}/shard.json').read_text()) for r in range(len(selected))]
        receipts=[json.loads((output/f'rank_{r}/receipt.json').read_text()) for r in range(len(selected))]
        preflight=json.loads((output/'preflight/receipt.json').read_text())
        for receipt in receipts:
            if not receipt['inference_performed']:raise ValueError('Incomplete rank')
            for key in ('job_sha256','checkpoint_receipt_sha256','plans','source_sha256','primary_common_receiver_ids'):
                if receipt[key]!=preflight[key]:raise ValueError('Worker provenance changed: '+key)
        full,wrong=evaluation.merge_shards(plans,shards)
        evaluation.save_metrics(job,plans,output,full,wrong)
        note.update(status='completed',inference_performed=True,unique_actual_forwards=len(full),
            unique_swapped_forwards=len(wrong),primary_common_anchors=len(preflight['primary_common_receiver_ids']))
        write(output/'receipt.json',note)
        return note
    except BaseException as exc:
        note.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write(output/'receipt.json',note)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--job',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-ids',default=os.environ.get('GPU_IDS',''))
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--inspect-existing',action='store_true',help='Read existing receipt and log tails only; no GPU or writes')
    a=p.parse_args()
    if a.inspect_existing:
        print(json.dumps(inspect_existing(a.output),ensure_ascii=False,indent=2));return
    def interrupted(signum, frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        print(json.dumps(launch(a.job,a.output,a.gpu_ids,a.preflight_only)))
    except ExistingOutputError as exc:
        print('Existing diagnostic output; prior evidence follows:\n'+str(exc),file=sys.stderr)
        raise SystemExit(2) from None


if __name__=='__main__':main()
