"""Bounded correctness check; no full-pool load, benchmark or saved weights."""
import argparse
import json
import os
from pathlib import Path
import sys

from .benchmark_runtime import progress, run_owned, trace_waits
from .phase4_first_epoch import APP, RELEASE
from .server_readiness import source_check
from qwen3vl_local.sft_new_loop_phase4.identity import write_json

CPU_TESTS = (
    'qwen3vl_local/tests/test_batch_runtime.py',
    'qwen3vl_local/audit_joint/tests/test_batch_runtime_check.py',
    'qwen3vl_local/sft_new_loop_phase3/test_training_state.py',
    'qwen3vl_local/audit_joint/tests/test_phase4_training.py',
)


def gpu_worker(a):
    trace_waits()
    import torch
    from .phase4_gpu_smoke import example
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.model import load_bundle, encode_batch, generate, generate_batch
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import observation_contract
    from qwen3vl_local.batch_runtime import frozen_rgb_reuse
    torch.manual_seed(20261017)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    progress('4B load START; one GPU; no full-pool read')
    bundle=load_bundle(a.model_dir,device,dropout=0.,vision_scope='off',checkpointing=True)
    reports=[]
    for mode in (2,4):
        progress(f'RGB={mode} input/generation check START')
        row=example(a.native,a.data_root,mode)
        bundle.observation_contract=observation_contract(mode)
        bundle.model.eval()
        expected=generate(bundle,row,a.data_root)
        plain=generate_batch(bundle,[row,row],a.data_root)
        reused=generate_batch(bundle,[row,row],a.data_root,reuse_rgb=True)
        if plain != [expected,expected] or reused != plain:
            raise ValueError(f'RGB{mode} generation mismatch: single={expected!r}, batch={plain!r}, reuse={reused!r}')
        progress(f'RGB={mode} B2 baseline/reuse gradient check START')
        inputs=encode_batch(bundle,[row,row],a.data_root)
        bundle.model.train();bundle.model.zero_grad(set_to_none=True)
        baseline=bundle.model(**inputs,use_cache=False).loss
        if not torch.isfinite(baseline):raise ValueError('nonfinite baseline loss')
        baseline.backward()
        expected_grads={n:p.grad.detach().float().cpu().clone() for n,p in bundle.model.named_parameters() if p.grad is not None}
        bundle.model.zero_grad(set_to_none=True)
        with frozen_rgb_reuse(bundle.model) as stats:
            actual=bundle.model(**inputs,use_cache=False).loss
        actual.backward()
        torch.testing.assert_close(actual,baseline,atol=.005,rtol=.005)
        max_relative=0.
        actual_names={n for n,p in bundle.model.named_parameters() if p.grad is not None}
        if actual_names!=set(expected_grads):raise ValueError('gradient parameter membership changed')
        for n,p in bundle.model.named_parameters():
            if n not in expected_grads:continue
            reference=expected_grads[n];measured=p.grad.detach().float().cpu()
            error=float(torch.linalg.vector_norm(measured-reference))
            norm=float(torch.linalg.vector_norm(reference))
            if not torch.isfinite(measured).all() or error>1e-5+.02*norm:
                raise ValueError(f'RGB{mode} gradient mismatch: {n}, error={error}, norm={norm}')
            max_relative=max(max_relative,error/max(norm,1e-5))
        if stats['reused_images']<mode:raise ValueError(f'expected duplicate images were not reused: {stats}')
        bundle.model.zero_grad(set_to_none=True)
        reports.append(dict(rgb_mode=mode,example_id=row['id'],prompt_sha256=row['prompt_sha256'],
            generation=plain,loss=float(actual.detach()),
            baseline_loss=float(baseline.detach()),max_gradient_relative_error=max_relative,visual_reuse=stats))
        progress(f'RGB={mode} correctness PASSED')
    write_json(a.output/'result.json',dict(status='correctness_passed',modes=reports,
        gpu=torch.cuda.get_device_name(0),scope='one GPU, repeated single real example, dropout zero; not quality, DDP, full-pool or speed acceptance',
        full_pool_admitted=False))


def main():
    from .guard import no_core
    no_core()
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--check',choices=('cpu','gpu'),default='cpu')
    p.add_argument('--native',type=Path,default=Path('checkpoints/phase4_v57_20261014_232157_762998141/data4'))
    p.add_argument('--model-dir',type=Path,default=Path('checkpoints/Qwen3.5-4B'))
    p.add_argument('--data-root',type=Path,default=Path('lead_data'))
    p.add_argument('--timeout',type=int,default=180)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.timeout<1:p.error('timeout must be positive')
    if a.worker:
        if a.check!='gpu':p.error('worker is GPU-only')
        gpu_worker(a);return
    a.output=a.output.absolute()
    if any(a.output.resolve().is_relative_to(v.resolve()) for v in (a.native,a.model_dir,a.data_root)):
        p.error('output must be outside inputs')
    report=source_check(APP,RELEASE)
    if report['status']!='matched':p.error('source mismatch; synchronize the published release first')
    env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    if a.check=='gpu':
        for path in (a.native/'train.jsonl',a.model_dir/'config.json'):
            if not path.is_file():p.error(f'missing input: {path}')
        from qwen3vl_local.sft_new_loop_phase4.devices import select_gpus
        env['CUDA_VISIBLE_DEVICES']=select_gpus(1)
        cmd=[sys.executable,'-u','-m','qwen3vl_local.audit_joint.batch_runtime_check',
             '--worker','--check','gpu','--native',str(a.native),'--model-dir',str(a.model_dir),
             '--data-root',str(a.data_root),'--output',str(a.output)]
    else:
        env['CUDA_VISIBLE_DEVICES']=''
        cmd=[sys.executable,'-m','pytest','-q',*[str(APP/t) for t in CPU_TESTS]]
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'source_check.json',report)
    code=run_owned(cmd,a.output/'check.log',env,a.timeout)
    result=a.output/'result.json'
    completed=code==0
    if completed and a.check=='gpu':
        try:
            measured=json.loads(result.read_text())
            completed=measured['status']=='correctness_passed' and [r['rgb_mode'] for r in measured['modes']]==[2,4]
        except (OSError,ValueError,KeyError,TypeError):
            completed=False
    exit_code=0 if completed else code or 2
    status=dict(check=a.check,returncode=exit_code,worker_returncode=code,completed=completed,timeout_seconds=a.timeout,
                log=str(a.output/'check.log'),full_pool_admitted=False)
    write_json(a.output/'status.json',status);print(json.dumps(status),flush=True)
    raise SystemExit(exit_code)


if __name__=='__main__':main()
