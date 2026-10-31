"""Single-GPU native stream smoke test using frozen CPU-ready validation RGB.

No trained adapter is assumed. A newly initialized LoRA exercises the runtime;
this diagnostic does not score labels, update weights, or emulate CARLA traffic.
"""
import argparse
import hashlib
from pathlib import Path
import torch
from PIL import Image
from qwen3vl_local import runtime_version
from qwen3vl_local.qwen35 import online_benchmark, cache_policy
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices
from . import CONTRACT, plan, prompts, infer, integrity


def requests(items, mode, domain):
    """Keep native context fixed; replay two existing groups over two windows."""
    selected=[];seen=set()
    for item in items:
        if item.row.question_domain!=domain:continue
        paths=[str(item.row.history_rgb_paths[i]) for i in history_rgb_indices(mode)]
        hashes=[]
        for path in paths:
            with Image.open(path) as image:
                rgb=image.convert('RGB')
                try:hashes.append((rgb.size,hashlib.sha256(rgb.tobytes()).hexdigest()))
                finally:rgb.close()
        key=tuple(hashes)
        if key not in seen:selected.append(paths);seen.add(key)
        if len(selected)==2:break
    if len(selected)!=2:raise ValueError(f'Two distinct validation RGB groups required: {domain}')
    return [dict(images=list(selected[i]),context=dict(domain=domain)) for i in (0,1,0,1)]


def run(bundle, root, output, *, max_new_tokens=32):
    if (output/'result.json').exists():raise FileExistsError(output/'result.json')
    ready=proof.read(root/'ready.json')
    receipt=dict(status='running',optimizer_updates=0,trained_adapter=False,
        quality_certified=False,carla_certified=False,
        source_verified_after_measurement=False,
        device=str(bundle.device),dtype=str(next(bundle.model.parameters()).dtype),
        scope='fixed-context validation RGB replay; two groups repeated over two question windows; not a natural driving stream',
        max_new_tokens=max_new_tokens,cases=[])
    def save():proof.write(output/'result.json',receipt)
    save()
    try:
        items=list(plan.evaluation(root/'data/frame_index.jsonl',ready['data_root'],'val'))
        domains=sorted({item.row.question_domain for item in items})
        if set(domains)!=set(prompts.P2.QUESTION_DOMAINS):raise ValueError('Both question domains required')
        bundle.model.eval()
        for mode in ('2rgb_endpoints','4rgb'):
            infer.bind_online_contract(bundle,dict(contract=CONTRACT,mode=mode,
                prompt=prompts.identity(mode),text_cache_policy=ready['text_cache_policy']))
            for domain in domains:
                case=dict(mode=mode,domain=domain,status='running')
                receipt['cases'].append(case);save()
                rows=requests(items,mode,domain)
                target=output/mode/domain;target.mkdir(parents=True,exist_ok=False)
                proof.write(target/'requests.json',rows)
                print(f'[stream-probe] {mode}/{domain}: 4 groups, 2 question windows, 3 timing arms',flush=True)
                report=online_benchmark.benchmark(bundle,12,rows,window_frames=2,slots=1,
                    seed='stream-probe',max_new_tokens=max_new_tokens,repeats=2,
                    on_run=lambda name,value:proof.write(target/(name+'.json'),value),
                    on_trial=lambda number,value:print(f'[stream-probe] replay {number+1}/2 passed',flush=True))
                proof.write(target/'benchmark.json',report)
                cached=report['cold_cached']
                # The first window must actually reuse both questions after changing RGB.
                if not all(q['cache_class']=='full_hit' for q in cached['frames'][1]['queries']):
                    raise ValueError('Fixed-question cross-image window did not fully hit')
                if not cached['frames'][2]['window_changed']:
                    raise ValueError('Question window switch was not exercised')
                # At least one native permutation must change at the switch; no fake switch claim.
                if not any(q['cache_class']=='miss' for q in cached['frames'][2]['queries']):
                    raise ValueError('Question window switch rendered no new prefix')
                if not all(q['cache_class']=='full_hit' for q in cached['frames'][3]['queries']):
                    raise ValueError('New question window did not fully hit')
                case.update(status='passed',cache=cached['cache'],summary=report['summary'],
                    same_partition_saving_fraction=report['same_partition_saving_fraction'],
                    same_partition_model_saving_fraction=report['same_partition_model_saving_fraction'])
                save()
                summary=report['summary']['cached']
                print(f"[stream-probe] passed: full_hits={cached['cache']['full_hits']} "
                      f"same_split_saving={report['same_partition_saving_fraction']:.2%} "
                      f"token_limit={summary['token_limit_calls']}/{summary['generation_calls']}; quality pending",flush=True)
        receipt['status']='stream_reuse_and_timing_checked_quality_pending'
    except BaseException as exc:
        receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        if receipt['cases'] and receipt['cases'][-1]['status']=='running':receipt['cases'][-1]['status']='failed'
        raise
    finally:save()
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-new-tokens',type=int,default=32)
    args=parser.parse_args()
    if not 1<=args.max_new_tokens<=512:parser.error('Use 1..512 generated tokens')
    root=args.run_dir.resolve();version=Path(proof.read(root/'execution.json')['version'])
    if Path(__file__).resolve().parents[2]!=(version/'app').resolve():
        raise ValueError('Stream probe must execute CPU-ready frozen source; create a new CPU-ready run')
    runtime_version.verify(version);ready=integrity.verify_ready(root,execution=True)
    if (args.output/'result.json').exists():raise FileExistsError(args.output/'result.json')
    cache_policy.validate(ready['text_cache_policy'])
    if not ready['text_cache_policy'] or not ready['text_cache_policy']['enabled']:
        raise ValueError('Stream probe requires enabled text-prefix cache')
    from qwen3vl_local.sft_v2.train import load_model_with_lora
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.manual_seed(plan.SEED);torch.cuda.manual_seed_all(plan.SEED)
    bundle=load_model_with_lora(args.model_dir,device=device,lora_rank=16,lora_alpha=32,
        lora_dropout=.05,lora_vision_scope='off',strict_vision_scope=True,gradient_checkpointing=True)
    run(bundle,root,args.output,max_new_tokens=args.max_new_tokens)
    try:
        runtime_version.verify(version);integrity.verify_ready(root)
    except BaseException as exc:
        receipt=proof.read(args.output/'result.json')
        receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        proof.write(args.output/'result.json',receipt)
        raise
    receipt=proof.read(args.output/'result.json')
    receipt.update(ready_sha256=proof.sha(root/'ready.json'),execution_sha256=proof.sha(root/'execution.json'),
        source_verified_after_measurement=True,environment=proof.environment(),
        base_asset_sha256=getattr(bundle.model.config,'local_asset_sha256',None))
    proof.write(args.output/'result.json',receipt)


if __name__=='__main__':main()
