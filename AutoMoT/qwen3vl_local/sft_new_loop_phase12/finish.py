"""Reload the selected shared adapter on one GPU; evaluate and package full evidence."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
from types import SimpleNamespace
import torch
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from qwen3vl_local.qwen35.backend import LocalModel,LocalProcessor
from qwen3vl_local.qwen35.adapters import LocalPeftModel
from . import integrity,prompts,evaluate,reference,CONTRACT


def package(root,mode):
    root=Path(root);ready=integrity.verify_ready(root);training=root/'runs'/mode
    complete=integrity.verify_training(training,root,mode)
    # Recompute strict parser results and metrics from every original output before copying.
    from .audit import verify_evaluations
    verify_evaluations(root,mode)
    dest=root/(mode+'_audit_bundle');dest.mkdir(exist_ok=False)
    paths=[root/'ready.json',root/'execution.json']
    for folder in (root/'cpu',root/'evaluation'/mode,training):
        paths.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl'))
    for folder in (root/'data',root/'phase1/data',root/'phase1_previous/data',root/'phase2/data'):
        paths.extend(p for p in folder.glob('*.json') if p.is_file())
    for folder in (root/'phase1',root/'phase1_previous',root/'phase2'):
        p=folder/'selected_train_reviews.json'
        if p.exists():paths.append(p)
    for p in sorted(set(paths)):
        target=dest/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    # Source closure includes both native samplers/parsers, fusion modules, and local Qwen backend.
    sources=proof.sources(1)|proof.sources(2)
    package_dir=Path(__file__).parent
    for p in package_dir.iterdir():
        if p.is_file() and p.suffix in ('.py','.json'):sources[str(p.relative_to(proof.ROOT))]=proof.sha(p)
    for p in (proof.ROOT/'qwen3vl_local/audit_joint/phase12_retrain').iterdir():
        if p.is_file() and p.suffix in ('.py','.json'):
            sources[str(p.relative_to(proof.ROOT))]=proof.sha(p)
    for name,expected in sources.items():
        p=proof.ROOT/name
        if proof.sha(p)!=expected:raise ValueError('Source changed while packaging')
        target=dest/'code'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    proof.write(dest/'source_sha256.json',sources)
    evidence={str(p.relative_to(dest)):proof.sha(p) for p in dest.rglob('*') if p.is_file()}
    proof.write(dest/'bundle_manifest.json',dict(contract=CONTRACT,mode=mode,status='quality_pending_audit',files=evidence,
        selected_adapter=complete['selected'],weights_included=False,source_version=ready['version']))
    archive=root/(mode+'_audit_bundle.tar.gz')
    with tarfile.open(archive,'x:gz') as tar:tar.add(dest,arcname=dest.name)
    if archive.stat().st_size>30*1024*1024:raise ValueError('Audit bundle exceeds 30 MiB; evidence retained, no silent truncation')
    print('[BUNDLE] '+str(archive),flush=True)
    return archive


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--mode',choices=('2rgb_endpoints','4rgb'),required=True)
    a=p.parse_args();ready=integrity.verify_ready(a.run_dir,execution=True);training=a.run_dir/'runs'/a.mode
    done=integrity.verify_training(training,a.run_dir,a.mode);adapter=training/done['selected']
    config=proof.read(adapter/'fusion_config.json')
    if config['prompt']!=prompts.identity(a.mode) or config['ready_sha256']!=proof.sha(a.run_dir/'ready.json'):
        raise ValueError('Selected adapter prompt/data contract mismatch')
    model=LocalModel.from_pretrained(str(a.model_dir),dtype=torch.bfloat16)
    model=LocalPeftModel.from_pretrained(model,str(adapter),is_trainable=False,torch_device='cpu')
    device=torch.device('cuda:0');model=model.to(device).eval();processor=LocalProcessor.from_pretrained(str(a.model_dir))
    bundle=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device=device)
    integrity.bind_evaluation(a.run_dir,a.mode,device,next(model.parameters()).dtype)
    evaluate.run(bundle,a.run_dir/'data/frame_index.jsonl',ready['data_root'],a.mode,'test',a.run_dir/'evaluation'/a.mode/'fusion_test')
    for phase in (1,2):reference.run(bundle,a.run_dir,a.mode,phase)
    integrity.verify_ready(a.run_dir);package(a.run_dir,a.mode)

if __name__=='__main__':main()
