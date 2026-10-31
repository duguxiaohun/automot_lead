"""在当前服务器的本地模型上验证有序图片流的文本前缀缓存；不训练。

requests JSONL 每行包含 system、text、images（按时间排序的本地图像路径）。
必须使用调用方原生 prompt；本工具不生成/读取标签，也不重新排列请求。
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    """流式绑定本地来源，避免一次读入模型权重。"""
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def worker(args):
    """CPU 读取 adapter 后放到已选卡，分别报告输出等价和计时。"""
    import time
    from types import SimpleNamespace
    import torch
    from PIL import Image
    from . import text_prefix
    from .backend import LocalModel,LocalProcessor
    from .adapters import LocalPeftModel
    from qwen3vl_local.sft_new_loop_phase1.run_integrity import write,environment
    if not text_prefix.enabled():raise ValueError('Set AUTOMOT_TEXT_PREFIX=1 for the new input layout')
    rows=[json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
    if not 2<=len(rows)<=32:raise ValueError('Use 2..32 ordered requests including different images')
    report=dict(status='running',optimizer_updates=0,quality_certified=False,
        requests_sha256=sha(args.requests),source_sha256={p.name:sha(p) for p in
            (Path(__file__),Path(text_prefix.__file__))},input_layout=text_prefix.identity(),
        model_dir=str(args.model_dir),adapter_dir=str(args.adapter_dir) if args.adapter_dir else None)
    write(args.output/'report.json',report)
    try:
        if args.adapter_dir:
            configs=[p for name in ('fusion_config.json',*[f'sft_new_loop_phase{p}_adapter_config.json' for p in (1,2,3)])
                     if (p:=args.adapter_dir/name).is_file()]
            if len(configs)!=1:raise ValueError('Expected exactly one native phase/fusion adapter contract')
            config=json.loads(configs[0].read_text())
            layout=config.get('input_layout',config.get('prompt',{}).get('input_layout'))
            if layout!=text_prefix.CONTRACT:raise ValueError('Adapter was not trained with text-before-images layout')
            report['adapter_files']={str(p.relative_to(args.adapter_dir)):sha(p)
                for p in args.adapter_dir.rglob('*') if p.is_file()}
        device=torch.device('cuda:0');torch.cuda.set_device(device)
        model=LocalModel.from_pretrained(str(args.model_dir),dtype=torch.bfloat16)
        if args.adapter_dir:
            model=LocalPeftModel.from_pretrained(model,str(args.adapter_dir),is_trainable=False,torch_device='cpu')
        model=model.to(device).eval();processor=LocalProcessor.from_pretrained(str(args.model_dir))
        bundle=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device=device)
        encoded=[];images_sha={};start=time.perf_counter()
        for row in rows:
            if set(row)!={'system','text','images'} or not isinstance(row['system'],str) or not isinstance(row['text'],str):
                raise ValueError('Each request must contain only system/text/images')
            if not isinstance(row['images'],list) or len(row['images']) not in (1,2,4):
                raise ValueError('Expected 1, 2 or 4 ordered images')
            paths=[(args.requests.parent/Path(p)).resolve() for p in row['images']]
            images=[]
            try:
                for path in paths:
                    images_sha[str(path)]=sha(path)
                    with Image.open(path) as im:images.append(im.convert('RGB'))
                messages=[dict(role='system',content=row['system']),
                          dict(role='user',content=text_prefix.content(images,row['text']))]
                chat=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                item=processor(text=[chat],images=images,return_tensors='pt',padding=False)
                if item['input_ids'].shape[1]>8192:raise ValueError('Prompt exceeds context budget')
                encoded.append({k:v.to(device) for k,v in item.items()})
            finally:
                for im in images:im.close()
        torch.cuda.synchronize(device)
        report.update(encoding_seconds=time.perf_counter()-start,images_sha256=images_sha,environment=environment(),
            input_tokens=[x['input_ids'].shape[1] for x in encoded],
            base_asset_sha256=getattr(model.config,'local_asset_sha256',None))
        torch.cuda.reset_peak_memory_stats(device)
        report['benchmark']=text_prefix.benchmark(bundle,encoded,max_new_tokens=args.max_new_tokens)
        report.update(status='text_reuse_checked_full_control_recorded_quality_pending',
                      peak_memory_bytes=torch.cuda.max_memory_allocated(device),
                      memory_scope='encoded stream and both model paths; not cache-only peak')
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:write(args.output/'report.json',report)


def main():
    """只选一张空闲卡，在独立目录与超时进程组中执行。"""
    import resource
    resource.setrlimit(resource.RLIMIT_CORE,(0,resource.getrlimit(resource.RLIMIT_CORE)[1]))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,default=Path('checkpoints/Qwen3.5-4B'))
    p.add_argument('--adapter-dir',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--max-new-tokens',type=int,default=160)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    for name in ('requests','model_dir','adapter_dir','output'):
        if getattr(args,name) is not None:setattr(args,name,getattr(args,name).resolve())
    if args.worker:return worker(args)
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    idle=[r.split(',')[0].strip() for r in raw.splitlines() if float(r.split(',')[1])<256 and float(r.split(',')[2])<=5]
    selected=os.environ.get('GPU_IDS') or (idle[0] if idle else '')
    if selected not in idle:raise ValueError('One idle physical GPU required; GPU_IDS may pin one idle ID')
    args.output=args.output or Path('checkpoints/text_prefix_benchmark')/f'run_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}'
    args.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-B','-m',__spec__.name,'--worker','--requests',str(args.requests),
             '--model-dir',str(args.model_dir),'--output',str(args.output),'--max-new-tokens',str(args.max_new_tokens)]
    if args.adapter_dir:command+=['--adapter-dir',str(args.adapter_dir)]
    env={**os.environ,'GPU_IDS':selected,'CUDA_VISIBLE_DEVICES':selected,'PYTHONUNBUFFERED':'1'}
    for name in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE'):env.pop(name,None)
    from qwen3vl_local.sft_new_loop_phase1 import data_stage
    data_stage.run(command,args.output/'worker.json',timeout=1800,env=env)
    print('[TEXT_PREFIX_REPORT] '+str(args.output/'report.json'),flush=True)


if __name__=='__main__':main()
