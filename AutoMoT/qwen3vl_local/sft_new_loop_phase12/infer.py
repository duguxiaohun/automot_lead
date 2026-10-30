"""Ask either task alone or both via one shared image prefix; no GT input needed."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
import torch
from qwen3vl_local.qwen35.backend import LocalModel,LocalProcessor
from qwen3vl_local.qwen35.adapters import LocalPeftModel
from qwen3vl_local.qwen35 import text_prefix
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices
from qwen3vl_local.sft_v3.train import _kv_start_state,_student_generate_kv
from . import prompts,cache,evaluate,CONTRACT


def prepare_request(bundle,images,mode,domain,task='both',seed='inference',*,schedule=None):
    if domain not in prompts.P2.QUESTION_DOMAINS or task not in ('both','phase1','phase2'):
        raise ValueError('Unknown task/question domain')
    if len(images)!=len(history_rgb_indices(mode)):raise ValueError('Wrong RGB count')
    from qwen3vl_local.qwen35 import text_prefix, question_schedule
    ticket=None
    if schedule is not False and text_prefix.enabled():
        scheduler=schedule or question_schedule.for_bundle(bundle,'phase12:'+task,seed)
        if task=='both' and scheduler.slots!=1:
            raise ValueError('Two-task inference already uses two cache entries; set question slots=1')
        ticket=scheduler.next();seed=ticket['seed']
    # Dummy answers only construct the question order; they are never rendered as an answer/input.
    facts={k:False for k in prompts.P1.ANSWER_KEYS};event={k:False for k in prompts.P2.ANSWER_KEYS}
    event[prompts.P2.DOMAIN_ANSWER_KEYS[domain]]=True
    second=prompts.P2.make_prompt_spec(variant='all_random_order',answers=event,seed_key=seed)
    first,_=prompts.specs(SimpleNamespace(spec=second),facts);texts=prompts.texts(first,second,mode)
    return dict(specs=(first,second),texts=texts,schedule=ticket,
                messages=[prompts.messages(images,text) for text in texts])


@torch.no_grad()
def predict(bundle,images,mode,domain,task='both',seed='inference',*,schedule=None):
    prepared=prepare_request(bundle,images,mode,domain,task,seed,schedule=schedule)
    first,second=prepared['specs'];texts=prepared['texts'];ticket=prepared['schedule']
    bundle.model.eval()
    if task=='both':
        report=cache.generate(bundle,images,texts)
        return dict(phase1=dict(text=report['branches'][0]['text'],answers=evaluate._parse(report['branches'][0]['text'],first,1)),
            phase2=dict(text=report['branches'][1]['text'],answers=evaluate._parse(report['branches'][1]['text'],second,2)),performance=report,schedule=ticket)
    phase=1 if task=='phase1' else 2;spec=(first,second)[phase-1]
    state=_kv_start_state(bundle,prompts.messages(images,texts[phase-1]));text,_,tokens=_student_generate_kv(bundle,state,160)
    return dict(task=task,text=text,answers=evaluate._parse(text,spec,phase),tokens=int(tokens.numel()),shared_prefix=False,schedule=ticket,text_prefix=text_prefix.report(bundle))


def load_online_bundle(model_dir,adapter_dir,*,device,mode=None,dtype=torch.bfloat16):
    """Load one verified shared adapter; preserve its native prompt/history contract."""
    model_dir,adapter_dir=Path(model_dir),Path(adapter_dir)
    config=proof.read(adapter_dir/'fusion_config.json');proof.verify_adapter(adapter_dir)
    if (config['contract']!=CONTRACT or config['prompt']!=prompts.identity(config['mode'])
            or (mode is not None and mode!=config['mode'])):
        raise ValueError('Shared adapter/prompt/history contract mismatch')
    model=LocalModel.from_pretrained(str(model_dir),dtype=dtype)
    model=LocalPeftModel.from_pretrained(model,str(adapter_dir),is_trainable=False,torch_device='cpu').to(device).eval()
    processor=LocalProcessor.from_pretrained(str(model_dir))
    return SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,
                           device=torch.device(device),fusion_config=config)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True);p.add_argument('--rgb',type=Path,nargs=4,required=True)
    p.add_argument('--mode',choices=('2rgb_endpoints','4rgb'),required=True)
    p.add_argument('--domain',choices=prompts.P2.QUESTION_DOMAINS,default=prompts.P2.ROAD_DOMAIN)
    p.add_argument('--task',choices=('both','phase1','phase2'),default='both');a=p.parse_args()
    bundle=load_online_bundle(a.model_dir,a.adapter,device='cuda:0',mode=a.mode)
    images=[Image.open(a.rgb[i]).convert('RGB') for i in history_rgb_indices(a.mode)]
    try:print(json.dumps(predict(bundle,images,a.mode,a.domain,a.task),ensure_ascii=False,indent=2))
    finally:
        for im in images:im.close()

if __name__=='__main__':main()
