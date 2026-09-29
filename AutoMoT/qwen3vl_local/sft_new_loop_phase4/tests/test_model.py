"""真实本地Qwen3.5小网络/官方模板；不是4B或GPU效果验收。"""
from types import SimpleNamespace
from pathlib import Path
import pytest
pytest.importorskip('transformers')
import torch
from PIL import Image
from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.sft_new_loop_phase4.model import encode,load_images
from qwen3vl_local.sft_new_loop_phase4.identity import digest,file_sha
from qwen3vl_local.sft_new_loop_phase4.prompts import prompt
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.observation import observation_contract


def example(tmp_path,frames=2,target='YES'):
    ids=[6,10] if frames==2 else [4,6,8,10]
    paths=[]
    for fid in ids:
        path=tmp_path/f'{fid:04d}.jpg';Image.new('RGB',(32,32),(fid,30,70)).save(path);paths.append(path)
    ep=dict(event='U-E4',instance_id='test',state='YIELD',longitudinal='HOLD')
    obs=dict(frame_id=10,history_frames=ids,speed_mps=0.)
    return dict(episode=ep,edge='proceed',observation=obs,images=[p.name for p in paths],
                image_sha256=[file_sha(p) for p in paths],target=target,
                prompt_sha256=digest(prompt(Episode(**ep),'proceed',obs)))


@pytest.mark.parametrize('frames',[2,4])
@pytest.mark.parametrize('target',['YES','NO'])
def test_real_processor_supervises_only_answer_and_eos(tmp_path,processor,frames,target):
    bundle=SimpleNamespace(processor=processor,tokenizer=processor.tokenizer,device=torch.device('cpu'),observation_contract=observation_contract(frames))
    row=example(tmp_path,frames,target)
    inputs=encode(bundle,row,tmp_path,max_length=20000)
    active=inputs['labels'][0][inputs['labels'][0]!=-100]
    assert processor.tokenizer.decode(active)==target+'<|im_end|>'
    inference=encode(bundle,row,tmp_path,supervise=False,max_length=20000)
    assert 'labels' not in inference
    assert inputs['image_grid_thw'].shape[0]==frames
    with pytest.raises(ValueError,match='token limit'):
        encode(bundle,row,tmp_path,max_length=10)


def test_real_multimodal_lora_backward(tmp_path,processor,model):
    from peft import LoraConfig,get_peft_model
    from qwen3vl_local.sft_v2.train import _lora_target_modules
    model.config.image_token_id=processor.image_token_id
    model.config.video_token_id=processor.video_token_id
    model.config.vision_start_token_id=processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    model.config.vision_end_token_id=processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    for p in model.parameters():p.requires_grad=False
    targets=_lora_target_modules(model,vision_scope='off',strict_vision_scope=True)
    model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,target_modules=targets,task_type='CAUSAL_LM'))
    bundle=SimpleNamespace(processor=processor,tokenizer=processor.tokenizer,device=torch.device('cpu'),model=model,observation_contract=observation_contract(2))
    inputs=encode(bundle,example(tmp_path),tmp_path,max_length=20000)
    model.train()
    loss=model(**inputs,use_cache=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_rgb_identity_and_timestamp(tmp_path):
    row=example(tmp_path)
    row['images'][0]='0010.jpg'
    row['image_sha256'][0]=row['image_sha256'][1]
    with pytest.raises(ValueError,match='timestamp'):
        load_images(row,tmp_path)
