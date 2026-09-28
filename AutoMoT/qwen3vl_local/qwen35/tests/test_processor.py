from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from tokenizers.pre_tokenizers import ByteLevel

from qwen3vl_local.qwen35.processor import Qwen35Processor
from qwen3vl_local.qwen35.backend import assistant_header_text, LocalProcessor
from qwen3vl_local.qwen35.vendor.tokenization_qwen3_5 import Qwen3_5Tokenizer
from qwen3vl_local.qwen35.vendor.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from qwen3vl_local.qwen35.vendor.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from qwen3vl_local.qwen35.tests.test_backend import model


@pytest.fixture
def processor():
    specials = ['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<|vision_start|>',
                '<|vision_end|>', '<|image_pad|>', '<|video_pad|>', '<think>', '</think>']
    vocab = {t: i for i, t in enumerate(specials + sorted(ByteLevel.alphabet()))}
    tokenizer = Qwen3_5Tokenizer(vocab=vocab, merges=[], additional_special_tokens=specials,
                                 eos_token='<|im_end|>')
    image = Qwen2VLImageProcessorFast(patch_size=16, temporal_patch_size=2, merge_size=2,
        size={'shortest_edge': 1024, 'longest_edge': 442368}, image_mean=[.5]*3, image_std=[.5]*3)
    video = Qwen3VLVideoProcessor(patch_size=16, temporal_patch_size=2, merge_size=2)
    template = (Path(__file__).parents[1]/'vendor/official_chat_template.jinja').read_text()
    return Qwen35Processor(image_processor=image, tokenizer=tokenizer, video_processor=video,
                           chat_template=template)


def test_train_generation_header_and_multiturn_prefix(processor):
    user = [{'role': 'system', 'content': 'Drive carefully.'}, {'role': 'user', 'content': 'STOP?'}]
    prefix = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    assert prefix.endswith(assistant_header_text(processor))
    complete = processor.apply_chat_template(user+[{'role': 'assistant', 'content': 'YES'}], tokenize=False)
    assert complete == prefix + 'YES<|im_end|>\n'
    continued = processor.apply_chat_template(user+[{'role': 'assistant', 'content': 'YES'},
        {'role': 'user', 'content': 'Why?'}], tokenize=False, add_generation_prompt=True)
    assert continued.startswith(complete)
    system = processor.apply_chat_template(user[:1], tokenize=False)
    assert prefix.startswith(system)
    thinking = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    assert thinking.endswith('<think>\n')


@pytest.mark.parametrize('frames', [1, 2, 4])
def test_actual_stitched_image_grid(processor, frames):
    images = [Image.new('RGB', (1152,384), (64,128,192)) for _ in range(frames)]
    messages = [{'role':'user', 'content': [{'type':'image','image':im} for im in images]
                + [{'type':'text','text':'Which action?'}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors='pt')
    assert inputs['image_grid_thw'].tolist() == [[1,24,72]]*frames
    assert (inputs['input_ids'] == processor.image_token_id).sum().item() == frames*432
    assert inputs['pixel_values'].shape == (frames*1728, 1536)
    # Official 3.5 normalization is mean/std .5, rather than old CLIP constants.
    assert torch.isfinite(inputs['pixel_values']).all()


def test_loss_mask_header_matches_real_template(processor):
    from qwen3vl_local.sft_v2.train import _assert_inside_assistant_turn, _find_subsequence
    messages=[{'role':'user','content':'YES in prompt is not a label.'}, {'role':'assistant','content':'YES'}]
    text=processor.apply_chat_template(messages, tokenize=False)
    ids=processor.tokenizer(text, add_special_tokens=False)['input_ids']
    header=processor.tokenizer(assistant_header_text(processor), add_special_tokens=False)['input_ids']
    target=processor.tokenizer('YES',add_special_tokens=False)['input_ids']
    pos=_find_subsequence(ids, target, _find_subsequence(ids,header,0)+len(header))
    _assert_inside_assistant_turn(ids,pos,header,0)
    with pytest.raises(ValueError):
        _assert_inside_assistant_turn(ids,_find_subsequence(ids,target,0),header,0)


@pytest.mark.parametrize('mode', ['binary', 'choice'])
@pytest.mark.parametrize('rgb_mode,frames', [('2rgb_endpoints', 2), ('4rgb', 4)])
def test_phase3_actual_loss_mask(processor, mode, rgb_mode, frames):
    from qwen3vl_local.sft_new_loop_phase3.train import _build_inputs
    from qwen3vl_local.sft_new_loop_phase3.test_binary_keep import spec_for
    from qwen3vl_local.sft_new_loop_phase3.prompts import build_action_target
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS
    labels=dict.fromkeys(ACTION_KEYS,False)
    labels['STOP']=True
    spec=spec_for('STATIC_BLOCKAGE',labels,mode)
    bundle=SimpleNamespace(processor=processor,tokenizer=processor.tokenizer)
    packed=_build_inputs(bundle, images=[Image.new('RGB',(32,32)) for _ in range(frames)],
        history_rgb_mode=rgb_mode,target=build_action_target(spec),spec=spec,
        max_length=20000,format_loss_weight=.1)
    assert packed is not None
    assert (packed['loss_weights'] > 0).sum() > 0
    think_ids=processor.tokenizer.convert_tokens_to_ids(['<think>','</think>'])
    for token in think_ids:
        assert packed['loss_weights'][packed['input_ids']==token].eq(0).all()


def test_processor_local_roundtrip(processor, tmp_path, monkeypatch):
    import json, socket
    monkeypatch.setattr(socket.socket,'connect',lambda *a,**k: (_ for _ in ()).throw(AssertionError('network')))
    # The model repository stores these components separately.
    processor.tokenizer.save_pretrained(tmp_path)
    processor.image_processor.save_pretrained(tmp_path)
    processor.video_processor.save_pretrained(tmp_path)
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'qwen3_5'}))
    (tmp_path/'chat_template.jinja').write_text(processor.chat_template)
    restored=LocalProcessor.from_pretrained(tmp_path)
    messages=[{'role':'user','content':'STOP?'}]
    assert restored.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)==processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)


def test_remote_media_rejected(processor):
    with pytest.raises(ValueError, match='local media'):
        processor.apply_chat_template([{'role':'user','content':[{'type':'image','url':'https://example.invalid/img.png'}]}])
    with pytest.raises(ValueError, match='local media'):
        processor(images=['https://example.invalid/img.png'], text=['hello'])


@pytest.mark.parametrize('mode', ['binary', 'choice'])
def test_phase3_real_loss_and_checkpointed_lora_step(processor, model, mode):
    from peft import LoraConfig, get_peft_model, TaskType
    from qwen3vl_local.sft_v2.train import _lora_target_modules
    from qwen3vl_local.sft_new_loop_phase3.train import _build_inputs, _loss_one
    from qwen3vl_local.sft_new_loop_phase3.test_binary_keep import spec_for
    from qwen3vl_local.sft_new_loop_phase3.prompts import build_action_target
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS
    for key, token in [('image_token_id','<|image_pad|>'),('video_token_id','<|video_pad|>'),
                       ('vision_start_token_id','<|vision_start|>'),('vision_end_token_id','<|vision_end|>')]:
        setattr(model.config,key,processor.tokenizer.convert_tokens_to_ids(token))
    targets=_lora_target_modules(model,vision_scope='off',strict_vision_scope=True)
    model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,target_modules=targets,task_type=TaskType.CAUSAL_LM))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    model.train()
    labels=dict.fromkeys(ACTION_KEYS,False);labels['STOP']=True
    spec=spec_for('STATIC_BLOCKAGE',labels,mode)
    bundle=SimpleNamespace(processor=processor,tokenizer=processor.tokenizer,model=model,device=torch.device('cpu'))
    packed=_build_inputs(bundle,images=[Image.new('RGB',(32,32))]*2,history_rgb_mode='2rgb_endpoints',
                         target=build_action_target(spec),spec=spec,max_length=20000,format_loss_weight=.1)
    loss,stats=_loss_one(bundle,packed,spec)
    assert torch.isfinite(loss) and stats['denom']>0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'in_proj_qkv.lora_B' in n)
