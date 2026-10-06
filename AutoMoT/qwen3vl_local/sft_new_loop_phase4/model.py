"""本地Qwen3.5 LoRA的条件SFT与自由生成；不使用Phase3动作输出头。"""
from pathlib import Path
from .controller import Episode
from .route_prompts import messages, parse_answer
from .identity import file_sha, digest
from .observation import check_observation_contract,validate_observation


def load_images(row,data_root):
    from PIL import Image
    if 'condition_provenance' in row:
        from .condition_builder import validate_row
        validate_row(row,data_root)
    root = Path(data_root).resolve()
    images = []
    pixel_hashes = row.get('image_rgb_sha256')
    if pixel_hashes is not None and len(pixel_hashes)!=len(row['images']):
        raise ValueError('decoded RGB identity count mismatch')
    if len(row['images']) != len(row['observation']['history_frames']):
        raise ValueError('image/history count mismatch')
    for index,(rel,expected) in enumerate(zip(row['images'],row['image_sha256'],strict=True)):
        path = (root/rel).resolve()
        if path.stem != f"{row['observation']['history_frames'][index]:04d}":
            raise ValueError('image timestamp does not match causal history')
        if not path.is_relative_to(root) or file_sha(path) != expected:
            raise ValueError('RGB source mismatch')
        with Image.open(path) as source:
            rgb = source.convert('RGB')
            if pixel_hashes is not None:
                from .input_identity import rgb_content_sha
                if rgb_content_sha(rgb)!=pixel_hashes[index]:
                    raise ValueError('decoded RGB content mismatch')
            images.append(rgb)
    return images


def load_bundle(model_dir,device,*,rank=16,alpha=32,dropout=.05,vision_scope='off',checkpointing=True):
    from qwen3vl_local.sft_v2.train import load_model_with_lora
    return load_model_with_lora(Path(model_dir),device=device,lora_rank=rank,lora_alpha=alpha,
        lora_dropout=dropout,lora_vision_scope=vision_scope,strict_vision_scope=True,
        gradient_checkpointing=checkpointing)


def encode(bundle,row,data_root,*,supervise=True,max_length=8192):
    import torch
    from qwen3vl_local.sft_v2.train import _find_subsequence, _assert_inside_assistant_turn
    from qwen3vl_local.qwen35.backend import assistant_header_text
    from .route_prompts import prompt
    validate_observation(getattr(bundle,'observation_contract',None),row['observation'],len(row['images']))
    ep = Episode(**row['episode'])
    if digest(prompt(ep,row['edge'],row['observation'])) != row['prompt_sha256']:
        raise ValueError('prompt identity mismatch')
    images = load_images(row,data_root)
    msgs = messages(ep,row['edge'],row['observation'],images,row['target'] if supervise else None)
    text = bundle.processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=not supervise)
    inputs = bundle.processor(text=[text],images=images,return_tensors='pt',padding=True)
    if inputs['input_ids'].shape[1] > max_length:
        raise ValueError('sample exceeds token limit; truncation would remove visual/answer evidence')
    if supervise:
        ids = inputs['input_ids'][0].tolist()
        header = bundle.tokenizer(assistant_header_text(bundle.processor),add_special_tokens=False)['input_ids']
        start = _find_subsequence(ids,header,0,last=True)
        target = bundle.tokenizer(row['target'],add_special_tokens=False)['input_ids']
        pos = _find_subsequence(ids,target,start+len(header))
        _assert_inside_assistant_turn(ids,pos,header,0)
        labels = torch.full_like(inputs['input_ids'],-100)
        labels[0,pos:pos+len(target)] = inputs['input_ids'][0,pos:pos+len(target)]
        eos = bundle.tokenizer.convert_tokens_to_ids('<|im_end|>')
        tail = pos+len(target)
        if tail >= len(ids) or ids[tail] != eos:
            raise ValueError('unexpected assistant answer boundary')
        labels[0,tail] = eos
        inputs['labels'] = labels
    return {k:v.to(bundle.device) if hasattr(v,'to') else v for k,v in inputs.items()}


def generate(bundle,row,data_root,max_length=8192):
    import torch
    inputs = encode(bundle,row,data_root,supervise=False,max_length=max_length)
    model = bundle.model.module if hasattr(bundle.model,'module') else bundle.model
    with torch.inference_mode():
        tokens = model.generate(**inputs,max_new_tokens=8,do_sample=False,use_cache=True)
    text = bundle.tokenizer.decode(tokens[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
    try:
        answer = parse_answer(text)
    except ValueError:
        answer = 'MALFORMED'
    return answer,text


def load_for_inference(model_dir,adapter_dir,device):
    from types import SimpleNamespace
    from qwen3vl_local.qwen35.backend import AutoProcessor, LocalModel
    from qwen3vl_local.qwen35.adapters import LocalPeftModel
    from .identity import check_contract
    import json
    import torch
    adapter_dir = Path(adapter_dir)
    recorded=json.loads((adapter_dir/'phase4_contract.json').read_text())
    check_contract(recorded['source'])
    observation=check_observation_contract(recorded.get('observation_contract'))
    base = LocalModel.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16).to(device)
    model = LocalPeftModel.from_pretrained(base,adapter_dir).eval()
    processor = AutoProcessor.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False)
    return SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device=device,observation_contract=observation)
