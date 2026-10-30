"""Independent greedy generation and transaction-local frozen RGB feature reuse.

No cross-rank or cross-forward cache. Only the local Qwen3.5 image tower is
supported; trainable/stochastic vision and diagnostic hidden-state outputs fall
back to the original forward. Reuse compares the actual processed tensors.
"""
from contextlib import contextmanager


def left_collate(rows, pad_token_id):
    import torch
    from .sequence_batch import collate
    # Use the training collator's strict key/shape validation, then move only
    # sequence padding. Image tensors keep their original flattened ordering.
    batch = collate(rows, pad_token_id)
    if any(not torch.all(row['attention_mask'] == 1) for row in rows):
        raise ValueError('generation collator requires unpadded single-example inputs')
    width = batch['input_ids'].shape[1]
    for key in ('input_ids', 'attention_mask', 'token_type_ids', 'mm_token_type_ids'):
        if key in batch:
            batch[key] = torch.cat([torch.roll(batch[key][i:i+1],
                width-row['input_ids'].shape[1], dims=1) for i,row in enumerate(rows)])
    if any(key in batch for key in ('labels', 'loss_weights', 'component_ids')):
        raise ValueError('generation inputs must not contain supervision')
    return batch


@contextmanager
def frozen_rgb_reuse(model, *, enabled=True):
    """Restore the tower method even if forward raises; never retain features."""
    import torch
    stats = dict(images=0, encoded_images=0, reused_images=0, reason='disabled')
    tower = next((m for m in model.modules() if type(m).__name__ == 'Qwen3_5VisionModel'
                  and type(m).__module__ == 'qwen3vl_local.qwen35.vendor.modeling_qwen3_5'), None) if enabled else None
    if tower is None:
        stats['reason'] = 'unsupported_model' if enabled else 'disabled'
        yield stats
        return
    if any(p.requires_grad for p in tower.parameters()):
        stats['reason'] = 'trainable_vision'
        yield stats
        return
    if any(m.training and ((isinstance(m, torch.nn.Dropout) and m.p > 0)
                           or getattr(m, 'attention_dropout', 0) > 0) for m in tower.modules()):
        stats['reason'] = 'stochastic_vision'
        yield stats
        return
    original = tower.forward
    had_instance = 'forward' in tower.__dict__
    instance_value = tower.__dict__.get('forward')
    stats['reason'] = 'frozen_exact_processed_rgb'

    def forward(hidden_states, grid_thw, **kwargs):
        if (hidden_states.requires_grad or kwargs.get('output_hidden_states', getattr(tower.config,'output_hidden_states',False))
                or kwargs.get('output_attentions', getattr(tower.config,'output_attentions',False)) or kwargs.get('return_dict') is False):
            stats['reason'] = 'unsupported_forward_options'
            return original(hidden_states, grid_thw, **kwargs)
        grids = grid_thw.tolist()
        if any(g[0] != 1 for g in grids):
            stats['reason'] = 'video_not_supported'
            return original(hidden_states, grid_thw, **kwargs)
        sizes = [g[0]*g[1]*g[2] for g in grids]
        chunks = hidden_states.split(sizes)
        unique, mapping = [], []
        for i, chunk in enumerate(chunks):
            match = next((j for j,u in enumerate(unique)
                          if grids[i] == grids[u] and torch.equal(chunk, chunks[u])), None)
            if match is None:
                match = len(unique)
                unique.append(i)
            mapping.append(match)
        stats['images'] += len(chunks)
        stats['encoded_images'] += len(unique)
        stats['reused_images'] += len(chunks)-len(unique)
        if len(unique) == len(chunks):
            return original(hidden_states, grid_thw, **kwargs)
        out = original(torch.cat([chunks[i] for i in unique]), grid_thw[unique], **kwargs)
        last = out.last_hidden_state.split([sizes[i] for i in unique])
        pooled = out.pooler_output.split([sizes[i]//tower.spatial_merge_size**2 for i in unique])
        # Preserve all original image slots: the language model and rotary
        # coordinates still see independent samples, including shared images.
        out.last_hidden_state = torch.cat([last[i] for i in mapping])
        out.pooler_output = torch.cat([pooled[i] for i in mapping])
        return out

    tower.forward = forward
    try:
        yield stats
    finally:
        if had_instance:
            tower.forward = instance_value
        else:
            del tower.forward


def greedy_batch(bundle, rows, *, max_new_tokens, phase3=False, reuse_rgb=False):
    """Return ordered texts. P3 keeps its EOS set and 1.05 repetition penalty."""
    import torch
    from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList
    if not rows:
        return []
    if max_new_tokens < 1:
        raise ValueError('max_new_tokens must be positive')
    from .qwen35 import text_prefix
    if text_prefix.enabled() and phase3:
        from .sft_v3.train import KVState, _student_generate_kv
        answers=[]
        with torch.no_grad():
            for row in rows:
                out=text_prefix.prefill(bundle,row)
                state=KVState(row['input_ids'],row['input_ids'],row['attention_mask'],
                              out.past_key_values,out.rope_deltas,out.logits[:,-1,:])
                answer,_,_=_student_generate_kv(bundle,state,max_new_tokens)
                answers.append(answer)
        return answers
    inputs = left_collate(rows, bundle.tokenizer.pad_token_id)
    model = bundle.model.module if hasattr(bundle.model, 'module') else bundle.model
    options = dict(max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    if phase3:
        from .sft_v3.train import _eos_token_ids
        prompt_ids = inputs['input_ids']
        prompt_mask = inputs['attention_mask'].bool()
        width = prompt_ids.shape[1]
        class ValidTokenPenalty(LogitsProcessor):
            def __call__(self, input_ids, scores):
                result = scores.clone()
                for i in range(len(rows)):
                    seen = torch.unique(torch.cat((prompt_ids[i][prompt_mask[i]], input_ids[i, width:])))
                    values = result[i, seen]
                    result[i, seen] = torch.where(values < 0, values*1.05, values/1.05)
                return result
        # The legacy P3 loop uses only argmax, EOS and this penalty. Do not
        # inherit unrelated checkpoint generation defaults or penalize pads.
        options['generation_config'] = GenerationConfig(pad_token_id=bundle.tokenizer.pad_token_id,
            eos_token_id=sorted(_eos_token_ids(bundle)))
        options['logits_processor'] = LogitsProcessorList([ValidTokenPenalty()])
    with torch.inference_mode(), frozen_rgb_reuse(model, enabled=reuse_rgb):
        tokens = model.generate(**inputs, **options)
    return [bundle.tokenizer.decode(row[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            for row in tokens]


def encode_messages(bundle, messages):
    images = [part['image'] for msg in messages for part in msg.get('content', [])
              if isinstance(part, dict) and part.get('type') == 'image']
    text = bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    row = bundle.processor(text=[text], images=images, return_tensors='pt', padding=True)
    return {k:v.to(bundle.device) for k,v in row.items()}


def distributed_identity():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def batch_indices(count, batch_size, rank, world):
    if min(batch_size, world) < 1 or not 0 <= rank < world:
        raise ValueError('invalid distributed batch geometry')
    return [list(range(start, min(start+batch_size, count)))
            for start in range(0, count, batch_size) if (start//batch_size) % world == rank]


def distributed_batches(rows, batch_size, generate, *, identities):
    """Whole consecutive batches per rank; no padding/repetition of eval cases.

    All ranks participate, including ranks with no cases. Only predictions and
    failure receipts are gathered; model forward never calls DDP collectives.
    """
    import hashlib
    import json
    import torch.distributed as dist
    rank, world = distributed_identity()
    invalid = len(identities) != len(rows) or len(set(identities)) != len(rows)
    signature = hashlib.sha256(json.dumps([identities,batch_size],ensure_ascii=False).encode()).hexdigest()
    if world > 1:
        signatures = [None]*world
        dist.all_gather_object(signatures,(invalid,signature))
        if any(value[0] for value in signatures):
            raise ValueError('evaluation identities must be unique and complete on every rank')
        if len(set(signatures)) != 1:
            raise ValueError('evaluation membership/order/batch differs across ranks')
    if invalid:raise ValueError('evaluation identities must be unique and complete')
    outputs=[];error=None
    try:
        for indices in batch_indices(len(rows),batch_size,rank,world):
            values=generate([rows[i] for i in indices])
            if len(values)!=len(indices):raise ValueError('generation output count mismatch')
            outputs.extend(zip(indices,values))
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    receipt=dict(rank=rank,outputs=outputs,error=error)
    ranks=[receipt]
    if world>1:
        ranks=[None]*world;dist.all_gather_object(ranks,receipt)
    failures=[f"rank {r['rank']}: {r['error']}" for r in ranks if r['error']]
    if failures:raise RuntimeError('distributed evaluation failed: '+'; '.join(failures))
    flat=[pair for r in ranks for pair in r['outputs']]
    if sorted(i for i,_ in flat)!=list(range(len(rows))):
        raise ValueError('evaluation consumption missing or duplicated')
    ordered=[v for _,v in sorted(flat,key=lambda p:p[0])]
    audit=dict(world_size=world,batch_per_rank=batch_size,selected_count=len(rows),
               per_rank_counts=[len(r['outputs']) for r in ranks],ordered_membership_sha256=signature,
               consumed_once=True,partition='consecutive_batches_round_robin')
    return ordered,audit


def initialize_evaluation(device):
    """Respect launcher mask; never reselect physical GPUs inside torchrun."""
    import os
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    world=int(os.environ.get('WORLD_SIZE','1'))
    if world>1:
        if device=='cuda':
            local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
            device=f'cuda:{local}'
        if not dist.is_initialized():
            dist.init_process_group('nccl' if device.startswith('cuda') else 'gloo',
                                    timeout=timedelta(seconds=600))
    return device


def launch_four_evaluation(module, device):
    """Direct CUDA CLI calls join the same four-card path as shell launchers."""
    import os
    import subprocess
    import sys
    if device=='cpu':return None  # Explicit CPU diagnostics remain possible.
    world=int(os.environ.get('WORLD_SIZE','1'))
    if world==4:return None
    if world!=1 or 'LOCAL_RANK' in os.environ:
        raise ValueError('formal CUDA evaluation requires exactly four ranks')
    if device not in ('auto','cuda',''):
        raise ValueError('use GPU_IDS with four cards, not a single CUDA device')
    from .sft_new_loop_phase4.devices import select_gpus
    cards=select_gpus(4)
    if len(cards.split(','))!=4:raise ValueError('four visible idle GPUs or four explicit GPU_IDS required')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m',module,*sys.argv[1:]]
    return subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=cards,GPU_IDS=cards)).returncode
