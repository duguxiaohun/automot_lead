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
