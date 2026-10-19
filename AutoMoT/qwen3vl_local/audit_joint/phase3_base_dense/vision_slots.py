"""Probe eight source-frame slots while retaining the real twenty-slot expansion.

Equal pixel content at different source times must not collapse the probe. Reuse
of the same source time across strict windows remains enabled. No global cache.
"""
from contextlib import contextmanager
import torch


@contextmanager
def source_slot_reuse(model):
    tower=next((m for m in model.modules() if type(m).__name__=='Qwen3_5VisionModel'
                and type(m).__module__=='qwen3vl_local.qwen35.vendor.modeling_qwen3_5'),None)
    if tower is None or any(p.requires_grad for p in tower.parameters()):
        raise ValueError('Source-slot probe requires frozen native Qwen3.5 vision')
    if any(m.training and (getattr(m,'attention_dropout',0)>0 or
            isinstance(m,torch.nn.Dropout) and m.p>0) for m in tower.modules()):
        raise ValueError('Source-slot probe requires deterministic vision')
    original=tower.forward;had_instance='forward' in tower.__dict__;saved=tower.__dict__.get('forward')
    stats=dict(images=0,encoded_images=0,reused_images=0,reason='eight_source_slots_no_cross_time_dedup')
    def forward(hidden_states,grid_thw,**kwargs):
        if hidden_states.requires_grad or len(grid_thw)!=20 or any(g[0]!=1 for g in grid_thw.tolist()):
            raise ValueError('Expected twenty frozen image presentations from five windows')
        sizes=[int(g.prod()) for g in grid_thw];chunks=hidden_states.split(sizes)
        mapping=[i+j for i in range(5) for j in range(4)];unique=[]
        for slot in range(8):unique.append(mapping.index(slot))
        for i,slot in enumerate(mapping):
            source=unique[slot]
            if not torch.equal(grid_thw[i],grid_thw[source]) or not torch.equal(chunks[i],chunks[source]):
                raise ValueError('Overlapping source frame differs between strict windows')
        out=original(torch.cat([chunks[i] for i in unique]),grid_thw[unique],**kwargs)
        last=out.last_hidden_state.split([sizes[i] for i in unique])
        pooled=out.pooler_output.split([sizes[i]//tower.spatial_merge_size**2 for i in unique])
        out.last_hidden_state=torch.cat([last[i] for i in mapping])
        out.pooler_output=torch.cat([pooled[i] for i in mapping])
        stats.update(images=20,encoded_images=8,reused_images=12)
        return out
    tower.forward=forward
    try:yield stats
    finally:
        if had_instance:tower.forward=saved
        else:del tower.forward
