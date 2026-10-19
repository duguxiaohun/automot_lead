"""Restore historical selection without changing shared Phase3/Phase4 modules on disk.

The adapter lives only in the new worker process. Generic rank partitioning,
recovery and evaluation utilities are retained; sequence selection is never run.
"""
from collections import Counter
import hashlib
import json

POLICY = 'event_action_independent_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def independent_work(rows, args, *, epoch, world_size, history=None):
    from qwen3vl_local.sft_new_loop_phase3 import train
    from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity
    if args.batch_size != 1 or args.grad_accum != 1 or args.action_output_mode != 'choice':
        raise ValueError('Independent base requires choice, B1 and accumulation 1')
    if epoch < 0 or world_size < 1:
        raise ValueError('Invalid epoch/world')
    if history is not None and history != dict(policy=POLICY, next_epoch=epoch):
        raise ValueError('Not an independent-base checkpoint; sequence exposure history is incompatible')
    seed = int(args.seed) + epoch * 1_000_003
    work = train._balanced_work(rows, target_per_bin=args.focus_balance_count,
        seed=seed, route_diverse=True, action_output_mode='choice')
    if len(work) % world_size:
        raise ValueError('Equal independent rank shards require divisible work')
    ids = [digest(case_identity(x)) for x in work]
    return work, dict(policy=POLICY, epoch=epoch, seed=seed, ordered_ids=ids,
        action_counts=dict(Counter(train.choice_action_for_answers(x.spec) for x in work)),
        event_counts=dict(Counter(x.row.context_id for x in work)),
        event_action_counts=dict(Counter(x.row.context_id+'/'+train.choice_action_for_answers(x.spec) for x in work)),
        unique_cases=len(set(ids)), selected_cases=len(ids),
        least_seen_priority=False, contiguous_priority=False, anchor_priority=False,
        hard_action_quotas=True, augmentation='historical_per_epoch_seed',
        next_history=dict(policy=POLICY, next_epoch=epoch+1))


def single_loss(bundle, packed_items, specs):
    """One example, one forward: no batch padding, no visual reuse."""
    from qwen3vl_local.sft_new_loop_phase3.train import _loss_from_logits
    if len(packed_items) != 1 or len(specs) != 1:
        raise ValueError('Independent base training uses B1')
    p = packed_items[0]
    inputs = {k:p[k].unsqueeze(0).to(bundle.device) for k in ('input_ids','attention_mask')}
    inputs.update({k:v.to(bundle.device) for k,v in p['vision'].items()})
    out = bundle.model(**inputs, use_cache=False, return_dict=True)
    loss, stats = _loss_from_logits(out.logits, p['labels'][None].to(bundle.device),
        p['loss_weights'][None].to(bundle.device), p['component_ids'][None].to(bundle.device), specs[0])
    return loss, [stats]


def install_baseline():
    from qwen3vl_local.sft_new_loop_phase3 import train
    train._contiguous_work = independent_work
    train._loss_batch = single_loss
    train.SEQUENCE_POLICY = POLICY
