"""Shared contiguous-window planning and true multimodal minibatch collation.

No tensor concatenation along time: each row remains an independent example.
History overlap (including transition boundaries) is allowed within a split.
"""
from collections import Counter, defaultdict
import hashlib
import json

POLICY = 'event_equal_contiguous_unseen_first_v3'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tail_piece_length(by_length, remaining):
    """Exact tail packing with whole pieces; minimize joins, then prefer long pieces.

    Counts are bounded by availability. If no complete packing exists, preserve
    the original longest-piece choice rather than create extra tiny fragments.
    This only sees pieces in the current question/anchor exposure tier.
    """
    best = {0: ()}
    for length in sorted(by_length, reverse=True):
        if length > remaining:
            continue
        count = min(len(by_length[length]), remaining // length)
        for _ in range(count):
            for size, combo in sorted(list(best.items()), reverse=True):
                total = size + length
                if total > remaining:
                    continue
                candidate = tuple(sorted((*combo, length), reverse=True))
                previous = best.get(total)
                if previous is None or (len(candidate), tuple(-n for n in candidate)) < (len(previous), tuple(-n for n in previous)):
                    best[total] = candidate
    return best[remaining][0] if remaining in best else max(by_length)


def plan(records, *, per_event, epoch=0, seed=0, world_size=1, batch_size=1,
         stride=2, history=None):
    """Records contain id/event/sequence/frame/action/split/group, no model tensors.

    Equal events are hard. Exposure is lexicographically first, then longest
    pieces at batch starts and complete-piece packing for batch tails, then soft
    action deficit. Short pieces join independent sequences WITHOUT fake history.
    Both parity streams are eligible when stride=2, so odd frames are not lost.
    """
    if not records or any(type(n) is not int or n < 1 for n in (world_size, batch_size, stride)):
        raise ValueError('invalid batch/stride/world or empty sampling pool')
    if type(epoch) is not int or epoch < 0:
        raise ValueError('invalid epoch')
    ids = [r['id'] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate stable sample identity')
    pools = defaultdict(list)
    group_splits = defaultdict(set)
    for i, r in enumerate(records):
        if type(r['frame']) is not int:
            raise ValueError('noninteger anchor frame')
        pools[r['event']].append(i)
        group_splits[r['group']].add(r['split'])
    if any(len(s) != 1 for s in group_splits.values()) or len({r['split'] for r in records}) != 1:
        raise ValueError('sampling cannot mix dataset splits or physical groups across splits')
    targets = dict(per_event) if isinstance(per_event, dict) else dict.fromkeys(pools, per_event)
    block = world_size * batch_size
    if set(targets) != set(pools) or any(type(n) is not int or n < 1 or n % block for n in targets.values()):
        raise ValueError(f'each event quota must be positive and divisible by global batch {block}; no silent padding/drop')
    binding = dict(policy=POLICY, pool_sha256=digest(sorted(records, key=lambda r:r['id'])),
                   seed=seed, targets=targets, world_size=world_size, batch_size=batch_size, stride=stride)
    if history is None:
        history = dict(**binding, next_epoch=0, counts=dict.fromkeys(ids, 0))
        for previous in range(epoch):
            _, prior = plan(records, per_event=targets, epoch=previous, seed=seed,
                world_size=world_size, batch_size=batch_size, stride=stride, history=history)
            history = prior['next_history']
    if (not isinstance(history, dict) or any(history.get(k) != v for k,v in binding.items())
            or history.get('next_epoch') != epoch or set(history.get('counts', {})) != set(ids)):
        raise ValueError('sampling history pool/config/epoch mismatch')
    counts = dict(history['counts'])
    if (any(type(n) is not int or n < 0 for n in counts.values()) or
            any(sum(counts[ids[i]] for i in pool) != epoch * targets[e] for e,pool in pools.items())):
        raise ValueError('invalid actual exposure history')
    before = dict(counts)
    frame_keys=[(r['event'],r['sequence'],r['frame']) for r in records]
    frame_counts=Counter()
    for i,ident in enumerate(ids):frame_counts[frame_keys[i]]+=counts[ident]
    queues, pieces_report, action_report, packing_report = {}, {}, {}, {}
    for event, pool in sorted(pools.items()):
        selected, segments = [], []
        packing = Counter()
        actions = Counter(records[i]['action'] for i in pool)
        # Soft action targets; no label editing or rare-action repetition before coverage.
        from qwen3vl_local.sft_new_loop_phase3.sampling import support_aware_quota
        desired = support_aware_quota(actions, targets[event])
        actual = Counter()
        while len(selected) < targets[event]:
            level = min(counts[ids[i]] for i in pool)
            frame_level=min(frame_counts[frame_keys[i]] for i in pool if counts[ids[i]]==level)
            groups = defaultdict(lambda: defaultdict(list))
            for i in pool:
                if counts[ids[i]] == level and frame_counts[frame_keys[i]]==frame_level:
                    r = records[i]
                    groups[(r['sequence'], r['frame'] % stride)][r['frame']].append(i)
            pieces = []
            for key, frames in sorted(groups.items()):
                # Preserve alternatives until consumption. All candidates here
                # share the same question/anchor exposure tier. Choosing another
                # question at this anchor cannot shorten the temporal window.
                run = []
                for frame in sorted(frames):
                    values = sorted(frames[frame], key=lambda i:ids[i])
                    if run and records[run[-1][0]]['frame'] + stride != frame:
                        pieces.extend(run[k:k+block] for k in range(0,len(run),block)); run=[]
                    run.append(values)
                pieces.extend(run[k:k+block] for k in range(0,len(run),block))
            # Stable seeded ties, independent of input list order and RGB mode.
            pieces.sort(key=lambda p:digest([seed, epoch, ids[p[0][0]]]))
            by_length = defaultdict(list)
            for piece in pieces:
                by_length[len(piece)].append(piece)
            opportunities = Counter(action for piece in pieces for alternatives in piece
                for action in {records[i]['action'] for i in alternatives})
            while by_length and len(selected) < targets[event]:
                remaining = min(block-len(selected)%block, targets[event]-len(selected))
                longest = max(by_length)
                length = tail_piece_length(by_length, remaining) if remaining < block else longest
                if length != longest:
                    packing['whole_piece_tail_choices'] += 1
                candidates = by_length[length]
                # Within equally continuous choices, compensate action deficit.
                def resolve(piece):
                    provisional = actual.copy()
                    available = opportunities.copy()
                    take, score = [], 0.
                    def deficit(i):
                        action = records[i]['action']
                        return max(0, desired[action]-provisional[action]) / max(1, desired[action])
                    for alternatives in piece[:remaining]:
                        # Account for scarce remaining opportunities: consuming
                        # an anchor can discard its only rare-action alternative.
                        def urgency(i):
                            action = records[i]['action']
                            return (max(0, desired[action]-provisional[action]) / max(1, available[action]), deficit(i))
                        i = max(alternatives, key=urgency)
                        # Keep the existing between-piece score; only selection
                        # within an anchor uses the running provisional deficit.
                        action = records[i]['action']
                        score += max(0, desired[action]-actual[action]) / max(1, desired[action])
                        take.append(i)
                        provisional[records[i]['action']] += 1
                        for action in {records[j]['action'] for j in alternatives}:
                            available[action] -= 1
                    return take, score
                choices = [resolve(piece) for piece in candidates]
                best = max(range(len(candidates)), key=lambda j:choices[j][1])
                piece = candidates.pop(best)
                if not candidates:
                    del by_length[length]
                take, rest = choices[best][0], piece[remaining:]
                for alternatives in piece[:remaining]:
                    for action in {records[i]['action'] for i in alternatives}:
                        opportunities[action] -= 1
                if rest:
                    packing['truncated_pieces'] += 1
                    by_length[len(rest)].append(rest)
                segments.append(dict(offset=len(selected), length=len(take), sequence=records[take[0]]['sequence'],
                                     first_frame=records[take[0]]['frame'], last_frame=records[take[-1]]['frame']))
                selected.extend(take)
                for i in take:
                    counts[ids[i]] += 1
                    frame_counts[frame_keys[i]]+=1
                    actual[records[i]['action']] += 1
        queues[event] = selected
        pieces_report[event] = segments
        action_report[event] = dict(target=desired, actual=dict(actual))
        packing_report[event] = dict(whole_piece_tail_choices=packing['whole_piece_tail_choices'],
                                     truncated_pieces=packing['truncated_pieces'])
    # Contiguous global blocks; contiguous per-rank chunks within each block.
    result=[]
    events=sorted(queues, key=lambda e:digest([seed,epoch,e]))
    for offset in range(0,max(targets.values()),block):
        for event in events:
            result.extend(queues[event][offset:offset+block])
    repeats=Counter(result)
    return result, dict(policy=POLICY, epoch=epoch, presentations=len(result), event_counts=targets,
        per_event_quota=next(iter(targets.values())) if len(set(targets.values()))==1 else None,
        batch_size=batch_size, world_size=world_size, anchor_stride=stride,
        global_batch=block, unique_questions=len(repeats), max_question_repeat=max(repeats.values()),
        new_questions=sum(before[ids[i]]==0 for i in repeats),
        cumulative_unique_questions=sum(n>0 for n in counts.values()),
        cumulative_missing_questions=sum(n==0 for n in counts.values()),
        event_coverage={e:dict(pool=len(v),seen=sum(counts[ids[i]]>0 for i in v)) for e,v in pools.items()},
        segments=pieces_report, action_balance=action_report, tail_packing=packing_report,
        priority='least question exposure, least anchor exposure, longest piece at batch start; whole-piece exact tail packing; soft action deficit',
        unique_anchors=len({frame_keys[i] for i in repeats}),
        overlap='real causal histories may overlap within split; no synthetic cross-route windows',
        ordered_ids_sha256=digest([ids[i] for i in result]),
        next_history=dict(**binding, next_epoch=epoch+1, counts=counts))


def rank_batches(indices, *, rank, world_size, batch_size):
    block=world_size*batch_size
    if len(indices)%block or not 0<=rank<world_size:
        raise ValueError('incomplete global batch or invalid rank')
    return [indices[k+rank*batch_size:k+(rank+1)*batch_size] for k in range(0,len(indices),block)]


def phase4_records(rows):
    return [dict(id=r['id'],event=r['episode']['event'],
        sequence=json.dumps([r['scenario'],r['route_id'],r['episode']['instance_id']]),
        frame=r['observation']['frame_id'],action=r['edge']+'/'+r['target'],
        split=r['split'],group=r['physical_group']) for r in rows]


def collate(inputs, pad_token_id):
    """Right padding and flattened Qwen image patches/grids in sample order.

    Position IDs are recomputed by the model from each padded text/image layout.
    Refuse unknown tensor layouts rather than silently reorder visual evidence.
    """
    import torch
    if not inputs or pad_token_id is None:
        raise ValueError('empty batch or missing pad token')
    keys=set(inputs[0])
    if any(set(x)!=keys for x in inputs):
        raise ValueError('inconsistent model input fields')
    widths=[x['input_ids'].shape[1] for x in inputs]
    width=max(widths)
    result={}
    fills={'input_ids':pad_token_id,'attention_mask':0,'labels':-100,
           'loss_weights':0,'component_ids':0,'token_type_ids':0,'mm_token_type_ids':0}
    for key in keys:
        values=[x[key] for x in inputs]
        if key in fills:
            if any(v.ndim!=2 or v.shape!=(1,n) for v,n in zip(values,widths)):
                raise ValueError('unexpected token tensor layout: '+key)
            result[key]=torch.cat([torch.nn.functional.pad(v,(0,width-v.shape[1]),value=fills[key]) for v in values])
        elif key in ('pixel_values','image_grid_thw','pixel_values_videos','video_grid_thw'):
            result[key]=torch.cat(values,dim=0)
        else:
            raise ValueError('unsupported batched model input: '+key)
    return result
