"""Exact fallback for geometry's one-group-per-event split coverage contract.

Original holdouts stay fixed; unexposed original train groups may be reassigned.
Only event presence matters. This is a covering problem, not frame matching:
one route can cover multiple events but can belong to only one split.
"""
from collections import defaultdict
import hashlib


def solve(group_masks, origins, development, *, event_count, seed, max_states=100000):
    if type(max_states) is not int or max_states < 1:
        raise ValueError('positive_coverage_search_state_limit_required')
    if type(event_count) is not int or not 1 <= event_count <= 10:
        raise ValueError('coverage_search_supports_one_to_ten_events')
    names = ('train', 'val', 'test')
    full = (1 << event_count)-1
    covered = [0, 0, 0]; buckets = defaultdict(list)
    for group, mask in group_masks.items():
        if type(mask) is not int or not 1 <= mask <= full:
            raise ValueError('invalid_event_presence_mask')
        origin = origins[group]
        if origin not in names: raise ValueError('unknown_split')
        if group in development and origin != 'train':
            raise ValueError('development_group_in_holdout')
        if origin == 'train' and group not in development:
            buckets[mask].append(group)
        else:
            covered[names.index(origin)] |= mask
    masks = sorted(buckets)
    for mask in masks:
        buckets[mask].sort(key=lambda g: (hashlib.sha256(f'{seed}:coverage_exact:{g}'.encode()).hexdigest(), g))
    # Equal event masks are exchangeable. At most one representative per split
    # is ever needed at the current minimum of one group/frame per event.
    capacity = tuple(min(3, len(buckets[m])) for m in masks)
    remaining = list(capacity)
    failed = set(); visited = 0

    class SearchLimit(Exception):
        pass

    def search(coverage):
        nonlocal visited
        if all(c == full for c in coverage): return []
        key = (coverage, bytes(remaining))
        if key in failed: return None
        if visited >= max_states: raise SearchLimit
        visited += 1
        target = None; choices = None
        # Necessary shared-supply bound, and branch on the most constrained cell.
        for event in range(event_count):
            bit = 1 << event
            missing = [s for s in range(3) if not coverage[s] & bit]
            if not missing: continue
            candidates = [i for i, mask in enumerate(masks) if remaining[i] and mask & bit]
            if sum(remaining[i] for i in candidates) < len(missing):
                failed.add(key); return None
            for split in missing:
                if choices is None or len(candidates) < len(choices):
                    target, choices = split, candidates
        choices = sorted(choices, key=lambda i: (-(masks[i] & ~coverage[target]).bit_count(), masks[i]))
        for index in choices:
            remaining[index] -= 1
            updated = list(coverage); updated[target] |= masks[index]
            tail = search(tuple(updated))
            remaining[index] += 1
            if tail is not None: return [(index, target), *tail]
        failed.add(key)
        return None

    try:
        witness = search(tuple(covered))
        status = 'feasible' if witness is not None else 'infeasible'
    except SearchLimit:
        witness = None; status = 'unknown'
    assignment = dict(origins)
    if witness is not None:
        used = defaultdict(int)
        for index, split in witness:
            mask = masks[index]; group = buckets[mask][used[index]]; used[index] += 1
            assignment[group] = names[split]
        # Unused routes remain in train. Verify the witness independently of DFS.
        observed = [0, 0, 0]
        for group, mask in group_masks.items(): observed[names.index(assignment[group])] |= mask
        if observed != [full]*3: raise RuntimeError('coverage_search_invalid_witness')
    return assignment if status == 'feasible' else None, dict(
        method='exact_event_mask_backtracking_v1', status=status, visited_states=visited,
        max_states=max_states, event_signatures=len(masks),
        movable_groups=sum(len(v) for v in buckets.values()),
        representatives=sum(capacity),
        scope='original holdouts fixed; development train-only; one group/frame per event/split')
