"""固定哈希划分的空桶修补：只移动未曝光 train 物理组，不按动作值挑验证集。"""
from collections import Counter, defaultdict
import hashlib


def complete_context_splits(rows, *, contexts, splits, development, group_of, seed,
                            min_holdout_frames=1, min_holdout_groups=1):
    """保留非空桶；为缺失 holdout context 分配独立物理路线并保护 train 覆盖。"""
    if min_holdout_frames < 1 or min_holdout_groups < 1:
        raise ValueError("holdout frame/group minimums must be positive")
    groups = defaultdict(list)
    for row in rows:
        groups[group_of(row)].append(row)
    counts = {split: Counter() for split in splits}
    group_counts = {split: Counter() for split in splits}
    group_contexts = {}
    origins = {}
    for group, values in groups.items():
        origin = {r["split"] for r in values}
        if len(origin) != 1:
            raise ValueError(f"physical route leakage before coverage planning: {group}")
        origins[group] = next(iter(origin))
        if group in development and origins[group] != "train":
            raise ValueError(f"development route in holdout: {group}")
        group_contexts[group] = Counter(r["context_id"] for r in values)
        if origins[group] in counts:
            counts[origins[group]].update(group_contexts[group])
            group_counts[origins[group]].update(group_contexts[group].keys())
    before = {s: {c: counts[s][c] for c in contexts} for s in splits}
    groups_before = {s: {c: group_counts[s][c] for c in contexts} for s in splits}
    # Never fix holdout diversity by leaving train with a lone route. Small source
    # contexts keep their existing train support and report the remaining deficit.
    train_group_floor = {c: min(min_holdout_groups, group_counts['train'][c]) for c in contexts}
    moves = {}
    if any(not counts["train"][c] for c in contexts):
        return dict(policy="unexposed_train_empty_context_repair_v1", before=before,
                    after=before, moves={}, unresolved={"train": [c for c in contexts if not counts['train'][c]]})
    while True:
        deficits = [(s, c) for s in splits if s != "train" for c in contexts
                    if counts[s][c] < min_holdout_frames or group_counts[s][c] < min_holdout_groups]
        if not deficits:
            break
        # 每条路线移出后 train 各 context 仍须至少有一条候选；禁止移动曝光组。
        available = [g for g in groups if origins[g] == "train" and g not in development
                     and g not in moves and all(counts['train'][c] > n
                         and group_counts['train'][c] > train_group_floor[c]
                         for c, n in group_contexts[g].items())]
        eligible = {(s, c): [g for g in available if c in group_contexts[g]] for s, c in deficits}
        feasible = [d for d in deficits if eligible[d]]
        if not feasible:
            break
        target = min(feasible, key=lambda d: (len(eligible[d]), d))
        split, context = target
        missing = {c for s, c in deficits if s == split}
        chosen = min(eligible[target], key=lambda g: (
            -len(missing.intersection(group_contexts[g])),
            -min(max(0, min_holdout_frames - counts[split][context]), group_contexts[g][context]),
            hashlib.sha256(f"{seed}:coverage:{split}:{g}".encode()).hexdigest(), g,
        ))
        counts['train'].subtract(group_contexts[chosen])
        counts[split].update(group_contexts[chosen])
        group_counts['train'].subtract(group_contexts[chosen].keys())
        group_counts[split].update(group_contexts[chosen].keys())
        moves[chosen] = dict(from_split="train", to_split=split,
                            fills=sorted(missing.intersection(group_contexts[chosen])))
    for group, move in moves.items():
        for row in groups[group]:
            row["split"] = move["to_split"]
    after = {s: {c: counts[s][c] for c in contexts} for s in splits}
    floors = {s: (1 if s == 'train' else min_holdout_frames) for s in splits}
    return dict(policy="unexposed_train_context_route_capacity_v2", before=before, after=after,
                min_context_frames=floors,
                physical_groups_before=groups_before,
                physical_groups_after={s: {c: group_counts[s][c] for c in contexts} for s in splits},
                min_holdout_physical_groups=min_holdout_groups, train_group_floor=train_group_floor,
                group_support={s: {c: ("supported" if group_counts[s][c] >= min_holdout_groups
                                      else "insufficient_support") for c in contexts}
                               for s in splits if s != 'train'},
                group_deficits={s: {c: min_holdout_groups-group_counts[s][c] for c in contexts
                                    if group_counts[s][c] < min_holdout_groups}
                                for s in splits if s != 'train'},
                moves=moves, unresolved={s: [c for c in contexts if counts[s][c] < floors[s]]
                                        for s in splits if any(counts[s][c] < floors[s] for c in contexts)},
                selection_fields="context presence and physical route identity only; no action/RGB/model score",
                capacity_claim="frame capacity plus physical-route support; minimum groups is not proof of generalization")
