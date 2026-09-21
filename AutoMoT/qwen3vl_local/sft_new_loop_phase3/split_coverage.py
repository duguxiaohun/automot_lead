"""固定哈希划分的空桶修补：只移动未曝光 train 物理组，不按动作值挑验证集。"""
from collections import Counter, defaultdict
import hashlib


def complete_context_splits(rows, *, contexts, splits, development, group_of, seed, min_holdout_frames=1):
    """保留非空桶；为缺失 holdout context 分配独立物理路线并保护 train 覆盖。"""
    if min_holdout_frames < 1:
        raise ValueError("min_holdout_frames must be positive")
    groups = defaultdict(list)
    for row in rows:
        groups[group_of(row)].append(row)
    counts = {split: Counter() for split in splits}
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
    before = {s: {c: counts[s][c] for c in contexts} for s in splits}
    moves = {}
    if any(not counts["train"][c] for c in contexts):
        return dict(policy="unexposed_train_empty_context_repair_v1", before=before,
                    after=before, moves={}, unresolved={"train": [c for c in contexts if not counts['train'][c]]})
    while True:
        deficits = [(s, c) for s in splits if s != "train" for c in contexts if counts[s][c] < min_holdout_frames]
        if not deficits:
            break
        # 每条路线移出后 train 各 context 仍须至少有一条候选；禁止移动曝光组。
        available = [g for g in groups if origins[g] == "train" and g not in development
                     and g not in moves and all(counts['train'][c] > n for c, n in group_contexts[g].items())]
        eligible = {(s, c): [g for g in available if c in group_contexts[g]] for s, c in deficits}
        target = min(deficits, key=lambda d: (len(eligible[d]), d))
        if not eligible[target]:
            break
        split, context = target
        missing = {c for s, c in deficits if s == split}
        chosen = min(eligible[target], key=lambda g: (
            -len(missing.intersection(group_contexts[g])),
            -min(min_holdout_frames - counts[split][context], group_contexts[g][context]),
            hashlib.sha256(f"{seed}:coverage:{split}:{g}".encode()).hexdigest(), g,
        ))
        counts['train'].subtract(group_contexts[chosen])
        counts[split].update(group_contexts[chosen])
        moves[chosen] = dict(from_split="train", to_split=split,
                            fills=sorted(missing.intersection(group_contexts[chosen])))
    for group, move in moves.items():
        for row in groups[group]:
            row["split"] = move["to_split"]
    after = {s: {c: counts[s][c] for c in contexts} for s in splits}
    floors = {s: (1 if s == 'train' else min_holdout_frames) for s in splits}
    return dict(policy="unexposed_train_context_capacity_repair_v1", before=before, after=after,
                min_context_frames=floors,
                moves=moves, unresolved={s: [c for c in contexts if counts[s][c] < floors[s]]
                                        for s in splits if any(counts[s][c] < floors[s] for c in contexts)},
                selection_fields="context presence and physical route identity only; no action/RGB/model score",
                capacity_claim="distinct recorded frames; not independent physical-route statistical support")
