"""空验证桶补齐必须以未曝光物理路线为单位，并保护既有覆盖。"""
from copy import deepcopy
from .split_coverage import complete_context_splits


def plan(rows, development=()):
    return complete_context_splits(rows, contexts=("A", "B"), splits=("train", "val", "test"),
                                   development=set(development), group_of=lambda r: r['group'], seed=17)


def test_missing_holdout_contexts_use_distinct_unexposed_whole_groups():
    rows = [dict(group=g, split="train", context_id=c, repeat=rep)
            for g in ("exposed", "g1", "g2", "g3") for c in ("A", "B") for rep in (0, 1)]
    original = deepcopy(rows)
    report = plan(rows, ["exposed"])
    assert not report['unresolved'] and len(report['moves']) == 2
    assert all(r['split'] == 'train' for r in rows if r['group'] == 'exposed')
    for g in {r['group'] for r in rows}:
        assert len({r['split'] for r in rows if r['group'] == g}) == 1
    assert {r['split'] for r in rows} == {'train', 'val', 'test'}
    original.reverse()
    assert plan(original, ["exposed"])['moves'] == report['moves']


def test_impossible_capacity_is_reported_without_using_exposed_routes():
    rows = [dict(group='exposed', split='train', context_id=c) for c in ('A', 'B')]
    report = plan(rows, ['exposed'])
    assert report['moves'] == {}
    assert report['unresolved'] == {'val': ['A', 'B'], 'test': ['A', 'B']}


def test_nonempty_split_is_not_rebalanced_and_last_train_support_is_protected():
    rows = [dict(group=s+c, split=s, context_id=c)
            for s in ('train', 'val', 'test') for c in ('A', 'B')]
    assert plan(rows)['moves'] == {}
    rows = [r for r in rows if not (r['split'] == 'val' and r['context_id'] == 'A')]
    report = plan(rows)
    assert report['moves'] == {} and report['unresolved'] == {'val': ['A']}


def test_holdout_frame_floor_is_met_from_real_source_capacity():
    rows = [dict(group=g, split='train', context_id=c)
            for g in ('a', 'b', 'c', 'd', 'e') for c in ('A', 'B') for _ in range(3)]
    report = complete_context_splits(rows, contexts=('A', 'B'), splits=('train', 'val', 'test'),
                                    development=set(), group_of=lambda r: r['group'], seed=17,
                                    min_holdout_frames=5)
    assert not report['unresolved']
    assert report['after']['val'] == {'A': 6, 'B': 6}
    assert report['after']['test'] == {'A': 6, 'B': 6}
    assert len(report['moves']) == 4


def test_route_support_floor_cannot_be_filled_with_frames_from_one_route():
    rows = [dict(group=g, split='train', context_id='A')
            for g in range(15) for _ in range(40)]
    report = complete_context_splits(rows, contexts=('A',), splits=('train','val','test'),
        development={0}, group_of=lambda r:r['group'], seed=17,
        min_holdout_frames=32, min_holdout_groups=5)
    assert not report['unresolved']
    assert report['physical_groups_after'] == {s:{'A':5} for s in ('train','val','test')}
    assert all(r['split']=='train' for r in rows if r['group']==0)


def test_group_shortage_is_reported_and_does_not_drain_train_or_block_other_contexts():
    rows = [dict(group=s+'A', split=s, context_id='A') for s in ('train','val','test')]
    rows += [dict(group=str(g)+'B', split='train', context_id='B') for g in range(15)]
    report = complete_context_splits(rows, contexts=('A','B'), splits=('train','val','test'),
        development=set(), group_of=lambda r:r['group'], seed=17,
        min_holdout_frames=1, min_holdout_groups=5)
    assert not report['unresolved']
    assert report['group_deficits'] == {'val':{'A':4},'test':{'A':4}}
    assert all(report['physical_groups_after'][s]['B']==5 for s in ('train','val','test'))
