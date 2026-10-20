from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qwen3vl_local.sft_new_loop_phase1 import train as p1
from qwen3vl_local.sft_new_loop_phase2 import train as p2
from qwen3vl_local.sft_new_loop_phase1.test_training_policy import phase1_rows, phase2_rows
from qwen3vl_local.sft_new_loop_phase1.training_policy import ActionHints, identity, UNKNOWN
from qwen3vl_local.sft_new_loop_phase1.exposure_sampling import (
    CoverageSampler, ExposureLedger, semantic_signature,
)


def pool(n=120):
    base = phase1_rows(16)[0]
    return [replace(base, frame_id=i, route_id=f"route_{i % 12}") for i in range(n)]


def work_for(rows, n):
    return [p1._make_all_item(rows[i % len(rows)], 'RS1', seed=i) for i in range(n)]


def test_three_epochs_cover_unseen_and_plans_do_not_consume(tmp_path):
    rows = pool()
    sampler, ledger = CoverageSampler(rows, ActionHints()), ExposureLedger()
    all_seen = set()
    for epoch in range(3):
        work, report = sampler.select(work_for(rows, 40), ledger, seed=epoch)
        assert report['unique_frames'] == report['new_frames'] == 40
        assert len(ledger.frames) == epoch * 40
        assert sampler.select(work_for(rows, 40), ledger, seed=epoch)[0] == work
        ids = {identity(x.row) for x in work}
        assert not ids & all_seen
        for item in work:
            ledger.record(item, valid=True)
        ledger.optimizer_applied()
        actual = ledger.finish_epoch(tmp_path, rank=0, step=(epoch+1)*40, epoch=epoch+1, planned=40)
        assert actual['new_frames'] == 40
        all_seen |= ids
    assert len(all_seen) == 120
    work, report = sampler.select(work_for(rows, 40), ledger, seed=4)
    assert report['new_frames'] == 0
    assert report['max_frame_repeat'] == 1


def test_action_cannot_override_unseen_priority(tmp_path):
    rows = pool(120)
    hints = ActionHints({identity(r): 'STOP' if i < 100 else 'RESUME' for i,r in enumerate(rows)})
    ledger = ExposureLedger()
    # All rare-action frames have already been trained once.
    ledger.frames.update({identity(r): 1 for r in rows[100:]})
    work, report = CoverageSampler(rows, hints).select(work_for(rows, 80), ledger, seed=7)
    assert report['after'] == {'STOP': 80}
    assert report['new_frames'] == 80
    assert [x.spec for x in work] == [x.spec for x in work_for(rows, 80)]
    # With equal exposure, soft action weighting improves the rare action share.
    _, report = CoverageSampler(rows, hints).select(work_for(rows, 80), ExposureLedger(), seed=7)
    assert 80*20/120 < report['after']['RESUME'] <= 20


def test_partial_action_coverage_and_subtypes_preserved():
    rows = pool(120)
    hints = ActionHints({identity(r): 'STOP' for r in rows[:20]})
    original = work_for(rows, 80)
    result, report = CoverageSampler(rows, hints).select(original, ExposureLedger(), seed=13)
    assert report['before'][UNKNOWN] == report['after'][UNKNOWN] == 60
    assert [semantic_signature(x.row) for x in original] == [semantic_signature(x.row) for x in result]
    assert all((hints.label(a.row) == UNKNOWN) == (hints.label(b.row) == UNKNOWN) for a,b in zip(original,result))


def test_repeat_cap_fail_closed_and_least_seen_after_exhaustion():
    rows = pool(4)
    sampler = CoverageSampler(rows, ActionHints())
    ledger = ExposureLedger()
    ledger.frames.update({identity(r): i for i,r in enumerate(rows)})
    _, report = sampler.select(work_for(rows, 12), ledger, seed=1)
    assert report['max_frame_repeat'] <= 5
    with pytest.raises(ValueError, match='compatible frame capacity'):
        sampler.select(work_for(rows, 41), ledger, seed=1)
    assert not ledger.local_frames  # Even a rejected plan never records training.


def test_no_train_from_heldout():
    with pytest.raises(ValueError, match='only accepts train'):
        CoverageSampler([replace(pool(1)[0], split='val')], ActionHints())


def test_skips_padding_partial_epoch_and_pending_gradients(tmp_path):
    work = work_for(pool(), 5)
    ledger = ExposureLedger()
    ledger.record(work[0], valid=False)
    ledger.record(work[1], valid=True)
    ledger.record(work[1], valid=True, padded=True)
    ledger.optimizer_applied()
    ledger.record(work[2], valid=True)
    report = ledger.finish_epoch(tmp_path, rank=0, step=4, epoch=1, planned=120)
    assert (report['actual_presentations'], report['new_frames']) == (3,2)
    assert report['skipped_cumulative'] == report['padded_cumulative'] == report['pending_backward'] == 1
    assert identity(work[0].row) not in ledger.frames
    ledger.optimizer_applied()
    ledger.finish_run(tmp_path, rank=0, step=4)
    summary = json.loads((tmp_path/'exposure/summary.json').read_text())
    assert summary['successful_backward'] == summary['applied_backward'] == 3
    assert summary['pending_backward'] == 0


@pytest.mark.parametrize('phase', [1,2])
def test_real_samplers_three_epochs_quota_specs_and_coverage(tmp_path, phase):
    module = p1 if phase == 1 else p2
    rows = phase1_rows(2048) if phase == 1 else phase2_rows(2048)
    hints = ActionHints({identity(r): 'STOP' if r.frame_id % 16 < 14 else 'RESUME' for r in rows})
    sampler, ledger = CoverageSampler(rows, hints), ExposureLedger()
    native_seen = set()
    for epoch in range(3):
        seed = 20260810 + epoch * 1000003
        kwargs = dict(target_per_bin=768, variant_weights=p1.TRAIN_VARIANT_WEIGHTS) if phase == 1 else dict(target_per_bin=1536, regular_multiplier=2., invalid_multiplier=1., route_diverse=True)
        original = module._balanced_work(rows, seed=seed, **kwargs)
        result, report = sampler.select(original, ledger, seed=seed)
        native_seen.update(identity(x.row) for x in original)
        assert len(result) == (12288 if phase == 1 else 10752)
        assert [x.spec for x in result] == [x.spec for x in original]
        assert [x.balance_key for x in result] == [x.balance_key for x in original]
        assert [semantic_signature(x.row) for x in result] == [semantic_signature(x.row) for x in original]
        assert report['max_frame_repeat'] <= 10
        if phase == 2:
            assert p2.invalid_subgroup_report(result) == p2.invalid_subgroup_report(original)
        for item in result:
            ledger.record(item, valid=True)
        ledger.optimizer_applied()
        ledger.finish_epoch(tmp_path, rank=0, step=epoch+1, epoch=epoch+1, planned=len(result))
    # A heterogeneous fixture exercises quota constraints as well as duplicates.
    assert len(ledger.frames) >= len(native_seen)


def test_two_gloo_ranks_merge_actual_not_planned_and_select_identically(tmp_path):
    script = tmp_path/'worker.py'
    script.write_text('''import json,sys
from pathlib import Path
import torch.distributed as dist
from dataclasses import dataclass
from types import SimpleNamespace
@dataclass
class Item:
    row: object
    spec: object
    balance_key: str = 'UE1'
def pool():
    return [SimpleNamespace(scenario='s',route_id=f'r{i%12}',frame_id=i,split='train',
                            event='UE1',answers={'UE1':True}) for i in range(120)]
def work_for(rows,n):
    return [Item(rows[i],SimpleNamespace(variant='event')) for i in range(n)]
from qwen3vl_local.sft_new_loop_phase1.exposure_sampling import CoverageSampler,ExposureLedger
from qwen3vl_local.sft_new_loop_phase1.training_policy import ActionHints,identity
dist.init_process_group('gloo')
rank=dist.get_rank();out=Path(sys.argv[1]);rows=pool()
sampler=CoverageSampler(rows,ActionHints());ledger=ExposureLedger()
work,_=sampler.select(work_for(rows,40),ledger,seed=1)
# Each rank actually runs just two assigned items, one of four is skipped.
for pos in (rank,rank+2): ledger.record(work[pos],valid=pos!=1)
ledger.optimizer_applied()
report=ledger.finish_epoch(out,rank=rank,step=2,epoch=1,distributed=dist,planned=40)
assert report['actual_presentations']==report['new_frames']==3
assert len(ledger.frames)==3
next_work,_=sampler.select(work_for(rows,40),ledger,seed=2)
assert not set(ledger.frames)&{identity(x.row) for x in next_work}
all_ids=[None,None];dist.all_gather_object(all_ids,[identity(x.row) for x in next_work])
assert all_ids[0]==all_ids[1]
ledger.finish_run(out,rank=rank,step=2,distributed=dist)
dist.destroy_process_group()
''')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(p1.__file__).resolve().parents[2])
    result = subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','--',str(script),str(tmp_path)], env=env, capture_output=True,text=True,timeout=45)
    assert result.returncode == 0, result.stdout+result.stderr
    assert json.loads((tmp_path/'exposure/summary.json').read_text())['successful_backward'] == 3
