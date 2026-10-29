"""Native Phase2 quotas/coverage, with independent Phase1 supervision per slot."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from qwen3vl_local.sft_new_loop_phase2 import train as native, eval as native_eval
from qwen3vl_local.sft_new_loop_phase1.exposure_sampling import identity, task_identity, semantic_signature
from qwen3vl_local.sft_new_loop_phase1.training_policy import UNKNOWN
from qwen3vl_local.sft_new_loop_phase1.run_integrity import digest, write
from . import CONTRACT, EPOCHS, QUOTA, WORLD, UPDATES
from .data import labels, relative_rgb
import json

SEED = 20260810


def arguments(output, action_index='auto'):
    return SimpleNamespace(action_index=action_index, _training_world_size=WORLD,
        max_steps=0,num_epochs=EPOCHS,output_dir=str(output),action_balance_strength=.5,max_train_frame_repeat=10)


def work(rows,args,epoch):
    items=native._training_work(rows,args=args,seed=SEED+epoch*1000003,target_per_bin=QUOTA,
        regular_multiplier=2.,invalid_multiplier=1.,highway_regular_fraction=.25,
        highway_ue3_fraction=.125,route_diverse=True)
    args._fusion_coverage=coverage(args._coverage_sampler,args._exposure,items)
    return items


def coverage(sampler,ledger,items):
    """Read-only diagnostics around the byte-preserved native sampler."""
    used={identity(i.row) for i in items}
    pool={identity(r) for rows in sampler.pools.values() for r in rows}
    keys={(semantic_signature(i.row),sampler.hints.label(i.row)!=UNKNOWN) for i in items}
    compatible={identity(r) for key in keys for r in sampler.pools[key]}
    unseen={k for k in pool if ledger.frames[k]==0};remaining=unseen-used
    return dict(candidate_unique_frames=len(pool),unseen_before=len(unseen),unseen_after=len(remaining),
        compatible_unseen_after=len(remaining&compatible),pool_exhausted=not remaining,
        new_frames=len(used&unseen),previously_seen_frames=len(used-unseen),
        unseen_policy='zero prior/current exposure first within native semantic/subtype/availability quotas')


def descriptor(item):
    return dict(key=identity(item.row),domain=item.row.question_domain,answers=item.row.answers,
                seed_key=item.spec.seed_key,output_keys=list(item.spec.output_keys),balance_key=item.balance_key,
                true_rs=item.row.true_rs,event=item.row.event,ue3_subtype=item.row.ue3_subtype,
                invalid_source=item.row.invalid_source,town=item.row.town)


def summarize(items,facts):
    counts=Counter(i.balance_key.rsplit('/',1)[-1] for i in items)
    if counts!=Counter(UE1=QUOTA,UE3=QUOTA,UE5=QUOTA,UE6=QUOTA,RE=QUOTA*2,INVALID=QUOTA):
        raise ValueError('Native Phase2 quota changed')
    return dict(presentations=len(items),updates=len(items)//WORLD,
        member_sha256=digest([descriptor(i) for i in items]),class_counts=dict(counts),
        phase1_positive_presentations=dict(Counter(k for i in items for k,v in facts[identity(i.row)].items() if v)),
        unique_frames=len({identity(i.row) for i in items}))


def evaluation(index,data_root,split):
    rows=native_eval._read_rows(Path(index),split=split,data_root=Path(data_root))
    # Joint holdout protection can move the old P2 val highway route to test.
    # Never move an old P1 test route back to val merely to fill a validation bin.
    if split=='test':native_eval._require_highway_ue3_rows(rows,split=split)
    return native_eval._balanced_cases(rows,cases_per_bin=32 if split=='val' else 64,
        seed=20260831,highway_ue3_fraction=.125,route_diverse=True)


def prepare(index,data_root,output,action_index='auto'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=native._read_rows(Path(index),split='train',data_root=Path(data_root));facts=labels(index)
    args=arguments(output,action_index);epochs=[]
    for epoch in range(EPOCHS):
        items=work(rows,args,epoch);summary=summarize(items,facts)
        summary['coverage']=args._fusion_coverage
        epochs.append(summary)
        args._exposure.frames.update(identity(i.row) for i in items)
        args._exposure.tasks.update(task_identity(i) for i in items)
    ev={split:[descriptor(i) for i in evaluation(index,data_root,split)] for split in ('val','test')}
    if sum(e['updates'] for e in epochs)!=UPDATES:raise ValueError('Unexpected optimizer budget')
    result=dict(contract=CONTRACT,world_size=WORLD,epochs=epochs,optimizer_updates=UPDATES,
        branch_supervisions_per_pair=2,phase1_loss_weight=.5,phase2_loss_weight=.5,
        invalid_share='1/7 logical pairs; INVALID flag belongs only to Phase2',
        action_source=args._action_hints.receipt,evaluation_sha256={k:digest(v) for k,v in ev.items()},
        evaluation_counts={k:len(v) for k,v in ev.items()},scope='projection_not_actual_training')
    result['validation_highway_ue3_positive_cases']=sum(
        i.row.ue3_subtype==native.HIGHWAY_UE3_SUBTYPE and native._target_class(i.row)=='UE3'
        for i in evaluation(index,data_root,'val'))
    if not result['validation_highway_ue3_positive_cases']:
        result['validation_limitation']='No highway cut-in positive after protecting both original holdouts; checkpoint selection cannot certify this subgroup. Final original-test audit is mandatory.'
    write(output/'plan.json',result);write(output/'evaluation_members.json',ev)
    bindings={}
    for raw in Path(index).open():
        row=json.loads(raw);bindings[identity(row)]=dict(key=identity(row),answers=row['phase1_answers'],
            paths=[relative_rgb(p) for p in row['history_rgb_paths']],split=row['split'],phase2_rs=row['true_rs'])
    write(output/'fusion_labels.json',list(bindings.values()))
    return result
