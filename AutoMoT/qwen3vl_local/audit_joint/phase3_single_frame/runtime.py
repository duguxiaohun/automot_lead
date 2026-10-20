"""Process-local native trainer adapter; no sequence sampling or old trainer edits."""
from collections import Counter
from pathlib import Path
import json
import math
import os
import re
import torch
from . import POLICY, dataset, objective, planning, protocol
from qwen3vl_local.audit_joint import phase3_base_trial as trial, phase3_base_trial_worker as base
from qwen3vl_local.audit_joint.phase3_base_dense import baseline
from qwen3vl_local.sft_new_loop_phase3 import train


def make_args(job, artifact):
    args = base.make_args(job)
    args.generation_eval_steps = 0
    args.base_trial['generation']='separate single-frame four-query evaluation; no native phrase generation'
    args.base_trial['validation']='frozen anchor IDs; incomplete windows masked, zero-label anchors reported skipped'
    args.single_frame_contract = dict(policy=POLICY,config=artifact['config'],sources=artifact['sources'],
        dataset_sha256=trial.file_sha(job['dataset']),input_rgb_count=1,query_count=4,
        loss='mean complete-window soft cross entropy per anchor; full vocabulary',
        initialization='same_qwen_no_base_adapter',history_predictions=False,
        inference=protocol.contract(artifact))
    # The unchanged native index validator still uses its historical schema.
    # Actual saved inference metadata is entirely replaced by protocol.annotate.
    return args


def install(artifact, output):
    """Install only in this worker. Caller must not reuse the process for other modes."""
    baseline.install_baseline()
    originals = {name:getattr(train,name) for name in ('_make_item','_append_jsonl',
        'verify_epoch_consumption','_save_adapter','_write_run_metadata','_validation_work')}
    by_seed = {}; state = {}; records = artifact['records']
    def make_item(row, **kwargs):
        item = originals['_make_item'](row,**kwargs)
        by_seed[item.spec.seed_key] = row
        return item
    def select(rows,args,**kwargs):
        work,audit = planning.plan(rows,records,args,**kwargs)
        state.clear();state.update(audit=audit,counts=Counter(),mass=Counter(),positions=[],query_histogram=Counter())
        return work,audit
    def validation(*args,**kwargs):
        work,audit=originals['_validation_work'](*args,**kwargs)
        for item in work:by_seed[item.spec.seed_key]=item.row
        return work,audit
    def pack(bundle,**kwargs):
        row = by_seed[kwargs['spec'].seed_key]
        rec = records[dataset.identity(row)]
        if row.split!='train' and not any(w['valid'] for w in rec['windows']):
            # Native validation reports this as skipped, not a zero loss/correct answer.
            return None
        packed = objective.pack(bundle,rec,kwargs['images'][-1],max_length=kwargs['max_length'])
        packed['record'] = rec
        return packed
    def loss(bundle,packed,specs):
        if len(packed)!=1:
            raise ValueError('Single-frame independent protocol requires B1')
        value,report = objective.loss(bundle,packed[0])
        rec = packed[0]['record']
        if bundle.model.training and rec['split']=='train':
            state['pending'] = rec,report
        hit = report['correct']/report['supervised']
        return value,[dict(denom=report['supervised'],token_acc=hit,value_token_acc=hit,
            choice_action_exact=hit,supervised_queries=report['supervised'],
            input_tokens=report['tokens'],target_entropy=report['target_entropy'],
            kl_to_training_target=report['kl_to_training_target'])]
    def loss_one(bundle,packed,spec):
        # Native B1 periodic validation calls _loss_one rather than _loss_batch.
        value,stats=loss(bundle,[packed],[spec])
        return value,stats[0]
    def append(path,value):
        if re.fullmatch(r'epoch_\d+_rank\d+_batches.jsonl',Path(path).name):
            rec,report = state.pop('pending')
            if value['ids']!=[rec['id']]:
                raise ValueError('Actual single-frame record differs from sampled anchor')
            state['positions'].extend(value['positions'])
            state['query_histogram'][str(report['supervised'])]+=1
            for i,(w,mass) in enumerate(zip(rec['windows'],report['query_weights'])):
                if w['valid']:
                    state['counts'][f'{rec["event"]}/{i}'] += 1
                    for a,q in zip(planning.ACTIONS,w['target']):
                        state['mass'][rec['event']+'/'+a] += mass*q
            value = dict(value,single_frame=dict(history=rec['history'],
                history_original=rec['history_original'],history_imputed=rec['history_imputed'],
                history_changed=rec['history_changed'],
                window_targets=[w['target'] for w in rec['windows']],
                raw_distributions=[w['raw'] for w in rec['windows']],
                source_ids=[w['source_ids'] for w in rec['windows']],
                query_weights=report['query_weights'],supervised_windows=report['supervised'],rgb_count=1,
                phase='backward_completed_before_optimizer_step'))
        originals['_append_jsonl'](path,value)
    def verify(work,consumed,*,device,world_size):
        result = originals['verify_epoch_consumption'](work,consumed,device=device,world_size=world_size)
        rank=int(os.environ.get('RANK','0'))
        local=dict(counts=dict(state['counts']),mass=dict(state['mass']),
                   query_histogram=dict(state['query_histogram']),
                   positions_ok=state['positions']==list(range(rank,len(work),world_size)))
        all_ranks=[None]*world_size
        if world_size>1:torch.distributed.all_gather_object(all_ranks,local)
        else:all_ranks=[local]
        counts=Counter();mass=Counter();query_histogram=Counter()
        for r in all_ranks:
            counts.update(r['counts']);mass.update(r['mass']);query_histogram.update(r['query_histogram'])
        audit=state['audit']
        if (not all(r['positions_ok'] for r in all_ranks) or dict(counts)!=audit['planned_query_exposure'] or
            any(not math.isclose(mass[k],v,rel_tol=1e-5,abs_tol=1e-4) for k,v in audit['preferred_future_loss_mass'].items())):
            raise ValueError('Actual post-backward single-frame query consumption differs from plan')
        if rank==0:
            trial.write(Path(output)/f'train/balance/single_frame_consumed_epoch_{audit["epoch"]+1:04d}.json',
                dict(status='verified_after_backward',actual_query_exposure=dict(counts),
                     actual_training_loss_mass=dict(mass),anchor_presentations=len(work),ranks=world_size,
                     supervised_windows_per_anchor_histogram=dict(query_histogram),
                     total_supervised_windows=sum(counts.values()),
                     mean_supervised_windows_per_optimizer_update=sum(counts.values())/(len(work)/world_size)))
        return result
    def annotate(path):
        value=trial.read(path)
        value=protocol.annotate(value,artifact)
        protocol.validate(value,artifact)
        trial.write(path,value)
    def save(*args,**kwargs):
        path=originals['_save_adapter'](*args,**kwargs)
        annotate(path/'sft_new_loop_phase3_adapter_config.json')
        return path
    def metadata(*args,**kwargs):
        originals['_write_run_metadata'](*args,**kwargs)
        annotate(Path(output)/'train/train_run_manifest.json')
    train._make_item=make_item;train._contiguous_work=select
    train._validation_work=validation;train._loss_one=loss_one
    train._build_inputs=pack;train._loss_batch=loss
    train._append_jsonl=append;train.verify_epoch_consumption=verify
    train._save_adapter=save;train._write_run_metadata=metadata
    # Native index still carries the original four paths, but neither loading
    # nor the transformer consumes historical RGB in this experiment.
    train.select_history_rgb_paths=lambda paths,mode:[paths[-1]]
    base.install_reference(trial.reference())


EVALUATION_VARIANTS = {
    'full': ('filled','actual'), 'no_history': ('none','actual'),
    'native_history': ('native','actual'), 'blank_image': ('filled','blank'),
}


def evaluation_input(rec,image,history_mode='filled',image_mode='actual'):
    if history_mode not in ('filled','none','native') or image_mode not in ('actual','blank'):
        raise ValueError('Unsupported single-frame evaluation intervention')
    if image_mode=='blank':
        from PIL import Image
        image=Image.new('RGB',image.size,(0,0,0))
    values=rec['history'] if history_mode=='filled' else rec['history_original'] if history_mode=='native' else ['UNKNOWN']*4
    return dict(rec,history=values),image


@torch.no_grad()
def predict(bundle,records,*,use_history=True,history_mode=None,image_mode='actual'):
    was_training=bundle.model.training;bundle.model.eval()
    results=[]
    try:
        for rec in records:
            image=train._load_images([rec['image']])[0]
            view,image=evaluation_input(rec,image,history_mode or ('filled' if use_history else 'none'),image_mode)
            packed=objective.pack(bundle,view,image)
            logits=objective.old.readout_logits(bundle.model,packed,bundle.device)
            indices=logits.argmax(-1).tolist()
            from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID
            allowed=(*CONTEXT_BY_ID[rec['event']].action_keys,'KEEP')
            valid_indices=[i for i,a in enumerate(planning.ACTIONS) if a in allowed]
            chosen=logits[:,[packed['choices'][i] for i in valid_indices]].argmax(-1).tolist()
            predictions=[planning.ACTIONS[valid_indices[i]] for i in chosen]
            raw_predictions=[planning.ACTIONS[packed['choices'].index(i)] if i in packed['choices'] else 'INVALID' for i in indices]
            results.append(dict(id=rec['id'],event=rec['event'],history=rec['history'],
                                history_original=rec['history_original'],history_imputed=rec['history_imputed'],
                                history_changed=rec['history_changed'],
                                history_input=view['history'],image_mode=image_mode,
                                windows=rec['windows'],predictions=predictions,raw_predictions=raw_predictions,
                                probabilities=logits.softmax(-1)[:,packed['choices']].cpu().tolist()))
    finally:bundle.model.train(was_training)
    return results,objective.metrics(results)
