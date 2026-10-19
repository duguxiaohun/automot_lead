"""Opt-in dense objectives over the independent event/action training schedule."""
import json
import math
import re
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import torch
from . import baseline, objectives, planning
from .data import Pool, POLICY, premise
from qwen3vl_local.sft_new_loop_phase3 import train
from qwen3vl_local.audit_joint import phase3_base_trial as trial, phase3_base_trial_worker as base_worker


def load_pool(job,*,recompute_aux=False):
    rows=[]
    for split in ('train','val','test'):
        rows.extend(train._read_rows(Path(job['index']),split,data_root=Path(job['data_root'])))
    auxiliary=[]
    if job.get('aux_index'):
        from qwen3vl_local.sft_new_loop_phase3.preflight import check_index
        check_index(job['aux_index'],action_output_mode='choice',prompt_variant=trial.reference()['prompt_variant'])
        for split in ('train','val','test'):
            auxiliary.extend(train._read_rows(Path(job['aux_index']),split,data_root=Path(job['data_root'])))
        p=Path(job['aux_index'])
        if job['aux_binding']!=dict(index_sha256=trial.file_sha(p),manifest_sha256=trial.file_sha(p.with_name('manifest.json'))):
            raise ValueError('Auxiliary labels changed while loading')
    pool=Pool(rows,auxiliary,offsets=planning.options(job)['offset_frames'])
    if auxiliary:
        from . import provenance
        if recompute_aux:provenance.recompute(job)
        else:provenance.verify_cached(job)
    return pool


def contract(job):
    return dict(policy=POLICY,mode=job['mode'],output='A-F existing vocabulary, external parallel readouts',
        current_action_in_prompt=False,ground_truth_in_input=False,
        query_time_semantics='all_native_rule_windows_reset_at_each_prediction_origin_v1',
        resource_probe_policy='exact_query_positions_and_eight_source_slot_upper_workload_v2',
        forecast_contract=planning.options(job),aux_index=job.get('aux_index'),aux_binding=job.get('aux_binding'),
        auxiliary_provenance='native_meta_recomputed_and_split_preserved_v1',
        evaluation_policy='fixed_anchor_matched_readout_primary_four_frame_transfer_secondary_v1',
        native_validation_loss_role='secondary_four_frame_transfer_diagnostic',
        min_eval_coverage=job.get('min_eval_coverage',.8),
        observation='4 frames' if job['mode']=='future' else
            ('causal prefixes of 4,5,6,7,8 frames' if job['mode']=='stream' else 'five strict four-frame windows'),
        frame_position_stride=512 if job['mode']=='windows_bank' else None,
        query_recurrence='text_prefix_only' if job['mode']=='windows_bank' else 'native',
        architecture_protocol=('independent_frame_kv_query_prefix_recurrence_v1' if job['mode']=='windows_bank' else 'native_causal'),
        language_cache_shared=job['mode'] in ('stream','windows_prefix','windows_bank'),
        language_cache_scope=('text_before_first_image_only' if job['mode']=='windows_prefix' else
            'independent_frames_with_strict_four_frame_readout' if job['mode']=='windows_bank' else
            'growing_prefix' if job['mode']=='stream' else 'none'),
        image_language_states_shared=job['mode'] in ('stream','windows_bank'),
        lora_dropout=0. if job['mode'] in ('windows_prefix','windows_bank') else .05,
        gradient_checkpointing=job['mode'] not in ('windows_prefix','windows_bank'),
        cache_scope='one optimizer forward only; no stale cache across updates',
        max_length_per_sequence=8192,max_padded_batch_tokens=job.get('max_batch_tokens',16384),max_pixels_per_image=442368,
        loss_normalization='event/action quota-weighted query CE sum per anchor; average anchors across ranks',
        init_adapter=job['init_adapter'],init_adapter_files=job['init_adapter_files'],
        release_sha256=job['release_sha256'])


def make_args(job):
    args=base_worker.make_args(job)
    args.dense_contract=contract(job)
    if job['mode'] in ('windows_prefix','windows_bank'):
        args.lora_dropout=0.
        args.gradient_checkpointing=False
    # Generation uses a separate explicit A-F readout evaluator, not the native
    # action-phrase parser. Native loss validation is retained on fixed 160 IDs.
    args.generation_eval_steps=0
    return args


def common_record(row,mode='future'):
    action=Pool.action(row)
    return dict(mode='future',images=row.history_rgb_paths,actions=[action],offsets_seconds=[0],
                premises=[premise(row,history=mode=='future')])


def initialize(bundle,job):
    from qwen3vl_local.qwen35.adapters import validate_adapter
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    validate_adapter(job['init_adapter'])
    weights=load_peft_weights(job['init_adapter'],device='cpu',local_files_only=True)
    result=set_peft_model_state_dict(bundle.model,weights,adapter_name='default')
    if result.unexpected_keys or any('lora_' in key for key in result.missing_keys):
        raise ValueError('Base adapter does not match dense LoRA topology')
    return bundle


def install(job,pool,shape_audit=None):
    baseline.install_baseline()
    original_item=train._make_item
    original_load=train.load_model_with_lora
    original_save=train._save_adapter
    original_metadata=train._write_run_metadata
    original_append=train._append_jsonl
    original_verify=train.verify_epoch_consumption
    by_seed={}
    state={}
    def make_item(row,**kwargs):
        item=original_item(row,**kwargs)
        by_seed[item.spec.seed_key]=row
        return item
    def select(rows,args,**kwargs):
        work,audit,records=planning.make_plan(pool,args,job,**kwargs)
        state.clear();state.update(audit=audit,records=records,counts=Counter(),mass=Counter(),positions=[])
        return work,audit
    def pack(bundle,**kwargs):
        row=by_seed[kwargs['spec'].seed_key]
        from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity
        identity=baseline.digest(case_identity(train._make_item(row,seed=20260904)))
        if row.split=='train':
            record=state['records'][identity]
        else:record=common_record(row,job['mode'])
        if record is None:
            raise ValueError('Dense training eligibility changed')
        images=train._load_images(record['images'])
        packed=objectives.pack(bundle,record,images,max_length=kwargs['max_length'],max_batch_tokens=job.get('max_batch_tokens',16384))
        if shape_audit is not None:
            lengths=[p['tokens'] for p in packed] if isinstance(packed,list) else [packed['tokens']]
            shape_key=record['id'] if row.split=='train' else f'eval/{row.split}/loss/{identity}'
            if shape_audit['shapes'][shape_key]['sequence_tokens']!=lengths:
                raise ValueError('Training input changed since admitted shape audit/probe')
        (packed[0] if isinstance(packed,list) else packed)['consumption_record']=record if row.split=='train' else None
        return packed
    def loss(bundle,packed,specs):
        if len(packed)!=1:
            raise ValueError('Dense schedule uses one independent anchor per rank')
        p=packed[0]
        value,report=(objectives.window_loss(bundle,p) if isinstance(p,list)
                      else objectives.readout_loss(bundle.model,p,bundle.device))
        valid=[(a,b) for a,b,w in zip(report['predictions'],report['targets'],report['query_weights']) if b!=-100 and w>0]
        index=-1 if job['mode']!='future' and len(valid)>1 else 0
        stats=dict(denom=len(valid),token_acc=report['correct']/report['supervised'],
                   value_token_acc=report['correct']/report['supervised'],
                   choice_action_exact=float(valid[index][0]==valid[index][1]),
                   supervised_queries=report['supervised'],
                   input_tokens=report.get('visual_reuse',{}).get('language_tokens_computed',report['tokens']))
        rec=(p[0] if isinstance(p,list) else p).get('consumption_record')
        if bundle.model.training and rec is not None:
            state['pending']=(rec,report)
        return value,[stats]
    def append(path,value):
        match=re.fullmatch(r'epoch_(\d+)_rank(\d+)_batches.jsonl',Path(path).name)
        if match:
            rec,report=state.pop('pending')
            if value['ids']!=[rec['id']]:raise ValueError('Actual dense batch does not match loss record')
            value=dict(value,dense_query_event=rec['event'],dense_query_actions=rec['actions'],
                       dense_query_weights=report['query_weights'],
                       query_record_phase='backward_completed_before_optimizer_step')
            if report.get('visual_reuse'):
                value['dense_compute']={k:report['visual_reuse'][k] for k in
                    ('protocol','encoded_images','frame_language_encodings','shared_prefix_tokens',
                     'language_tokens_computed','live_cache_token_slots','image_language_states_shared')
                    if k in report['visual_reuse']}
            state['positions'].extend(value['positions'])
            for i,(action,w) in enumerate(zip(rec['actions'],report['query_weights'])):
                if w>0 and action is not None:
                    state['counts'][f'{rec["event"]}/{i}/{action}']+=1
                    state['mass'][f'{rec["event"]}/{action}']+=w
        original_append(path,value)
    def verify(work,consumed,*,device,world_size):
        result=original_verify(work,consumed,device=device,world_size=world_size)
        rank=int(os.environ.get('RANK','0'));audit=state['audit']
        expected_positions=list(range(rank,len(work),world_size))
        local=dict(counts=dict(state['counts']),mass=dict(state['mass']),
                   positions_verified=state['positions']==expected_positions)
        gathered=[None]*world_size
        if world_size>1:torch.distributed.all_gather_object(gathered,local)
        else:gathered=[local]
        counts=Counter();mass=Counter()
        for item in gathered:counts.update(item['counts']);mass.update(item['mass'])
        if (not all(x['positions_verified'] for x in gathered) or dict(counts)!=audit['planned_query_exposure']
                or any(not math.isclose(mass[k],v,rel_tol=1e-5,abs_tol=1e-4)
                       for k,v in audit['planned_query_loss_mass'].items())):
            raise ValueError('Actual post-backward query consumption differs from epoch plan')
        if rank==0:
            trial.write(Path(job['output'])/f'train/balance/query_consumed_epoch_{audit["epoch"]+1:04d}.json',
                dict(status='verified_after_backward',epoch=audit['epoch'],actual_query_exposure=dict(counts),
                     actual_query_loss_mass=dict(mass),supervised_queries=sum(counts.values()),
                     anchor_presentations=len(work),plan_sha256=baseline.digest(audit),ranks=world_size))
        return result
    def save(*args,**kwargs):
        path=original_save(*args,**kwargs)
        p=path/'sft_new_loop_phase3_adapter_config.json'
        cfg=json.loads(p.read_text())
        cfg.update(prompt_name=POLICY,prompt_variant=POLICY,dense_contract=contract(job),
                   production_prompt_sha256=baseline.digest(objectives.INSTRUCTION))
        p.write_text(json.dumps(cfg,indent=2)+'\n')
        return path
    def metadata(*args,**kwargs):
        original_metadata(*args,**kwargs)
        p=Path(job['output'])/'train/train_run_manifest.json'
        value=json.loads(p.read_text())
        value.update(dense_contract=contract(job),prompt_name=POLICY,
                     production_prompt_sha256=baseline.digest(objectives.INSTRUCTION))
        p.write_text(json.dumps(value,indent=2)+'\n')
    train._make_item=make_item
    train._contiguous_work=select
    train._build_inputs=pack
    train._loss_batch=loss
    train.load_model_with_lora=lambda *a,**kw:initialize(original_load(*a,**kw),job)
    train._save_adapter=save
    train._write_run_metadata=metadata
    train._append_jsonl=append
    train.verify_epoch_consumption=verify
    base_worker.install_reference(trial.reference())


def create_bundle(job,device):
    args=make_args(job)
    return initialize(train.load_model_with_lora(Path(job['model_dir']),device=device,
        lora_rank=args.lora_rank,lora_alpha=args.lora_alpha,lora_dropout=args.lora_dropout,
        lora_vision_scope='off',strict_vision_scope=True,gradient_checkpointing=args.gradient_checkpointing),job)


@torch.no_grad()
def evaluate(job,split='val'):
    from qwen3vl_local.sft_new_loop_phase3 import training_state
    from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity
    from . import evaluation
    root=Path(job['output'])
    pool=load_pool(job)
    eval_plan,_=evaluation.split_plan(job,pool,split)
    if trial.read(root/'dense_evaluation_plan.json')[split]!=eval_plan:
        raise ValueError('Evaluation coverage/membership changed since admission')
    input_shapes=trial.read(root/'dense_input_shapes.json')['shapes']
    # Quick checkpoint holds trainable tensors, not an inference adapter.
    bundle=create_bundle(job,torch.device('cuda:0'))
    checkpoint=root/'train/training_states'/f'epoch_{1 if job["stage"]=="quick" else 3:04d}'
    receipt=trial.read(checkpoint/'receipt.json')
    if receipt['files']!=training_state.files(checkpoint):
        raise ValueError('Evaluation checkpoint is incomplete or changed')
    state=torch.load(checkpoint/'training_state.pt',
                     map_location='cpu',weights_only=True)
    if state['binding']!=receipt['binding']:
        raise ValueError('Evaluation state/receipt binding mismatch')
    if state['binding']['configuration'].get('dense_contract')!=contract(job):
        raise ValueError('Dense evaluation checkpoint contract changed')
    params=dict(bundle.unwrap().named_parameters())
    for name,value in state['trainable'].items():
        params[name].copy_(value.to(params[name].device))
    bundle.model.eval()
    selected=evaluation.selected(pool,split)
    reports=[];correct=0;constrained_correct=0
    dense_records=[];dense_totals=Counter()
    for item in selected:
        identity=baseline.digest(case_identity(item))
        record=common_record(item.row,job['mode'])
        packed=objectives.pack(bundle,record,train._load_images(record['images']))
        if [packed['tokens']]!=input_shapes[f'eval/{split}/transfer/{identity}']['sequence_tokens']:
            raise ValueError('Transfer evaluation input changed since admission')
        _,report=objectives.readout_loss(bundle.model,packed,bundle.device)
        correct+=report['correct']
        constrained_correct+=report['constrained_correct']
        reports.append(dict(scenario=item.row.scenario,route=item.row.route_id,frame=item.row.frame_id,
            event=item.row.context_id,action=Pool.action(item.row),**report))
        dense=pool.record(item.row,job['mode'])
        if dense is None:
            dense_totals['missing_complete_segment']+=1
            continue
        predictions=objectives.predict(bundle,dense,train._load_images(dense['images']),max_batch_tokens=job.get('max_batch_tokens',16384),
            expected_tokens=input_shapes[f'eval/{split}/matched/{identity}']['sequence_tokens'])
        dense_records.append(dict(id=identity,frame=item.row.frame_id,event=item.row.context_id,
            scenario=item.row.scenario,route=item.row.route_id,frames=dense['frames'],
            targets=dense['actions'],predictions=predictions))
        for i,(target,pred) in enumerate(zip(dense['actions'],predictions)):
            prefix=f'query_{i}'
            if target is None:
                dense_totals[prefix+'/missing']+=1
            else:
                dense_totals[prefix+'/support']+=1
                dense_totals[prefix+'/correct']+=int(target==pred['action'])
                dense_totals[prefix+'/constrained_correct']+=int(target==pred['constrained_action'])
                dense_totals[f'{prefix}/{target}/support']+=1
                dense_totals[f'{prefix}/{target}/correct']+=int(target==pred['action'])
    trial.write(root/f'dense_{split}_transfer_cases.json',reports)
    trial.write(root/f'dense_{split}_cases.json',dense_records)
    matched=evaluation.score_matched(dense_records,eval_plan['primary_query'])
    trial.write(root/f'dense_{split}_summary.json',dict(**matched,contract=contract(job),
        metric_role='primary_matched_training_readout',evaluation_plan=eval_plan,
        observation=contract(job)['observation'],fixed_anchor_total=len(selected),old_models_evaluated=False))
    trial.write(root/f'dense_{split}_transfer_summary.json',dict(correct=correct,total=len(reports),exact=correct/len(reports),
        constrained_correct=constrained_correct,constrained_exact=constrained_correct/len(reports),
        contract=contract(job),metric_role='secondary_four_frame_transfer',
        comparison='not the matched dense-supervision comparison; stream single trains query 4 after eight images',
        common_observation='four frames, one current-action query; no additional past or future images',
        old_models_evaluated=False))
    trial.write(root/f'dense_{split}_sequence_metrics.json',dict(counts=dense_totals,cases=dense_records,
        selection='same frozen anchor IDs, no replacement of missing segments',
        note='overlapping targets correlated; missing native labels excluded, not imputed'))
