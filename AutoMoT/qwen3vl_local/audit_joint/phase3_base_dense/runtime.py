"""Opt-in dense objectives over the independent event/action training schedule."""
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import torch
from . import baseline, objectives
from .data import Pool, POLICY, premise
from qwen3vl_local.sft_new_loop_phase3 import train
from qwen3vl_local.audit_joint import phase3_base_trial as trial, phase3_base_trial_worker as base_worker


def load_pool(job):
    rows=[]
    for split in ('train','val','test'):
        rows.extend(train._read_rows(Path(job['index']),split,data_root=Path(job['data_root'])))
    return Pool(rows)


def contract(job):
    return dict(policy=POLICY,mode=job['mode'],output='A-F existing vocabulary, external parallel readouts',
        current_action_in_prompt=False,ground_truth_in_input=False,
        future_offsets_seconds=[0,1,2] if job['mode']=='future' else None,
        observation='4 frames' if job['mode']=='future' else
            ('causal prefixes of 4,5,6,7,8 frames' if job['mode']=='stream' else 'five strict four-frame windows'),
        language_cache_shared=job['mode']=='stream',
        cache_scope='one optimizer forward only; no stale cache across updates',
        max_length=8192,max_pixels_per_image=442368,
        loss_normalization='mean supported query CE per independent anchor; mean anchors across ranks',
        init_adapter=job['init_adapter'],init_adapter_files=job['init_adapter_files'],
        release_sha256=job['release_sha256'])


def make_args(job):
    args=base_worker.make_args(job)
    args.dense_contract=contract(job)
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


def install(job,pool):
    baseline.install_baseline()
    original_item=train._make_item
    original_load=train.load_model_with_lora
    original_save=train._save_adapter
    original_metadata=train._write_run_metadata
    by_seed={}
    def make_item(row,**kwargs):
        item=original_item(row,**kwargs)
        by_seed[item.spec.seed_key]=row
        return item
    def select(rows,args,**kwargs):
        valid,_=pool.eligible(job['mode'],'train')
        return baseline.independent_work(valid,args,**kwargs)
    def pack(bundle,**kwargs):
        row=by_seed[kwargs['spec'].seed_key]
        record=pool.record(row,job['mode']) if row.split=='train' else common_record(row,job['mode'])
        if record is None:
            raise ValueError('Dense training eligibility changed')
        images=train._load_images(record['images'])
        return objectives.pack(bundle,record,images,max_length=kwargs['max_length'])
    def loss(bundle,packed,specs):
        if len(packed)!=1:
            raise ValueError('Dense schedule uses one independent anchor per rank')
        p=packed[0]
        value,report=(objectives.window_loss(bundle,p) if isinstance(p,list)
                      else objectives.readout_loss(bundle.model,p,bundle.device))
        valid=[(a,b) for a,b in zip(report['predictions'],report['targets']) if b!=-100]
        index=-1 if job['mode']!='future' and len(valid)>1 else 0
        stats=dict(denom=len(valid),token_acc=report['correct']/report['supervised'],
                   value_token_acc=report['correct']/report['supervised'],
                   choice_action_exact=float(valid[index][0]==valid[index][1]),
                   supervised_queries=report['supervised'],input_tokens=report['tokens'])
        return value,[stats]
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
    base_worker.install_reference(trial.reference())


def create_bundle(job,device):
    args=make_args(job)
    return initialize(train.load_model_with_lora(Path(job['model_dir']),device=device,
        lora_rank=args.lora_rank,lora_alpha=args.lora_alpha,lora_dropout=args.lora_dropout,
        lora_vision_scope='off',strict_vision_scope=True,gradient_checkpointing=True),job)


@torch.no_grad()
def evaluate(job,split='val'):
    from qwen3vl_local.sft_new_loop_phase3 import training_state
    root=Path(job['output'])
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
    rows=train._read_rows(Path(job['index']),split,data_root=Path(job['data_root']))
    work=[train._make_item(r,seed=20260904,action_output_mode='choice') for r in rows]
    work=[x for x in work if train.choice_rejection_reason(x.spec) is None]
    selected,_=trial.select_reference(work,trial.reference(),seed=20260904,limit=32 if split=='val' else 0,split=split)
    reports=[];correct=0;constrained_correct=0
    pool=load_pool(job);dense_records=[];dense_totals=Counter()
    for item in selected:
        record=common_record(item.row,job['mode'])
        packed=objectives.pack(bundle,record,train._load_images(record['images']))
        _,report=objectives.readout_loss(bundle.model,packed,bundle.device)
        correct+=report['correct']
        constrained_correct+=report['constrained_correct']
        reports.append(dict(scenario=item.row.scenario,route=item.row.route_id,frame=item.row.frame_id,
            event=item.row.context_id,action=Pool.action(item.row),**report))
        dense=pool.record(item.row,job['mode'])
        if dense is None:
            dense_totals['missing_complete_segment']+=1
            continue
        predictions=objectives.predict(bundle,dense,train._load_images(dense['images']))
        dense_records.append(dict(frame=item.row.frame_id,event=item.row.context_id,
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
    trial.write(root/f'dense_{split}_cases.json',reports)
    trial.write(root/f'dense_{split}_summary.json',dict(correct=correct,total=len(reports),exact=correct/len(reports),
        constrained_correct=constrained_correct,constrained_exact=constrained_correct/len(reports),
        contract=contract(job),comparison='fixed historical identities, new unified prompt and A-F readout',
        common_observation='four frames, one current-action query; no additional past or future images',
        old_models_evaluated=False))
    trial.write(root/f'dense_{split}_sequence_metrics.json',dict(counts=dense_totals,cases=dense_records,
        selection='same frozen anchor IDs, no replacement of missing segments',
        note='overlapping targets correlated; missing native labels excluded, not imputed'))
