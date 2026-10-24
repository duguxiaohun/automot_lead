"""Prepare full validation, then replay the same completed single adapter with fresh inputs."""
import argparse
import copy
import json
import os
from pathlib import Path
from .audit import APP,sha,write,items,verify_snapshot
from . import prompt
from .identity import digest
from .paths import image_path

MODES=('original','clarified','original_history_unknown','clarified_history_unknown','original_image_black','clarified_image_black')

def guard_candidates(buckets, record, per_stratum=4):
    """Deterministic train controls: four different groups per event/pure class.

    These exposed training examples are regression controls, never holdout evidence.
    Selection uses no model prediction, and does not remove or relabel any member.
    """
    if record['split'] != 'train':return
    for target in record['targets']:
        if not target['valid'] or target['target'].count(1.) != 1:continue
        key=(record['event'],target['target'].index(1.))
        candidates=buckets.setdefault(key,{})
        group=record['physical_group'];old=candidates.get(group)
        if old is None or record['id']<old['id']:candidates[group]=record
        if len(candidates)>per_stratum:
            worst=max(candidates,key=lambda g:candidates[g]['id']);del candidates[worst]

def guard_records(buckets):
    unique={r['id']:dict(r,variant='six') for group in buckets.values() for r in group.values()}
    return sorted(unique.values(),key=lambda r:(r['event'],r['id']))

def code():
    root=Path(__file__).parent
    files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.txt','.json','.jsonl','.sh') and 'tests' not in p.parts]
    # Bind inference helpers as well as the independent rule/prompt snapshot.
    for rel in ('geometry_v1/objective.py','geometry_v1/codec.py','phase3_suite.py','phase3_completion.py','abc_delivery.py','../sft_new_loop_phase3/train.py'):
        files.append(root.parent/rel)
    return {str(p.resolve().relative_to(APP)):sha(p) for p in sorted(files)}

def require_received_source():
    source=verify_snapshot()['received_binding']['code']
    changed=[p for p,h in source.items() if not (APP/p).is_file() or sha(APP/p)!=h]
    if changed:raise ValueError('baseline_runtime_changed_use_matching_single_source: '+str(changed))

def load_ready(ready):
    r=json.loads(ready.read_text())
    if r.get('status')!='ready':raise ValueError('single_ready_required')
    ds=Path(r['dataset'])
    if sha(ds)!=r['dataset_sha256']:raise ValueError('ready_dataset_changed')
    return r,ds

def prepare(ready,run,data_root,output):
    from . import objective
    require_received_source();r,ds=load_ready(ready);before=sha(ds)
    complete=json.loads((run/'training_complete.json').read_text());receipt=json.loads((run/'epoch_3/receipt.json').read_text())
    binding=complete['binding']
    if complete.get('selection')!='final_completed_epoch' or complete['selected_checkpoint']!='epoch_3' or complete['total_steps']!=9600 or receipt['step']!=9600 or receipt['status']!='epoch_complete':raise ValueError('completed_final_single_required')
    if binding!=receipt['binding'] or binding['dataset_sha256']!=before or binding['code']!=verify_snapshot()['received_binding']['code']:raise ValueError('run_dataset_source_mismatch')
    if sha(run/'epoch_3/receipt.json')!=complete['checkpoint_receipts']['epoch_3']:raise ValueError('checkpoint_receipt_changed')
    from ..phase3_single_v3 import dataset as original_dataset
    verified=original_dataset.load(ds,variant='six',check_sources=True,expected_sha256=before)
    del verified
    records=[];guard_buckets={}
    for rec in items(ds):
        if rec['split']=='val':records.append(dict(rec,variant='six'))
        guard_candidates(guard_buckets,rec)
    records.sort(key=lambda x:(x['event'],x['id']))
    if not records or {x['id'] for x in records}!=set(binding['fixed_validation_ids']):raise ValueError('full_validation_membership_mismatch')
    guards=guard_records(guard_buckets)
    refs=[];guard_refs=[];examples=[]
    for rec in records+guards:
        image=image_path(data_root,rec['image'])
        (refs if rec['split']=='val' else guard_refs).append(dict(id=rec['id'],image=rec['image'],sha256=sha(image)))
        changed=copy.deepcopy(rec);changed['targets']=[];changed['six_audit']={};changed['sampling_mass']={}
        for v in prompt.VARIANTS:
            if objective.messages(rec,'RGB',prompt_variant=v)!=objective.messages(changed,'RGB',prompt_variant=v):raise ValueError('target_leaks_into_prompt')
        if not examples:examples=[objective.messages(rec,'CURRENT_RGB',prompt_variant=v) for v in prompt.VARIANTS]
    if sha(ds)!=before:raise ValueError('dataset_changed_during_prepare')
    output.mkdir(parents=True,exist_ok=False)
    write(output/'records.json',records);write(output/'images.json',refs);write(output/'prompt_examples.json',examples)
    write(output/'guards.json',guards);write(output/'guard_images.json',guard_refs)
    write(output/'guard_support.json',dict(scope='exposed train regression controls; not validation or independent holdout',
        strata={f'{event}/{cls}':len(group) for (event,cls),group in sorted(guard_buckets.items())}))
    m=dict(schema='single_full_val_prompt_replay_v1',dataset_sha256=before,binding=binding,
        checkpoint_receipt_sha256=sha(run/'epoch_3/receipt.json'),state_sha256=receipt['state_sha256'],
        records_sha256=sha(output/'records.json'),images_sha256=sha(output/'images.json'),sources=code(),
        guards_sha256=sha(output/'guards.json'),guard_images_sha256=sha(output/'guard_images.json'),
        guard_support_sha256=sha(output/'guard_support.json'),
        members=len(records),guard_members=len(guards),modes=MODES,training_admitted=False,model_inference=False)
    write(output/'manifest.json',m);return m

def load_plan(plan,data_root,run):
    m=json.loads((plan/'manifest.json').read_text());require_received_source()
    if m['schema']!='single_full_val_prompt_replay_v1' or m['sources']!=code():raise ValueError('plan_source_mismatch')
    for name in ('records','images','guards','guard_images','guard_support'):
        if sha(plan/(name+'.json'))!=m[name+'_sha256']:raise ValueError('plan_changed')
    records=json.loads((plan/'records.json').read_text());images=json.loads((plan/'images.json').read_text())
    if len(records)!=m['members'] or len({r['id'] for r in records})!=len(records) or set(r['id'] for r in records)!=set(m['binding']['fixed_validation_ids']):raise ValueError('plan_members_changed')
    if [r['id'] for r in records]!=[r['id'] for r in images]:raise ValueError('image_members_changed')
    for r,im in zip(records,images):
        p=image_path(data_root,im['image'])
        if r['split']!='val' or r['image']!=im['image'] or sha(p)!=im['sha256']:raise ValueError('validation_rgb_changed')
    if sha(run/'epoch_3/receipt.json')!=m['checkpoint_receipt_sha256'] or sha(run/'epoch_3/state.pt')!=m['state_sha256']:raise ValueError('single_checkpoint_changed')
    guards=json.loads((plan/'guards.json').read_text());guard_images=json.loads((plan/'guard_images.json').read_text())
    if len(guards)!=m['guard_members'] or len({r['id'] for r in guards})!=len(guards) or {r['id'] for r in guards}&{r['id'] for r in records}:raise ValueError('guard_membership_changed')
    if [r['id'] for r in guards]!=[r['id'] for r in guard_images]:raise ValueError('guard_images_changed')
    for r,im in zip(guards,guard_images):
        p=image_path(data_root,im['image'])
        if r['split']!='train' or r['image']!=im['image'] or sha(p)!=im['sha256']:raise ValueError('guard_rgb_changed')
    return m,records,images

def intervention(record,mode):
    r=copy.deepcopy(record)
    if mode.endswith('_history_unknown'):
        for h in r['observation']['history']:h.update(valid=False,token_id=-1,name='UNKNOWN')
    return r

def paired_run(bundle,records,images,data_root,stream):
    from PIL import Image
    from . import objective
    results={m:[] for m in MODES}
    for n,(r,im) in enumerate(zip(records,images),1):
        if sha(data_root/im['image'])!=im['sha256']:raise ValueError('image_changed_before_forward')
        with Image.open(data_root/im['image']) as f:rgb=f.convert('RGB')
        for mode in MODES:
            view=intervention(r,mode);image=Image.new('RGB',rgb.size) if mode.endswith('_image_black') else rgb
            # objective.predict uses use_cache=False and resets rope_deltas before and after every full forward.
            pred=objective.predict(bundle,view,image,prompt_variant=mode.split('_')[0])
            result=dict(id=r['id'],token_ids=pred['token_ids'],prediction=pred,input_sha256=digest(objective.messages(view,'CURRENT_RGB',prompt_variant=mode.split('_')[0])))
            results[mode].append(result);stream.write(json.dumps(dict(mode=mode,**result))+'\n');stream.flush()
        if n%20==0:print(f'[single refine] {n}/{len(records)} anchors x {len(MODES)} modes',flush=True)
    return results

def replay(plan,data_root,run,model,output):
    m,records,images=load_plan(plan,data_root,run)
    # Local files only; select GPU before importing the model runtime.
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    selected=os.environ.get('GPU_IDS','0')
    if not selected.isdigit():raise ValueError('GPU_IDS_must_be_one_physical_GPU')
    os.environ['CUDA_VISIBLE_DEVICES']=selected
    import torch
    from qwen3vl_local.sft_new_loop_phase3 import train as native
    from ..phase3_suite import model_snapshot
    from .regression import compare
    identity={k:v['sha256'] for k,v in model_snapshot(model).items()}
    if identity!=m['binding']['model']:raise ValueError('base_model_mismatch')
    state=torch.load(run/'epoch_3/state.pt',map_location='cpu',weights_only=True)
    if state['binding']!=m['binding'] or state['step']!=9600:raise ValueError('checkpoint_binding_mismatch')
    output.mkdir(parents=True,exist_ok=False);write(output/'started.json',dict(plan=m,model=identity,physical_gpu=selected))
    bundle=native.load_model_with_lora(model,device=torch.device('cuda:0'),lora_rank=16,lora_alpha=32,lora_dropout=.05,lora_vision_scope='off',strict_vision_scope=True,gradient_checkpointing=False)
    params={n:p for n,p in bundle.model.named_parameters() if p.requires_grad}
    if set(params)!=set(state['trainable']):raise ValueError('checkpoint_topology_mismatch')
    with torch.no_grad():
        for n,p in params.items():
            if p.shape!=state['trainable'][n].shape:raise ValueError('checkpoint_shape_mismatch')
            p.copy_(state['trainable'][n].to(p.device,p.dtype))
    del state
    with (output/'predictions.jsonl').open('x') as stream:results=paired_run(bundle,records,images,data_root,stream)
    report=compare(records,results['original'],results['clarified'])
    for v in prompt.VARIANTS:
        for mode in ('history_unknown','image_black'):write(output/(v+'_'+mode+'_review.json'),compare(records,results[v],results[v+'_'+mode]))
    guards=json.loads((plan/'guards.json').read_text());guard_images=json.loads((plan/'guard_images.json').read_text())
    guard_report=None
    if guards:
        with (output/'guard_predictions.jsonl').open('x') as stream:guard_results=paired_run(bundle,guards,guard_images,data_root,stream)
        guard_report=compare(guards,guard_results['original'],guard_results['clarified'])
        write(output/'guard_regression.json',guard_report)
    report['train_guard_status']=guard_report['status'] if guard_report else 'unavailable'
    report['validation_status']=report['status']
    if guard_report and guard_report['status']=='tradeoff_review_required':report['status']='tradeoff_review_required'
    # Recheck bound weights, plan, RGB and runtime after inference before claiming completion.
    load_plan(plan,data_root,run)
    if {k:v['sha256'] for k,v in model_snapshot(model).items()}!=identity:raise ValueError('model_changed_during_replay')
    write(output/'regression.json',report);write(output/'completed.json',dict(status='paired_inference_complete',model_inference=True,
        plan_sha256=sha(plan/'manifest.json'),results_sha256=sha(output/'predictions.jsonl'),regression_sha256=sha(output/'regression.json'),
        guard_results_sha256=sha(output/'guard_predictions.jsonl') if guards else None,
        guard_regression_sha256=sha(output/'guard_regression.json') if guards else None,training_admitted=False))
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('prepare','replay'))
    p.add_argument('--ready',type=Path,default=APP/'checkpoints/phase3_single_v3/ready_1280.json')
    p.add_argument('--run',type=Path,default=APP/'checkpoints/phase3_single_v3/runs/six_1280')
    p.add_argument('--data-root',type=Path,default=APP/'lead_data');p.add_argument('--model',type=Path,default=APP/'checkpoints/Qwen3.5-4B')
    p.add_argument('--plan',type=Path,default=APP/'checkpoints/phase3_single_refine/plan');p.add_argument('--output',type=Path)
    args=p.parse_args()
    output=args.output or (args.plan if args.operation=='prepare' else APP/'checkpoints/phase3_single_refine/replay')
    protected=[args.run.resolve(),args.data_root.resolve(),args.model.resolve()]
    if any(output.resolve().is_relative_to(x) or x.is_relative_to(output.resolve()) for x in protected):raise ValueError('output_overlaps_inputs')
    if args.operation=='prepare':r=prepare(args.ready,args.run,args.data_root,output)
    else:r=replay(args.plan,args.data_root,args.run,args.model,output)
    print(json.dumps({k:v for k,v in r.items() if k in ('status','members','model_inference','training_admitted')},ensure_ascii=False))
    return 2 if r.get('status')=='tradeoff_review_required' else 0

if __name__=='__main__':raise SystemExit(main())
