"""Independent prepare/replay/train/evaluate/pack orchestration for ABC refinement."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from ..abc_delivery import digest,sha,write,pack,lock
from ..geometry_abc_v2 import dataset,admission,planning,objective,train
from . import runtime,prompt,regression,bindings,VERSION


def records_digest(records):
    import hashlib
    h=hashlib.sha256()
    for r in records:
        h.update(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode());h.update(b'\n')
    return h.hexdigest()


def verify_training(run,dataset_path,variant):
    from collections import Counter
    run=Path(run);completion=json.loads((run/'training_complete.json').read_text())
    if completion['total_steps']!=9600 or completion['selected_checkpoint']!='epoch_3' or completion['selection']!='final_completed_epoch':
        raise ValueError('refine_training_incomplete')
    binding=completion['binding']
    if binding['code']!=runtime.identity() or binding['contract']!=runtime.contract(variant) or binding['dataset_sha256']!=sha(dataset_path):
        raise ValueError('refine_training_binding_changed')
    cumulative=Counter();projection=json.loads((run/'plan.json').read_text())
    for epoch,p in enumerate(projection['plans'],1):
        receipt_path=run/f'epoch_{epoch}'/'receipt.json';receipt=json.loads(receipt_path.read_text())
        if (completion['checkpoint_receipts'][f'epoch_{epoch}']!=sha(receipt_path)
            or receipt['state_sha256']!=sha(receipt_path.with_name('state.pt')) or receipt['binding']!=binding
            or receipt['status']!='epoch_complete' or receipt['step']!=epoch*3200):
            raise ValueError('refine_checkpoint_evidence_changed')
        cumulative.update(p['ordered_ids'])
        actual=json.loads((run/f'epoch_{epoch}.json').read_text())
        if actual['plan']!=p or actual['result']['actual_exposure']!=dict(cumulative) or actual['result']['steps']!=3200:
            raise ValueError('refine_actual_consumption_changed')
    return completion


def prepare(parent,output):
    from .streaming import prepare as streamed_prepare
    return streamed_prepare(parent,output)


def replay(args):
    """All val members with original adapter, independent forward/cache per text."""
    from PIL import Image
    import torch
    from qwen3vl_local.sft_new_loop_phase3 import train as native
    from ..phase3_base_prompt_experiment import select_replay_device
    artifact=dataset.load(args.baseline_dataset,variant=args.variant)
    model=bindings.model_identity(args.model_dir)
    adapter=bindings.adapter_identity(args.checkpoint,model,args.variant)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    device=select_replay_device('cuda:0')
    bundle=native.load_model_with_lora(Path(args.model_dir),device=torch.device(device),
        lora_rank=16,lora_alpha=32,lora_dropout=.05,lora_vision_scope='off',strict_vision_scope=True,gradient_checkpointing=False)
    train.restore(bundle,args.checkpoint,artifact=artifact,dataset_path=args.baseline_dataset)
    from ..phase3_suite import model_snapshot
    receipt=json.loads((Path(args.checkpoint)/'receipt.json').read_text())
    if receipt['binding']['model']!={k:v['sha256'] for k,v in model_snapshot(Path(args.model_dir)).items()}:
        raise ValueError('replay_base_model_changed')
    old=[];new=[];tokens=[];images={}
    records=planning.evaluation(artifact['records'],'val')
    for r in records:
        path=Path(args.data_root or artifact['data_root'])/r['image'];images[r['image']]=sha(path)
        with Image.open(path) as image:rgb=image.convert('RGB')
        for target,candidate in ((old,False),(new,True)):
            saved=objective.messages
            try:
                objective.messages=prompt.messages if candidate else prompt.ORIGINAL_MESSAGES
                # readout supplies use_cache=False and never accepts past_key_values.
                packed=objective.pack(bundle,r,rgb)
                tokens.append(packed['tokens'])
                prediction=objective.predict(bundle,r,rgb)
                target.append(dict(record=r,prediction=prediction))
            finally:objective.messages=saved
    report=regression.compare(old,new)
    report.update(source=runtime.identity(),baseline_dataset_sha256=sha(args.baseline_dataset),
        adapter_sha256=sha(Path(args.checkpoint)/'state.pt'),image_sha256=images,
        model=model,variant=args.variant,adapter_receipt_sha256=adapter['receipt_sha256'],
        full_validation=True,processor_checked=True,max_tokens=max(tokens),fresh_KV=True)
    if bindings.model_identity(args.model_dir)!=model or bindings.adapter_identity(args.checkpoint,model,args.variant)!=adapter:
        raise ValueError('replay_model_or_adapter_changed_during_inference')
    if any(sha(Path(args.data_root or artifact['data_root'])/p)!=value for p,value in images.items()):
        raise ValueError('replay_images_changed')
    write(out/'original_cases.json',old);write(out/'candidate_cases.json',new);write(out/'regression.json',report)
    if args.received_cases:
        received=json.loads(Path(args.received_cases).read_text())
        write(out/'received_replay_difference.json',regression.compare(received,old))
    print(json.dumps(dict(status=report['status'],cases=len(old),output=str(out.resolve()))))
    return 2 if report['status']=='tradeoff_review_required' else 0


def compact(cases):
    return [[c['record']['id'],[t['token_id'] if t['valid'] else -1 for t in c['record']['targets']],
             c['prediction']['token_ids']] for c in cases]


def preserve_partial_evaluation(root,evaluation):
    """Keep failed native outputs, then allow an explicit evaluation retry."""
    import time
    root=Path(root).resolve();evaluation=Path(evaluation)
    if not evaluation.exists():return None
    if evaluation.is_symlink() or evaluation.parent.resolve()!=root or not evaluation.is_dir():
        raise ValueError('unsafe_partial_evaluation_directory')
    if (evaluation/'metrics.json').exists():raise ValueError('completed_evaluation_must_not_be_replaced')
    archive=root/'failed_evaluations'/f'{evaluation.name}_{time.time_ns()}'
    archive.parent.mkdir(parents=True,exist_ok=True);evaluation.rename(archive)
    return str(archive)


def package(root):
    root=Path(root);required={};summary={}
    state=json.loads((root/'pipeline.json').read_text())
    required.update(admission_evidence(root,state))
    for name in ('status.json','receipt.json','admission.json','members.jsonl'):
        required['data/'+name]=root/'data'/name
    required['pipeline.json']=root/'pipeline.json'
    for v in 'abc':
        completion=verify_training(root/f'train_{v}',root/'data'/'dataset.json',v)
        if completion['binding']['model']!=state['model_binding']['model']:raise ValueError('training_model_differs_from_admission')
        evaldir=root/f'eval_{v}';cases=json.loads((evaldir/'cases.json').read_text())
        indexed=regression.indexed(cases)
        with runtime.activated():artifact=dataset.load(root/'data'/'dataset.json',variant=v,check_sources=False)
        expected={r['id']:r for r in planning.evaluation(artifact['records'],'test')}
        if indexed.keys()!=expected.keys() or any(indexed[k]['record']!=r for k,r in expected.items()):
            raise ValueError('final_test_cohort_incomplete_or_changed')
        baseline=state['baseline_test'][v]
        if sha(baseline['path'])!=baseline['sha256']:raise ValueError('baseline_test_changed')
        report=regression.compare(json.loads(Path(baseline['path']).read_text()),cases)
        if report!=json.loads((root/f'regression_{v}.json').read_text()):raise ValueError('saved_regression_not_recomputed')
        if state['groups'].get(v)!=report['status']:raise ValueError('pipeline_regression_status_changed')
        with runtime.activated():
            metrics=__import__('qwen3vl_local.audit_joint.geometry_abc_v2.metrics',fromlist=['evaluate']).evaluate(cases)
        if metrics!=json.loads((evaldir/'metrics.json').read_text()):raise ValueError('test_metrics_not_recomputed')
        path=root/'compact'/f'{v}.json';write(path,dict(columns=['id','truth_masked','prediction'],rows=compact(cases)))
        required[f'{v}/cases.json']=path
        required[f'{v}/metrics.json']=evaldir/'metrics.json'
        required[f'{v}/training_complete.json']=root/f'train_{v}'/'training_complete.json'
        required[f'{v}/regression.json']=root/f'regression_{v}.json'
        summary[v]=dict(test_cases=len(indexed),full_cases_sha256=sha(evaldir/'cases.json'),
            plan_sha256=sha(root/f'train_{v}'/'plan.json'),checkpoint_sha256=sha(root/f'train_{v}'/'epoch_3'/'state.pt'))
    return pack(root/'delivery'/'abc_refine_audit.zip',required,summary=summary)


def admission_evidence(root,state):
    """Revalidate and include the checks that actually authorized long training."""
    proof=state['model_binding'];model=bindings.model_identity(proof['path'])
    if model!=proof['model']:raise ValueError('suite_base_model_changed')
    baseline=bindings.verify_file(state['baseline_dataset'])
    processor=root/'processor.json';bindings.verify_processor(processor,root/'data/dataset.json',model)
    required={'admission/processor.json':processor}
    for v in 'abc':
        adapter=state['baseline_adapters'][v]
        if bindings.adapter_identity(adapter['path'],model,v)!=adapter:raise ValueError('baseline_adapter_changed')
        evidence=state['paired_validation'][v]
        for item in evidence.values():bindings.verify_file(item)
        path=Path(evidence['regression']['path'])
        artifact=dataset.load(baseline,variant=v,check_sources=False)
        old,new,report=bindings.verify_replay(path,baseline,artifact,model,v,adapter)
        output=root/'compact'/f'{v}_paired_validation.json'
        write(output,dict(columns=['id','truth_masked','prediction'],original=compact(old),candidate=compact(new),
                          original_cases_sha256=evidence['original']['sha256'],candidate_cases_sha256=evidence['candidate']['sha256']))
        required[f'admission/{v}/paired_validation.json']=output
        required[f'admission/{v}/regression.json']=path
        probe=root/'probes'/v/'probe_complete.json'
        bindings.verify_probe(probe,root/'data/dataset.json',model,v,processor)
        required[f'admission/{v}/probe_complete.json']=probe
        required[f'admission/{v}/checkpoint_receipt.json']=probe.parent/'checkpoint/receipt.json'
        for i in range(4):required[f'admission/{v}/rank{i}.json']=probe.parent/f'rank{i}.json'
    return required


def workflow(args):
    from qwen3vl_local.action_prior.launch import prepare_run_directory,run_logged
    if args.operation=='pack' and not args.run_dir:raise ValueError('pack_requires_existing_run_dir')
    root=Path(args.run_dir).resolve(strict=True) if args.run_dir else prepare_run_directory(args.output).resolve()
    with lock(root):
        state=json.loads((root/'pipeline.json').read_text()) if (root/'pipeline.json').exists() else dict(schema=VERSION,status='checking',groups={})
        write(root/'pipeline.json',state)
        try:
            if args.operation=='all':
                if not args.baseline_runs_root:raise ValueError('all_requires_baseline_runs_root_for_frozen_adapters_and_cases')
                model=bindings.model_identity(args.model_dir)
                model_binding=dict(path=str(Path(args.model_dir).resolve()),model=model)
                adapters={v:bindings.adapter_identity(Path(args.baseline_runs_root)/f'{v}_1280/epoch_3',model,v) for v in 'abc'}
                baseline_proof=bindings.file_proof(args.baseline_dataset)
                for key,value in (('model_binding',model_binding),('baseline_adapters',adapters),('baseline_dataset',baseline_proof)):
                    if state.get(key,value)!=value:raise ValueError('frozen_suite_inputs_changed: '+key)
                    state[key]=value
                write(root/'pipeline.json',state)
                if not (root/'data').exists():prepare(args.baseline_dataset,root/'data')
                # Confirm the complete baseline test cohort BEFORE any GPU work.
                baseline_test={}
                for v in 'abc':
                    artifact=dataset.load(args.baseline_dataset,variant=v,check_sources=False)
                    baseline=Path(args.baseline_runs_root)/f'{v}_1280'/'test'/'cases.json'
                    cases=regression.indexed(json.loads(baseline.read_text()))
                    expected={r['id']:r for r in planning.evaluation(artifact['records'],'test')}
                    if cases.keys()!=expected.keys() or any(cases[k]['record']!=r for k,r in expected.items()):
                        raise ValueError('baseline_test_cohort_mismatch_before_training: '+v)
                    baseline_test[v]=dict(path=str(baseline.resolve()),sha256=sha(baseline))
                if state.get('baseline_test',baseline_test)!=baseline_test:raise ValueError('frozen_baseline_test_changed')
                state['baseline_test']=baseline_test;write(root/'pipeline.json',state)
                del artifact,expected,cases
                core={'members.jsonl':root/'data'/'members.jsonl','receipt.json':root/'data'/'receipt.json'}
                for v in 'abc':
                    baseline=Path(args.baseline_runs_root)/f'{v}_1280'/'test'/'cases.json'
                    compact_path=root/'package_drill'/f'{v}_cases.json'
                    write(compact_path,compact(json.loads(baseline.read_text())))
                    core[f'{v}/cases.json']=compact_path
                pack(root/'package_drill'/'delivery'/'size_drill.zip',core,limit=26_000_000,
                     summary={'baseline_predictions_only':True,'reserved_bytes':4_000_000})
                replay_root=Path(args.replay_root) if args.replay_root else root/'replay'
                for v in 'abc':
                    path=replay_root/v/'regression.json'
                    if not path.exists():
                        run_logged([sys.executable,'-B','-m','qwen3vl_local.audit_joint.phase3_abc_refine','replay',
                            '--baseline-dataset',args.baseline_dataset,'--variant',v,'--model-dir',args.model_dir,
                            '--checkpoint',str(Path(args.baseline_runs_root)/f'{v}_1280'/'epoch_3'),
                            '--output',str(replay_root/v)],root/'logs'/f'replay_{v}.log')
                    artifact=dataset.load(args.baseline_dataset,variant=v,check_sources=False)
                    bindings.verify_replay(path,args.baseline_dataset,artifact,model,v,adapters[v])
                    evidence={name:bindings.file_proof(file) for name,file in (
                        ('regression',path),('original',path.with_name('original_cases.json')),
                        ('candidate',path.with_name('candidate_cases.json')))}
                    previous=state.setdefault('paired_validation',{}).get(v,evidence)
                    if previous!=evidence:raise ValueError('frozen_paired_validation_changed')
                    state['paired_validation'][v]=evidence;write(root/'pipeline.json',state)
                    del artifact
                with runtime.activated():dataset.load(root/'data'/'dataset.json')
                from .preflight import processor_check
                processor_path=root/'processor.json'
                if not processor_path.exists():processor_check(root/'data'/'dataset.json',args.model_dir,processor_path)
                bindings.verify_processor(processor_path,root/'data/dataset.json',model)
                from qwen3vl_local.action_prior.launch import ensure_gpu
                os.environ.pop('ACTION_PRIOR_GPU_READY',None)
                count=ensure_gpu(4)
                if count!=4:raise ValueError('four_GPU_required')
                env=dict(os.environ,STUDY_GPU_SELECTED='1',PYTHONDONTWRITEBYTECODE='1')
                for v in 'abc':
                    probe_root=root/'probes'/v
                    if not (probe_root/'probe_complete.json').exists():
                        run_logged([sys.executable,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                            '-m','qwen3vl_local.audit_joint.phase3_abc_refine','probe','--output',str(probe_root),
                            '--dataset',str(root/'data'/'dataset.json'),'--variant',v,'--model-dir',args.model_dir,
                            '--processor-report',str(processor_path)],
                            root/'logs'/f'probe_{v}.log')
                    bindings.verify_probe(probe_root/'probe_complete.json',root/'data/dataset.json',model,v,processor_path)
                core.update(admission_evidence(root,state))
                pack(root/'package_drill/delivery/size_drill.zip',core,limit=26_000_000,
                     summary={'baseline_predictions_only':True,'all_pretraining_evidence':True,'reserved_bytes':4_000_000})
                for v in 'abc':
                    training=root/f'train_{v}';evaluation=root/f'eval_{v}'
                    common=['--dataset',str(root/'data'/'dataset.json'),'--variant',v,'--model-dir',args.model_dir]
                    if bindings.model_identity(args.model_dir)!=model:raise ValueError('suite_base_model_changed')
                    if not (training/'training_complete.json').exists():
                        state.update(status='training',active=v);write(root/'pipeline.json',state)
                        cmd=[sys.executable,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                             '-m','qwen3vl_local.audit_joint.phase3_abc_refine','train','--output',str(training),*common]
                        os.environ.update({k:env[k] for k in ('STUDY_GPU_SELECTED','PYTHONDONTWRITEBYTECODE')})
                        run_logged(cmd,root/'logs'/f'train_{v}.log')
                    completion=verify_training(training,root/'data'/'dataset.json',v)
                    if completion['binding']['model']!=model:raise ValueError('training_base_model_changed')
                    if not (evaluation/'metrics.json').exists():
                        archived=preserve_partial_evaluation(root,evaluation)
                        if archived:
                            state.setdefault('preserved_partial_evaluations',[]).append(archived)
                            write(root/'pipeline.json',state)
                        subprocess.run([sys.executable,'-B','-m','qwen3vl_local.audit_joint.phase3_abc_refine','evaluate',
                            '--output',str(evaluation),'--checkpoint',str(training/'epoch_3'),'--split','test',*common],env=env,check=True)
                    baseline=Path(args.baseline_runs_root)/f'{v}_1280'/'test'/'cases.json'
                    report=regression.compare(json.loads(baseline.read_text()),json.loads((evaluation/'cases.json').read_text()))
                    write(root/f'regression_{v}.json',report);state['groups'][v]=report['status'];write(root/'pipeline.json',state)
            state['status']='audited';write(root/'pipeline.json',state)
            receipt=package(root)
            state.update(status='completed' if all(v=='observed_cohort_protection_passed' for v in state['groups'].values())
                         else 'tradeoff_review_required',delivery=receipt);write(root/'pipeline.json',state)
            return 0 if state['status']=='completed' else 2
        except BaseException as exc:
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}');write(root/'pipeline.json',state)
            pack(root/'failure_delivery'/'diagnostic.zip',{'pipeline.json':root/'pipeline.json'},summary={'final_result':False})
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['prepare','replay','probe','train','evaluate','all','pack'])
    p.add_argument('--output',required=True);p.add_argument('--dataset');p.add_argument('--baseline-dataset')
    p.add_argument('--model-dir');p.add_argument('--checkpoint');p.add_argument('--variant',choices=list('abc'))
    p.add_argument('--data-root');p.add_argument('--received-cases');p.add_argument('--baseline-cases-root')
    p.add_argument('--baseline-runs-root');p.add_argument('--run-dir')
    p.add_argument('--processor-report');p.add_argument('--replay-root');p.add_argument('--split',choices=['val','test'],default='val')
    args=p.parse_args()
    if args.operation=='prepare':print(json.dumps(prepare(args.baseline_dataset,args.output)));return
    if args.operation=='replay':raise SystemExit(replay(args))
    if args.operation in ('all','pack'):raise SystemExit(workflow(args))
    if args.operation=='probe':
        from .preflight import probe
        probe(args);return
    if os.environ.get('STUDY_GPU_SELECTED')!='1':raise ValueError('train/evaluate_use_orchestrator_for_shared_GPU_selection')
    for key,value in dict(per_event=1280,epochs=3,world_size=4,seed=20260904,learning_rate=1e-5,
                          max_epoch_hours=10.,max_startup_hours=1.,ablations=False).items():setattr(args,key,value)
    with runtime.activated():
        if args.operation=='train':train.run(args)
        else:train.evaluate(args)


if __name__=='__main__':main()
