"""Fresh-base ABC training, post-training regression audit and bounded delivery."""
import argparse
import json
import os
from pathlib import Path
import sys
from ..abc_delivery import sha,write,pack,lock
from ..geometry_abc_v2 import dataset,planning,train
from . import runtime,regression,bindings,VERSION


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
    # 旧 adapter 的换提示词实验不是新 LoRA 的开训准入，禁止继续加载它。
    raise ValueError('legacy_adapter_replay_disabled_use_fresh_all_and_posttraining_regression')


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
    require_fresh_state(state)
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
        path=root/'compact'/f'{v}.json';write(path,dict(columns=['id','truth_masked','prediction'],rows=compact(cases),
            baseline_rows=compact(json.loads(Path(baseline['path']).read_text())),baseline_cases_sha256=baseline['sha256']))
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
    require_fresh_state(state)
    bindings.verify_file(state['baseline_dataset'])
    processor=root/'processor.json';bindings.verify_processor(processor,root/'data/dataset.json',model)
    required={'admission/processor.json':processor}
    for v in 'abc':
        bindings.verify_file(state['baseline_test'][v])
        probe=root/'probes'/v/'probe_complete.json'
        bindings.verify_probe(probe,root/'data/dataset.json',model,v,processor)
        required[f'admission/{v}/probe_complete.json']=probe
        required[f'admission/{v}/checkpoint_receipt.json']=probe.parent/'checkpoint/receipt.json'
        for i in range(4):required[f'admission/{v}/rank{i}.json']=probe.parent/f'rank{i}.json'
    return required


def require_fresh_state(state):
    if (state.get('workflow')!=runtime.WORKFLOW or state.get('initialization')!=runtime.INITIALIZATION
        or 'baseline_adapters' in state or 'paired_validation' in state):
        raise ValueError('legacy_or_changed_workflow_use_new_output_keep_old_run_untouched')


def failure_delivery(root,state):
    """保留可诊断的小型证据，而不是只交付一份 CalledProcessError。"""
    required={'pipeline.json':root/'pipeline.json'};optional={}
    for name in ('processor.json','data/status.json','data/admission.json','data/receipt.json'):
        p=root/name
        if p.is_file():required[name]=p
    for v in 'abc':
        for name in (f'regression_{v}.json',f'probes/{v}/probe_complete.json'):
            p=root/name
            if p.is_file():required[name]=p
        # 兼容打包旧重放失败证据，仅读 JSON，不加载任何模型权重。
        replay=root/'replay'/v
        if (replay/'regression.json').is_file():
            required[f'replay/{v}/regression.json']=replay/'regression.json'
            for name in ('original_cases.json','candidate_cases.json'):
                p=replay/name
                if p.is_file():optional[f'replay/{v}/{name}']=p
    for p in sorted((root/'logs').glob('*.log')):
        with p.open('rb') as stream:
            stream.seek(max(0,p.stat().st_size-64000));tail=stream.read()
        target=root/'failure_delivery'/'log_tails'/p.name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(tail);required['log_tails/'+p.name]=target
    receipt=pack(root/'failure_delivery'/'diagnostic.zip',required,optional=optional,
        summary=dict(final_result=False,status=state['status'],error=state.get('error'),log_tail_bytes=64000))
    print(json.dumps(dict(status=state['status'],result_directory=str(root),**receipt),ensure_ascii=False),flush=True)
    return receipt


def workflow(args):
    from qwen3vl_local.action_prior.launch import prepare_run_directory,run_logged
    if args.operation=='pack' and not args.run_dir:raise ValueError('pack_requires_existing_run_dir')
    if getattr(args,'replay_root',None):raise ValueError('legacy_replay_is_not_a_training_gate')
    if getattr(args,'checkpoint',None):raise ValueError('workflow_forbids_checkpoint_initialization')
    root=Path(args.run_dir).resolve(strict=True) if args.run_dir else prepare_run_directory(args.output).resolve()
    with lock(root):
        state=json.loads((root/'pipeline.json').read_text()) if (root/'pipeline.json').exists() else dict(
            schema=VERSION,workflow=runtime.WORKFLOW,initialization=runtime.INITIALIZATION,status='checking',groups={})
        # 先拒绝旧运行，再写状态；原失败目录及其证据保持原样。
        require_fresh_state(state)
        write(root/'pipeline.json',state)
        try:
            if args.operation=='all':
                baseline_root=getattr(args,'baseline_cases_root',None) or getattr(args,'baseline_runs_root',None)
                if not baseline_root:raise ValueError('all_requires_baseline_cases_root_JSON_predictions_only')
                model=bindings.model_identity(args.model_dir)
                model_binding=dict(path=str(Path(args.model_dir).resolve()),model=model)
                baseline_proof=bindings.file_proof(args.baseline_dataset)
                for key,value in (('model_binding',model_binding),('baseline_dataset',baseline_proof)):
                    if state.get(key,value)!=value:raise ValueError('frozen_suite_inputs_changed: '+key)
                    state[key]=value
                write(root/'pipeline.json',state)
                if not (root/'data').exists():prepare(args.baseline_dataset,root/'data')
                # Confirm the complete baseline test cohort BEFORE any GPU work.
                baseline_test={}
                for v in 'abc':
                    artifact=dataset.load(args.baseline_dataset,variant=v,check_sources=False)
                    baseline=Path(baseline_root)/f'{v}_1280'/'test'/'cases.json'
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
                    baseline=Path(baseline_root)/f'{v}_1280'/'test'/'cases.json'
                    compact_path=root/'package_drill'/f'{v}_cases.json'
                    write(compact_path,compact(json.loads(baseline.read_text())))
                    core[f'{v}/cases.json']=compact_path
                pack(root/'package_drill'/'delivery'/'size_drill.zip',core,limit=26_000_000,
                     summary={'baseline_predictions_only':True,'reserved_bytes':4_000_000})
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
                os.environ.update({k:env[k] for k in ('STUDY_GPU_SELECTED','PYTHONDONTWRITEBYTECODE')})
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
                     summary={'baseline_predictions_only':True,'all_pretraining_evidence':True,'reserved_bytes':4_000_000,
                              'initialization':runtime.INITIALIZATION,'old_adapter_loaded':False})
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
                        run_logged([sys.executable,'-B','-m','qwen3vl_local.audit_joint.phase3_abc_refine','evaluate',
                            '--output',str(evaluation),'--checkpoint',str(training/'epoch_3'),'--split','test',*common],
                            root/'logs'/f'evaluate_{v}.log')
                    baseline=bindings.verify_file(state['baseline_test'][v])
                    report=regression.compare(json.loads(baseline.read_text()),json.loads((evaluation/'cases.json').read_text()))
                    write(root/f'regression_{v}.json',report);state['groups'][v]=report['status'];write(root/'pipeline.json',state)
            state['status']='completed' if len(state['groups'])==3 and all(
                v=='observed_cohort_protection_passed' for v in state['groups'].values()) else 'tradeoff_review_required'
            state.pop('error',None);state.pop('active',None);state.pop('delivery',None)
            write(root/'pipeline.json',state)
            receipt=package(root)
            state['delivery']=receipt;write(root/'pipeline.json',state)
            print(json.dumps(dict(status=state['status'],result_directory=str(root),**receipt),ensure_ascii=False),flush=True)
            return 0 if state['status']=='completed' else 2
        except BaseException as exc:
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}');write(root/'pipeline.json',state)
            try:failure_delivery(root,state)
            except Exception as pack_error:
                print(f'failure_delivery_error: {pack_error}; original error retained; results: {root}',file=sys.stderr)
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['prepare','replay','probe','train','evaluate','all','pack'])
    p.add_argument('--output',required=True);p.add_argument('--dataset');p.add_argument('--baseline-dataset')
    p.add_argument('--model-dir');p.add_argument('--checkpoint');p.add_argument('--variant',choices=list('abc'))
    p.add_argument('--data-root');p.add_argument('--received-cases');p.add_argument('--baseline-cases-root',
        help='JSON only: root containing a_1280/test/cases.json, b_1280/test/cases.json, c_1280/test/cases.json; no old weights read')
    p.add_argument('--baseline-runs-root',help='compatibility alias for --baseline-cases-root; JSON only');p.add_argument('--run-dir')
    p.add_argument('--processor-report');p.add_argument('--replay-root');p.add_argument('--split',choices=['val','test'],default='val')
    args=p.parse_args()
    if args.operation=='prepare':print(json.dumps(prepare(args.baseline_dataset,args.output)));return
    if args.operation=='replay':p.error('legacy adapter replay disabled; use all for fresh LoRA training and post-training regression')
    if args.operation in ('all','pack'):raise SystemExit(workflow(args))
    if args.operation in ('train','probe') and args.checkpoint:
        p.error('fresh training/probe forbids checkpoint initialization')
    if args.operation=='probe':
        from .preflight import probe
        probe(args);return
    if os.environ.get('STUDY_GPU_SELECTED')!='1':raise ValueError('train/evaluate_use_orchestrator_for_shared_GPU_selection')
    for key,value in dict(per_event=1280,epochs=3,world_size=4,seed=20260904,learning_rate=1e-5,
                          max_epoch_hours=10.,max_startup_hours=1.,ablations=False).items():setattr(args,key,value)
    bindings.model_identity(args.model_dir)
    with runtime.activated():
        if args.operation=='train':
            with runtime.fresh_training():train.run(args)
        else:train.evaluate(args)


if __name__=='__main__':main()
