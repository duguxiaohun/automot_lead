"""Six-group CPU admission → GPU probes → train/test/audit/automatic delivery."""
import argparse
import json
import os
from pathlib import Path
import sys
from qwen3vl_local.audit_joint.abc_delivery import sha,write,pack,lock


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['prepare','check','probe','all','evaluate','pack','status'])
    p.add_argument('--data-root',default='lead_data');p.add_argument('--collection',default='keyframe_filter/collection_output')
    p.add_argument('--maps');p.add_argument('--data-dir',default='checkpoints/abc_trajectory_v1')
    p.add_argument('--output',default='checkpoints/action_expert_ablation/geometry_abc')
    p.add_argument('--recover-from',help='preparation-only old run; create a new run and reuse only compatible processor evidence')
    p.add_argument('--run-dir',help='explicit existing suite for probe/evaluation/pack retries')
    p.add_argument('--model-dir',default='checkpoints/Qwen3.5-4B')
    p.add_argument('--lead-bev-ckpt',default='checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--branch',choices=['both','bev_only','qwen_simple'],default=None,
                   help='new suites default to both; existing suites retain their admitted branch scope')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    from . import recovery
    if args.recover_from:
        if args.run_dir or args.operation not in ('all','check'):raise ValueError('recover_requires_new_all_or_check_run')
        recovery.inspect_old(args.recover_from)
        args.data_dir=str(recovery.data_directory(args.data_dir))
    from . import data,preflight,audit
    from qwen3vl_local.action_prior.launch import prepare_run_directory,ensure_gpu
    from qwen3vl_local.action_prior.host_memory import run_logged
    if args.operation=='status':
        path=Path(args.run_dir)/'pipeline.json' if args.run_dir else Path(args.data_dir)/'status.json'
        print(path.read_text());return
    if args.operation=='prepare':
        if not args.maps:raise ValueError('prepare_requires_explicit_verified_maps')
        result=data.prepare(args.data_root,args.collection,args.maps,args.data_dir,workers=args.workers)
        audit.drill(args.data_dir,Path(args.data_dir)/'package_drill')
        print(json.dumps(result));return
    from . import group_scope
    previous=None
    if args.run_dir and (Path(args.run_dir)/'admission.json').exists():
        previous=json.loads((Path(args.run_dir)/'admission.json').read_text())
    selected_groups=group_scope.resolve(args.branch,previous)
    destination=Path(args.run_dir or args.output).resolve()
    for source in (args.data_dir,args.data_root,args.model_dir,args.lead_bev_ckpt):
        protected=Path(source).resolve()
        if destination.is_relative_to(protected) or protected.is_relative_to(destination):
            raise ValueError('suite_output_overlaps_protected_input: '+str(protected))
    root=Path(args.run_dir).resolve(strict=True) if args.run_dir else prepare_run_directory(args.output).resolve()
    with lock(root):
        status_path=root/'pipeline.json'
        if os.environ.get('QWEN_EXECUTION_VERSION'):
            write(root/'execution_version.json',dict(directory=os.environ['QWEN_EXECUTION_VERSION'],
                origin=os.environ.get('QWEN_EXECUTION_ORIGIN'),schema='qwen_local_execution_v1'))
        state=json.loads(status_path.read_text()) if status_path.exists() else dict(schema='abc_six_group_pipeline_v1',groups={})
        try:
            if args.operation in ('all','check'):
                state.update(status='checking');write(status_path,state)
                if not (Path(args.data_dir)/'status.json').exists():
                    if not args.maps:raise ValueError('missing_prepared_dataset_and_maps')
                    data.prepare(args.data_root,args.collection,args.maps,args.data_dir,workers=args.workers)
                gate=preflight.check(args.data_dir,args.model_dir,args.lead_bev_ckpt,args.data_root,root/'admission.json',
                                     collection=args.collection,maps=args.maps)
                gate['selected_groups']=list(selected_groups)
                audit.drill(args.data_dir,root/'package_drill');gate['package_ready']=True
                if gate['resources']['ready']:
                    reused=args.recover_from and recovery.reuse_processor(args.recover_from,gate,root/'processor.json')
                    if not reused:preflight.processor_check(args.data_dir,args.model_dir,args.data_root,root/'processor.json')
                    gate['processor_ready']=True;gate['processor_sha256']=sha(root/'processor.json')
                else:gate['processor_ready']=False
                write(root/'admission.json',gate)
                if gate['processor_ready']:
                    recovery.require_current(gate,output=root/'source_mismatch.json')
                    from .probe_inputs import verify_processor
                    verify_processor(root/'processor.json',gate)
                state.update(status='cpu_admitted' if gate['processor_ready'] else 'resources_missing',admission=gate)
                write(status_path,state)
                if args.operation=='check' or not gate['processor_ready']:
                    print(json.dumps(dict(output=str(root),admission=gate),ensure_ascii=False))
                    if not gate['processor_ready']:raise SystemExit(2)
                    return
            if args.operation in ('all','probe','evaluate'):
                gate=json.loads((root/'admission.json').read_text())
                if group_scope.admitted(gate)!=selected_groups:raise ValueError('admitted_group_scope_changed')
                if not gate.get('processor_ready'):raise ValueError('complete_CPU_admission_required')
                recovery.require_current(gate,output=root/'source_mismatch.json')
                from .probe_inputs import verify_processor
                verify_processor(root/'processor.json',gate)
                if ensure_gpu(4)!=4:raise ValueError('six_group_suite_requires_four_GPUs')
                def command(group,operation,output):
                    v,branch=group.split('_',1)
                    cmd=[sys.executable,'-B','-m','torch.distributed.run','--standalone','--max_restarts=0','--nproc_per_node=4',
                        '-m','qwen3vl_local.action_expert_ablation.geometry_abc.worker','--operation',operation,
                        '--abc-variant',v,'--branch',branch,'--data-dir',str(Path(args.data_dir).resolve()),
                        '--data-root',str(Path(args.data_root).resolve()),'--output-dir',str(output),
                        '--model-dir',str(Path(args.model_dir).resolve()),'--lead-bev-ckpt',str(Path(args.lead_bev_ckpt).resolve()),
                        '--suite-root',str(root)]
                    if operation=='evaluate':cmd+=['--checkpoint',str(root/group/'best.pt')]
                    return cmd
                if args.operation in ('all','probe'):
                    # Finish every selected short probe before its suite trains.
                    for group in selected_groups:
                        output=root/'probes'/group;receipt=output/'probe_complete.json'
                        if receipt.exists():
                            if json.loads(receipt.read_text())['admission_sha256']!=sha(root/'admission.json'):
                                raise ValueError('stale_probe_use_new_suite')
                        else:
                            state.update(status='probing',active=group);write(status_path,state)
                            run_logged(command(group,'probe',output),root/'logs'/(group+'_probe.log'))
                    state.update(status='selected_probes_passed',selected_groups=list(selected_groups));write(status_path,state)
                    if args.operation=='probe':print(json.dumps(state));return
                for group in selected_groups:
                    run=root/group
                    if args.operation=='all' and not (run/'training_complete.json').exists():
                        state.update(status='training',active=group);write(status_path,state)
                        run_logged(command(group,'train',run),root/'logs'/(group+'_train.log'))
                    if not (run/'training_complete.json').exists():raise ValueError('training_not_complete: '+group)
                    if not all((run/'test'/name).exists() for name in ('metrics.json','condition_sensitivity.json')):
                        state.update(status='evaluating',active=group);write(status_path,state)
                        run_logged(command(group,'evaluate',run/'test'),root/'logs'/(group+'_test.log'))
                    state['groups'][group]='test_complete';write(status_path,state)
            state.update(status='auditing_and_packing');write(status_path,state)
            receipt=audit.build(root,args.data_dir)
            state.update(status='condition_use_review_required' if receipt.get('quality_status')=='condition_use_review_required' else 'completed',
                         delivery=receipt);write(status_path,state)
            print(json.dumps(dict(result_directory=str(root),**receipt),ensure_ascii=False))
            if state['status']=='condition_use_review_required':raise SystemExit(2)
        except BaseException as exc:
            state.update(status='failed' if not isinstance(exc,SystemExit) else state.get('status','rejected'),
                         error=f'{type(exc).__name__}: {exc}')
            write(status_path,state)
            try:recovery.failure_package(root,state)
            except Exception as packing_error:print(f'[ABC failure delivery] {packing_error}; original error retained',file=sys.stderr)
            raise


if __name__=='__main__':main()
