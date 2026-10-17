"""Selected B2 multi-epoch training/recovery for both modes, with fresh output.

Epochs is an absolute target, not the number of extra epochs. Uses the existing
formal trainer, validation, fresh-process reload and fixed test paths.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from .guard import no_core
from .io import file_sha
from . import phase4_first_epoch as first
from .rule_baseline import budget_check
from .training_budget import plan
from qwen3vl_local.sft_new_loop_phase4.identity import write_json,digest


def checkpoint(path,mode,epochs):
    import torch
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import check_contract
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.train import adapter_identity,restore_best
    path=Path(path).resolve(strict=True)
    run=json.loads((path/'phase4_contract.json').read_text());check_contract(run['source'])
    if run['world_size']!=4 or run['observation_contract']['rgb_mode']!=mode:
        raise ValueError('resume RGB mode/world mismatch')
    expected=dict(batch_size=2,anchor_stride=2 if mode==2 else 1,accumulation=4,epoch_samples=10240,
                  seed=20260929,lr=2e-5,max_length=8192,lora_rank=16,lora_alpha=32,vision_scope='off',
                  cap=8,require_complete_coverage=False,sampling_policy='contiguous_event_equal',max_question_repeat=None)
    expected.update(eval_batch_size=int(os.environ.get('EVAL_BATCH_SIZE','1')),
                    reuse_frozen_rgb=os.environ.get('REUSE_FROZEN_RGB','0')=='1')
    if any(run['training'].get(k, 1 if k=='eval_batch_size' else False if k=='reuse_frozen_rgb' else None)!=v for k,v in expected.items()):raise ValueError('resume differs from selected B2 training configuration')
    if run['training']['evaluation']['mode']!='rule_consistency':raise ValueError('automatic entry requires rule-consistency resume')
    receipt=path.parent/(path.name+'_checkpoint.json')
    before=file_sha(receipt);record=json.loads(receipt.read_text())
    epoch=record['epoch']
    if type(epoch) is not int or epoch<0 or path.name!=f'epoch_{epoch:03d}' or record['adapter']!=path.name:
        raise ValueError('resume epoch folder/receipt mismatch')
    if record['files']!=adapter_identity(path):raise ValueError('resume checkpoint assets changed')
    state_path=path/'training_state.pt';state_sha=file_sha(state_path)
    if record['training_state']!=dict(bytes=state_path.stat().st_size,sha256=state_sha):
        raise ValueError('resume optimizer/history bytes changed')
    state=torch.load(state_path,map_location='cpu',weights_only=True)
    next_epoch,*_=restore_best(path,state,run)
    if next_epoch!=epoch+1 or next_epoch>=epochs or state.get('sampling_history',{}).get('next_epoch')!=next_epoch:
        raise ValueError('resume epoch/history mismatch or target already completed')
    if file_sha(receipt)!=before or file_sha(state_path)!=state_sha or record['files']!=adapter_identity(path):
        raise ValueError('resume checkpoint changed during inspection')
    return dict(path=str(path),next_epoch=next_epoch,receipt_sha256=before),run


def resume_check(path,mode,epochs,view,references,data_root):
    record,run=checkpoint(path,mode,epochs)
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.data import load_dataset
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.evaluation_policy import prepare
    data,manifest=load_dataset(view,rgb_mode=mode)
    if digest(manifest)!=run['dataset']:raise ValueError('resume dataset/view differs')
    data,evaluation=prepare(data,mode='rule_consistency',references=references,rgb_mode=mode,
                         short=True,seed=20260929,data_root=data_root)
    if evaluation!=run['training']['evaluation']:raise ValueError('resume evaluation references/selection differs')
    # Validate actual history with the formal sampler before starting GPU workers.
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.batch_sampling import plan as sampling
    from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
    import torch
    state=torch.load(Path(path)/'training_state.pt',map_location='cpu',weights_only=True)
    sampling(data['train'],expected_events=EVENTS,epoch=record['next_epoch'],seed=20260929,cap=8,budget=10240,
             world_size=4,batch_size=2,anchor_stride=2 if mode==2 else 1,policy='contiguous_event_equal',
             history=state['sampling_history'],max_question_repeat=None)
    if checkpoint(path,mode,epochs)[0]!=record:raise ValueError('resume changed during data preflight')
    return record


def extended_budget(request,output,epochs):
    baseline=budget_check(request,output)
    if baseline['sufficient'] is not True:raise ValueError('first-epoch budget insufficient')
    expanded=deepcopy(request);expanded['retained_checkpoints_per_mode']=epochs
    expanded['free_bytes']=min(request['free_bytes'],shutil.disk_usage(output).free)
    expanded['components']['logs_and_reports']['upper_bound_bytes']*=epochs
    expanded['limits']='two modes; all target epochs retained, including conservative allowance for resumed epochs'
    report=plan(expanded)
    if report['sufficient'] is not True:raise ValueError('multi-epoch disk budget insufficient')
    return report


def run(prepared,view,output,data_root,model_dir,*,epochs,resume_rgb2=None,resume_rgb4=None):
    if type(epochs) is not int or epochs<1:raise ValueError('epochs must be a positive absolute total')
    if os.environ.get('REUSE_FROZEN_RGB','0') not in ('0','1') or int(os.environ.get('EVAL_BATCH_SIZE','1'))<1:
        raise ValueError('invalid EVAL_BATCH_SIZE or REUSE_FROZEN_RGB')
    prepared,view,output,data_root,model_dir=[Path(v).absolute() for v in (prepared,view,output,data_root,model_dir)]
    resumes={m:Path(v).absolute() if v is not None else None for m,v in ((2,resume_rgb2),(4,resume_rgb4))}
    cards=os.environ.get('GPU_IDS','').split(',') if os.environ.get('GPU_IDS') else []
    if cards and (len(cards)!=4 or len(set(cards))!=4 or any(not c.isdigit() for c in cards)):
        raise ValueError('specify four distinct numeric GPU_IDS')
    if any(output.resolve().is_relative_to(p.resolve()) for p in (prepared,view,data_root,model_dir,
            *(Path(p).parent for p in resumes.values() if p))):raise ValueError('output must be outside inputs and previous runs')
    output.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',epochs=epochs,stages=[],mode_configuration={},training_accepted=False,
               scope='multi-epoch execution receipt; no new GPU capacity or model-quality certification')
    def save():write_json(output/'training_workflow.json',state)
    code=2
    old={n:signal.signal(n,lambda n,f:(_ for _ in ()).throw(first.Interrupted(n))) for n in (signal.SIGINT,signal.SIGTERM)}
    try:
        save();sources=first.source_check(first.APP,first.RELEASE);write_json(output/'source_check.json',sources)
        if sources['status']!='matched':raise ValueError('source mismatch; synchronize published code')
        from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import check_contract
        descriptor=json.loads((view/'student_view.json').read_text());check_contract(descriptor['source'])
        if Path(descriptor['parent']['logical_path']).resolve()!=(prepared/'data4').resolve():raise ValueError('view parent mismatch')
        for p in (prepared/'references4/references.json',prepared/'references4/val.jsonl',prepared/'references4/test.jsonl',model_dir/'config.json'):
            if not p.is_file():raise ValueError(f'missing input: {p}')
        budget=json.loads((prepared/'first_epoch_budget.json').read_text())
        write_json(output/'launch_budget.json',extended_budget(budget,output,epochs))
        for mode,path in resumes.items():
            recovery=checkpoint(path,mode,epochs)[0] if path else None
            state['mode_configuration'][str(mode)]=dict(batch_size=2,accumulation=4,update_samples=32,
                eval_batch_size=int(os.environ.get('EVAL_BATCH_SIZE','1')),
                reuse_frozen_rgb=os.environ.get('REUSE_FROZEN_RGB','0')=='1',
                first_epoch=recovery['next_epoch'] if recovery else 0,resume=recovery)
        env=dict(os.environ,PYTHON=sys.executable,PYTHONUNBUFFERED='1',DATASET=str(view),DATA_ROOT=str(data_root),
                 MODEL_DIR=str(model_dir),OUTPUT_DIR=str(output),RUN_TAG='training',BATCH_SIZE='2',
                 P4_VALIDATED_DATA=os.environ.get('P4_VALIDATED_DATA') or str(output/'validated_data'))
        commands=[]
        commands.append(('admission',[sys.executable,'-m','qwen3vl_local.audit_joint.rule_baseline',
            '--dataset',str(view),'--teacher-references',str(prepared/'references4'),'--data-root',str(data_root),
            '--batch-size','2','--budget',str(prepared/'first_epoch_budget.json'),'--space-path',str(output),
            '--coverage-policy','existing_data_development','--output',str(output/'admission.json')],None))
        # Check BOTH recoveries before either mode can train.
        for mode,path in resumes.items():
            if path:commands.append((f'resume_check_rgb{mode}',[sys.executable,'-m','qwen3vl_local.audit_joint.phase4_training','--check-resume',str(path),
                '--rgb-mode',str(mode),'--epochs',str(epochs),'--view',str(view),'--prepared',str(prepared),
                '--data-root',str(data_root)],mode))
        for mode in (2,4):
            args=['bash','qwen3vl_local/sft_new_loop_phase4/rgb_short/run.sh','--selection-mode','rule_consistency',
                '--teacher-references',str(prepared/'references4'),'--epochs',str(epochs),'--batch-size','2',
                '--accumulation','4','--sampling-policy','contiguous_event_equal','--epoch-samples','10240','--seed','20260929']
            if resumes[mode]:args+=['--resume',str(resumes[mode])]
            commands.append((f'train_rgb{mode}',args,mode))
            folder=output/f'run_training_rgb{mode}'
            commands.append((f'test_rgb{mode}',['bash','qwen3vl_local/audit_joint/run_guarded.sh','--space-path',str(output),
                '--min-free-gib','20','--',sys.executable,'-m','qwen3vl_local.sft_new_loop_phase4.rgb_short.evaluate',
                '--dataset',str(view),'--data-root',str(data_root),'--model-dir',str(model_dir),'--rgb-mode',str(mode),
                '--adapter',str(folder/f'epoch_{epochs-1:03d}'),'--split','test','--output-dir',str(folder/'test_fixed')],mode))
        for name,command,mode in commands:
            record=dict(stage=name,status='running',log=name+'.log');state['stages'].append(record);save()
            print(f'[phase4 training] {name}: {output/record["log"]}',flush=True);started=time.monotonic()
            stage_env=dict(env,RGB_MODE=str(mode or 2))
            if name=='admission' or name.startswith('resume_check'):
                stage_env.update(CUDA_VISIBLE_DEVICES='',WORLD_SIZE='1',RANK='0',LOCAL_RANK='0')
            code=first.execute(command,output/record['log'],stage_env)
            record.update(returncode=code,seconds=time.monotonic()-started,status='finished' if code==0 else 'failed');save()
            if code:break
        state['status']='completed_execution_not_audited' if code==0 else 'failed'
    except first.Interrupted as ex:code=ex.code;state.update(status='interrupted',error=f'signal exit {code}')
    except (OSError,ValueError,KeyError,TypeError) as ex:code=2;state.update(status='failed',error=f'{type(ex).__name__}: {ex}')
    finally:
        state['returncode']=code;save()
        for n,handler in old.items():signal.signal(n,handler)
    return code


def main():
    no_core();p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--view',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--data-root',type=Path,default=Path('lead_data'))
    p.add_argument('--model-dir',type=Path,default=Path('checkpoints/Qwen3.5-4B'))
    p.add_argument('--epochs',type=int,required=True)
    p.add_argument('--resume-rgb2',type=Path);p.add_argument('--resume-rgb4',type=Path)
    p.add_argument('--check-resume',type=Path,help=argparse.SUPPRESS);p.add_argument('--rgb-mode',type=int,choices=(2,4))
    a=p.parse_args()
    if a.epochs<1:p.error('epochs must be positive')
    if a.check_resume:
        if not a.rgb_mode:p.error('check-resume requires rgb-mode')
        print(json.dumps(resume_check(a.check_resume,a.rgb_mode,a.epochs,a.view,a.prepared/'references4',a.data_root)));return 0
    if a.output is None:p.error('output is required for training')
    return run(a.prepared,a.view,a.output,a.data_root,a.model_dir,epochs=a.epochs,
               resume_rgb2=a.resume_rgb2,resume_rgb4=a.resume_rgb4)


if __name__=='__main__':raise SystemExit(main())
