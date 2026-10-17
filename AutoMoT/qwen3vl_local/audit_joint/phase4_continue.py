"""Finish a source-compatible RGB2 epoch: reload, test, then RGB4; never retrain RGB2."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

from . import phase4_first_epoch as workflow
from .io import file_sha
from .phase4_epoch_handoff import check_consumption, collect
from .python_dispatch import environment
from qwen3vl_local.sft_new_loop_phase4.identity import write_json
from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import check_contract


def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verified_reload(run, record):
    """Reuse complete reload evidence only while its inputs and weights match."""
    folder=run/'reload_check'
    if not folder.exists():return False
    report=read(folder/'checkpoint_reload.json')
    cases=rows(run/'epoch_000_cases.jsonl')
    checks=dict(status='verified',stage='complete',epoch=0,world_size=4,
        count=len(cases),expected_count=len(cases),prediction_mismatch_ids=[],
        raw_text_differences=0,checkpoint_assets_verified=True,optimizer_bytes_verified=True,
        source=record['source'],dataset=record['dataset'],
        expected_cases_sha256=file_sha(run/'epoch_000_cases.jsonl'),
        checkpoint_receipt_sha256=file_sha(run/'epoch_000_checkpoint.json'))
    if any(report.get(k)!=v for k,v in checks.items()):
        raise ValueError('existing reload is incomplete or its evidence changed')
    if rows(folder/'cases.jsonl')!=cases:
        raise ValueError('reload cases differ from saved validation')
    checkpoint=read(run/'epoch_000_checkpoint.json');adapter=run/'epoch_000'
    actual={p.relative_to(adapter).as_posix():file_sha(p) for p in sorted(adapter.rglob('*'))
            if p.is_file() and p.name!='training_state.pt'}
    state=adapter/'training_state.pt'
    if (checkpoint['epoch']!=0 or checkpoint['adapter']!='epoch_000'
            or checkpoint['files']!=actual or checkpoint['training_state']!=
                dict(bytes=state.stat().st_size,sha256=file_sha(state))):
        raise ValueError('checkpoint assets changed after successful reload')
    return True


def preflight(root):
    state=read(root/'workflow.json');run=root/'run_first_epoch_rgb2'
    if state.get('stage')!='rgb2' or state.get('command_exit_code') not in (1,2):
        raise ValueError('expected the completed-RGB2/failed-reload workflow')
    if any((root/p).exists() for p in ('continuation.json','workflow_before_continuation.json',
            'run_first_epoch_rgb4','rgb4.log',
            'run_first_epoch_rgb2/test_fixed','handoff_continued')):
        raise FileExistsError('continuation outputs already exist; preserve them for audit')
    record=read(run/'run.json');check_contract(record['source'])
    if not (run/'epoch_000/adapter_model.safetensors').is_file():
        raise ValueError('RGB2 adapter weights missing; do not silently retrain')
    if record!=read(run/'epoch_000/phase4_contract.json'):
        raise ValueError('run and checkpoint contracts differ')
    view=Path(state['dataset']);prepared=Path(state['prepared'])
    descriptor=read(view/'student_view.json');check_contract(descriptor['source'])
    if Path(descriptor['parent']['logical_path']).resolve()!=(prepared/'data4').resolve():
        raise ValueError('view belongs to a different native pool')
    if record['world_size']!=4 or record['observation_contract']['rgb_mode']!=2:
        raise ValueError('not a four-rank RGB2 run')
    check=check_consumption(rows(run/'epoch_000_samples.jsonl'),read(run/'epoch_000_sampling.json'),
                           [rows(run/f'epoch_000_rank{i}_steps.jsonl') for i in range(4)])
    if check['status']!='matched_recorded_plan':raise ValueError(check)
    expected=record['training']['evaluation']['plans']['val']['selected_ids']
    cases=rows(run/'epoch_000_cases.jsonl')
    if len(expected)!=370 or [c['id'] for c in cases]!=expected:
        raise ValueError('RGB2 validation incomplete or wrong membership')
    if not all((root/'validated_data'/name).is_file() for name in ('rgb2.json','rgb2.receipt.json')):
        raise ValueError('missing prepared RGB2 runtime snapshot; refusing implicit rebuilding')
    verified_reload(run,record)
    return state,prepared,view,run


def run(root,data_root,model_dir):
    root,data_root,model_dir=[Path(p).absolute() for p in (root,data_root,model_dir)]
    state,prepared,view,rgb2=preflight(root)
    reuse_reload=(rgb2/'reload_check').exists()
    source=workflow.source_check(workflow.APP,workflow.RELEASE)
    if source['status']!='matched':raise ValueError('current release source check failed')
    env=dict(os.environ,PYTHONUNBUFFERED='1',DATASET=str(view),DATA_ROOT=str(data_root),
        MODEL_DIR=str(model_dir),TEACHER_REFERENCES=str(prepared/'references4'),
        FIRST_EPOCH_BUDGET=str(prepared/'first_epoch_budget.json'),OUTPUT_DIR=str(root),
        RUN_TAG='first_epoch',P4_VALIDATED_DATA=str(root/'validated_data'),
        RGB_MODE='4',BATCH_SIZE='2',ACCUMULATION='4',UPDATE_SAMPLES='32',
        EVAL_BATCH_SIZE='2',REUSE_FROZEN_RGB='1',AUTOMATIC_COVERAGE_POLICY='existing_data_development')
    env.update(environment())
    evaluator=['bash','qwen3vl_local/audit_joint/run_guarded.sh','--space-path',str(root),
        '--min-free-gib','20','--','bash','qwen3vl_local/sft_new_loop_phase4/rgb_short/evaluate.sh']
    common=['--dataset',str(view),'--data-root',str(data_root),'--model-dir',str(model_dir)]
    # Validate the existing snapshot once on CPU before reserving GPUs. This
    # reuses preparation; it does not regenerate the native pool or its proofs.
    commands=[('snapshot_check',['bash','qwen3vl_local/audit_joint/run_guarded.sh',
              '--space-path',str(root),'--min-free-gib','20','--',sys.executable,
              '-m','qwen3vl_local.sft_new_loop_phase4.rgb_short.runtime_data',
              '--dataset',str(view),'--rgb-mode','2'])]
    if not reuse_reload:
        commands.append(('rgb2_reload',evaluator+['reload_check','--run',str(rgb2),*common]))
    commands.extend([('rgb2_test',evaluator+['evaluate',*common,'--adapter',str(rgb2/'epoch_000'),
                  '--rgb-mode','2','--split','test','--output-dir',str(rgb2/'test_fixed')]),
              ('rgb4',['bash','qwen3vl_local/sft_new_loop_phase4/rgb_short/run_automatic.sh'])])
    original_sha=file_sha(root/'workflow.json')
    write_json(root/'workflow_before_continuation.json',state)
    write_json(root/'continuation_source_check.json',source)
    receipt=dict(status='in_progress',original_workflow_sha256=original_sha,
        rgb2_training_reused=True,rgb2_reload_reused=reuse_reload,stages=[],training_accepted=False)
    def save():
        write_json(root/'continuation.json',receipt);write_json(root/'workflow.json',state)
    def on_signal(number,frame):raise workflow.Interrupted(number)
    previous={n:signal.signal(n,on_signal) for n in (signal.SIGINT,signal.SIGTERM)}
    code=2
    try:
        for name,command in commands:
            log='rgb4.log' if name=='rgb4' else f'continuation_{name}.log'
            stage=dict(stage=name,status='running',log=log)
            receipt['stages'].append(stage);state['stages'].append(stage)
            state.update(stage=name,status='running');save()
            print(f'[phase4 continue] {name}: {root/log}',flush=True)
            started=time.monotonic();code=workflow.execute(command,root/log,env)
            stage.update(exit_code=code,elapsed_seconds=time.monotonic()-started,
                         status='finished' if code==0 else 'failed');save()
            if code:break
    except workflow.Interrupted as exc:code=exc.code
    except (OSError,ValueError,RuntimeError) as exc:
        code=2;receipt['error']=str(exc)
    finally:
        status='completed_execution_not_audited' if code==0 else 'failed'
        receipt.update(status=status,command_exit_code=code)
        state.update(status=status,command_exit_code=code,continuation='continuation.json');save()
        for n,handler in previous.items():signal.signal(n,handler)
    packed=collect(root,root/'handoff_continued')
    print(f'[phase4 continue] exit={code}; evidence={root}/handoff_continued',flush=True)
    return code or (0 if packed['status']=='collected_not_semantically_verified' else 2)


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('lead_data'))
    p.add_argument('--model-dir',type=Path,default=Path('checkpoints/Qwen3.5-4B'))
    a=p.parse_args();workflow.no_core()
    raise SystemExit(run(a.root,a.data_root,a.model_dir))


if __name__=='__main__':main()
