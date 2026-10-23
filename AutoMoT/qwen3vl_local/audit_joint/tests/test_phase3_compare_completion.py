"""Real compatibility subprocess + synthetic completed-run evidence, no GPU."""
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from qwen3vl_local.audit_joint import phase3_compare as entry
from qwen3vl_local.audit_joint import phase3_base_bounded as bounded
from qwen3vl_local.audit_joint import phase3_base_bounded_sampling as sampling
from qwen3vl_local.audit_joint import phase3_base_coverage as coverage
from qwen3vl_local.audit_joint import phase3_native_runtime as native
from qwen3vl_local.audit_joint import phase3_base_trial as trial
from qwen3vl_local.audit_joint.tests.test_phase3_base_coverage import rec
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture(scope='module')
def completed_template(tmp_path_factory):
    folder=tmp_path_factory.mktemp('base_completion')
    runtime=native.prepare(folder/'cache')
    result=subprocess.run([sys.executable,'-B','-c',
        'import json; from qwen3vl_local.audit_joint.phase3_base_bounded import contract; print(json.dumps(contract()))'],
        cwd=runtime,env=dict(os.environ,PYTHONPATH=str(runtime),PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True,text=True,check=True)
    contract=json.loads(result.stdout)
    run=folder/'template';experiment=run/'experiment';train=experiment/'train'
    write(run/'launch_workflow.json',dict(status='completed_quality_pending'))
    write(train/'train_run_manifest.json',dict(total_steps_global=9600,batch_size=1,world_size=4,
        train_work_global=12800,learning_rate_schedule=dict(total_optimizer_updates=9600,warmup_updates=2500)))
    records=[rec(f'{event}/{i}',event,n*1280+i) for n,event in enumerate(CONTEXT_IDS) for i in range(1280)]
    plans=[];exposure=Counter()
    for epoch in range(3):
        _,plan=sampling.plan(records,per_event=1280,world_size=4,
                             exposure=exposure,seed=epoch)
        plan['policy']=sampling.POLICY
        plans.append(plan);ids=plan['ordered_ids'];exposure.update(ids)
        write(train/f'balance/plan_epoch_{epoch+1:04d}.json',plan)
        write(train/f'balance/consumed_epoch_{epoch+1:04d}.json',dict(status='verified',
            actual=dict(**dict.fromkeys(CONTEXT_IDS,1280),INVALID=0)))
        write(train/f'balance/exposure_epoch_{epoch+1:04d}.json',dict(policy=sampling.POLICY,
            next_epoch=epoch+1,exposure=dict(exposure)))
        for rank in range(4):
            (train/f'epoch_{epoch+1:04d}_rank{rank}_batches.jsonl').write_text('\n'.join(
                json.dumps(dict(epoch=epoch,step=step,batch_size=1,positions=[4*step+rank],
                                ids=[ids[4*step+rank]],loss=1.)) for step in range(3200)))
    bounded.validate_plans(plans)
    write(experiment/'preflight.json',dict(coverage_contract=contract,epoch_plans=plans))
    write(train/'final/sft_new_loop_phase3_adapter_config.json',dict(global_step=9600,sampling_policy=sampling.POLICY))
    # Synthetic weight bytes: this tests integrity verification, not model quality.
    (train/'final/adapter_model.safetensors').write_bytes(b'synthetic completion fixture')
    write(train/'final/coverage_receipt.json',dict(coverage_contract=contract,index_sha256='fixture',
        files=coverage.adapter_files(train/'final'),global_step=9600))
    write(experiment/'job.json',dict(output=str(experiment),coverage_contract=contract,index_sha256='fixture'))
    return runtime,run


@pytest.fixture
def completed(tmp_path,monkeypatch,completed_template):
    runtime,template=completed_template
    run=entry.root_for(tmp_path,'base')/'run_base_bounded_v3'
    shutil.copytree(template,run)
    job=run/'experiment/job.json';state=json.loads(job.read_text());state['output']=str(job.parent);write(job,state)
    def prepare(checkpoints,*,app):
        assert app==tmp_path and checkpoints==tmp_path/'checkpoints'
        return runtime
    monkeypatch.setattr(native,'prepare',prepare)
    monkeypatch.setattr(entry,'APP',tmp_path)
    return tmp_path,run,runtime


def test_real_compatible_verification_succeeds_where_main_tree_rejects(completed):
    app,run,runtime=completed
    # Reproduce the reported fault without substituting the release SHA check.
    with pytest.raises(ValueError,match='Published trial source mismatch'):
        bounded.verify_training(json.loads((run/'experiment/job.json').read_text()))
    result=entry.verify_completion(app,'base')
    assert result['verified_runtime']==str(runtime)
    assert result['training_verification']['verified_rank_presentations']==38400
    assert result['optimizer_steps']==9600


def test_train_all_finishes_base_receipt_and_continues_other_four(completed,monkeypatch):
    app,run,_=completed
    from qwen3vl_local.audit_joint.tests.test_phase3_study_completion import completed_study
    for method in ('single','a','b','c'):
        completed_study(app,method,monkeypatch)
    monkeypatch.setattr(entry,'preflight',lambda *a:'0,1,2,3')
    monkeypatch.setattr(sys,'argv',['compare','train','all'])
    actual_run=subprocess.run;launched=[]
    def run_command(cmd,**kwargs):
        if cmd[0]=='bash':
            launched.append(cmd);return SimpleNamespace(returncode=0)
        return actual_run(cmd,**kwargs)
    monkeypatch.setattr(entry.subprocess,'run',run_command)
    assert entry.main()==0
    assert len(launched)==5
    for method in entry.METHODS:
        receipt=json.loads((app/f'checkpoints/phase3_compare/{method}_completed.json').read_text())
        assert receipt['method']==method and receipt['optimizer_steps']==9600


def test_verify_recovers_outer_receipt_without_gpu_build_or_training(completed,monkeypatch):
    app,run,_=completed
    before={str(p):trial.file_sha(p) for p in run.rglob('*') if p.is_file()}
    monkeypatch.setattr(entry,'preflight',lambda *a:pytest.fail('no GPU/model preflight'))
    monkeypatch.setattr(entry,'commands',lambda *a,**k:pytest.fail('no data or training launch'))
    monkeypatch.setattr(sys,'argv',['compare','verify','base'])
    assert entry.main()==0
    assert (app/'checkpoints/phase3_compare/base_completed.json').is_file()
    assert before=={str(p):trial.file_sha(p) for p in run.rglob('*') if p.is_file()}


@pytest.mark.parametrize('damage',['source_contract','weights','consumption','incomplete','wrong_output'])
def test_failed_base_completion_never_launches_next_arm_or_writes_receipt(completed,monkeypatch,damage):
    app,run,_=completed;experiment=run/'experiment'
    if damage in ('source_contract','wrong_output'):
        job=json.loads((experiment/'job.json').read_text())
        job['coverage_contract' if damage=='source_contract' else 'output']={} if damage=='source_contract' else str(app/'other')
        write(experiment/'job.json',job)
    elif damage=='weights':(experiment/'train/final/adapter_model.safetensors').write_bytes(b'changed')
    elif damage=='consumption':(experiment/'train/epoch_0003_rank2_batches.jsonl').write_text('')
    else:write(run/'launch_workflow.json',dict(status='failed_or_interrupted'))
    monkeypatch.setattr(entry,'preflight',lambda *a:'0,1,2,3')
    monkeypatch.setattr(sys,'argv',['compare','train','all'])
    actual_run=subprocess.run;launched=[]
    def run_command(cmd,**kwargs):
        if cmd[0]=='bash':launched.append(cmd);return SimpleNamespace(returncode=0)
        return actual_run(cmd,**kwargs)
    monkeypatch.setattr(entry.subprocess,'run',run_command)
    with pytest.raises((ValueError,subprocess.CalledProcessError)):entry.main()
    assert len(launched)==1
    assert not list((app/'checkpoints/phase3_compare').glob('*_completed.json'))


def test_changed_runtime_source_still_rejected(completed,monkeypatch,tmp_path):
    app,_,runtime=completed
    corrupt=tmp_path/'corrupt_runtime';shutil.copytree(runtime,corrupt)
    source=corrupt/'qwen3vl_local/leadmot/config.py'
    source.write_text(source.read_text()+'\n# integrity regression\n')
    monkeypatch.setattr(native,'prepare',lambda *a,**k:corrupt)
    with pytest.raises(subprocess.CalledProcessError):entry.verify_completion(app,'base')
