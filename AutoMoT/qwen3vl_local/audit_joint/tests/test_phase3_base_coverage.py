from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

from qwen3vl_local.audit_joint import phase3_base_coverage as coverage
from qwen3vl_local.audit_joint import phase3_base_coverage_sampling as sampling
from qwen3vl_local.audit_joint import phase3_base_trial as trial, phase3_base_trial_worker as base
from qwen3vl_local.audit_joint import phase3_launch as launch, phase3_native_runtime as native
from qwen3vl_local.audit_joint.phase3_base_dense import baseline
from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_base_dense import rows, args
from qwen3vl_local.audit_joint.tests.test_phase3_sequence_trial import job
from qwen3vl_local.sft_new_loop_phase3 import train


def test_new_budget_is_explicit_and_restores_old_hooks(tmp_path):
    old=(trial.profile,base.make_args,baseline.independent_work,train._contiguous_work,train.SEQUENCE_POLICY)
    with coverage.activate():
        j=job(tmp_path);j['coverage_contract']=coverage.contract()
        cfg=base.make_args(j)
        assert (cfg.focus_balance_count,cfg.warmup_steps,cfg.eval_steps)==(1280,2500,3200)
        assert trial.profile()['optimizer_updates']==9600
        assert cfg.coverage_contract==coverage.contract()
        base.install_baseline()
        assert train._contiguous_work is sampling.independent_work
        assert train._loss_batch is baseline.single_loss
    assert old==(trial.profile,base.make_args,baseline.independent_work,train._contiguous_work,train.SEQUENCE_POLICY)
    assert trial.profile()['optimizer_updates']==7680


def test_sampling_three_epochs_new_frames_first_and_frame_caps(tmp_path):
    cfg=args(tmp_path);cfg.focus_balance_count=4
    history=None; seen=set()
    for epoch in range(3):
        work,report=sampling.independent_work(rows(),cfg,epoch=epoch,world_size=4,history=history)
        frames={(x.row.scenario,x.row.route_id,x.row.frame_id) for x in work}
        assert len(frames)==40 and not frames & seen
        assert report['event_counts']==dict.fromkeys(train.CONTEXT_IDS,4)
        assert report['max_frame_repeat']==1 and report['unseen_physical_frames']==40
        assert report['contiguous_priority'] is False
        assert report['hard_action_quotas'] is False
        again=sampling.independent_work(list(reversed(rows())),cfg,epoch=epoch,world_size=4,history=history)
        assert again[1]==report
        seen|=frames;history=report['next_history']
    assert sum(history['exposure'].values())==120


def test_capacity_refuses_repeat_and_foreign_history(tmp_path):
    cfg=args(tmp_path);cfg.focus_balance_count=100000
    with pytest.raises(ValueError):sampling.independent_work(rows(),cfg,epoch=0,world_size=4)
    cfg.focus_balance_count=4
    with pytest.raises(ValueError,match='Missing'):sampling.independent_work(rows(),cfg,epoch=1,world_size=4)
    with pytest.raises(ValueError,match='Foreign'):
        sampling.independent_work(rows(),cfg,epoch=1,world_size=4,history={'policy':'sequence','next_epoch':1})
    with pytest.raises(ValueError,match='train'):
        sampling.independent_work([replace(x,split='test') for x in rows()],cfg,epoch=0,world_size=4)


def rec(key,event,frame):
    return dict(id=key,event=event,frame=frame,scenario='s',route_id='r',split='train',
                physical_group='g',targets=[dict(valid=True)],sampling_mass={'current/0/A':1})


def test_shared_frame_capacity_augmenting_path():
    source=[rec('a1','a',1),rec('a2','a',2),rec('b1','b',1)]
    for seed in range(20):
        chosen,report=sampling.plan(source,per_event=1,world_size=1,seed=seed)
        assert {(x['event'],x['frame']) for x in chosen}=={('a',2),('b',1)}
    with pytest.raises(ValueError,match='capacity'):
        sampling.plan([rec('a','a',1),rec('b','b',1)],per_event=1,world_size=1)


def test_real_9600_sampling_projection_not_7680(tmp_path):
    cfg=args(tmp_path);cfg.focus_balance_count=1280
    originals={r.context_id:r for r in rows()}
    source=[replace(r,frame_id=i%32,route_id=f'{r.context_id}_Town01_route_{i//32:06d}_0')
            for r in originals.values() for i in range(3850)]
    history=None
    for epoch in range(3):
        work,report=sampling.independent_work(source,cfg,epoch=epoch,world_size=4,history=history)
        assert len(work)==12800 and report['optimizer_steps']==3200
        assert report['planned_total_steps']==9600 and report['unseen_physical_frames']==12800
        history=report['next_history']


def test_common_and_coverage_bootstrap_keeps_torchrun_boundary(tmp_path):
    from qwen3vl_local.audit_joint import phase3_common_runtime as common
    from torch.distributed.run import get_args_parser
    for name,cmd in launch.schedule('base',tmp_path):
        wrapped=coverage.wrap(common.wrap(cmd,tmp_path/'reference.json',tmp_path/'data'))
        assert 'qwen3vl_local.audit_joint.phase3_base_coverage' in wrapped
        if 'torch.distributed.run' in wrapped:
            parsed=get_args_parser().parse_args(wrapped[3:])
            assert parsed.training_script=='qwen3vl_local.audit_joint.phase3_base_coverage'
            assert parsed.nproc_per_node=='4'


def test_real_synced_runtime_original_contract_and_new_worker_help(tmp_path):
    runtime=native.prepare(tmp_path/'checkpoints')
    assert native.prepare(tmp_path/'checkpoints')==runtime
    code='''from qwen3vl_local.audit_joint import phase3_base_trial as t, phase3_base_coverage as c
assert t.verified_sources()
assert t.profile()['optimizer_updates']==7680
with c.activate():
 assert t.verified_sources()
 assert t.profile()['optimizer_updates']==9600
assert t.profile()['optimizer_updates']==7680
'''
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=runtime,
        env=dict(os.environ,PYTHONPATH=str(runtime),PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def complete_reports(tmp_path):
    root=tmp_path/'train';(root/'balance').mkdir(parents=True);(root/'final').mkdir()
    def write(p,v):p.write_text(json.dumps(v))
    write(root/'train_run_manifest.json',dict(total_steps_global=9600,batch_size=1,world_size=4,
        train_work_global=12800,learning_rate_schedule=dict(total_optimizer_updates=9600,warmup_updates=2500)))
    plans=[];exposure=Counter()
    for e in range(1,4):
        ids=[f'{e}/{i}' for i in range(12800)]
        plan=dict(ordered_ids=ids);plans.append(plan)
        write(root/f'balance/plan_epoch_{e:04d}.json',plan)
        write(root/f'balance/consumed_epoch_{e:04d}.json',dict(status='verified',actual=dict(**dict.fromkeys(train.CONTEXT_IDS,1280),INVALID=0)))
        exposure.update(ids)
        write(root/f'balance/exposure_epoch_{e:04d}.json',dict(policy=sampling.POLICY,next_epoch=e,exposure=dict(exposure)))
        for rank in range(4):
            (root/f'epoch_{e:04d}_rank{rank}_batches.jsonl').write_text('\n'.join(json.dumps(dict(
                epoch=e-1,step=i,batch_size=1,positions=[4*i+rank],ids=[ids[4*i+rank]],loss=1.)) for i in range(3200)))
    write(tmp_path/'preflight.json',dict(coverage_contract=coverage.contract(),epoch_plans=plans))
    write(root/'final/sft_new_loop_phase3_adapter_config.json',dict(global_step=9600,sampling_policy=sampling.POLICY))
    write(root/'final/coverage_receipt.json',dict(coverage_contract=coverage.contract(),
          index_sha256='fixture',files=coverage.adapter_files(root/'final'),global_step=9600))
    return dict(output=str(tmp_path),coverage_contract=coverage.contract(),index_sha256='fixture')


@pytest.mark.parametrize('damage',[None,'budget','consumption','exposure','source','weights'])
def test_completion_checks_actual_budget_and_rank_ids(tmp_path,damage):
    j=complete_reports(tmp_path)
    if damage=='budget':
        p=tmp_path/'train/train_run_manifest.json';v=json.loads(p.read_text());v['total_steps_global']=7680;p.write_text(json.dumps(v))
    if damage=='consumption':
        p=tmp_path/'train/epoch_0002_rank1_batches.jsonl';p.write_text(p.read_text().replace('2/1"','wrong"',1))
    if damage=='exposure':(tmp_path/'train/balance/exposure_epoch_0003.json').write_text('{}')
    if damage=='source':j['coverage_contract']={}
    if damage=='weights':(tmp_path/'train/final/foreign.safetensors').write_bytes(b'foreign')
    if damage:
        with pytest.raises(ValueError):coverage.verify_training(j)
    else:assert coverage.verify_training(j)['verified_rank_presentations']==38400


@pytest.mark.parametrize('fail_optimizer',[False,True])
def test_native_cpu_loop_only_commits_exposure_after_success(tmp_path,monkeypatch,fail_optimizer):
    import torch
    from qwen3vl_local.sft_new_loop_phase3 import preflight
    from qwen3vl_local.sft_new_loop_phase3.test_entry_admission import fixture
    index=fixture(tmp_path,'valid');model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    def load(*a,**kw):
        m=torch.nn.Linear(1,1)
        return SimpleNamespace(model=m,unwrap=lambda:m,lora_target_modules=['weight'])
    monkeypatch.setattr(train,'load_model_with_lora',load)
    monkeypatch.setattr(train,'_load_images',lambda *a,**kw:[])
    monkeypatch.setattr(train,'_build_inputs',lambda *a,**kw:{})
    with coverage.activate():
        base.install_baseline()
        monkeypatch.setattr(train,'_loss_batch',lambda b,p,s:(b.model(torch.ones(1,1)).square().mean(),[{}]))
        j=job(tmp_path);j.update(index=str(index),model_dir=str(model),output=str(tmp_path/'run'),
                               resume=None,stop_after_epoch=None,coverage_contract=coverage.contract())
        cfg=base.make_args(j);cfg.device='cpu';cfg.focus_balance_count=1;cfg.warmup_steps=2
        cfg.prompt_variant='baseline';cfg.eval_steps=cfg.generation_eval_steps=0;cfg.save_final=False
        if fail_optimizer:
            def fail(**kw):raise RuntimeError('optimizer failed')
            monkeypatch.setattr(train,'_optimizer_step',fail)
            with pytest.raises(RuntimeError,match='optimizer failed'):train.train(cfg)
            assert not list(Path(cfg.output_dir).glob('balance/exposure_epoch_*.json'))
        else:
            train.train(cfg)
            saved=torch.load(Path(cfg.output_dir)/'training_states/epoch_0003/training_state.pt',weights_only=True)
            assert saved['global_step']==30
            assert sum(saved['sampling_history']['exposure'].values())==30
            assert saved['sampling_history']['policy']==sampling.POLICY
            for e in range(1,4):
                r=json.loads((Path(cfg.output_dir)/f'balance/plan_epoch_{e:04d}.json').read_text())
                assert r['max_frame_repeat']==1


def test_common_reference_parent_and_worker_hook_order_and_summary(tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint.tests.test_phase3_common_data import common_fixture
    from qwen3vl_local.audit_joint import phase3_common_runtime as common, phase3_launch_worker as own
    from qwen3vl_local.sft_new_loop_phase3.prompts import prompt_spec_to_json,choice_target_action,parse_action_output
    j,root,ref=common_fixture(tmp_path,monkeypatch)
    opts=SimpleNamespace(index=root/'native/frame_index.jsonl',data_root=Path(j['data_root']),
                         model_dir=Path(j['model_dir']),resume=None,stage='complete',experiment='base')
    with common.activate(root/'reference.json',j['data_root']),coverage.activate():
        bound=trial.build_job(opts,tmp_path/'coverage')
        base.validate_job(bound);parent_args=vars(base.make_args(bound)).copy()
    with coverage.activate(),common.activate(root/'reference.json',j['data_root']):
        base.validate_job(bound);assert vars(base.make_args(bound))==parent_args
        base.install_reference(trial.reference())
        val=train._read_rows(opts.index,'val',data_root=opts.data_root)
        selected,_=train._validation_work(val,equal_events=True,history_rgb_mode='4rgb',target_per_bin=32,
                                          seed=20260904,action_output_mode='choice')
        records=[]
        for item in selected:
            r=item.row;target=choice_target_action(item.spec)
            records.append(dict(step=9600,prompt_spec=prompt_spec_to_json(item.spec),answers=r.answers,
                scenario=r.scenario,route_id=r.route_id,frame_id=r.frame_id,context_id=r.context_id,
                raw_output=target,parsed=parse_action_output(target,spec=item.spec),all_ok=True,format_valid=True))
        folder=Path(bound['output'])/'train';folder.mkdir(parents=True)
        (folder/'final_generation_val_cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        own.summarize(bound)
        result=trial.read(Path(bound['output'])/'validation_summary.json')
        assert result['optimizer_updates']==result['planned_optimizer_updates']==9600
        assert result['correct']==result['total']==320
        bad=dict(bound,coverage_contract={})
        with pytest.raises(ValueError,match='Coverage'):base.validate_job(bad)


def test_real_adapter_save_records_coverage(tmp_path,monkeypatch):
    from qwen3vl_local.qwen35 import adapters
    from qwen3vl_local.sft_new_loop_phase3 import eval as evaluate
    from qwen3vl_local.audit_joint import phase3_launch_worker as own
    monkeypatch.setattr(adapters,'save_adapter',lambda m,p:(Path(p)/'adapter_model.safetensors').write_bytes(b'test weights'))
    j=job(tmp_path);j['coverage_contract']=coverage.contract()
    with coverage.activate():
        base.install_baseline();cfg=base.make_args(j);cfg.training_index_sha256='index'
        folder=train._save_adapter(SimpleNamespace(unwrap=lambda:None,lora_target_modules=['q_proj']),
                                   tmp_path/'train',cfg,step=9600)
        receipt=trial.read(folder/'coverage_receipt.json')
        assert receipt['coverage_contract']==coverage.contract()
        assert receipt['files']==coverage.adapter_files(folder)
        saved=trial.read(folder/'sft_new_loop_phase3_adapter_config.json')
        assert saved['global_step']==9600 and saved['sampling_policy']==sampling.POLICY


@pytest.mark.parametrize('fail_stage',[None,'probe'])
def test_coordinator_runs_only_coverage_pipeline_and_stops_first_error(tmp_path,monkeypatch,fail_stage):
    from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_second_review import raw_fixture
    from qwen3vl_local.sft_new_loop_phase3 import preflight
    j,_,_=raw_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    monkeypatch.setattr(trial,'gpu_ids',lambda _: '0,1,2,3')
    monkeypatch.setattr(coverage,'verify_training',lambda _:dict(optimizer_updates=9600))
    seen=[]
    def run(command,log,env,**kw):
        seen.append(log.stem)
        assert 'qwen3vl_local.audit_joint.phase3_base_coverage' in command
        bound=trial.read(log.parent/'experiment/job.json')
        assert bound['policy']==coverage.POLICY
        assert bound['profile']['optimizer_updates']==9600
        assert bound['coverage_contract']==coverage.contract()
        return dict(exit_code=7 if log.stem==fail_stage else 0,seconds=0,log=str(log))
    monkeypatch.setattr(trial,'run_stage',run)
    out=tmp_path/'new_coverage'
    monkeypatch.setattr(sys,'argv',['launch','--method','base','--base-budget','coverage_v2',
        '--index',j['index'],'--data-root',j['data_root'],'--model-dir',j['model_dir'],'--output',str(out)])
    assert launch.main()==(7 if fail_stage else 0)
    assert seen==(['prepare','probe'] if fail_stage else ['prepare','probe','train','summarize','test'])
    assert (out/'run_base_coverage_v2/handoff/phase3_package.json').is_file()
    assert (out/'latest').is_symlink()==(fail_stage is None)
    with pytest.raises(FileExistsError):launch.main()
