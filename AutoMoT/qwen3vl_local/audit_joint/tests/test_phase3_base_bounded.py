from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from qwen3vl_local.audit_joint import phase3_base_bounded as bounded
from qwen3vl_local.audit_joint import phase3_base_bounded_sampling as sampling
from qwen3vl_local.audit_joint import phase3_base_coverage as coverage, phase3_base_trial as trial
from qwen3vl_local.audit_joint import phase3_base_trial_worker as base
from qwen3vl_local.audit_joint.tests.test_phase3_base_coverage import rec


def test_scarce_only_exhausts_distinct_pool_then_balances_actual_repeats_across_epochs():
    rows=[rec(f'a{i}','a',i) for i in range(3)]+[rec(f'b{i}','b',10+i) for i in range(8)]
    exposure=Counter()
    for epoch in range(3):
        selected,p=sampling.plan(rows,per_event=4,world_size=2,exposure=exposure,seed=epoch)
        assert p==sampling.plan(list(reversed(rows)),per_event=4,world_size=2,exposure=exposure,seed=epoch)[1]
        assert p['event_counts']==dict(a=4,b=4) and p['selected']==8
        assert p['event_repeats']['a']['selected_distinct_frames']==3
        assert p['event_repeats']['a']['repeated_presentations']==1
        assert p['event_repeats']['b']['max_frame_repeat']==1
        assert p['scarce_events']==['a'] and p['repeated_presentations']==1
        if epoch==0:assert p['unseen_physical_frames']==7
        exposure.update(r['id'] for r in selected)
    assert {exposure[f'a{i}'] for i in range(3)}=={4}
    assert max(exposure[f'b{i}'] for i in range(8))==2


def test_cannot_manufacture_capacity_with_duplicate_case_ids_for_same_frame():
    rows=[rec('one','a',1),rec('alias','a',1)]
    with pytest.raises(ValueError,match='even_with_two'):sampling.plan(rows,per_event=3,world_size=1)
    _,p=sampling.plan(rows,per_event=2,world_size=1)
    assert p['event_repeats']['a']['available_distinct_frames']==1
    assert p['max_frame_repeat']==2 and p['unique']==1


def test_shared_physical_frames_never_enable_repetition_in_sufficient_events():
    rows=[rec('a1','a',1),rec('a2','a',2),rec('b1','b',1),rec('b2','b',2)]
    _,p=sampling.plan(rows,per_event=2,world_size=2)
    assert p['max_frame_repeat']==2 and p['shared_context_presentations']==2
    assert all(r['max_case_repeat']==1 for r in p['event_repeats'].values())


def test_shared_frame_rerouting_keeps_scarce_repeat_on_own_frames():
    rows=[rec('a1','a',1),rec('a2','a',2),rec('a3','a',3),rec('z1','z',1)]
    selected,p=sampling.plan(rows,per_event=2,world_size=2)
    assert p['event_repeats']['z']['repeated_presentations']==1
    assert all(r['frame']!=1 for r in selected if r['event']=='a')
    assert p['repeated_presentations']==1


def test_shared_capacity_minimum_matches_small_exhaustive_oracle():
    import itertools
    for masks in itertools.product(range(4),repeat=3):
        event_frames={e:[i for i,m in enumerate(masks) if m&bit] for e,bit in [('a',1),('b',2)]}
        if not all(event_frames.values()):continue
        for quota in (2,3):
            possibilities=[]
            for e,frames in event_frames.items():
                scarce=len(frames)<quota
                options=[]
                for values in itertools.product(range(1 if scarce else 0,3 if scarce else 2),repeat=len(frames)):
                    if sum(values)==quota:options.append(dict(zip(frames,values)))
                possibilities.append(options)
            best=None
            for left,right in itertools.product(*possibilities):
                counts=Counter(left);counts.update(right)
                if max(counts.values(),default=0)>2:continue
                repeated=2*quota-sum(n>0 for n in counts.values())
                best=repeated if best is None else min(best,repeated)
            rows=[rec(f'{e}{f}',e,f) for e,frames in event_frames.items() for f in frames]
            if best is None:
                with pytest.raises(ValueError):sampling.plan(rows,per_event=quota,world_size=1)
            else:
                _,p=sampling.plan(rows,per_event=quota,world_size=1)
                assert p['repeated_presentations']==best,(masks,quota,p)


def test_shared_capacity_never_uses_three_presentations():
    rows=[rec(f'{e}{i}',e,i) for e in 'abc' for i in (1,2)]
    with pytest.raises(ValueError,match='shared_frame_capacity_exceeded'):
        sampling.plan(rows,per_event=2,world_size=1)


def test_activation_binds_new_policy_and_restores_frozen_coverage(tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint.tests.test_phase3_sequence_trial import job
    from qwen3vl_local.sft_new_loop_phase3 import train
    monkeypatch.setattr(trial,'verified_sources',lambda:'fixture')
    before=coverage.contract,coverage.sampling,coverage.POLICY,trial.profile,train._contiguous_work
    with bounded.activate():
        j=job(tmp_path);j['coverage_contract']=bounded.contract();cfg=base.make_args(j)
        base.install_baseline()
        assert cfg.focus_balance_count==1280 and cfg.warmup_steps==2500
        assert train._contiguous_work is sampling.independent_work
        assert trial.profile()['coverage_policy']==bounded.POLICY
        assert cfg.coverage_contract['case_cap_per_epoch']==dict(sufficient=1,scarce=2)
    assert before==(coverage.contract,coverage.sampling,coverage.POLICY,trial.profile,train._contiguous_work)


@pytest.mark.parametrize('fail_optimizer',[False,True])
def test_real_native_cpu_loop_commits_repeats_only_after_success(tmp_path,monkeypatch,fail_optimizer):
    import torch
    from qwen3vl_local.sft_new_loop_phase3 import preflight,train
    from qwen3vl_local.sft_new_loop_phase3.test_entry_admission import fixture
    from qwen3vl_local.audit_joint.tests.test_phase3_sequence_trial import job
    index=fixture(tmp_path,'valid');model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    monkeypatch.setattr(trial,'verified_sources',lambda:'fixture')
    def load(*a,**kw):
        m=torch.nn.Linear(1,1)
        return SimpleNamespace(model=m,unwrap=lambda:m,lora_target_modules=['weight'])
    monkeypatch.setattr(train,'load_model_with_lora',load)
    monkeypatch.setattr(train,'_load_images',lambda *a,**kw:[])
    monkeypatch.setattr(train,'_build_inputs',lambda *a,**kw:{})
    with bounded.activate():
        base.install_baseline()
        monkeypatch.setattr(train,'_loss_batch',lambda b,p,s:(b.model(torch.ones(1,1)).square().mean(),[{}]))
        j=job(tmp_path);j.update(index=str(index),model_dir=str(model),output=str(tmp_path/'run'),
            resume=None,stop_after_epoch=None,coverage_contract=bounded.contract())
        cfg=base.make_args(j);cfg.device='cpu';cfg.focus_balance_count=2;cfg.warmup_steps=2
        cfg.prompt_variant='baseline';cfg.eval_steps=cfg.generation_eval_steps=0;cfg.save_final=False
        if fail_optimizer:
            def fail(**kw):raise RuntimeError('optimizer failed')
            monkeypatch.setattr(train,'_optimizer_step',fail)
            with pytest.raises(RuntimeError,match='optimizer failed'):train.train(cfg)
            assert not list(Path(cfg.output_dir).glob('balance/exposure_epoch_*.json'))
        else:
            train.train(cfg)
            saved=torch.load(Path(cfg.output_dir)/'training_states/epoch_0003/training_state.pt',weights_only=True)
            assert saved['global_step']==60 and saved['sampling_history']['policy']==sampling.POLICY
            assert set(saved['sampling_history']['exposure'].values())=={6}
            for epoch in range(1,4):
                p=json.loads((Path(cfg.output_dir)/f'balance/plan_epoch_{epoch:04d}.json').read_text())
                assert p['selected']==20 and p['max_frame_repeat']==2


def test_worker_wrapper_preserves_common_runtime_order():
    cmd=['python','-m','qwen3vl_local.audit_joint.phase3_common_runtime','--module','phase3_launch_worker','--','prepare','--job','j']
    out=bounded.wrap(cmd)
    assert out[:5]==['python','-m','qwen3vl_local.audit_joint.phase3_base_bounded','worker','--module']
    assert out[-3:]==['prepare','--job','j']


def test_launcher_routes_preparation_and_worker_to_same_bounded_version(tmp_path,monkeypatch):
    import sys
    from qwen3vl_local.audit_joint.tests.test_phase3_common_data import common_fixture
    from qwen3vl_local.audit_joint import phase3_launch as launch
    from qwen3vl_local.sft_new_loop_phase3 import preflight
    j,root,_=common_fixture(tmp_path,monkeypatch);seen=[]
    def ensure(target,*args,**kwargs):
        assert target==root.parent/'phase3_base_bounded_v3';seen.append('pool')
        return root/'native/frame_index.jsonl',root/'reference.json'
    monkeypatch.setattr(bounded,'ensure',ensure)
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    monkeypatch.setattr(trial,'gpu_ids',lambda *a:'0,1,2,3')
    monkeypatch.setattr(launch.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    def run(cmd,log,*args,**kwargs):
        assert 'qwen3vl_local.audit_joint.phase3_base_bounded' in cmd and 'worker' in cmd
        job=trial.read(tmp_path/'launch/run_base_bounded_v3/experiment/job.json')
        assert job['policy']==bounded.POLICY and job['profile']['optimizer_updates']==9600
        log.write_text('intentional fixture stop before model');seen.append('prepare')
        return dict(exit_code=2,log=str(log),seconds=0)
    monkeypatch.setattr(trial,'run_stage',run)
    monkeypatch.setattr(sys,'argv',['launch','--method','base','--base-budget','bounded_v3',
        '--data-root',j['data_root'],'--model-dir',j['model_dir'],'--common-data-dir',str(root),
        '--output',str(tmp_path/'launch')])
    assert launch.main()==2 and seen==['pool','prepare']


def test_saved_adapter_and_native_eval_keep_bounded_policy_and_four_images(tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_second_review import raw_fixture
    from qwen3vl_local.sft_new_loop_phase3 import train,eval as evaluate
    from qwen3vl_local.qwen35 import adapters
    j,_,_=raw_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(adapters,'save_adapter',lambda *a,**kw:None)
    bundle=SimpleNamespace(unwrap=lambda:None,lora_target_modules=['q_proj'])
    with bounded.activate():
        base.install_baseline()
        j['coverage_contract']=bounded.contract();cfg=base.make_args(j)
        cfg.training_index_sha256=trial.file_sha(j['index'])
        folder=train._save_adapter(bundle,Path(j['output'])/'train',cfg,step=9600)
        saved=evaluate._validate_action_adapter(folder,Path(j['model_dir']))
        assert saved['global_step']==9600 and saved['history_rgb_count']==4
        assert saved['sampling_policy']==sampling.POLICY
        receipt=trial.read(folder/'coverage_receipt.json')
        assert receipt['coverage_contract']==bounded.contract()
        assert receipt['files']==coverage.adapter_files(folder)
