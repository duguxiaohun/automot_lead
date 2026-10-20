"""Budget, immutable data reuse, real samplers and legacy-action isolation."""
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from qwen3vl_local.sft_new_loop_phase1 import train as p1
from qwen3vl_local.sft_new_loop_phase2 import train as p2
from qwen3vl_local.sft_new_loop_phase2.test_direct_event_contract import _balance_rows
from qwen3vl_local.sft_new_loop_phase1.training_policy import (
    ActionHints, RuntimeBudget, UNKNOWN, _action, budget_plan, check_runtime,
    identity, rebalance_actions, signature,
)
from qwen3vl_local.sft_new_loop_phase1.local_assets import ensure_data, inspect_index, bind_data


def args(**kwargs):
    return SimpleNamespace(**(dict(num_epochs=3, max_steps=0, grad_accum=1, warmup_steps=-1,
        epoch_step_limit=4096, total_step_limit=12288, epoch_hours_limit=8., budget_probe_steps=100,
        action_balance_strength=.5, action_index='off') | kwargs))


def phase1_rows(n=256):
    rows=[]
    for rs in ('R1','R2','R3','R4','R5'):
        for i in range(n):
            answers={k: bool((i >> j) & 1) for j,k in enumerate(p1.PHASE1_ANSWER_KEYS)}
            answers.update({k: rs == 'R'+k[2:] for k in p1.PHASE2_ANSWER_KEYS})
            rows.append(p1.FrameRow('fixture',f'{rs}_route_{i//16}','Town01',i,rs,'event','train',
                ['0.jpg','1.jpg','2.jpg','3.jpg'],'3.jpg',answers))
    return rows


def phase2_rows(n=128):
    return [replace(row, split='train', frame_id=i,
        ue3_subtype=p2.HIGHWAY_UE3_SUBTYPE if row.event=='UE3' and i%8==0 else '')
        for row in _balance_rows(p2) for i in range(n)]


@pytest.mark.parametrize('samples,reference,steps,total,warmup',[(12288,110592,3072,9216,167),(10752,10752,2688,8064,1500)])
def test_default_budgets(samples,reference,steps,total,warmup):
    plan=budget_plan(args(warmup_reference_steps=reference),samples,4)
    assert (plan['steps_per_epoch'],plan['total_steps'],plan['warmup_updates'])==(steps,total,warmup)


@pytest.mark.parametrize('world,kwargs',[(4,{}),(1,{}),(4,{'grad_accum':32})])
def test_old_110592_budget_rejected(world,kwargs):
    with pytest.raises(ValueError,match='BEFORE model'):
        budget_plan(args(**kwargs),147456,world)


@pytest.mark.parametrize('kwargs', [dict(warmup_steps=7000),dict(epoch_hours_limit=float('nan')),
    dict(action_balance_strength=2),dict(grad_accum=0),dict(epoch_step_limit=0)])
def test_bad_budget_rejected(kwargs):
    with pytest.raises(ValueError): budget_plan(args(**kwargs),8192,4)


def test_clock_estimate_and_preserved_adapter_stop(tmp_path,monkeypatch):
    plan=budget_plan(args(),8192,4)
    guard=RuntimeBudget(plan,100)
    monkeypatch.setattr('qwen3vl_local.sft_new_loop_phase1.training_policy.time.monotonic',lambda:guard.started+1500)
    assert not guard.report(99)['exceeded']
    assert guard.report(100)['exceeded']
    with pytest.raises(RuntimeError,match='NOT an optimizer resume'):
        check_runtime(guard,100,rank=0,distributed=None,device='cpu',output_dir=tmp_path,
                      save_adapter=lambda:(tmp_path/'budget_stop_adapter').write_text('saved'))
    assert (tmp_path/'budget_stop_adapter').read_text()=='saved'
    assert json.loads((tmp_path/'budget_stop.json').read_text())['resumable_optimizer'] is False


def test_guard_checks_first_applied_step_after_probe_not_lcm(tmp_path,monkeypatch):
    guard=RuntimeBudget(budget_plan(args(),8192,4),100)
    monkeypatch.setattr('qwen3vl_local.sft_new_loop_phase1.training_policy.time.monotonic',lambda:guard.started+500)
    check_runtime(guard,102,rank=0,distributed=None,device='cpu',output_dir=tmp_path,save_adapter=lambda:None)
    assert json.loads((tmp_path/'training_eta.json').read_text())['step']==102


def test_action_loader_train_only_conflict_unknown_and_sha(tmp_path):
    rows=phase1_rows(16)[:5]
    data=[dict(scenario=r.scenario,route_id=r.route_id,frame_id=r.frame_id,split='train',primary_action='STOP') for r in rows]
    data[1]['primary_action']='bogus'
    data += [dict(data[2],primary_action='RESUME')]
    # A different physical route in val must not affect train rows.
    data += [dict(data[0],route_id='heldout',split='val')]
    path=tmp_path/'actions.jsonl'; path.write_text(''.join(json.dumps(x)+'\n' for x in data))
    hints=ActionHints.load(str(path),rows,tmp_path)
    assert hints.label(rows[0])=='STOP'
    assert hints.label(rows[1])==hints.label(rows[2])==UNKNOWN
    assert len(hints.receipt['sha256'])==64
    # Any frame of the same physical recording in heldout excludes its hints.
    data += [dict(data[0],frame_id=999,split='test')]
    path.write_text(''.join(json.dumps(x)+'\n' for x in data))
    assert not ActionHints.load(str(path),rows,tmp_path).values


def test_missing_explicit_no_fallback_or_relabel(tmp_path):
    assert ActionHints.load('auto',[],tmp_path).receipt['status']=='unavailable'
    with pytest.raises(ValueError,match='no generation or fallback'):
        ActionHints.load(str(tmp_path/'missing'),[],tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_legacy_answers_missing_is_unknown_not_keep():
    assert _action({'answers':{}})==UNKNOWN
    answers={k:False for k in ('STOP','LANE_CHANGE_LEFT','LANE_CHANGE_RIGHT','DECELERATE','RESUME')}
    assert _action({'answers':answers})=='KEEP'
    assert _action({'answers':dict(answers, STOP=True,RESUME=True)})==UNKNOWN
    assert _action({'answers':answers,'invalid_action_context':True})==UNKNOWN


def test_soft_actions_change_selection_without_growing_or_relabelling():
    row=phase1_rows(16)[0]
    rows=[replace(row,frame_id=i) for i in range(120)]
    work=[p1._make_all_item(r,'RS1',seed=i) for i,r in enumerate(rows[:80])]
    hints=ActionHints({identity(r): 'STOP' if i<100 else 'RESUME' for i,r in enumerate(rows)})
    result,report=rebalance_actions(work,rows,hints,seed=42)
    assert len(result)==len(work)==80
    assert report['after']['RESUME']>0 and report['after']['RESUME']<40
    assert [signature(x.row) for x in result]==[signature(x.row) for x in work]
    assert [x.spec for x in result]==[x.spec for x in work]
    assert report['max_frame_repeat']<=10
    assert result==rebalance_actions(work,rows,hints,seed=42)[0]
    assert work==rebalance_actions(work,rows,ActionHints(),seed=42)[0]


def test_partial_coverage_does_not_replace_unknown_slots():
    row=phase1_rows(16)[0]
    rows=[replace(row,frame_id=i) for i in range(50)]
    work=[p1._make_all_item(r,'RS1',seed=i) for i,r in enumerate(rows)]
    hints=ActionHints({identity(r): 'STOP' if i<5 else 'RESUME' for i,r in enumerate(rows[:10])})
    result,report=rebalance_actions(work,rows,hints,seed=7)
    assert result[10:]==work[10:]
    assert report['before'][UNKNOWN]==report['after'][UNKNOWN]==40


@pytest.mark.parametrize("sampling_seed", [20260810, 21260813, 22260816])
def test_legacy_phase1_sampler_quota_and_action_invariants(tmp_path, sampling_seed):
    rows=phase1_rows()
    work=p1._balanced_work(rows,target_per_bin=576,seed=sampling_seed,variant_weights=p1.TRAIN_VARIANT_WEIGHTS)
    assert len(work)==9216
    assert set(Counter(x.balance_key for x in work).values())=={576}
    hints=ActionHints({identity(r): 'STOP' if r.frame_id%16<14 else 'RESUME' for r in rows})
    balanced,report=rebalance_actions(work,rows,hints,seed=8)
    assert Counter((x.balance_key,x.augment_balance_key,x.spec.variant) for x in work)==Counter((x.balance_key,x.augment_balance_key,x.spec.variant) for x in balanced)
    assert Counter(tuple(sorted(x.row.answers.items())) for x in work)==Counter(tuple(sorted(x.row.answers.items())) for x in balanced)
    p1._assert_repeat_limit(balanced,max_repeat=10,context='test')
    for world in (1,4):
        shards=[p1._split_work_for_rank(balanced,rank=rank,world_size=world) for rank in range(world)]
        assert sum(map(len,shards))==9216


def test_legacy_phase2_preserves_events_highway_invalid_and_routes():
    rows=phase2_rows()
    work=p2._balanced_work(rows,target_per_bin=1024,seed=7,regular_multiplier=2.,invalid_multiplier=1.,route_diverse=True)
    assert Counter(p2._target_class(x.row) for x in work)==dict(UE1=1024,UE3=1024,UE5=1024,UE6=1024,RE=2048,INVALID=1024)
    hints=ActionHints({identity(r): 'STOP' if r.frame_id%16<14 else 'RESUME' for r in rows})
    balanced,_=rebalance_actions(work,rows,hints,seed=8)
    assert Counter(signature(x.row) for x in balanced)==Counter(signature(x.row) for x in work)
    assert p2.invalid_subgroup_report(balanced)==p2.invalid_subgroup_report(work)
    assert p2.route_diversity_report(balanced)==p2.route_diversity_report(work)


def write_dataset(path,phase):
    path.mkdir()
    module=p1 if phase==1 else p2
    rows=phase1_rows(16) if phase==1 else phase2_rows(8)
    records=[dict(asdict(row),dataset_name=module.DATASET_NAME,split=split) for split in ('train','val','test') for row in rows]
    (path/'frame_index.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    (path/'manifest.json').write_text(json.dumps(dict(dataset_name=module.DATASET_NAME,
        format='sft_new_loop_phase2_frame_index_v2_highway_ue3' if phase==2 else 'sft_new_loop_phase1_frame_index',
        counts={f'frames/{s}':len(rows) for s in ('train','val','test')})))
    return path/'frame_index.jsonl'


@pytest.mark.parametrize('phase',[1,2])
def test_fixed_dataset_reuse_bind_and_no_overwrite(tmp_path,phase):
    index=write_dataset(tmp_path/'old_timestamp',phase)
    before=index.read_bytes()
    for flag in ('auto','0','1'):
        assert ensure_data(index,phase,flag)[0]=='reuse'
    bind_data(index.parent,tmp_path/'method',phase)
    assert inspect_index(tmp_path/'method/frame_index.jsonl',phase)['index_sha256']==inspect_index(index,phase)['index_sha256']
    with pytest.raises(ValueError,match='already exists'): bind_data(index.parent,tmp_path/'method',phase)
    assert index.read_bytes()==before
    module=__import__(f'qwen3vl_local.sft_new_loop_phase{phase}.build_dataset',fromlist=['build_dataset'])
    with pytest.raises(ValueError,match='already exists'): module.build_dataset(SimpleNamespace(output_dir=str(index.parent)))


def test_incomplete_and_bad_dataset_never_rebuilt(tmp_path):
    with pytest.raises(ValueError,match='RUN_BUILD=0'): ensure_data(tmp_path/'missing/frame_index.jsonl',1,'0')
    assert ensure_data(tmp_path/'missing/frame_index.jsonl',1,'auto')[0]=='build'
    path=tmp_path/'bad'; path.mkdir(); (path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='incomplete'): ensure_data(path/'frame_index.jsonl',1,'1')


def test_cli_rejects_old_budget_without_reading_index_or_loading_model(tmp_path):
    command=[sys.executable,str(Path(p1.__file__)), '--plan-only','--focus-balance-count','9216',
             '--index',str(tmp_path/'does_not_exist'), '--output-dir',str(tmp_path/'out')]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode!=0 and 'BEFORE model loading' in result.stderr
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('phase',[1,2])
def test_real_pipeline_cpu_plan_reuses_data_and_rejects_second_run(tmp_path,phase):
    import os
    import shutil
    source=Path(p1.__file__).resolve().parents[1]
    root=tmp_path/'AutoMoT'
    package=root/'qwen3vl_local';package.mkdir(parents=True)
    for child in source.iterdir():
        if child.name in ('sft_new_loop_phase1','sft_new_loop_phase2'):
            shutil.copytree(child,package/child.name,ignore=shutil.ignore_patterns('__pycache__','probe_output','.pytest_cache'))
        else:
            (package/child.name).symlink_to(child,target_is_directory=child.is_dir())
    data=root/'data';data.mkdir()
    index=write_dataset(data/'fixed',phase)
    # Full bounded quota needs enough distinct frames for the repeat ceiling.
    if phase in (1,2):
        rows=phase1_rows(512) if phase==1 else phase2_rows(1024)
        records=[dict(asdict(r),dataset_name=(p1 if phase==1 else p2).DATASET_NAME,split=split) for split in ('train','val','test') for r in rows]
        index.write_text(''.join(json.dumps(x)+'\n' for x in records))
        m=json.loads((index.parent/'manifest.json').read_text());m['counts']={f'frames/{s}':len(rows) for s in ('train','val','test')}
        (index.parent/'manifest.json').write_text(json.dumps(m))
    before={x.name:x.read_bytes() for x in index.parent.iterdir()}
    env=os.environ.copy();env.update(PATH=str(Path(sys.executable).parent)+':'+env['PATH'],
        INDEX=str(index),DATA_OUTPUT_DIR=str(index.parent),GPU_IDS='0,1,2,3',
        RUN_BUILD='1',PLAN_ONLY='1',ACTION_INDEX='off',HISTORY_RGB_MODES='2rgb_endpoints')
    command=['bash',str(package/f'sft_new_loop_phase{phase}/run_full_pipeline.sh')]
    result=subprocess.run(command,env=env,cwd=root,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert '[dataset] reuse ' in result.stdout
    assert 'starting optimizer' not in result.stdout and '[training-budget]' in result.stdout
    plans=list(root.glob('checkpoints/**/training_budget.json'))
    assert len(plans)==1
    assert not list(root.glob('checkpoints/sft_new_loop_phase*_runs/**/latest'))
    plan=json.loads(plans[0].read_text())
    assert (plan['steps_per_epoch'],plan['total_steps'])==((3072,9216) if phase==1 else (2688,8064))
    assert {x.name:x.read_bytes() for x in index.parent.iterdir()}==before
    second=subprocess.run(command,env=env,cwd=root,capture_output=True,text=True,timeout=10)
    assert second.returncode!=0 and 'Output already exists' in second.stderr
    assert {x.name:x.read_bytes() for x in index.parent.iterdir()}==before


@pytest.mark.parametrize('phase',[1,2])
@pytest.mark.parametrize('max_steps',[5,0])
def test_real_training_loop_cpu_optimizer_and_time_guard(tmp_path,monkeypatch,phase,max_steps):
    import torch
    module=p1 if phase==1 else p2
    monkeypatch.setattr(sys,'argv',['train','--output-dir',str(tmp_path/'run'),'--device','cpu',
        '--focus-balance-count','6','--max-steps',str(max_steps),'--grad-accum','2',
        '--eval-steps','0','--generation-eval-steps','0','--save-steps','0',
        '--action-index','off','--no-tb','--budget-probe-steps','2'])
    config=module.parse_args()
    rows=phase1_rows(32) if phase==1 else phase2_rows(8)
    monkeypatch.setattr(module,'_read_rows',lambda *a,**k:rows)
    monkeypatch.setattr(module,'_load_images',lambda *a:[None])
    build_calls=[]
    def pack(*a,**k):
        build_calls.append(1)
        return None if len(build_calls)==2 else {}
    monkeypatch.setattr(module,'_build_inputs',pack)
    model=torch.nn.Linear(2,1,bias=False)
    before=model.weight.detach().clone()
    bundle=SimpleNamespace(model=model,unwrap=lambda:model)
    monkeypatch.setattr(module,'load_model_with_lora',lambda *a,**k:bundle)
    monkeypatch.setattr(module,'_loss_one',lambda *a,**k:((model(torch.ones(1,2))-1).square().sum(),{}))
    if phase==1:
        monkeypatch.setattr(module,'_dummy_ddp_forward_zero_loss',lambda *a,**k:(model(torch.ones(1,2)).sum()*0,{}))
    else:
        monkeypatch.setattr(module,'_ddp_sync_zero_loss',lambda *a,**k:model(torch.ones(1,2)).sum()*0)
    saved=[]
    def save(bundle,output,args,*,step,name='final'):
        path=output/name;path.mkdir()
        (path/f'sft_new_loop_phase{phase}_adapter_config.json').write_text(json.dumps({'step':step}))
        saved.append((name,step));return path
    monkeypatch.setattr(module,'_save_adapter',save)
    module.train(config)
    assert not torch.equal(model.weight,before)
    total=max_steps or (288 if phase==1 else 126)
    assert saved==[('final',total)]
    plan=json.loads((tmp_path/'run/training_budget.json').read_text())
    assert plan['optimizer_updates']==(total+1)//2
    receipt=json.loads((tmp_path/'run/exposure_summary.json').read_text())
    assert receipt['successful_backward']==receipt['applied_backward']==total-1
    assert receipt['skipped']==1 and receipt['pending_backward']==0
    epochs=json.loads((tmp_path/'run/exposure_epochs.json').read_text())
    assert len(epochs)==(1 if max_steps else 3)
    assert sum(x['actual_presentations'] for x in epochs)==total-1
    assert (tmp_path/'run/training_eta.json').is_file()
    # Same output never restarts over the existing adapter.
    with pytest.raises(ValueError,match='already contains'): module.train(config)


def test_two_real_gloo_ranks_wait_for_saved_adapter_before_budget_exit(tmp_path):
    script=tmp_path/'worker.py'
    script.write_text('''import os,sys,time
from pathlib import Path
import torch.distributed as dist
from qwen3vl_local.sft_new_loop_phase1.training_policy import RuntimeBudget,check_runtime
dist.init_process_group('gloo')
rank=dist.get_rank()
plan={'steps_per_epoch':2304,'total_steps':6912,'epoch_hours_limit':8}
guard=RuntimeBudget(plan,100)
guard.started-=1500 if rank==1 else 10
out=Path(sys.argv[1])
def save():
    time.sleep(.1)
    (out/'adapter_saved').write_text('complete')
try:
    check_runtime(guard,100,rank=rank,distributed=dist,device='cpu',output_dir=out,save_adapter=save)
except RuntimeError:
    assert (out/'adapter_saved').read_text()=='complete'
    (out/f'rank{rank}.stopped').write_text('ok')
else:
    raise AssertionError('must stop all ranks')
finally:
    dist.destroy_process_group()
''')
    result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','--',str(script),str(tmp_path)],capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
    assert len(list(tmp_path.glob('rank*.stopped')))==2
    assert json.loads((tmp_path/'budget_stop.json').read_text())['any_rank_exceeded']
