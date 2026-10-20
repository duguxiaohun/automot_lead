"""Production parser boundaries and early failures without loading a 4B model."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys
import zipfile

import pytest

from qwen3vl_local.audit_joint import phase3_launch as launch, phase3_launch_worker as worker
from qwen3vl_local.audit_joint import phase3_base_trial as trial, phase3_base_trial_worker as base
from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_second_review import raw_fixture
from qwen3vl_local.audit_joint.phase3_single_frame.tests.test_single_frame import fixture
from qwen3vl_local.qwen35.tests.test_processor import processor


def test_real_base_eval_parser_uses_saved_rgb_contract(tmp_path, monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import eval as evaluate
    cfg = dict(history_rgb_mode='4rgb', action_output_mode='choice')
    folder = tmp_path/'train/final'; folder.mkdir(parents=True)
    (folder/'sft_new_loop_phase3_adapter_config.json').write_text('{}')
    monkeypatch.setattr(evaluate, '_validate_action_adapter', lambda *a:cfg)
    j = dict(output=str(tmp_path), index='index', data_root='data', model_dir='model')
    argv = worker.test_arguments(j)
    monkeypatch.setattr(sys, 'argv', ['evaluate', *argv])
    args = evaluate.parse_args()
    assert args.history_rgb_mode is None and args.max_new_tokens == 256
    assert evaluate._resolve_history_rgb_mode(args.history_rgb_mode,cfg)[0]=='4rgb'
    assert args.expected_total_cases == 380 and args.batch_size == 1
    monkeypatch.setattr(sys, 'argv', ['evaluate', *argv, '--history-rgb-mode', '4rgb'])
    with pytest.raises(ValueError, match='only for base-Qwen'):
        evaluate.parse_args()


def test_native_base_saved_contract_passes_final_eval_parser(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import train, eval as evaluate
    from qwen3vl_local.qwen35 import adapters
    j,_,_=raw_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(adapters,'save_adapter',lambda *a,**kw:None)
    bundle=SimpleNamespace(unwrap=lambda:None,lora_target_modules=['q_proj'])
    train._save_adapter(bundle,Path(j['output'])/'train',base.make_args(j),step=7680)
    monkeypatch.setattr(sys,'argv',['evaluate',*worker.test_arguments(j)])
    parsed=evaluate.parse_args()
    cfg=evaluate._validate_action_adapter(Path(parsed.adapter_dir),Path(j['model_dir']))
    assert cfg['global_step']==7680 and cfg['history_rgb_count']==4
    assert cfg['prompt_variant']==trial.reference()['prompt_variant']


@pytest.mark.parametrize('method', ['base', 'single_frame'])
def test_schedule_real_torchrun_parser_and_cpu_only(tmp_path, method):
    from torch.distributed.run import get_args_parser
    steps = launch.schedule(method, tmp_path)
    for name, cmd in steps:
        if 'torch.distributed.run' in cmd:
            parsed = get_args_parser().parse_args(cmd[3:])
            assert parsed.module and parsed.nproc_per_node == '4'
            assert parsed.training_script.startswith('qwen3vl_local.audit_joint.')
            assert '--job' in parsed.training_script_args
            assert parsed.max_restarts == 0
    only = launch.schedule(method, tmp_path, prepare_only=True)
    assert [n for n,_ in only] == (['prepare'] if method=='base' else [])
    assert not any('phase3_suite' in arg for _,cmd in steps for arg in cmd)


def test_base_all_epoch_and_fixed_test_inputs_checked_before_gpu(tmp_path, monkeypatch, processor):
    j,_,_=raw_fixture(tmp_path,monkeypatch)
    original=base.make_args
    def args(job):
        cfg=original(job);cfg.focus_balance_count=4
        cfg.eval_balance_count=cfg.generation_eval_balance_count=1
        return cfg
    monkeypatch.setattr(base,'make_args',args)
    monkeypatch.setattr(base,'preflight',lambda *a:None)
    report=worker.prepare(j,processor)
    assert {s['kind'] for s in report['shapes'].values()}=={'train','loss','generation','test'}
    assert all(len(s['grid'])==4 for s in report['shapes'].values())
    assert {report['shapes'][k]['kind'] for k in report['probe_ids']}=={'train','loss','generation','test'}
    work=worker.items(j)
    item=next(x for kind,x in work.values() if kind=='test')
    Path(item.row.history_rgb_paths[-1]).unlink()
    with pytest.raises(FileNotFoundError):worker.prepare(j,processor)


def test_single_frame_manifest_rejected_before_shape_scan(tmp_path,monkeypatch,processor):
    from qwen3vl_local.audit_joint.phase3_single_frame import worker as single
    from qwen3vl_local.sft_new_loop_phase3 import preflight
    j,_,art,_=fixture(tmp_path,monkeypatch);j['dataset']=str(tmp_path/'single/dataset.json')
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    manifest=Path(j['index']).with_name('manifest.json');manifest.unlink()
    monkeypatch.setattr(single,'planned',lambda *a:pytest.fail('Shape scan started before native admission'))
    with pytest.raises(ValueError,match='manifest required'):single.preflight(j,art,processor)


def test_package_keeps_reports_and_excludes_weights_and_derived_data(tmp_path):
    for name in ('data/dataset.json','experiment/train/final/adapter_config.json',
                 'experiment/train/training_states/epoch_0001/training_state.pt'):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('SECRET_LARGE_ASSET')
    p=tmp_path/'experiment/single_frame_val_cases.json';p.write_text('[{"id":"fixture"}]')
    (tmp_path/'train.log').write_text('x'*100000)
    state=dict(status='failed_or_interrupted',method='single_frame')
    launch.package(tmp_path,state)
    with zipfile.ZipFile(tmp_path/'handoff/phase3_report.zip') as z:
        assert set(z.namelist())=={'launch_workflow.json','train.log','experiment/single_frame_val_cases.json'}
        assert len(z.read('train.log'))==60000
    receipt=trial.read(tmp_path/'handoff/phase3_package.json')
    assert receipt['archive_sha256']==trial.file_sha(tmp_path/'handoff/phase3_report.zip')


@pytest.mark.parametrize('flag',['SKIP_TRAIN','PIPELINE_CHECK_ONLY'])
def test_shell_check_flags_and_arbitrary_cwd(tmp_path,flag):
    fake=tmp_path/'python';fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n');fake.chmod(0o755)
    env=dict(os.environ,PYTHON=str(fake),SKIP_TRAIN='0',PIPELINE_CHECK_ONLY='0')
    env[flag]='1'
    cmd=['bash',str(Path(launch.__file__).with_suffix('.sh')),'--method','base']
    result=subprocess.run(cmd,cwd=tmp_path,env=env,text=True,capture_output=True)
    assert result.returncode==0 and '--prepare-only' in result.stdout
    env[flag]='yes'
    assert subprocess.run(cmd,cwd=tmp_path,env=env,capture_output=True).returncode==2


def test_launch_failure_stops_before_gpu_and_packages(tmp_path,monkeypatch):
    j,_,_=raw_fixture(tmp_path,monkeypatch)
    from qwen3vl_local.sft_new_loop_phase3 import preflight
    def bad_model(*a):raise ValueError('missing model shard')
    monkeypatch.setattr(preflight,'check_model',bad_model)
    monkeypatch.setattr(trial,'gpu_ids',lambda *a:pytest.fail('GPU selected before CPU admission'))
    monkeypatch.setattr(sys,'argv',['launch','--method','base','--index',j['index'],
        '--data-root',j['data_root'],'--model-dir',j['model_dir'],'--output',str(tmp_path/'launch'), '--run-tag','test'])
    assert launch.main()==2
    root=tmp_path/'launch/run_test'
    state=trial.read(root/'launch_workflow.json')
    assert 'missing model shard' in state['error'] and not state['stages']
    assert (root/'handoff/phase3_report.zip').is_file()


@pytest.mark.parametrize('method',['base','single_frame'])
@pytest.mark.parametrize('fail_at',[None,'probe','train','test'])
def test_coordinator_order_failure_and_fixed_output(tmp_path,monkeypatch,method,fail_at):
    j,_,_=raw_fixture(tmp_path,monkeypatch)
    from qwen3vl_local.sft_new_loop_phase3 import preflight
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    monkeypatch.setattr(trial,'gpu_ids',lambda *a:'0,1,2,3')
    monkeypatch.setattr(trial,'verify_training',lambda job:dict(optimizer_updates=7680))
    monkeypatch.setattr(launch.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    observed=[]
    def run(cmd,log,env,**kw):
        name=log.stem;observed.append(name);log.write_text('fixture stage '+name)
        assert env['CUDA_VISIBLE_DEVICES']=='0,1,2,3'
        assert env['HF_HUB_OFFLINE']=='1'
        if name=='prepare' and method=='single_frame':
            exp=log.parent/'experiment';exp.mkdir()
            trial.write(exp/'single_frame_job.json',dict(j,output=str(exp),mode='base'))
        return dict(exit_code=17 if name==fail_at else 0,seconds=0,log=str(log))
    monkeypatch.setattr(trial,'run_stage',run)
    root=tmp_path/'runs'
    argv=['launch','--method',method,'--index',j['index'],'--data-root',j['data_root'],
          '--model-dir',j['model_dir'],'--output',str(root)]
    if method=='single_frame':argv+=['--build-data']
    monkeypatch.setattr(sys,'argv',argv)
    assert launch.main()==(17 if fail_at else 0)
    expected=['prepare','probe','train','summarize','test'] if method=='base' else ['derive_data','prepare','probe','train','val','test']
    assert observed==(expected[:expected.index(fail_at)+1] if fail_at else expected)
    folder=root/('run_base_independent_v1' if method=='base' else 'run_single_frame_nearest_v1')
    assert (folder/'handoff/phase3_package.json').is_file()
    assert (root/'latest').is_symlink()==(fail_at is None)
    with pytest.raises(FileExistsError):launch.main()


def test_existing_latest_directory_rejected_before_work(tmp_path,monkeypatch):
    j,_,_=raw_fixture(tmp_path,monkeypatch)
    out=tmp_path/'runs';(out/'latest').mkdir(parents=True)
    monkeypatch.setattr(sys,'argv',['launch','--method','base','--index',j['index'],
        '--data-root',j['data_root'],'--model-dir',j['model_dir'],'--output',str(out)])
    with pytest.raises(ValueError,match='not a symlink'):launch.main()
    assert list(out.iterdir())==[out/'latest']
