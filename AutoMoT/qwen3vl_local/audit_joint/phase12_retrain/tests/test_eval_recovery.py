import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from qwen3vl_local.audit_joint.phase12_retrain import recover_eval as recovery, cpu_adapter_entry as entry
from qwen3vl_local.qwen35.tests.test_backend import model


@pytest.mark.parametrize('phase',[1,2])
def test_real_qwen_peft_cpu_read_even_if_library_infers_cuda(model,tmp_path,monkeypatch,phase):
    from peft import LoraConfig,get_peft_model,TaskType
    from peft.utils import save_and_load
    import peft.peft_model
    from qwen3vl_local.qwen35 import backend,adapters
    base=tmp_path/'base';adapter=tmp_path/'adapter'
    model.save_pretrained(base)
    original=backend.LocalModel.from_pretrained(base)
    lora=get_peft_model(original,LoraConfig(r=2,target_modules=['q_proj'],task_type=TaskType.CAUSAL_LM))
    adapters.save_adapter(lora,adapter)
    real_load=save_and_load.safe_load_file;seen=[]
    def checked(path,device=None,**kw):
        seen.append(device)
        if device!='cpu':raise RuntimeError('simulated Exclusive_Process default-device collision')
        return real_load(path,device=device,**kw)
    monkeypatch.setattr(save_and_load,'safe_load_file',checked)
    monkeypatch.setattr(peft.peft_model,'infer_device',lambda:'cuda')
    # Negative control: original implicit adapter loading really hits this trap.
    with pytest.raises(RuntimeError,match='collision'):
        adapters.LocalPeftModel.from_pretrained(backend.LocalModel.from_pretrained(base),adapter)
    module=importlib.import_module(f'qwen3vl_local.sft_new_loop_phase{phase}.eval')
    monkeypatch.setattr(module,'_validate_phase1_adapter' if phase==1 else '_validate_event_adapter',lambda *a:None)
    monkeypatch.setattr(backend.AutoProcessor,'from_pretrained',lambda *a,**kw:SimpleNamespace(tokenizer=None))
    loaded=module.load_eval_bundle(base,adapter,torch.device('cpu'),merge_lora=False)
    assert seen[-1]=='cpu' and next(loaded.model.parameters()).device.type=='cpu'
    assert set(k for k in loaded.model.state_dict() if 'lora_' in k)==set(k for k in lora.state_dict() if 'lora_' in k)


def test_external_supplement_keeps_validation_and_forces_cpu(tmp_path):
    calls=[]
    class Local:
        @classmethod
        def from_pretrained(cls,model,path,*a,**kw):
            calls.append((model,path,kw));return 'loaded'
    module=SimpleNamespace(LocalPeftModel=Local)
    entry.install(module,tmp_path)
    assert Local.from_pretrained('model','adapter',is_trainable=False,torch_device='cuda:3')=='loaded'
    assert calls[0][2]==dict(is_trainable=False,torch_device='cpu')
    assert json.loads(next(tmp_path.glob('*.jsonl')).read_text())['torch_device']=='cpu'


def test_two_rank_real_torchrun_shim_and_supplement(tmp_path):
    tree=tmp_path/'old';package=tree/'qwen3vl_local';(package/'qwen35').mkdir(parents=True)
    (package/'__init__.py').write_text('');(package/'qwen35/__init__.py').write_text('')
    adapter=package/'qwen35/adapters.py'
    adapter.write_text('class LocalPeftModel:\n @classmethod\n def from_pretrained(cls,model,path,**kw):return kw["torch_device"]\n')
    native=package/'sft_new_loop_phase2/eval.py';native.parent.mkdir()
    native.write_text('import os, torch.distributed as dist\nfrom qwen3vl_local.qwen35.adapters import LocalPeftModel\ndist.init_process_group("gloo")\nassert LocalPeftModel.from_pretrained(None,"test")=="cpu"\ndist.barrier()\nprint("RANK_OK",os.environ["LOCAL_RANK"])\ndist.destroy_process_group()\n')
    traces=tmp_path/'traces';traces.mkdir();shim=tmp_path/'torchrun'
    shim.write_text(recovery.torchrun_shim(sys.executable,Path(entry.__file__).resolve()));shim.chmod(0o755)
    env={**os.environ,'PYTHONPATH':str(tree),'PHASE12_RECOVERY_TREE':str(tree),
         'PHASE12_RECOVERY_TRACE':str(traces),'PHASE12_RECOVERY_ADAPTER_SHA':hashlib.sha256(adapter.read_bytes()).hexdigest(),
         'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    result=subprocess.run([str(shim),'--standalone','--nproc_per_node=2','--',str(native)],cwd=tree,env=env,capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
    assert len(list(traces.glob('*.jsonl')))==2
    assert {json.loads(p.read_text())['local_rank'] for p in traces.glob('*.jsonl')}=={'0','1'}


@pytest.mark.parametrize('mode,adapter,expected',[('2rgb_endpoints','old/best_generation','0'),('4rgb',None,'1')])
def test_recovery_runs_old_two_only_eval_and_new_four_train(tmp_path,monkeypatch,mode,adapter,expected):
    from qwen3vl_local.audit_joint.phase12_retrain.launch import environment,APP
    monkeypatch.setenv('PHASE12_RECOVERY_ADAPTER_SHA','bound')
    monkeypatch.setenv('GPU_IDS','0,1,2,3')
    info=dict(index=str(tmp_path/'index'),data_root=str(tmp_path/'rgb'),model_dir=str(tmp_path/'base'))
    env=recovery.mode_environment(environment,APP,tmp_path/'old',tmp_path/'recovery',info,mode,adapter,tmp_path/'four' if mode=='4rgb' else None)
    assert env['RUN_TRAIN']==expected and env['HISTORY_RGB_MODES']==mode
    assert env['TRAIN_FOCUS_BALANCE_COUNT']=='1024' and env['MAX_TRAIN_FRAME_REPEAT']=='10'
    assert env['RUN_BUILD']=='0' and env['GPU_IDS']=='0,1,2,3'
    assert env['PIPELINE_ROOT']==str(tmp_path/'recovery/pipeline'/mode)
    if adapter:assert env['ADAPTER_DIR']==adapter
    else:assert 'ADAPTER_DIR' not in env


@pytest.mark.parametrize('bad',['incomplete','budget','phase','data','mode'])
def test_incomplete_or_wrong_completed_training_rejected(tmp_path,bad):
    (tmp_path/'final').mkdir();(tmp_path/'final/sft_new_loop_phase2_adapter_config.json').write_text(json.dumps(dict(history_rgb_mode='4rgb' if bad=='mode' else '2rgb_endpoints')))
    complete=dict(phase=1 if bad=='phase' else 2,budget={'optimizer_updates':999 if bad=='budget' else 5376},adapters={'final':{}})
    def verify(*a):
        if bad=='incomplete':raise ValueError('No valid training completion receipt')
        return complete
    proof=SimpleNamespace(verify_training=verify,read=lambda p:dict(data={}),same_data=lambda *a:bad!='data',data_identity=lambda p:{})
    with pytest.raises(ValueError):recovery.validate_completed(proof,tmp_path,'index','2rgb_endpoints')


@pytest.mark.parametrize('existing_four',[False,True])
def test_recovery_orchestration_with_explicit_gpu_stage_doubles(tmp_path,monkeypatch,existing_four):
    from qwen3vl_local import runtime_version
    from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof,data_stage,delivery
    from qwen3vl_local.audit_joint.phase12_retrain import launch
    old=tmp_path/'original_run';old.mkdir();(old/'data').mkdir()
    for name in ['execution.json','ready.json','data/training_revision.json','selected_train_reviews.json']:
        (old/name).write_text('{}')
    tree=tmp_path/'version/app';tree.mkdir(parents=True);version=tree.parent
    adapter_source=tree/'adapter.py';adapter_source.write_text('original')
    profile={'qwen3vl_local/qwen35/adapters.py':hashlib.sha256(b'original').hexdigest()}
    target=tree/'qwen3vl_local/qwen35/adapters.py';target.parent.mkdir(parents=True);target.write_text('original')
    two=tree/'checkpoints/sft_new_loop_phase2_retrain_runs/run_direct_event_format_supervised_2rgb_endpoints'/old.name
    four=tree/'checkpoints/sft_new_loop_phase2_retrain_runs/run_direct_event_format_supervised_4rgb'/old.name
    (two/'final').mkdir(parents=True);(two/'training_complete.json').write_text('{}')
    if existing_four:(four/'final').mkdir(parents=True);(four/'training_complete.json').write_text('{}')
    info=dict(adapter_dir=str(two/'final'),index=str(old/'data/frame_index.jsonl'),split='test',data_root=str(tmp_path/'rgb'),model_dir=str(tmp_path/'base'))
    failed=old/'pipeline/2rgb_endpoints/eval_review/lora_production/evaluation_inputs.json';failed.parent.mkdir(parents=True)
    failed.write_text(json.dumps(dict(arguments=info,adapter_identity={})))
    old_before={str(p.relative_to(old)):p.read_bytes() for p in old.rglob('*') if p.is_file()}
    out=tmp_path/'recovery';(out/'code').mkdir(parents=True);(out/'code/cpu_adapter_entry.py').write_text('# explicit orchestration double')
    monkeypatch.setattr(proof,'ROOT',tree)
    monkeypatch.setattr(runtime_version,'verify',lambda p:{})
    monkeypatch.setattr(launch,'verify_ready',lambda *a:dict(index=info['index'],data={}))
    monkeypatch.setattr(proof,'verify_adapter',lambda p:{})
    validated=[]
    def complete(proof,run,index,mode):
        assert (run/'training_complete.json').exists()
        validated.append(mode);return run/'final',{}
    monkeypatch.setattr(recovery,'validate_completed',complete)
    monkeypatch.setattr(delivery,'verify_package',lambda p:{'status':'verified'})
    calls=[]
    def stage(cmd,status,*,timeout,env):
        mode=env['HISTORY_RGB_MODES'];calls.append((mode,env['RUN_TRAIN']))
        if mode=='4rgb' and not existing_four:
            assert not four.exists();(four/'final').mkdir(parents=True);(four/'training_complete.json').write_text('{}')
        # Exercise the real native exclusive output reservation, including both launches.
        from qwen3vl_local.sft_new_loop_phase1 import local_assets
        folder=Path(env['PIPELINE_ROOT'])
        with monkeypatch.context() as m:
            m.setattr(sys,'argv',['local_assets','reserve-output','--path',str(folder)])
            local_assets.main()
        (folder/'fixture_audit_bundle.tar.gz').write_bytes(b'GPU stage double')
        (out/'traces'/f'{mode}.jsonl').write_text('{"torch_device":"cpu"}\n')
    monkeypatch.setattr(data_stage,'run',stage)
    cwd=Path.cwd()
    try:recovery.worker(old,out,version,profile)
    finally:os.chdir(cwd)
    assert calls==[('2rgb_endpoints','0'),('4rgb','0' if existing_four else '1')]
    assert (out/'recovery_audit.tar.gz').is_file()
    assert json.loads((out/'completion.json').read_text())['two_training_repeated'] is False
    assert old_before=={str(p.relative_to(old)):p.read_bytes() for p in old.rglob('*') if p.is_file()}
