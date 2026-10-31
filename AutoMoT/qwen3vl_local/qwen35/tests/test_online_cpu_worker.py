"""CPU CLI regression: real tiny BF16 weights/LoRA, synthetic visual processor."""
import importlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from qwen3vl_local.qwen35 import backend, cache_policy, online_benchmark as bench
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35.tests.test_online_benchmark import Processor, setup


@pytest.mark.parametrize('phase',[1,2,3,12])
def test_cpu_cli_executes_bf16_worker_and_writes_real_reuse_report(model,tmp_path,monkeypatch,phase):
    from qwen3vl_local.sft_new_loop_phase1 import data_stage
    for name in ('AUTOMOT_TEXT_PREFIX','AUTOMOT_TEXT_PREFIX_CACHE','AUTOMOT_ONLINE_TEXT_PREFIX'):
        monkeypatch.delenv(name,raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    _,rows,_=setup(model,tmp_path,phase)
    base=tmp_path/'base';model.save_pretrained(base)
    loaded=backend.LocalModel.from_pretrained(base,dtype=torch.bfloat16)
    adapted=get_peft_model(loaded,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj'],task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name,p in adapted.named_parameters():
            if 'lora_B' in name:p.normal_(0,.03)
    bundle=SimpleNamespace(model=adapted,lora_target_modules=['q_proj'],unwrap=lambda:adapted)
    cache_policy.start_training(bundle)
    if phase==12:
        from qwen3vl_local.sft_new_loop_phase12 import train,cache_checks
        (tmp_path/'ready.json').write_text(json.dumps(dict(text_cache_policy=cache_policy.current(),
            text_cache_limits=cache_policy.current()['limits'],cache_acceptance_protocol=cache_checks.protocol())))
        train.save(bundle,tmp_path/'final',tmp_path,'2rgb_endpoints',1)
    else:
        train=importlib.import_module(f'qwen3vl_local.sft_new_loop_phase{phase}.train')
        monkeypatch.setattr(sys,'argv',['train','--device','cpu','--model-dir',str(base),
                                      '--history-rgb-mode','2rgb_endpoints'])
        train._save_adapter(bundle,tmp_path,train.parse_args(),step=1)
    requests=tmp_path/'requests.jsonl';requests.write_text('\n'.join(map(json.dumps,rows)))
    output=tmp_path/'benchmark'
    # Keep official model/adapter loaders and all SHA/contract checks. Only image
    # tokenization is synthetic; the tiny base has no official tokenizer assets.
    monkeypatch.setattr(backend.LocalProcessor,'from_pretrained',lambda *a,**kw:Processor())
    command=['benchmark','--phase',str(phase),'--requests',str(requests),'--model-dir',str(base),
        '--adapter-dir',str(tmp_path/'final'),'--output',str(output),'--device','cpu',
        '--max-new-tokens','1','--repeats','2','--window-frames','2']
    monkeypatch.setattr(sys,'argv',command+['--check-only']);bench.main()
    assert not output.exists()
    launches=[]
    def run_worker(argv,status,*,timeout,env):
        # Execute the forwarded worker CLI in-process so the processor fixture is
        # available. Real subprocess isolation/timeout is covered separately.
        launches.append(argv)
        assert env['CUDA_VISIBLE_DEVICES']==env['GPU_IDS']==''
        with monkeypatch.context() as child:
            child.setattr(sys,'argv',argv[3:])
            bench.main()
    monkeypatch.setattr(data_stage,'run',run_worker)
    monkeypatch.setattr(sys,'argv',command);bench.main()
    assert len(launches)==1 and '--worker' in launches[0]
    report=json.loads((output/'report.json').read_text())
    assert report['status']=='text_reuse_checked_full_control_recorded_quality_pending'
    assert report['source_files_verified_after_load'] and report['source_files_verified_after_measurement']
    assert report['optimizer_updates']==0 and not report['quality_certified'] and not report['carla_certified']
    result=report['benchmark']
    assert result['device']=='cpu' and result['dtype']=='torch.bfloat16'
    assert result['same_partition_equivalence'] and result['cold_cached']['cache']['full_hits']>0
    assert (output/'resident_text_stream.json').exists() and (output/'trial_01.json').exists()
