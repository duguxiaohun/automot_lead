"""Real tiny hybrid model, synthetic processor; no 4B speed/quality certification."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from PIL import Image
from qwen3vl_local.qwen35 import online_benchmark as bench, online, text_prefix as tp
from qwen3vl_local.qwen35.question_schedule import QuestionSchedule
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35.tests.test_online_contract import metadata as phase3_metadata
from qwen3vl_local.sft_new_loop_phase12 import prompts, CONTRACT


class Processor:
    tokenizer=SimpleNamespace(eos_token_id=1,convert_tokens_to_ids=lambda _:1)

    def apply_chat_template(self,messages,**kwargs):
        parts=[]
        for message in messages:
            content=message['content']
            parts.append(content if isinstance(content,str) else '\n'.join(
                part['text'] for part in content if part['type']=='text'))
        return '\n'.join(parts)

    def __call__(self,text,images,**kwargs):
        # Entire native text determines a collision-resistant small fixture prefix.
        prefix=[10]*48+[10+v for v in hashlib.sha256(text[0].encode()).digest()]
        ids=torch.tensor([prefix+[x for _ in images for x in (2,4,4,4,4,3)]+[21,22]])
        pixels=[]
        for image in images:
            seed=int.from_bytes(hashlib.sha256(image.tobytes()).digest()[:4],'big')
            pixels.append(torch.randn(16,1536,generator=torch.Generator().manual_seed(seed)))
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids),mm_token_type_ids=(ids==4).long(),
                    image_grid_thw=torch.tensor([[1,4,4]]*len(images)),pixel_values=torch.cat(pixels))

    def batch_decode(self,ids,**kwargs):return [str(ids.flatten().tolist())]


def setup(model,tmp_path,phase):
    b=SimpleNamespace(model=model,device=torch.device('cpu'),processor=Processor(),tokenizer=Processor.tokenizer)
    context={}
    if phase==12:
        cfg=dict(contract=CONTRACT,prompt=prompts.identity('2rgb_endpoints'),mode='2rgb_endpoints')
        from qwen3vl_local.sft_new_loop_phase12 import infer
        infer.bind_online_contract(b,cfg);context=dict(domain=prompts.P2.ROAD_DOMAIN)
    elif phase==3:
        from qwen3vl_local.sft_new_loop_phase3 import prompts as p3
        cfg=phase3_metadata(mode='2rgb_endpoints')
        context=dict(context_id=next(iter(p3.CONTEXT_BY_ID)),road_structure=next(iter(p3.ROAD_STRUCTURE_TEXT)),current_speed_mps=5)
    else:
        from qwen3vl_local.action_prior import prompt_versions as versions
        name=versions.DIRECT8_NAME if phase==1 else 'sft_new_loop_phase2_direct_event_visual_v7_evidence_20261030'
        module=versions._restore(phase,name)
        cfg=dict(prompt_name=name,history_rgb_mode='2rgb_endpoints',history_rgb_count=2,
                 history_rgb_selected_indices=[0,3],input_layout=tp.identity(),
                 production_prompt_sha256=(module.phase1_prompt_sha256 if phase==1 else module.event_prompt_sha256)(history_rgb_mode='2rgb_endpoints'))
        if phase==2:context=dict(domain=module.ROAD_DOMAIN)
    if phase!=12:online.bind_prompt_contract(b,phase,cfg)
    rows=[]
    for frame in range(4):
        path=tmp_path/f'{frame}.png';Image.new('RGB',(8,8),(30*frame,10,70)).save(path)
        rows.append(dict(images=[str(path)]*2,context=copy.deepcopy(context)))
    return b,rows,cfg


@pytest.mark.parametrize('phase',[1,2,3,12])
def test_paired_native_stream_cold_switch_and_independent_state(model,tmp_path,monkeypatch,phase):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    b,rows,_=setup(model,tmp_path,phase)
    production=tp.bank(b);schedule=QuestionSchedule();schedule.next()
    b._question_schedules={'production':schedule}
    if phase==3:rows[3]['context']['current_speed_mps']=12
    result=bench.benchmark(b,phase,rows,window_frames=2,max_new_tokens=3,repeats=2,deadline_ms=.001)
    calls=2 if phase==12 else 1
    assert result['complete_output_equivalence'] and not result['quality_certified'] and not result['carla_certified']
    assert result['policy']['generation_calls_per_frame']==calls and not result['policy']['prior_engine_rechecks']
    assert not result['policy']['equal_forward_call_count_required']
    assert result['trials'][0]['order']==['full','cached'] and result['trials'][1]['order']==['cached','full']
    assert b._question_schedules=={'production':schedule} and schedule.requests==1
    assert tp.bank(b) is production and production.stats['requests']==0
    for trial in result['trials']:
        for arm in ('full','cached'):
            run=trial[arm];summary=run['summary']
            assert summary['generation_calls']==4*calls
            assert summary['deadline_exceeded']==4
            assert summary['frame_groups']['first']['count']==1
            assert summary['frame_groups']['window_changed']['count']==1
            assert summary['frame_groups']['context_changed']['count']==int(phase==3)
            assert summary['input_verification_seconds']>0 and summary['encoding_seconds']>0
        cached=trial['cached']
        assert cached['frames'][0]['queries'][0]['cache_class']=='miss'
        assert cached['frames'][1]['queries'][0]['cache_class']=='full_hit'
        assert cached['cache']['requests']==4*calls
        if phase==3:assert cached['frames'][3]['queries'][0]['cache_class']!='full_hit'
    assert result['summary']['cached']['generation_calls']==8*calls
    assert result['saving_fraction']==1-sum(t['cached']['seconds'] for t in result['trials'])/sum(t['full']['seconds'] for t in result['trials'])


@pytest.mark.parametrize('field',['token_ids','stop_token_id','reached_token_limit','generated_tokens','text'])
def test_output_mismatch_rejected(field):
    answer=dict(token_ids=[5],stop_token_id=1,reached_token_limit=False,generated_tokens=1,text='x')
    a=dict(frames=[dict(schedule={},images_sha256=['a'],context_sha256='b',queries=[dict(input_sha256='c',answer=answer)])])
    b=copy.deepcopy(a);b['frames'][0]['queries'][0]['answer'][field]=None
    with pytest.raises(ValueError,match='token/EOS/text'):bench.assert_same(a,b)
    b=copy.deepcopy(a);b['frames'][0]['queries'][0]['input_sha256']='d'
    with pytest.raises(ValueError,match='Encoded input'):bench.assert_same(a,b)


@pytest.mark.parametrize('kwargs',[dict(repeats=1),dict(repeats=3),dict(deadline_ms=float('nan')),dict(max_new_tokens=0),dict(window_frames=0),dict(slots=3)])
def test_invalid_measurement_policy_rejected(model,tmp_path,monkeypatch,kwargs):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');b,rows,_=setup(model,tmp_path,1)
    with pytest.raises(ValueError):bench.benchmark(b,1,rows,**kwargs)


def test_replay_changed_pixels_or_weights_fails_and_keeps_production(model,tmp_path,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');b,rows,_=setup(model,tmp_path,1);production=tp.bank(b)
    def change_pixels(number,trial):Image.new('RGB',(8,8),(255,0,0)).save(rows[0]['images'][0])
    with pytest.raises(ValueError,match='input/policy mismatch'):
        bench.benchmark(b,1,rows,max_new_tokens=1,on_trial=change_pixels)
    assert tp.bank(b) is production and production.stats['requests']==0
    def change_weights(number,trial):
        with torch.no_grad():next(model.parameters()).add_(.01)
    with pytest.raises(ValueError,match='Weights changed'):
        bench.benchmark(b,1,rows,max_new_tokens=1,on_trial=change_weights)


@pytest.mark.parametrize('phase',[1,2,3,12])
def test_cpu_check_plan_and_relative_paths_use_native_contract(model,tmp_path,monkeypatch,phase):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');_,rows,cfg=setup(model,tmp_path,phase)
    adapter=tmp_path/'adapter';adapter.mkdir()
    name='fusion_config.json' if phase==12 else f'sft_new_loop_phase{phase}_adapter_config.json'
    file=adapter/name;file.write_text(json.dumps(cfg))
    source=tmp_path/'requests.jsonl'
    source.write_text('\n'.join(json.dumps(dict(images=[Path(p).name for p in row['images']],context=row['context'])) for row in rows))
    args=SimpleNamespace(phase=phase,adapter_dir=adapter,window_frames=2,slots=1,seed='check',rank=0)
    assert bench.check_plan(args,bench.read_requests(source,phase))==cfg
    if phase!=12:cfg['production_prompt_sha256']='bad'
    else:cfg['prompt']['system_sha256']='bad'
    file.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):bench.check_plan(args,rows)


def test_worker_failure_persists_receipt_without_certification(tmp_path):
    args=SimpleNamespace(output=tmp_path/'failed',requests=tmp_path/'absent',phase=1)
    with pytest.raises(FileNotFoundError):bench.worker(args)
    receipt=json.loads((args.output/'report.json').read_text())
    assert receipt['status']=='failed' and not receipt['carla_certified'] and not receipt['quality_certified']
    with pytest.raises(FileExistsError):bench.worker(args)


@pytest.mark.parametrize('problem',['mode','layout','metadata','model','adapter'])
def test_shared_production_rejects_contract_drift_before_scheduling(model,tmp_path,monkeypatch,problem):
    from qwen3vl_local.sft_new_loop_phase12 import infer
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');b,rows,cfg=setup(model,tmp_path,12)
    scheduler=QuestionSchedule();mode='2rgb_endpoints';images=[None]*2
    if problem=='mode':mode='4rgb';images=[None]*4
    elif problem=='layout':monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','0')
    elif problem=='metadata':b.fusion_config['mode']='4rgb'
    elif problem=='model':b.model=SimpleNamespace()
    else:
        b.model.active_adapters=['unexpected_adapter']
    with pytest.raises(ValueError):infer.predict(b,images,mode,prompts.P2.ROAD_DOMAIN,schedule=scheduler)
    assert scheduler.requests==0


def test_first_pair_mismatch_keeps_both_original_outputs(model,tmp_path,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');b,rows,_=setup(model,tmp_path,1)
    original=bench._decode;calls=[];saved={};production=tp.bank(b)
    def changed(*args):
        answer=original(*args);calls.append(1)
        if len(calls)==len(rows)+1:answer['text']+='changed'
        return answer
    monkeypatch.setattr(bench,'_decode',changed)
    with pytest.raises(ValueError,match='token/EOS/text'):
        bench.benchmark(b,1,rows,max_new_tokens=1,on_run=lambda name,run:saved.update({name:copy.deepcopy(run)}))
    assert set(saved)=={'cold_cached','cold_full','recomputed_split','resident_text_stream','trial_00_full'}
    assert all(len(run['frames'])==4 for name,run in saved.items() if name!='resident_text_stream')
    assert saved['cold_full']['frames'][0]['queries'][0]['answer']['text'].endswith('changed')
    assert tp.bank(b) is production and production.stats['requests']==0


def test_mid_stream_failure_keeps_completed_inputs_outputs(model,tmp_path,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');b,rows,_=setup(model,tmp_path,1)
    original=bench._prepared;calls=[];saved={}
    def fail(*args):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('injected render error')
        return original(*args)
    monkeypatch.setattr(bench,'_prepared',fail)
    with pytest.raises(RuntimeError,match='injected render'):
        bench.benchmark(b,1,rows,max_new_tokens=1,on_run=lambda name,run:saved.update({name:run}))
    partial=saved['cold_cached']
    assert partial['status']=='failed' and partial['failed_frame']==1 and len(partial['frames'])==1
    assert partial['frames'][0]['queries'][0]['answer']['token_ids']


@pytest.mark.parametrize('changed',['requests','adapter','source'])
def test_file_provenance_drift_rejected(tmp_path,changed):
    requests=tmp_path/'requests.jsonl';requests.write_text('{}')
    adapter=tmp_path/'adapter';adapter.mkdir();weight=adapter/'adapter_model.safetensors';weight.write_bytes(b'fixture')
    package=tmp_path/'code';package.mkdir();code=package/'source.py';code.write_text('before')
    args=SimpleNamespace(requests=requests,adapter_dir=adapter)
    expected=bench.file_bindings(args,package);bench.verify_file_bindings(args,package,expected)
    {'requests':requests,'adapter':weight,'source':code}[changed].write_bytes(b'changed')
    with pytest.raises(ValueError,match='files changed'):bench.verify_file_bindings(args,package,expected)


def test_cpu_plan_rejects_corrupt_image(model,tmp_path,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');_,rows,cfg=setup(model,tmp_path,1)
    (tmp_path/'sft_new_loop_phase1_adapter_config.json').write_text(json.dumps(cfg))
    Path(rows[0]['images'][0]).write_bytes(b'not an image')
    args=SimpleNamespace(phase=1,adapter_dir=tmp_path,window_frames=2,slots=1,seed='check',rank=0)
    with pytest.raises(OSError):bench.check_plan(args,rows)


def test_shared_single_task_predict_and_loader_preserve_contract(monkeypatch,tmp_path):
    from qwen3vl_local.sft_new_loop_phase12 import infer
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    model=SimpleNamespace(to=lambda device:model,eval=lambda:model)
    b=SimpleNamespace(model=model)
    infer.bind_online_contract(b,dict(contract=CONTRACT,mode='2rgb_endpoints',prompt=prompts.identity('2rgb_endpoints')))
    monkeypatch.setattr(infer,'_kv_start_state',lambda *a:None)
    monkeypatch.setattr(infer,'_student_generate_kv',lambda *a:('',None,torch.tensor([6])))
    for task in ('phase1','phase2'):
        result=infer.predict(b,[None]*2,'2rgb_endpoints',prompts.P2.ROAD_DOMAIN,task=task)
        assert result['task']==task and result['tokens']==1
    cfg=dict(contract=CONTRACT,mode='2rgb_endpoints',prompt=prompts.identity('2rgb_endpoints'))
    (tmp_path/'fusion_config.json').write_text(json.dumps(cfg))
    calls=[]
    monkeypatch.setattr(infer.proof,'verify_adapter',lambda p:calls.append('identity'))
    monkeypatch.setattr(infer.LocalModel,'from_pretrained',lambda *a,**k:model)
    monkeypatch.setattr(infer.LocalPeftModel,'from_pretrained',lambda *a,**k:calls.append(k['torch_device']) or model)
    monkeypatch.setattr(infer.LocalProcessor,'from_pretrained',lambda *a:Processor())
    loaded=infer.load_online_bundle(tmp_path,tmp_path,device='cpu')
    assert loaded.fusion_config==cfg and calls==['identity','cpu']
    with pytest.raises(ValueError,match='contract mismatch'):infer.load_online_bundle(tmp_path,tmp_path,device='cpu',mode='4rgb')


def test_cli_check_only_never_selects_gpu_or_loads_weights(model,tmp_path,monkeypatch,capsys):
    import sys
    import subprocess
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    _,rows,cfg=setup(model,tmp_path,3)
    (tmp_path/'sft_new_loop_phase3_adapter_config.json').write_text(json.dumps(cfg))
    requests=tmp_path/'requests.jsonl';requests.write_text('\n'.join(map(json.dumps,rows)))
    def forbidden(*a,**k):raise AssertionError('check-only selected GPU or loaded weights')
    monkeypatch.setattr(subprocess,'check_output',forbidden)
    monkeypatch.setattr(online,'load_online_bundle',forbidden)
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    monkeypatch.setattr(sys,'argv',['benchmark','--phase','3','--requests',str(requests),
        '--model-dir',str(tmp_path/'missing-base'),'--adapter-dir',str(tmp_path),'--check-only'])
    bench.main()
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='request_prompt_plan_valid' and not result['weights_validated'] and not result['inference_executed']
    assert not (tmp_path/'report.json').exists()


def test_two_slots_and_partial_prefixes_counted_honestly(model,tmp_path,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');b,rows,_=setup(model,tmp_path,1)
    factory=tp.TextPrefixCache
    monkeypatch.setattr(tp,'TextPrefixCache',lambda *args,**kw:factory(*args,**kw) if args or kw else factory(max_tokens=48))
    saved={}
    with pytest.raises(ValueError,match='no complete text cache hits'):
        bench.benchmark(b,1,rows,window_frames=4,slots=2,max_new_tokens=1,on_run=lambda n,r:saved.update({n:r}))
    run=saved['cold_cached']
    assert [f['schedule']['slot'] for f in run['frames']]==[0,1,0,1]
    assert run['summary']['cache_classes']['full_hit']['count']==0
    assert run['summary']['cache_classes']['partial_hit']['count']==3


def test_cpu_cli_worker_isolated_from_ddp_and_respects_timeout(model,tmp_path,monkeypatch):
    import sys
    from qwen3vl_local.sft_new_loop_phase1 import data_stage
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GPU_IDS','CUDA_VISIBLE_DEVICES'):
        monkeypatch.setenv(key,'7')
    _,rows,cfg=setup(model,tmp_path,1)
    (tmp_path/'sft_new_loop_phase1_adapter_config.json').write_text(json.dumps(cfg))
    requests=tmp_path/'requests.jsonl';requests.write_text('\n'.join(map(json.dumps,rows)))
    output=tmp_path/'output';calls=[]
    monkeypatch.setattr(data_stage,'run',lambda *a,**k:calls.append((a,k)))
    monkeypatch.setattr(sys,'argv',['benchmark','--phase','1','--requests',str(requests),
        '--model-dir',str(tmp_path/'base'),'--adapter-dir',str(tmp_path),'--device','cpu',
        '--output',str(output),'--timeout','123'])
    bench.main()
    (command,status),options=calls[0]
    assert '--worker' in command and command[command.index('--device')+1]=='cpu'
    assert status==output/'worker.json' and options['timeout']==123
    env=options['env']
    assert env['GPU_IDS']==env['CUDA_VISIBLE_DEVICES']==''
    assert not {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'}&set(env)
    with pytest.raises(FileExistsError):bench.main()


def test_stable_full_path_difference_is_recorded_without_waiving_reuse(model,tmp_path,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');b,rows,_=setup(model,tmp_path,1)
    forward=model.forward;decode=bench._decode
    def tagged(**kw):
        out=forward(**kw)
        out.full_control=kw.get('past_key_values') is None and 'pixel_values' in kw
        return out
    def changed(bundle,item,out,budget):
        result=decode(bundle,item,out,budget)
        if getattr(out,'full_control',False):result['text']+='full control difference'
        return result
    monkeypatch.setattr(model,'forward',tagged);monkeypatch.setattr(bench,'_decode',changed)
    result=bench.benchmark(b,1,rows,max_new_tokens=1)
    assert result['same_partition_equivalence'] and not result['complete_output_equivalence']
    assert len(result['full_output_differences'])==4
    assert result['protocol']=='native_online_stream_v3' and not result['quality_certified']
