import json
from pathlib import Path
from types import SimpleNamespace
import sys
import zipfile

import pytest
from PIL import Image
from qwen3vl_local.audit_joint import phase3_base_prompt_experiment as lab
from qwen3vl_local.sft_new_loop_phase3.test_prompt_candidate import case


def test_replay_gpu_explicit_pin_skips_query(monkeypatch):
    monkeypatch.setenv('GPU_IDS','3');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','7')
    monkeypatch.setattr(lab.subprocess,'check_output',lambda *a,**k:pytest.fail('explicit pin must not query'))
    assert lab.select_replay_device('cuda:0')=='cuda:0'
    assert lab.os.environ['CUDA_VISIBLE_DEVICES']=='3'


def test_replay_gpu_auto_selection_overrides_old_mask(monkeypatch):
    monkeypatch.delenv('GPU_IDS',raising=False);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','7')
    monkeypatch.setattr(lab.subprocess,'check_output',lambda *a,**k:'0, 100, 5\n2, 50, 10\n3, 50, 2\n')
    assert lab.select_replay_device('cuda')=='cuda:0'
    assert lab.os.environ['CUDA_VISIBLE_DEVICES']=='3'


def test_replay_cpu_does_not_select_gpu(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','7')
    monkeypatch.setattr(lab.subprocess,'check_output',lambda *a,**k:pytest.fail('CPU must not query'))
    assert lab.select_replay_device('cpu')=='cpu'
    assert lab.os.environ['CUDA_VISIBLE_DEVICES']=='7'


def test_replay_gpu_rejects_ambiguous_selection(monkeypatch):
    monkeypatch.setenv('GPU_IDS','0,1')
    with pytest.raises(ValueError):lab.select_replay_device('cuda:0')
    with pytest.raises(ValueError):lab.select_replay_device('cuda:2')


@pytest.fixture
def prepared(tmp_path):
    row=case('choice')
    row['action_user_prompt']=lab.render(row['prompt_spec'])[0]
    del row['history_rgb_sha256']  # Received training val omits remote RGB hashes.
    for path in row['history_rgb_paths_used']:
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(8,8)).save(p)
    archive=tmp_path/'source.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('experiment/common_reference.json',json.dumps(dict(prompt_variant=lab.CANDIDATE_NAME)))
        z.writestr('experiment/train/final_generation_val_cases.jsonl',json.dumps(row)+'\n')
    reference=tmp_path/'ref.json'
    reference.write_text(json.dumps([dict(key=['val','s','r',10,'LEAD_BRAKE'],gt='STOP',pred='STOP',correct=True)]))
    plan=tmp_path/'plan'
    lab.prepare(archive,reference,tmp_path,plan)
    return tmp_path,plan,archive,reference


def test_prepare_real_render_and_inputs(prepared):
    root,plan,_,_=prepared
    m,ref,req=lab.load_plan(plan,root)
    assert len(ref)==len(req)==m['cases']==1
    assert req[0]['candidate']==lab.NOTE+req[0]['original']
    assert not any(k in req[0] for k in ('gt','evidence','future_speeds'))
    assert len(req[0]['images'])==4


@pytest.mark.parametrize('target',['requests','reference','image','source'])
def test_tamper_rejected(prepared,monkeypatch,target):
    root,plan,_,_=prepared
    if target=='source':monkeypatch.setattr(lab,'source_hashes',lambda:{})
    elif target=='image':(root/'s/r/rgb/0007.jpg').write_bytes(b'changed')
    else:(plan/(target+'.json')).write_text('[]')
    with pytest.raises(ValueError):lab.load_plan(plan,root)


def test_existing_output_rejected(prepared):
    root,plan,archive,reference=prepared
    with pytest.raises(FileExistsError):lab.prepare(archive,reference,root,plan)


def test_reference_subset_rejected(prepared):
    root,_,archive,reference=prepared
    reference.write_text('[]')
    with pytest.raises(ValueError):lab.prepare(archive,reference,root,root/'bad')
    assert not (root/'bad').exists()


def test_real_prepare_rejects_saved_prompt_drift(prepared):
    root,_,_,reference=prepared
    row=case('choice');row['action_user_prompt']='wrong'
    archive=root/'drift.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('experiment/common_reference.json',json.dumps(dict(prompt_variant=lab.CANDIDATE_NAME)))
        z.writestr('experiment/train/final_generation_val_cases.jsonl',json.dumps(row))
    with pytest.raises(ValueError,match='prompt mismatch'):lab.prepare(archive,reference,root,root/'bad')


@pytest.mark.parametrize('limit',[4096,10])
def test_paired_replay_and_context_guard(prepared,monkeypatch,limit):
    import torch
    root,plan,_,_=prepared
    class Processor:
        def apply_chat_template(self,messages,**kwargs):return messages[-1]['content'][-1]['text']
        def __call__(self,**kwargs):
            assert kwargs['truncation'] is False
            return dict(input_ids=torch.zeros((1,100),dtype=torch.long))
    bundle=SimpleNamespace(processor=Processor(),model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=limit)))
    seen=[]
    def start(bundle,messages):
        seen.append(messages)
        return messages[-1]['content'][-1]['text']
    fake=SimpleNamespace(_validate_action_adapter=lambda *a,**kw:dict(history_rgb_mode='4rgb',prompt_variant=lab.CANDIDATE_NAME),
        load_eval_bundle=lambda *a,**kw:bundle,_kv_start_state=start,
        _student_generate_kv=lambda b,s,n:('RESUME' if s.startswith(lab.NOTE) else 'STOP',None,None))
    monkeypatch.setitem(sys.modules,'qwen3vl_local.sft_new_loop_phase3.eval',fake)
    model=root/'model';model.mkdir();(model/'config.json').write_text('{}')
    adapter=root/'adapter';adapter.mkdir();(adapter/'weights').write_bytes(b'fixture')
    out=root/'run'
    if limit==10:
        with pytest.raises(ValueError,match='context budget'):lab.replay(plan,root,model,adapter,out,'cpu')
        assert not (out/'report.json').exists()
        return
    lab.replay(plan,root,model,adapter,out,'cpu')
    report=json.loads((out/'report.json').read_text())
    assert len(seen)==2 and report['completed']
    assert seen[0][-1]['content'][0]['image'] is seen[1][-1]['content'][0]['image']
    assert report['saved_baseline_predictions_changed']==0
    assert report['paired_prompt_effect']['status']=='regression_detected'
    assert not report['promotion_authorized']
    # Exported candidate predictions must feed the standalone CLI unchanged,
    # paired to the fresh original inference, not the historical reference.
    import subprocess
    result=subprocess.run([sys.executable,'-B','-m','qwen3vl_local.audit_joint.phase3_base_regression_review',
        '--reference',str(out/'fresh_reference.json'),'--predictions',str(out/'candidate_predictions.json'),
        '--output',str(out/'independent_review.json')],capture_output=True,text=True)
    assert result.returncode==2,result.stderr
    assert json.loads((out/'independent_review.json').read_text())==report['paired_prompt_effect']
