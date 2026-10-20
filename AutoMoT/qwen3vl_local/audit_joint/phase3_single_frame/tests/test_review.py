import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from PIL import Image
from qwen3vl_local.audit_joint.phase3_single_frame.tests.test_single_frame import fixture,record
from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.audit_joint.phase3_single_frame import runtime,protocol,objective,worker,dataset,refresh_runtime
from qwen3vl_local.audit_joint import phase3_base_trial as trial
from qwen3vl_local.sft_new_loop_phase3 import train,preflight,training_state


def test_repeat_last_comparison_uses_same_denominator():
    a=dict(record(),id='a',history=['STOP']*4,predictions=['STOP']*4)
    b=dict(record(),id='b',predictions=['KEEP']*4)
    report=objective.metrics([a,b])
    assert report['total']['all']==8 and report['total']['repeat_last']==4
    p=report['paired_repeat_last']
    assert p['windows']==4 and p['model_set_accuracy']==p['repeat_last_set_accuracy']==1
    assert p['set_accuracy_delta']==0
    assert objective.metrics([b])['paired_repeat_last']['model_set_accuracy'] is None


def test_formal_input_ablations_fixed_ids_no_grad(tmp_path,monkeypatch,processor,model):
    _,_,art,_=fixture(tmp_path,monkeypatch)
    r=copy.deepcopy(next(iter(art['records'].values())))
    r.update(history=['STOP']*4,history_original=['UNKNOWN']*4,history_changed=[True]*4,history_imputed=[True]*4)
    monkeypatch.setattr(train,'_load_images',lambda *a:[Image.new('RGB',(32,32),'white')])
    bundle=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    before={k:v.detach().clone() for k,v in model.named_parameters()}
    model.train()
    report=worker.evaluate_variants(bundle,[r],tmp_path,'val')
    assert set(report['paired_input_ablations'])=={'no_history','native_history','blank_image'}
    for name in ('no_history','native_history','blank_image'):
        cases=trial.read(tmp_path/f'single_frame_val_{name}_cases.json')
        assert [x['id'] for x in cases]==[r['id']]
        assert cases[0]['history_input']==(['UNKNOWN']*4 if name!='blank_image' else ['STOP']*4)
        assert report['paired_input_ablations'][name]['windows']==4
    view,pic=runtime.evaluation_input(r,Image.new('RGB',(32,32),'white'),image_mode='blank')
    assert pic.getextrema()==((0,0),(0,0),(0,0)) and view['history']==r['history']
    assert model.training
    for n,p in model.named_parameters():
        assert p.grad is None
        torch.testing.assert_close(p,before[n])
    with pytest.raises(ValueError,match='order'):objective.paired_comparison([dict(r,predictions=['STOP']*4)],[])


@pytest.mark.parametrize('cutoff',(1,2))
def test_real_native_save_contract_and_epoch_resume_exact(tmp_path,monkeypatch,processor,cutoff):
    from qwen3vl_local.qwen35 import adapters
    job,rows,art,_=fixture(tmp_path,monkeypatch)
    job['dataset']=str(tmp_path/'single/dataset.json')
    names=('_make_item','_contiguous_work','_loss_batch','SEQUENCE_POLICY','_build_inputs',
        '_save_adapter','_write_run_metadata','_validation_work','_loss_one','_append_jsonl','verify_epoch_consumption','select_history_rgb_paths')
    originals={n:getattr(train,n) for n in names}
    for n,v in originals.items():monkeypatch.setattr(train,n,v)
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.values=torch.nn.Parameter(torch.zeros(320))
        def forward(self,input_ids,logits_to_keep,**kw):
            assert kw['image_grid_thw'].shape[0]==1
            logits=torch.nn.functional.dropout(self.values,p=.1,training=self.training)
            return SimpleNamespace(logits=logits[None,None].expand(1,len(logits_to_keep),-1))
    def load(*a,**kw):
        m=Tiny();return SimpleNamespace(model=m,unwrap=lambda:m,lora_target_modules=['values'],
            processor=processor,tokenizer=processor.tokenizer,device='cpu')
    monkeypatch.setattr(train,'load_model_with_lora',load)
    def save(m,folder):torch.save(m.state_dict(),Path(folder)/'adapter_model.pt')
    monkeypatch.setattr(adapters,'save_adapter',save)
    original_make_args=runtime.make_args
    def args(j,a):
        cfg=original_make_args(j,a);cfg.device='cpu';cfg.focus_balance_count=1;cfg.warmup_steps=2
        cfg.eval_steps=10;cfg.eval_balance_count=1;cfg.generation_eval_steps=0;cfg.save_final=True
        return cfg
    monkeypatch.setattr(runtime,'make_args',args)
    def run(name,resume=None,stop=None):
        for n,v in originals.items():setattr(train,n,v)
        j=dict(job,output=str(tmp_path/name),resume=str(resume) if resume else None,resume_epoch=cutoff if resume else 0,
               stop_after_epoch=stop)
        runtime.install(art,j['output'])
        if resume:
            proof=worker.check_resume(j,art,world_size=1,steps_per_epoch=10)
            assert proof['next_epoch']==cutoff and proof['remaining_epochs']==3-cutoff
        train.train(args(j,art))
        return j
    full=run('full');partial=run('partial',stop=cutoff)
    checkpoint=Path(partial['output'])/f'train/training_states/epoch_{cutoff:04d}'
    before=training_state.files(checkpoint)
    resumed=run('resumed',checkpoint)
    assert training_state.files(checkpoint)==before
    states=[torch.load(Path(j['output'])/'train/training_states/epoch_0003/training_state.pt',weights_only=True) for j in (full,resumed)]
    def same(a,b):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:same(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):same(x,y)
        else:assert a==b
    for field in ('trainable','optimizer','scheduler','rng','sampling_history','global_step'):same(states[0][field],states[1][field])
    assert states[1]['global_step']==30
    root=Path(resumed['output'])/'train'
    assert not (root/'training_states/epoch_0001').exists()
    for path in (root/'final/sft_new_loop_phase3_adapter_config.json',root/'train_run_manifest.json'):
        value=trial.read(path);protocol.validate(value,art)
        assert value['history_rgb_count']==1 and value['history_rgb_mode']=='1rgb_current'
        assert value['prompt_variant']==value['prompt_name'] and value['output_query_count']==4
        corrupted=dict(value,history_rgb_count=4)
        with pytest.raises(ValueError):protocol.validate(corrupted,art)
    consumed=trial.read(root/'balance/single_frame_consumed_epoch_0003.json')
    assert consumed['total_supervised_windows']==40
    assert consumed['supervised_windows_per_anchor_histogram']=={'4':10}
    damaged=copy.deepcopy(art);damaged['config']['lateral_weight']=2
    with pytest.raises(ValueError,match='mismatch'):worker.check_resume(resumed,damaged,world_size=1,steps_per_epoch=10)
    with (checkpoint/'training_state.pt').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='changed'):worker.check_resume(resumed,art,world_size=1,steps_per_epoch=10)


def test_runtime_only_refresh_preserves_labels_and_rejects_label_changes(tmp_path,monkeypatch):
    job,rows,art,_=fixture(tmp_path,monkeypatch)
    p=tmp_path/'single/dataset.json';receipt=p.with_name('dataset_summary.json')
    old=trial.read(p);r=trial.read(receipt)
    old['sources']['experiment_sources']['runtime.py']='older-runtime-sha'
    r['sources']=copy.deepcopy(old['sources']);trial.write(p,old)
    r['artifact_sha256']=trial.file_sha(p);trial.write(receipt,r)
    note=refresh_runtime.refresh(p,tmp_path/'refreshed')
    new=dataset.load(tmp_path/'refreshed/dataset.json')
    assert new['records']==old['records'] and note['labels_recomputed'] is False
    old['sources']['experiment_sources']['history.py']='changed-label-code';r['sources']=copy.deepcopy(old['sources'])
    trial.write(p,old);r['artifact_sha256']=trial.file_sha(p);trial.write(receipt,r)
    with pytest.raises(ValueError,match='Label/data/sampling'):refresh_runtime.refresh(p,tmp_path/'forbidden')
