import json
import random
import sys
from types import SimpleNamespace

import pytest
import torch

from . import train,training_state as recovery,preflight
from .test_entry_admission import fixture


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def test_actual_loop_epoch_resume_matches_uninterrupted_optimizer_rng_and_exposure(tmp_path,monkeypatch):
    index=fixture(tmp_path,'valid');base=tmp_path/'base';base.mkdir();(base/'config.json').write_text('{}')
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    bundles=[]
    def load(*a,**kw):
        model=torch.nn.Sequential(torch.nn.Linear(1,3),torch.nn.Dropout(.3),torch.nn.Linear(3,1))
        bundle=SimpleNamespace(model=model,unwrap=lambda:model,lora_target_modules=['0','2'])
        bundles.append(bundle);return bundle
    monkeypatch.setattr(train,'load_model_with_lora',load)
    monkeypatch.setattr(train,'_load_images',lambda *a,**kw:[])
    monkeypatch.setattr(train,'_build_inputs',lambda *a,**kw:{})
    def loss(bundle,packed,specs):
        x=torch.ones(len(packed),1)*(1+random.random())
        return bundle.model(x).square().mean(),[{} for _ in packed]
    monkeypatch.setattr(train,'_loss_batch',loss)
    common=['train','--index',str(index),'--model-dir',str(base),'--device','cpu',
        '--num-epochs','3','--focus-balance-count','4','--action-output-mode','choice',
        '--eval-steps','0','--generation-eval-steps','0','--save-steps','0','--no-save-final',
        '--no-save-best-val','--no-save-best-generation','--no-tb','--no-require-invalid-coverage',
        '--batch-size','2','--grad-accum','2','--save-training-state']
    def run(name,extra=()):
        output=tmp_path/name
        monkeypatch.setattr(sys,'argv',common+['--output-dir',str(output),*extra])
        train.train(train.parse_args());return output
    full=run('full');first=run('first',['--stop-after-epoch','1'])
    ckpt=first/'training_states/epoch_0001'
    resumed=run('resumed',['--resume',str(ckpt)])
    state=lambda out:torch.load(out/'training_states/epoch_0003/training_state.pt',weights_only=True)
    a,b=state(full),state(resumed)
    for key in ('trainable','optimizer','scheduler','rng','sampling_history','global_step','next_epoch'):equal(a[key],b[key])
    assert not (first/'final').exists()
    assert json.loads((first/'training_progress.json').read_text())['completed_epochs']==1
    for epoch in (2,3):
        assert (full/f'epoch_{epoch:04d}_rank0_batches.jsonl').read_bytes()==(resumed/f'epoch_{epoch:04d}_rank0_batches.jsonl').read_bytes()
    # Incompatible total LR horizon and corrupted checkpoints fail before model load.
    loaded=len(bundles)
    with pytest.raises(ValueError,match='mismatch'):run('bad_plan',['--resume',str(ckpt),'--num-epochs','4'])
    assert len(bundles)==loaded
    with (ckpt/'training_state.pt').open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='changed or incomplete'):run('corrupt',['--resume',str(ckpt)])
    assert len(bundles)==loaded


def test_checkpoint_selection_is_immutable_and_incomplete_state_is_rejected(tmp_path):
    model=torch.nn.Linear(1,1);opt=torch.optim.AdamW(model.parameters());scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:1.)
    output=tmp_path/'run';output.mkdir();best=output/'best_val';best.mkdir();(best/'weights').write_bytes(b'best')
    (output/'best_val.json').write_text('{"score":1}')
    binding=dict(configuration={'num_epochs':3},steps_per_epoch=4,world_size=1)
    folder=output/'state'
    recovery.save(folder,output,model,opt,scheduler,binding=binding,rng=[recovery.rng_state('cpu')],
                  state=dict(next_epoch=1,global_step=4,sampling_history={'next_epoch':1}))
    (best/'weights').write_bytes(b'later')
    recovery.read(folder,binding)
    new=tmp_path/'new';new.mkdir();recovery.restore_selection(folder,new)
    assert (new/'best_val/weights').read_bytes()==b'best'
    (folder/'receipt.json').unlink()
    with pytest.raises(FileNotFoundError):recovery.read(folder,binding)


def test_pending_gradients_cannot_be_published(tmp_path):
    model=torch.nn.Linear(1,1);opt=torch.optim.AdamW(model.parameters());scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:1.)
    model(torch.ones(1,1)).sum().backward()
    with pytest.raises(ValueError,match='cleared gradients'):
        recovery.save(tmp_path/'state',tmp_path,model,opt,scheduler,binding={},state={},rng=[])
    assert not (tmp_path/'state').exists()


def test_configuration_binding_is_json_stable_and_detects_base_changes(tmp_path):
    base=tmp_path/'base';base.mkdir();(base/'config.json').write_text('{}')
    args=SimpleNamespace(model_dir=str(base),num_epochs=3,validation_sampling={'tuple':(1,2)})
    before=recovery.binding(args,4,1280)
    assert before==json.loads(json.dumps(before))
    (base/'config.json').write_text('{"changed": true}')
    assert recovery.binding(args,4,1280)!=before


def test_failed_checkpoint_write_does_not_publish_a_receipt_or_latest(tmp_path,monkeypatch):
    model=torch.nn.Linear(1,1);opt=torch.optim.AdamW(model.parameters());scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:1.)
    def fail(*a,**kw):raise OSError('interrupted write')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError,match='interrupted write'):
        recovery.save(tmp_path/'state',tmp_path,model,opt,scheduler,binding={},state={},rng=[])
    assert not (tmp_path/'state').exists() and not (tmp_path/'state.pending').exists()
    assert not (tmp_path/'latest_training_state.json').exists()
