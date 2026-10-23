"""Actual small native Qwen3.5, one image, all four representations, no downloads."""
from types import SimpleNamespace
import copy
import pytest
import torch
from PIL import Image
from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.qwen35.tests.test_backend import model
from .. import objective, train, dataset, regions
from .test_single import record


@pytest.mark.parametrize('variant',regions.VARIANTS)
def test_native_qwen_backward_all_shared_letters_and_reload(processor,model,tmp_path,variant):
    bundle=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    r=record(variant=variant);image=Image.new('RGB',(32,32))
    count=len(regions.names(variant));before=len(bundle.tokenizer)
    for token in range(count):
        for h in r['observation']['history']:h['token_id']=token
        packed=objective.pack(bundle,r,image)
        assert len(packed['positions'])==4
    assert len(bundle.tokenizer)==before
    poisoned=copy.deepcopy(r);poisoned['targets']='NEVER_VISIBLE';poisoned['sampling_mass']='NEVER_VISIBLE'
    assert torch.equal(objective.pack(bundle,poisoned,image)['inputs']['input_ids'],packed['inputs']['input_ids'])
    opt=torch.optim.AdamW(model.parameters(),lr=.001);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    result=train.train_epoch(bundle,[r],opt,sched,lambda row:objective.pack(bundle,row,image))
    assert result['actual_exposure']=={r['id']:1}
    assert model.lm_head.weight.grad is not None and torch.isfinite(model.lm_head.weight.grad).all()
    artifact=dict(contract=regions.contract(variant),code={'tiny':'native_qwen35'})
    source=tmp_path/'dataset.json';source.write_text('{}')
    binding=dict(contract=artifact['contract'],code=artifact['code'],dataset_sha256=dataset.sources.file_sha(source))
    train.save_checkpoint(tmp_path/'ckpt',bundle,opt,sched,binding=binding,epoch=1,step=1,
        exposure=result['actual_exposure'],status='epoch_complete')
    model.eval();expected=objective.readout(bundle,packed).detach().clone()
    with torch.no_grad():model.lm_head.weight.add_(1)
    train.restore(bundle,tmp_path/'ckpt',artifact=artifact,dataset_path=source)
    torch.testing.assert_close(expected,objective.readout(bundle,packed),rtol=0,atol=0)
    wrong=dict(artifact,contract=dict(regions.contract(variant),version='legacy'))
    with pytest.raises(ValueError,match='contract'):train.restore(bundle,tmp_path/'ckpt',artifact=wrong,dataset_path=source)
