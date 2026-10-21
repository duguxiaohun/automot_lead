"""Actual native small Qwen3.5 image forward/backward; no downloaded weights."""
from types import SimpleNamespace
import copy
import pytest
import torch
from PIL import Image
from tokenizers.pre_tokenizers import ByteLevel

from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35.vendor.tokenization_qwen3_5 import Qwen3_5Tokenizer
from qwen3vl_local.audit_joint.geometry_v1 import codec, objective, train, dataset, expert
from .test_geometry import record


def test_all_nine_history_tokens_and_actual_qwen_backward_reload(processor,model,tmp_path):
    bundle=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    r=record();image=Image.new('RGB',(32,32))
    before=len(bundle.tokenizer)
    for token in range(9):
        for item in r['observation']['history']:item['token_id']=token
        packed=objective.pack(bundle,r,image)
        assert len(packed['positions'])==4 and len(packed['inputs']['image_grid_thw'])==1
    assert len(bundle.tokenizer)==before
    poisoned=copy.deepcopy(r);poisoned['targets']='NEVER_VISIBLE'
    assert torch.equal(objective.pack(bundle,poisoned,image)['inputs']['input_ids'],packed['inputs']['input_ids'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    result=train.train_epoch(bundle,[r],optimizer,scheduler,lambda row:objective.pack(bundle,row,image))
    assert result['steps']==1 and result['actual_exposure']=={r['id']:1}
    assert model.lm_head.weight.grad is not None and torch.isfinite(model.lm_head.weight.grad).all()
    model.eval()
    prediction=objective.predict(bundle,r,image)
    assert prediction['source']=='predicted' and len(prediction['token_ids'])==4
    assert expert.condition(r,prediction=prediction)['geometry_token_ids'].tolist()==[prediction['token_ids']]
    checkpoint=tmp_path/'checkpoint';source=tmp_path/'dataset.json';source.write_text('{}')
    art=dict(contract=codec.contract(),code={'tiny':'native_qwen35'})
    binding=dict(contract=art['contract'],code=art['code'],dataset_sha256=dataset.sources.file_sha(source))
    train.save_checkpoint(checkpoint,bundle,optimizer,scheduler,binding=binding,epoch=1,step=1,
        exposure=result['actual_exposure'],status='epoch_complete')
    original=objective.readout(bundle,packed).detach().clone()
    with torch.no_grad():model.lm_head.weight.add_(1)
    train.restore(bundle,checkpoint,artifact=art,dataset_path=source)
    torch.testing.assert_close(original,objective.readout(bundle,packed),rtol=0,atol=0)


def test_reject_context_merged_token_even_if_standalone_ids_are_valid(processor):
    vocab=processor.tokenizer.get_vocab().copy()
    for letter in codec.LETTERS:vocab['Ċ'+letter]=len(vocab)
    tokenizer=Qwen3_5Tokenizer(vocab=vocab,merges=[('Ċ',x) for x in codec.LETTERS],
        additional_special_tokens=list(processor.tokenizer.all_special_tokens),eos_token='<|im_end|>')
    tokenizer._tokenizer.pre_tokenizer=ByteLevel(add_prefix_space=False,use_regex=False)
    processor.tokenizer=tokenizer
    bundle=SimpleNamespace(processor=processor,tokenizer=tokenizer)
    r=record()
    assert len(objective.choice_ids(tokenizer))==9
    with pytest.raises(ValueError,match='identity_mismatch'):objective.pack(bundle,r,Image.new('RGB',(32,32)))
