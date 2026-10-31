"""Numerical interventions, cleanup and real hybrid inference on CPU."""
from types import SimpleNamespace
import copy
import pytest
import torch
from qwen3vl_local.sft_new_loop_phase12 import precision_controls as controls,cache,cache_diagnose
from qwen3vl_local.sft_v3 import train as native
from qwen3vl_local.qwen35.tests.test_backend import model


class TiedHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=torch.nn.Embedding(2,2,dtype=torch.bfloat16)
        self.head=torch.nn.Linear(2,2,bias=False,dtype=torch.bfloat16)
        self.head.weight=self.embedding.weight
        with torch.no_grad():self.head.weight.copy_(torch.tensor([[1.,0.],[1.,.03125]]))
        self.eval()
    def get_output_embeddings(self):return self.head


def test_real_fp32_projection_resolves_rounding_tie_without_converting_tied_weights():
    model=TiedHead();hidden=torch.tensor([[21.875,1.]],dtype=torch.bfloat16)
    saved=model.head.weight.detach().clone();weight=model.head.weight
    old_penalty=native._apply_repetition_penalty
    rounded=model.head(hidden)
    assert rounded.tolist()==[[21.875,21.875]] and rounded.argmax().item()==0
    # Casting rounded logits cannot recover the sub-BF16 difference.
    with controls.apply(model,'fp32_penalty') as profile:
        out=native._apply_repetition_penalty(model.head(hidden),torch.tensor([],dtype=torch.long),1.05)
        assert out.dtype==torch.float32 and out.argmax().item()==0
        assert profile['head_copy_bytes']==0
    with controls.apply(model,'fp32_head') as profile:
        out=model.head(hidden)
        assert out.dtype==torch.float32 and out.tolist()==[[21.875,21.90625]] and out.argmax().item()==1
        assert not out.requires_grad and profile['head_copy_bytes']==16
        assert model.head.weight is model.embedding.weight is weight
        assert weight.dtype==torch.bfloat16 and torch.equal(weight,saved)
    assert 'forward' not in model.head.__dict__ and native._apply_repetition_penalty is old_penalty
    assert torch.equal(model.head(hidden),rounded)


@pytest.mark.parametrize('control',['fp32_penalty','fp32_head'])
def test_penalty_uses_fp32_arithmetic_and_restores_on_exception(control):
    model=TiedHead();original=native._apply_repetition_penalty
    flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    logits=torch.tensor([[23.,-23.]],dtype=torch.bfloat16);seen=torch.tensor([0,1])
    expected=original(logits.float(),seen,1.05)
    with pytest.raises(RuntimeError,match='inference failed'):
        with controls.apply(model,control):
            actual=native._apply_repetition_penalty(logits,seen,1.05)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            assert not torch.equal(actual,original(logits,seen,1.05).float())
            with pytest.raises(ValueError,match='nested'):
                with controls.apply(model,'bf16'):pass
            raise RuntimeError('inference failed')
    assert native._apply_repetition_penalty is original
    assert not hasattr(native,'_precision_diagnostic_active')
    assert 'forward' not in model.head.__dict__
    assert flags==(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)


@pytest.mark.parametrize('custom',['forward','hook','dtype','training'])
def test_refuses_incompatible_heads_and_preserves_customizations(custom):
    model=TiedHead();original=native._apply_repetition_penalty
    if custom=='forward':model.head.forward=lambda hidden:hidden
    elif custom=='hook':model.head.register_forward_hook(lambda *args:None)
    elif custom=='dtype':model.float()
    else:model.train()
    before=dict(model.head.__dict__)
    with pytest.raises(ValueError):
        with controls.apply(model,'fp32_head'):pass
    assert model.head.__dict__==before
    assert native._apply_repetition_penalty is original
    assert not hasattr(native,'_precision_diagnostic_active')


@pytest.mark.parametrize('layout',['0','1'])
@pytest.mark.parametrize('control',['bf16','fp32_penalty','fp32_head'])
def test_real_hybrid_controls_preserve_body_and_production_cache(model,monkeypatch,layout,control):
    from qwen3vl_local.qwen35.tests.test_text_prefix import inputs
    from qwen3vl_local.qwen35 import text_prefix as tp
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX',layout)
    model.bfloat16();before={k:v.clone() for k,v in model.state_dict().items()}
    a=inputs(prefix=65);b={k:v.clone() for k,v in a.items()};b['input_ids'][0,-1]=23
    if layout=='1':b['input_ids'][0,50]=155
    bundle=SimpleNamespace(model=model,device=torch.device('cpu'),
        tokenizer=SimpleNamespace(eos_token_id=1,convert_tokens_to_ids=lambda _:1),
        processor=SimpleNamespace(batch_decode=lambda ids,**kw:[str(ids.flatten().tolist())]))
    bank=tp.bank(bundle);prior=copy.deepcopy(bank.report())
    with controls.apply(model,control):
        out=model(**a,use_cache=False)
        assert out.logits.dtype==(torch.float32 if control=='fp32_head' else torch.bfloat16)
        del out
        result=cache_diagnose.measure(bundle,[a,b],cache,max_new_tokens=4,warmup_inputs=([a,b],),
            reference_difference=dict(arm=1,common_token_ids=[100,101]))
    assert result['full_answers_identical'] and result['cached_repeat_identical'] and result['independent_repeat_identical']
    probe=result['fixed_history_probe']
    assert probe['conditional_only'] and probe['common_token_ids']==[100,101]
    assert probe['cached']['dtype']==('torch.bfloat16' if control=='bf16' else 'torch.float32')
    assert probe['raw_cached']['dtype']==('torch.float32' if control=='fp32_head' else 'torch.bfloat16')
    assert bank is tp.bank(bundle) and bank.report()==prior
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert 'forward' not in model.get_output_embeddings().__dict__
