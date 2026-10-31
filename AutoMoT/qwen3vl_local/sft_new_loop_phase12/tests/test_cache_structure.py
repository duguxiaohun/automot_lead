from types import SimpleNamespace
import copy
import pytest
import torch
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35.tests.test_text_prefix import inputs
from qwen3vl_local.qwen35 import text_prefix as tp
from qwen3vl_local.sft_new_loop_phase12 import cache,cache_structure as structure


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('prefix',[64,65])
def test_real_hybrid_distinguishes_partition_and_cross_image_reuse(model,monkeypatch,dtype,prefix):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');model.to(dtype=dtype)
    bundle=SimpleNamespace(model=model)
    a=inputs(prefix=prefix);b=inputs(prefix=prefix,grid=6)
    bank=tp.bank(bundle);before=copy.deepcopy(bank.report())
    state={k:v.clone() for k,v in model.state_dict().items()}
    report=structure.measure(bundle,a,cache,other_item=b)
    assert report['reuse_exercised'] and report['status'].startswith('reuse_matches')
    assert not report['cache_certified'] and report['scope']=='prefill_only_not_complete_generation'
    assert bank is tp.bank(bundle) and bank.report()==before
    for name,row in report['comparisons'].items():
        assert row['layer_coverage_complete'] and row['sequence_length_equal']
        assert len(row['full_final_states'])==8 # three hybrid + one attention layer
        if name!='cold_split_vs_full':assert row['exact_on_observed_tensors']
    assert report['comparisons']['cold_split_vs_full']['rope_delta']['exact']
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    assert all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in model.model.language_model.layers)


def test_no_retention_not_misreported_as_cache_hit(model,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    bundle=SimpleNamespace(model=model,_text_prefix_cache=tp.TextPrefixCache(max_bytes=1))
    result=structure.measure(bundle,inputs(),cache)
    assert result['status']=='reuse_not_exercised' and not result['reuse_exercised']
    assert not result['warm_cache']['hit']


def test_corrupted_retained_state_is_not_hidden_by_numerical_tolerance(model,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');bundle=SimpleNamespace(model=model)
    original=tp.TextPrefixCache.prefill;calls=[]
    def corrupted(bank,model,item):
        calls.append(1)
        if bank.entries:
            retained=next(iter(bank.entries.values()))[0]
            retained.recurrent_states[0].add_(1)
        return original(bank,model,item)
    monkeypatch.setattr(tp.TextPrefixCache,'prefill',corrupted)
    result=structure.measure(bundle,inputs(),cache)
    assert len(calls)==2 and result['reuse_exercised']
    assert not result['retained_prefix_unchanged_after_hit']
    assert not result['comparisons']['warm_vs_cold']['exact_on_observed_tensors']
    assert result['status']=='repeat_or_reuse_difference_requires_investigation'
    with pytest.raises(ValueError,match='mutated'):structure.verify_reuse(result)
    assert not hasattr(bundle,'_text_prefix_cache')


def test_hooks_and_delta_restored_after_failure(model,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');bundle=SimpleNamespace(model=model)
    delta=torch.tensor([[77]]);model.model.rope_deltas=delta
    def fail(*args,**kw):raise RuntimeError('fixture projection failure')
    monkeypatch.setattr(model.lm_head,'forward',fail)
    with pytest.raises(RuntimeError,match='projection failure'):structure.measure(bundle,inputs(),cache)
    assert model.model.rope_deltas is delta and not hasattr(bundle,'_text_prefix_cache')
    for module in model.modules():assert not module._forward_hooks and not module._forward_pre_hooks


def test_margin_bound_is_sufficient_not_necessary_and_not_acceptance():
    ref=torch.tensor([[1.,2.,10.]])
    large=structure.decision_margin(ref+.01,ref)
    assert large['sufficient_stability_bound'] and large['argmax_equal']
    tie=torch.tensor([[1.,1.]])
    unproven=structure.decision_margin(tie,tie)
    assert not unproven['sufficient_stability_bound'] and unproven['argmax_equal']
    changed=structure.decision_margin(torch.tensor([[1.001,1.]]),torch.tensor([[1.,1.001]]))
    assert not changed['sufficient_stability_bound'] and not changed['argmax_equal']
    assert structure.decision_margin(torch.tensor([[float('nan'),1.]]),tie)['status']=='nonfinite'


def test_metrics_preserve_shape_and_detect_nonfinite_or_missing_state():
    a=torch.zeros(2,3);b=a.clone();b[1,2]=2
    result=structure.difference(a,b)
    assert result['shape']==[2,3] and result['max_abs']==2 and result['differing_elements']==1
    assert not result['exact'] and result['relative_l2']==1
    b[1,2]=float('inf')
    assert structure.difference(a,b)['nonfinite']==1
    assert not structure.difference(a,b.reshape(6))['comparable']


@pytest.mark.parametrize('cut',[10,64,70])
@pytest.mark.parametrize('decay',[.01,1.])
def test_delta_rule_state_handoff_matches_true_fp64_recurrence(cut,decay):
    """Independent arithmetic-order control: lift ONLY local kernel FP32 scratch casts.

    This tests mathematical equivalence; it does not change production kernels
    or assert that BF16 kernels must produce identical bits/greedy trajectories.
    """
    import ast
    import inspect
    from qwen3vl_local.qwen35.vendor import modeling_qwen3_5 as vendor
    def double_kernel(function):
        tree=ast.parse(inspect.getsource(function))
        class ScratchPrecision(ast.NodeTransformer):
            def visit_Attribute(self,node):
                if isinstance(node.value,ast.Name) and node.value.id=='torch' and node.attr=='float32':
                    return ast.copy_location(ast.Attribute(node.value,'float64',node.ctx),node)
                return self.generic_visit(node)
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Attribute) and node.func.attr=='float' and not node.args and not node.keywords:
                    node.func.attr='double'
                return node
        tree=ast.fix_missing_locations(ScratchPrecision().visit(tree))
        scope=dict(function.__globals__);exec(compile(tree,'<test-only-fp64-kernel>','exec'),scope)
        return scope[function.__name__]
    chunk=double_kernel(vendor.torch_chunk_gated_delta_rule)
    recurrent=double_kernel(vendor.torch_recurrent_gated_delta_rule)
    torch.manual_seed(18)
    q,k,v=[torch.randn(1,83,2,4,dtype=torch.float64) for _ in range(3)]
    g=-torch.rand(1,83,2,dtype=torch.float64)*decay;beta=torch.rand(1,83,2,dtype=torch.float64)
    args=dict(output_final_state=True,use_qk_l2norm_in_kernel=True)
    full,state=chunk(q,k,v,g,beta,**args)
    serial,serial_state=recurrent(q,k,v,g,beta,initial_state=None,**args)
    first,initial=chunk(q[:,:cut],k[:,:cut],v[:,:cut],g[:,:cut],beta[:,:cut],**args)
    tail,tail_state=chunk(q[:,cut:],k[:,cut:],v[:,cut:],g[:,cut:],beta[:,cut:],initial_state=initial,**args)
    torch.testing.assert_close(full,serial,atol=2e-14,rtol=2e-14)
    torch.testing.assert_close(state,serial_state,atol=2e-14,rtol=2e-14)
    torch.testing.assert_close(torch.cat((first,tail),1),full,atol=2e-14,rtol=2e-14)
    torch.testing.assert_close(tail_state,state,atol=2e-14,rtol=2e-14)


def test_position_errors_are_structural_failures_even_when_reuse_is_repeatable(model,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');original=model.forward
    def incorrect_position(**kw):
        if 'position_ids' in kw:kw['position_ids']=kw['position_ids']+1
        return original(**kw)
    monkeypatch.setattr(model,'forward',incorrect_position)
    result=structure.measure(SimpleNamespace(model=model),inputs(),cache)
    assert not result['comparisons']['cold_split_vs_full']['position_metadata_equal']
    assert result['status']=='structural_contract_violation'
    with pytest.raises(ValueError,match='structure failure'):structure.verify_reuse(result)


def test_real_nonzero_lora_state_reuse_is_exact(model,monkeypatch):
    from peft import LoraConfig,get_peft_model
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    wrapped=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,target_modules=['in_proj_qkv'],lora_dropout=0.))
    with torch.no_grad():
        for name,param in wrapped.named_parameters():
            if 'lora_B' in name:param.normal_(0,.02)
    wrapped.eval();bundle=SimpleNamespace(model=wrapped)
    result=structure.measure(bundle,inputs(),cache,other_item=inputs(grid=6))
    assert result['status'].startswith('reuse_matches') and not hasattr(bundle,'_text_prefix_cache')
