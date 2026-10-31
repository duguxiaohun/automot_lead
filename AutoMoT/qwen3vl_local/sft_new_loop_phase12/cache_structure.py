"""Frozen-model prefill diagnostics: numerical partitioning vs retained-state reuse.

measure collects evidence; verify_reuse applies the new same-partition gate.
No import-time torch/CUDA access or production precision change.
"""
from contextlib import contextmanager
import hashlib

STATES=('key_cache','value_cache','conv_states','recurrent_states')


def difference(a,b):
    """Bounded-memory, whole-tensor metrics; no guessed numerical tolerance."""
    import torch
    if a.shape!=b.shape or a.dtype!=b.dtype:
        return dict(comparable=False,exact=False,shape_a=list(a.shape),shape_b=list(b.shape),
                    dtype_a=str(a.dtype),dtype_b=str(b.dtype))
    shape=list(a.shape)
    a=a.detach().reshape(-1).cpu();b=b.detach().reshape(-1).cpu()
    maximum=ss_delta=ss_reference=0.;nonfinite=0;different=0
    for start in range(0,a.numel(),262144):
        x=a[start:start+262144].double();y=b[start:start+262144].double()
        finite=torch.isfinite(x)&torch.isfinite(y);nonfinite+=int((~finite).sum())
        different+=int((x!=y).sum());d=(x[finite]-y[finite]).abs()
        if d.numel():maximum=max(maximum,float(d.max()));ss_delta+=float((d*d).sum());ss_reference+=float((y[finite]**2).sum())
    return dict(comparable=True,shape=shape,dtype=str(a.dtype),elements=a.numel(),nonfinite=nonfinite,
        exact=nonfinite==0 and different==0,differing_elements=different,max_abs=maximum,
        rms_abs=(ss_delta/max(1,a.numel()-nonfinite))**.5,
        relative_l2=(ss_delta/ss_reference)**.5 if ss_reference else (0. if ss_delta==0 else None))


def decision_margin(a,b):
    """m > 2 epsilon is a sufficient argmax-stability bound, NOT an error budget."""
    import torch
    if a.shape!=b.shape or a.ndim!=2 or a.shape[0]!=1 or a.shape[-1]<2:
        raise ValueError('Expected matched single-request next-token scores')
    x=a.detach().double();y=b.detach().double()
    if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):return dict(status='nonfinite',accepted=False)
    epsilon=float((x-y).abs().max());top=y.topk(2).values[0];margin=float(top[0]-top[1])
    return dict(status='measured',max_abs_delta=epsilon,reference_top2_margin=margin,
        sufficient_stability_bound=margin>2*epsilon,argmax_equal=int(x.argmax())==int(y.argmax()),
        candidate_token_id=int(x.argmax()),reference_token_id=int(y.argmax()),
        scope='same_history_scores_only_not_model_quality_or_cache_acceptance')


def _copy(tensor):
    return tensor.detach().to('cpu',copy=True).contiguous()


def _state(cache):
    result={}
    for name in STATES:
        for i,value in enumerate(getattr(cache,name)):
            if value is not None:result[f'{i:03d}.{name}']=_copy(value)
    return result


def _fingerprint(bank):
    """All retained prefix states, not just sequence lengths or tensor addresses."""
    h=hashlib.sha256()
    for key,(state,size) in sorted(bank.entries.items()):
        h.update(repr((key,size,state.get_seq_length())).encode())
        for name,tensor in _state(state).items():
            h.update(repr((name,str(tensor.dtype),tuple(tensor.shape))).encode())
            import torch
            h.update(tensor.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@contextmanager
def _observe(model,positions):
    """Sample each decoder stage at matching absolute positions; keep no GPU views."""
    import torch
    from qwen3vl_local.qwen35.text_prefix import _native
    language=_native(model).model.language_model
    samples={};handles=[];active={}
    def capture(name,value):
        if isinstance(value,tuple):value=value[0]
        if not isinstance(value,torch.Tensor) or value.ndim!=3:raise ValueError('Unexpected decoder stage output')
        for absolute,local in active['indices']:
            samples[f'{name}@{absolute}']=_copy(value[:,local,:])
    def before(index):
        def hook(module,args,kwargs):
            value=args[0] if args else kwargs['hidden_states']
            cp=kwargs['cache_position'];start=int(cp[0]);length=cp.numel()
            active['indices']=[(p,p-start) for p in positions if start<=p<start+length]
            capture(f'{index:03d}.input',value)
            for absolute,local in active['indices']:
                samples[f'{index:03d}.cache_position@{absolute}']=_copy(cp[local:local+1])
                samples[f'{index:03d}.position_ids@{absolute}']=_copy(kwargs['position_ids'][...,local:local+1])
        return hook
    def after(name):
        def hook(module,args,out):capture(name,out)
        return hook
    try:
        for i,layer in enumerate(language.layers):
            handles.append(layer.register_forward_pre_hook(before(i),with_kwargs=True))
            for name in ('input_layernorm','linear_attn','self_attn','post_attention_layernorm','mlp'):
                part=getattr(layer,name,None)
                if part is not None:handles.append(part.register_forward_hook(after(f'{i:03d}.{name}')))
            handles.append(layer.register_forward_hook(after(f'{i:03d}.output')))
        yield samples
    finally:
        for handle in handles:handle.remove()


def _capture(bundle,item,forward,positions,decode=None):
    with _observe(bundle.model,positions) as layers:
        out=forward(item)
        result=dict(layers=layers,states=_state(out.past_key_values),
            next_logits=_copy(out.logits[:,-1,:]),rope_delta=_copy(out.rope_deltas),
            sequence_length=out.past_key_values.get_seq_length())
    if decode is not None:result['answer']=decode(out)
    del out
    return result


def _compare(candidate,reference,*,from_position=0):
    layers={k:difference(v,reference['layers'][k]) for k,v in candidate['layers'].items()
        if int(k.rsplit('@',1)[1])>=from_position and k in reference['layers']}
    expected={k for k in reference['layers'] if int(k.rsplit('@',1)[1])>=from_position}
    names=set(candidate['states'])|set(reference['states'])
    states={k:difference(candidate['states'][k],reference['states'][k]) if k in candidate['states'] and k in reference['states']
        else dict(comparable=False,exact=False,missing=True) for k in sorted(names)}
    logits=difference(candidate['next_logits'],reference['next_logits']);rope=difference(candidate['rope_delta'],reference['rope_delta'])
    covered=bool(layers) and set(layers)==expected
    same_length=candidate['sequence_length']==reference['sequence_length']
    metrics=[*layers.values(),*states.values(),logits,rope]
    metadata=covered and same_length and rope['exact'] and all(v['exact'] for k,v in layers.items()
        if '.position_ids@' in k or '.cache_position@' in k)
    compatible=covered and same_length and all(v['comparable'] for v in metrics)
    finite=all(v.get('nonfinite',1)==0 for v in metrics)
    changed_layers=[int(k.split('.',1)[0]) for k,v in layers.items() if not v['exact']]
    return dict(sampled_layers=layers,full_final_states=states,next_logits=logits,rope_delta=rope,
        layer_coverage_complete=covered,sequence_length_equal=same_length,position_metadata_equal=metadata,
        shapes_and_dtypes_compatible=compatible,all_finite=finite,first_different_layer=min(changed_layers) if changed_layers else None,
        exact_on_observed_tensors=covered and same_length and all(v['exact'] for v in [*layers.values(),*states.values(),logits,rope]),
        first_different_sample=next((k for k,v in layers.items() if not v['exact']),None),
        next_token=decision_margin(candidate['next_logits'],reference['next_logits']))


def measure(bundle,item,cache,*,other_item=None,on_evidence=None):
    """Compare full/repeated full, fresh split/hit, and cross-image state reuse.

    All final K/V/conv/recurrent tensors are compared; hidden layers are sampled.
    Only prefill is executed. Greedy generation/semantic validity remain separate.
    """
    import torch
    from qwen3vl_local.qwen35 import text_prefix as tp
    if bundle.model.training or not tp.cache_enabled():raise ValueError('Structure diagnostic requires eval and enabled text cache')
    n=min(int(torch.where(item['input_ids'][0]==tp._native(bundle.model).config.vision_start_token_id)[0][0]),
          (getattr(bundle,'_text_prefix_cache',None) or tp.TextPrefixCache()).max_tokens)
    length=item['input_ids'].shape[1];positions=sorted({0,n-1,n,min(n+1,length-1),length-1})
    language=tp._native(bundle.model).model.language_model
    kernels={name:sorted({getattr(getattr(layer.linear_attn,name),'__name__',type(getattr(layer.linear_attn,name)).__name__)
        for layer in language.layers if hasattr(layer,'linear_attn')})
        for name in ('chunk_gated_delta_rule','recurrent_gated_delta_rule','causal_conv1d_update')}
    report=dict(scope='prefill_only_not_complete_generation',sampled_positions=positions,prefix_tokens=n,
        cache_certified=False,quality_certified=False,measurement_applies_no_gate=True,comparisons={},
        backend=dict(attention=getattr(language.config,'_attn_implementation',None),kernels=kernels,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            matmul_precision=torch.get_float32_matmul_precision(),
            bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark))
    def save():
        if on_evidence:on_evidence(report)
    full=lambda x:bundle.model(**x,use_cache=True,return_dict=True)
    native=tp._native(bundle.model);previous_delta=getattr(native.model,'rope_deltas',None)
    had_bank=hasattr(bundle,'_text_prefix_cache');previous_bank=getattr(bundle,'_text_prefix_cache',None)
    try:
        with torch.no_grad(),cache.isolated_text_cache(bundle):
            bank=tp.bank(bundle)
            report['cache_limits']=dict(max_tokens=bank.max_tokens,max_bytes=bank.max_bytes,max_entries=bank.max_entries)
            a=_capture(bundle,item,full,positions)
            b=_capture(bundle,item,full,positions)
            report['comparisons']['full_repeat_vs_full']=_compare(b,a);del b;save()
            cold=_capture(bundle,item,lambda x:tp.prefill(bundle,x),positions)
            report['cold_cache']=dict(bank.last)
            report['comparisons']['cold_split_vs_full']=_compare(cold,a);del a;save()
            fingerprint=_fingerprint(bank)
            warm=_capture(bundle,item,lambda x:tp.prefill(bundle,x),positions)
            report['warm_cache']=dict(bank.last)
            report['retained_prefix_unchanged_after_hit']=fingerprint==_fingerprint(bank)
            report['comparisons']['warm_vs_cold']=_compare(warm,cold,from_position=n);del warm;save()
            if other_item is not None:
                if (not torch.equal(item['input_ids'][:,:n],other_item['input_ids'][:,:n]) or
                        torch.equal(item['pixel_values'],other_item['pixel_values'])):
                    raise ValueError('Cross-image control requires same text prefix and different image tensors')
                other_length=other_item['input_ids'].shape[1]
                other_positions=sorted({0,n-1,n,min(n+1,other_length-1),other_length-1})
                cross=_capture(bundle,other_item,lambda x:tp.prefill(bundle,x),other_positions)
                report['cross_image_cache']=dict(bank.last)
                with cache.isolated_text_cache(bundle):
                    fresh=_capture(bundle,other_item,lambda x:tp.prefill(bundle,x),other_positions)
                report['comparisons']['cross_image_hit_vs_fresh_split']=_compare(cross,fresh,from_position=n)
                del cross,fresh
                report['retained_prefix_unchanged_after_other_image']=fingerprint==_fingerprint(bank);save()
                returned=_capture(bundle,item,lambda x:tp.prefill(bundle,x),positions)
                report['return_cache']=dict(bank.last)
                report['comparisons']['return_after_other_image_vs_cold']=_compare(returned,cold,from_position=n);del returned
                report['retained_prefix_unchanged_after_return']=fingerprint==_fingerprint(bank)
            del cold
            report['reuse_exercised']=all(report[name]['hit'] and report[name]['reused_tokens']==n
                for name in ('warm_cache','cross_image_cache','return_cache') if name in report)
            checks=[v['exact_on_observed_tensors'] for k,v in report['comparisons'].items() if k!='cold_split_vs_full']
            checks+=[v for k,v in report.items() if k.startswith('retained_prefix_unchanged')]
            structurally_valid=all(v['position_metadata_equal'] and v['shapes_and_dtypes_compatible'] and v['all_finite']
                for v in report['comparisons'].values())
            report['status']=('structural_contract_violation' if not structurally_valid else
                'reuse_not_exercised' if not report['reuse_exercised'] else
                'repeat_or_reuse_difference_requires_investigation' if not all(checks) else
                'reuse_matches_fresh_split_on_observed_tensors_not_full_forward_certification')
            save()
    finally:
        native.model.rope_deltas=previous_delta
        if not had_bank and hasattr(bundle,'_text_prefix_cache'):delattr(bundle,'_text_prefix_cache')
        elif had_bank:bundle._text_prefix_cache=previous_bank
    return report


def verify_reuse(report,*,require_cross=False):
    """New runs gate on same-partition evidence; full-vs-split values are diagnostic.

    Recompute acceptance from individual metrics, never trust a status string alone.
    A cache that did not retain an entry has not demonstrated reuse correctness.
    """
    required={'full_repeat_vs_full','cold_split_vs_full','warm_vs_cold'}
    comparisons=report.get('comparisons',{})
    if not required<=set(comparisons):raise ValueError('Missing text state comparisons')
    cross='cross_image_cache' in report
    if require_cross and not cross:raise ValueError('Missing cross-image reuse evidence')
    if cross:
        required|={'cross_image_hit_vs_fresh_split','return_after_other_image_vs_cold'}
        if not required<=set(comparisons):raise ValueError('Missing cross-image comparisons')
    for name in ('warm_cache','cross_image_cache','return_cache') if cross else ('warm_cache',):
        state=report[name]
        if (state.get('hit') is not True or state.get('reused_tokens')!=report['prefix_tokens']
                or not report['prefix_tokens']>0):raise ValueError('Text cache reuse not exercised')
    for name in ('retained_prefix_unchanged_after_hit','retained_prefix_unchanged_after_other_image',
                 'retained_prefix_unchanged_after_return') if cross else ('retained_prefix_unchanged_after_hit',):
        if report.get(name) is not True:raise ValueError('Retained text prefix mutated')
    for name,result in comparisons.items():
        verify_comparison(result,exact=name!='cold_split_vs_full')
    return True


def verify_comparison(result,*,exact=True):
    for field in ('layer_coverage_complete','sequence_length_equal','position_metadata_equal',
                  'shapes_and_dtypes_compatible','all_finite'):
        if result.get(field) is not True:raise ValueError('Text state structure failure: '+field)
    if not result.get('sampled_layers') or not result.get('full_final_states'):
        raise ValueError('Missing observed text state tensors')
    metrics={**result['sampled_layers'],**result['full_final_states'],
             'next_logits':result['next_logits'],'rope_delta':result['rope_delta']}
    for key,metric in metrics.items():
        if metric.get('comparable') is not True or metric.get('nonfinite')!=0:
            raise ValueError('Incompatible or nonfinite text state')
        exact_required=(exact or key=='rope_delta'
                        or '.position_ids@' in key or '.cache_position@' in key)
        if exact_required and (metric.get('exact') is not True or metric.get('differing_elements')!=0
                               or metric.get('max_abs')!=0):
            raise ValueError('Same-partition text state changed: '+key)


STREAM_PROTOCOL='resident_text_questions_cross_image_v1'


def measure_stream(bundle,items,other_items,cache,*,max_new_tokens=160,on_evidence=None):
    """One bank: A/X, B/X, A/Y, B/Y, A/X, B/X (or the one-question subset).

    Actual bounded cache, fresh references in separate banks, full decode included.
    Diagnostics are outside production timing and never warm its bank.
    """
    import torch
    from qwen3vl_local.qwen35 import text_prefix as tp
    if bundle.model.training or not tp.cache_enabled() or len(items) not in (1,2) or len(items)!=len(other_items):
        raise ValueError('Expected one or two text questions and alternate real images')
    report=dict(protocol=STREAM_PROTOCOL,questions=len(items),steps=[],status='running',quality_certified=False)
    def save():
        if on_evidence:on_evidence(report)
    native=tp._native(bundle.model);previous_delta=getattr(native.model,'rope_deltas',None)
    try:
        with torch.no_grad(),cache.isolated_text_cache(bundle):
            bank=tp.bank(bundle);report['limits']={k:getattr(bank,k) for k in ('max_entries','max_bytes','max_tokens')}
            prefixes=[];bindings=[]
            for a,b in zip(items,other_items):
                n=int(torch.where(a['input_ids'][0]==native.config.vision_start_token_id)[0][0])
                if not 0<n<=bank.max_tokens:raise ValueError(f'Full text prefix requires {n} tokens; configured max_tokens={bank.max_tokens}')
                if max(a['input_ids'].shape[1],b['input_ids'].shape[1])>8192:raise ValueError('Stream probe exceeds context limit')
                if (not torch.equal(a['input_ids'][:,:n],b['input_ids'][:,:n]) or
                        int(torch.where(b['input_ids'][0]==native.config.vision_start_token_id)[0][0])!=n or
                        torch.equal(a['pixel_values'],b['pixel_values'])):
                    raise ValueError('Stream probe requires same text and different real image tensors')
                key=tuple(a['input_ids'][0,:n].tolist());prefixes.append(n)
                def sha(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                bindings.append(dict(prefix_sha256=hashlib.sha256(repr(key).encode()).hexdigest(),
                    image_sha256=[sha(a['pixel_values']),sha(b['pixel_values'])]))
            if len({b['prefix_sha256'] for b in bindings})!=len(items):raise ValueError('Expected distinct complete question prefixes')
            report['inputs']=bindings;save()
            fingerprint=None
            for image_index,group in ((0,items),(1,other_items),(0,items)):
                for arm,item in enumerate(group):
                    n=prefixes[arm];length=item['input_ids'].shape[1];positions=sorted({0,n-1,n,min(n+1,length-1),length-1})
                    decode=lambda out:cache._decode(bundle,item,out,max_new_tokens)
                    actual=_capture(bundle,item,lambda x:tp.prefill(bundle,x),positions,decode)
                    step=dict(question=arm,image=image_index,cache=dict(bank.last),resident_entries=len(bank.entries),
                        resident_bytes=sum(v[1] for v in bank.entries.values()),prefix_fingerprint=_fingerprint(bank))
                    report['steps'].append(step);save()
                    if len(report['steps'])==len(items):
                        fingerprint=step['prefix_fingerprint']
                        report['required_bytes']=sum(s['cache']['prefix_bytes'] for s in report['steps'])
                        report['required_entries']=len(items);save()
                        if bank.max_entries<len(items) or report['required_bytes']>bank.max_bytes:
                            raise ValueError(f'Text questions cannot coexist: required_bytes={report["required_bytes"]}, max_bytes={bank.max_bytes}, required_entries={len(items)}')
                    with cache.isolated_text_cache(bundle):
                        fresh=_capture(bundle,item,lambda x:tp.prefill(bundle,x),positions,decode)
                    step.update(comparison=_compare(actual,fresh,from_position=n),answer=actual['answer'],reference=fresh['answer'])
                    del actual,fresh
                    if image_index==1 or len(report['steps'])>2*len(items):
                        step['resident_prefixes_unchanged']=fingerprint==step['prefix_fingerprint']
                    save()
            report['cache']=bank.report()
            verify_stream(report,questions=len(items))
            report['status']='passed';save()
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');save()
        exc.evidence=report
        raise
    finally:native.model.rope_deltas=previous_delta
    return report


def verify_stream(report,*,questions=2):
    from .cache import assert_answers
    if report.get('protocol')!=STREAM_PROTOCOL or report.get('questions')!=questions:
        raise ValueError('Missing resident cross-image stream acceptance')
    steps=report.get('steps',[]);bindings=report.get('inputs',[]);limits=report['limits']
    if len(steps)!=3*questions or len(bindings)!=questions:
        raise ValueError('Incomplete cross-image/return stream')
    if len({b['prefix_sha256'] for b in bindings})!=questions or any(b['image_sha256'][0]==b['image_sha256'][1] for b in bindings):
        raise ValueError('Missing distinct questions or alternate images')
    required=sum(s['cache']['prefix_bytes'] for s in steps[:questions])
    if report.get('required_bytes')!=required or report.get('required_entries')!=questions:
        raise ValueError('Missing or incorrect measured text capacity')
    if limits['max_entries']<questions or required>limits['max_bytes']:
        raise ValueError(f'Text questions cannot coexist: required_bytes={required}, max_bytes={limits["max_bytes"]}, required_entries={questions}')
    for i,step in enumerate(steps):
        state=step['cache'];warm=i>=questions
        if step['question']!=i%questions or step['image']!=(1 if questions<=i<2*questions else 0):
            raise ValueError('Wrong alternating question/image order')
        if (state['prefix_tokens']!=state['complete_text_tokens'] or state['stored'] is not True or
                state['hit'] is not warm or state['full_hit'] is not warm or
                state['reused_tokens']!=(state['prefix_tokens'] if warm else 0) or
                state['computed_tokens']!=(0 if warm else state['prefix_tokens'])):
            raise ValueError('Resident text stream did not exercise complete reuse')
        if step['resident_entries']!=min(i+1,questions) or step['resident_bytes']>limits['max_bytes']:
            raise ValueError('Text stream evicted required question')
        if warm and (step.get('resident_prefixes_unchanged') is not True or
                     step['prefix_fingerprint']!=steps[questions-1]['prefix_fingerprint']):
            raise ValueError('Cross-image/decode mutated retained questions')
        verify_comparison(step['comparison'])
        assert_answers({'branches':[step['answer']]*2},{'branches':[step['reference']]*2})
    return True
