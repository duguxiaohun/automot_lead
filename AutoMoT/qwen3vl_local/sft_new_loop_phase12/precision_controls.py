"""Temporary inference-only interventions for the frozen-run diagnostic worker.

No import-time torch/CUDA access; not used by production training or inference.
"""
from contextlib import contextmanager

CONTROLS=('bf16','fp32_penalty','fp32_head','fp32')


def validate_controls(controls,*,allow_single=False):
    if (len(controls)<(1 if allow_single else 2) or controls[0]!='bf16' or len(set(controls))!=len(controls)
            or any(c not in CONTROLS for c in controls)
            or ('fp32' in controls and controls[-1]!='fp32')):
        raise ValueError('Use unique controls: bf16 first, at least one control, whole-model fp32 last')
    return list(controls)


@contextmanager
def apply(model,control):
    """Restore methods, penalty and backend flags even on failure.

    fp32_head projects the BF16 hidden states with a temporary FP32 copy of
    the SAME output weights, then computes repetition penalties in FP32.
    Parameters (including tied embeddings), buffers and cache limits stay intact.
    Whole-model conversion, if requested, is performed explicitly by the worker
    before the terminal fp32 control; this context never converts the model.
    """
    import torch
    import torch.nn.functional as F
    from qwen3vl_local.sft_v3 import train as native
    if control not in CONTROLS or model.training:raise ValueError('Precision controls require known mode and eval model')
    dtype=next(model.parameters()).dtype
    if dtype!=(torch.float32 if control=='fp32' else torch.bfloat16):
        raise ValueError('Body dtype does not match precision control')
    if getattr(native,'_precision_diagnostic_active',False):raise ValueError('Precision controls cannot be nested/concurrent')
    original_penalty=native._apply_repetition_penalty
    matmul_tf32=torch.backends.cuda.matmul.allow_tf32;cudnn_tf32=torch.backends.cudnn.allow_tf32
    head=None;had_forward=False;old_forward=None;patched_head=False
    native._precision_diagnostic_active=True
    try:
        profile=dict(control=control,body_dtype=str(dtype),penalty_dtype='torch.float32' if control!='bf16' else 'torch.bfloat16',
            output_projection_dtype='torch.float32' if control in ('fp32_head','fp32') else 'torch.bfloat16',
            context_mutates_parameters=False,body_promoted_from_bf16=control=='fp32',
            head_copy_bytes=0,production_default_changed=False)
        if control in ('fp32_penalty','fp32_head'):
            def penalty(logits,seen_ids,value):
                return original_penalty(logits.float(),seen_ids,value)
            native._apply_repetition_penalty=penalty
        if control=='fp32_head':
            head=model.get_output_embeddings()
            if type(head) is not torch.nn.Linear or head.weight.dtype!=torch.bfloat16:
                raise ValueError('FP32 head diagnostic requires the unmodified BF16 Linear output head')
            if head._forward_hooks or head._forward_pre_hooks or 'forward' in head.__dict__:
                raise ValueError('Refusing to replace an already customized output head')
            # No .data edit / .to() on the module: tied embedding weights retain
            # their dtype and identity. Convert weights once, not every token.
            weight=head.weight.detach().float()
            bias=head.bias.detach().float() if head.bias is not None else None
            profile['head_copy_bytes']=weight.numel()*weight.element_size()+(bias.numel()*bias.element_size() if bias is not None else 0)
            had_forward='forward' in head.__dict__;old_forward=head.__dict__.get('forward')
            def project(hidden):
                with torch.autocast(device_type=hidden.device.type,enabled=False):
                    return F.linear(hidden.float(),weight,bias)
            head.forward=project
            patched_head=True
            torch.backends.cuda.matmul.allow_tf32=False
        if control=='fp32':
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        profile.update(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)
        with torch.no_grad():yield profile
    finally:
        if patched_head:
            if had_forward:head.forward=old_forward
            else:del head.forward
        native._apply_repetition_penalty=original_penalty
        torch.backends.cuda.matmul.allow_tf32=matmul_tf32;torch.backends.cudnn.allow_tf32=cudnn_tf32
        del native._precision_diagnostic_active
