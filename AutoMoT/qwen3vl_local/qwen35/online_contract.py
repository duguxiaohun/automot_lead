"""Bind online rendering to the saved native adapter prompt contract."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from . import text_prefix
from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices


def _model_identity(model):
    adapters = tuple((name, str(module.active_adapters), str(getattr(module, 'disable_adapters', None)))
                     for name, module in model.named_modules() if hasattr(module, 'active_adapters')) if hasattr(model, 'named_modules') else ()
    return id(model), adapters


@dataclass(frozen=True)
class OnlinePromptContract:
    phase: int
    model_identity: tuple
    input_layout: str
    history_rgb_mode: str
    action_output_mode: str | None
    prompt_variant: str | None
    module: object
    prompt_sha256: str
    metadata_sha256: str


def bind_prompt_contract(bundle, phase, metadata):
    """For an already loaded/validated adapter, bind an immutable rendering contract.

    This verifies prompt/RGB/layout metadata, not tensor provenance. Use
    load_online_bundle when loading from disk to validate both together.
    """
    if phase not in (1, 2, 3):raise ValueError('Expected phase 1, 2 or 3')
    mode = metadata['history_rgb_mode']
    indices = list(history_rgb_indices(mode))
    if metadata.get('history_rgb_count') != len(indices) or metadata.get('history_rgb_selected_indices') != indices:
        raise ValueError('Online adapter RGB metadata mismatch')
    layout = metadata.get('input_layout', 'images_before_text_legacy')
    if layout != text_prefix.identity():raise ValueError('Online adapter input layout mismatch')
    variant = output_mode = None
    if phase == 3:
        from qwen3vl_local.sft_new_loop_phase3 import prompts as module, prompt_contract
        from qwen3vl_local.sft_new_loop_phase3.prompts import validate_action_output_mode
        variant = prompt_contract.adapter_prompt_variant(metadata)
        output_mode = validate_action_output_mode(metadata['action_output_mode'])
        expected = prompt_contract.action_prompt_sha256(history_rgb_mode=mode,
            action_output_mode=output_mode, prompt_variant=variant)
    else:
        from qwen3vl_local.action_prior.prompt_versions import prompt_module
        module = prompt_module(phase, metadata)
        expected = metadata['production_prompt_sha256']  # checked by prompt_module
    if metadata.get('production_prompt_sha256') != expected:
        raise ValueError('Online adapter production prompt SHA mismatch')
    contract = OnlinePromptContract(phase, _model_identity(bundle.model), layout, mode,
        output_mode, variant, module, expected,
        hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest())
    bundle._online_prompt_contract = contract
    return contract


def bound_contract(bundle, phase, *, history_rgb_mode=None, action_output_mode=None, prompt_variant=None):
    contract = getattr(bundle, '_online_prompt_contract', None)
    if not isinstance(contract, OnlinePromptContract) or contract.phase != phase:
        raise ValueError('Bind the validated adapter prompt contract before online inference')
    if contract.model_identity != _model_identity(bundle.model):
        raise ValueError('Online model/active adapter changed; reload and bind its contract')
    if contract.input_layout != text_prefix.identity():raise ValueError('Online input layout changed')
    for name, requested in (('history_rgb_mode', history_rgb_mode), ('action_output_mode', action_output_mode),
                            ('prompt_variant', prompt_variant)):
        if requested not in (None, 'auto') and requested != getattr(contract, name):
            raise ValueError(f'Online {name} differs from adapter training contract')
    return contract


def load_online_bundle(model_dir, adapter_dir, phase, *, device, dtype=None):
    """Validate native/backend/base contracts, load the exact adapter, then bind.

    Device is explicit: importing this module never selects or occupies a GPU.
    No new training, download, label generation or checkpoint rewriting.
    """
    import torch
    from .backend import LocalModel, LocalProcessor
    from .adapters import LocalPeftModel, validate_adapter
    from .identity import base_asset_hashes
    model_dir, adapter_dir = Path(model_dir).resolve(), Path(adapter_dir).resolve()
    if phase in (1, 2):
        from qwen3vl_local.action_prior.contracts import inspect_adapter
        metadata = inspect_adapter(adapter_dir, phase, model_dir)['metadata']
    elif phase == 3:
        from qwen3vl_local.sft_new_loop_phase3.eval import _validate_action_adapter
        validate_adapter(adapter_dir, base_assets=base_asset_hashes(model_dir))
        metadata = _validate_action_adapter(adapter_dir, model_dir)
    else:raise ValueError('Expected phase 1, 2 or 3')
    # Fail metadata/layout errors before loading model weights.
    bind_prompt_contract(SimpleNamespace(model=None), phase, metadata)
    model = LocalModel.from_pretrained(str(model_dir), dtype=dtype or torch.bfloat16)
    model = LocalPeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False,
                                          torch_device='cpu').to(device).eval()
    processor = LocalProcessor.from_pretrained(str(model_dir))
    bundle = SimpleNamespace(model=model, processor=processor, tokenizer=processor.tokenizer,
                             device=torch.device(device))
    bind_prompt_contract(bundle, phase, metadata)
    return bundle
