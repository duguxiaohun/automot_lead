"""Recover only repository-owned prompt revisions and verify the saved surface SHA."""
from functools import lru_cache
from pathlib import Path
import sys
import types
from qwen3vl_local.sft_new_loop_phase1 import prompts as phase1
from qwen3vl_local.sft_new_loop_phase2 import prompts as phase2
from qwen3vl_local.action_prior import phase2_v3_prompts

DIRECT8_NAME = 'phase12_fresh_retrain_20261027_direct8_visual_20261029'
_REVISIONS = {
    (1, 'phase12_fresh_retrain_20261027_rs2_rs4_trained'): 'legacy',
    (1, DIRECT8_NAME): 'visual',
    (2, 'sft_new_loop_phase2_direct_event_visual_v6_temporal_20261029'): 'visual',
    (2, 'sft_new_loop_phase2_direct_event_visual_v7_evidence_20261030'): 'evidence',
}


@lru_cache(maxsize=4)
def _restore(phase, name):
    # Exactly the same transformations as snapshot.prepare. No code from an
    # adapter directory / migration bundle is imported or executed.
    from qwen3vl_local.audit_joint.phase12_retrain import snapshot, refinement
    native = phase1 if phase == 1 else phase2
    source = Path(native.__file__).read_text()
    revision = _REVISIONS[phase, name]
    if phase == 1:
        source = snapshot.training_prompt(source)
        if revision == 'visual':source = refinement.phase1_prompt(source)
    else:
        source = (refinement.phase2_prompt_v7 if revision == 'evidence' else refinement.phase2_prompt)(source)
    module_name = __name__ + f'._phase{phase}_{revision}'
    module = types.ModuleType(module_name)
    module.__package__ = native.__package__
    module.__file__ = native.__file__
    module._restored_source = source
    sys.modules[module_name] = module  # dataclasses resolve their defining module
    try:
        exec(compile(source, module_name, 'exec'), module.__dict__)
        if module.PROMPT_NAME != name:raise ValueError('Recovered prompt name mismatch')
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def prompt_source(module):
    """The actual restored source, also used by the archival LoRA packager."""
    return module._restored_source.encode() if hasattr(module, '_restored_source') else Path(module.__file__).read_bytes()


def phase1_recheck_policy(module):
    return 'direct8_reordered_consistency_v1' if module.PROMPT_NAME == DIRECT8_NAME else 'hierarchical_consistency_v2'


def prompt_module(phase, metadata):
    """Unknown revision, layout mismatch or changed prompt SHA always fails."""
    if phase not in (1, 2):raise ValueError('Expected Phase1 or Phase2')
    from qwen3vl_local.qwen35.text_prefix import identity
    saved_layout = metadata.get('input_layout', 'images_before_text_legacy')
    if saved_layout != identity():
        raise ValueError(f'Phase{phase}: saved input layout differs')
    name = metadata.get('prompt_name')
    modules = (phase1,) if phase == 1 else (phase2, phase2_v3_prompts)
    module = next((m for m in modules if m.PROMPT_NAME == name), None)
    if module is None and (phase, name) in _REVISIONS:module = _restore(phase, name)
    if module is None:raise ValueError(f'Phase{phase}: unsupported new-package prompt {name!r}')
    hash_fn = module.phase1_prompt_sha256 if phase == 1 else module.event_prompt_sha256
    expected = hash_fn(history_rgb_mode=metadata['history_rgb_mode'])
    if metadata.get('production_prompt_sha256') != expected:
        raise ValueError(f'Phase{phase}: saved prompt hash differs from {module.PROMPT_NAME}')
    return module
