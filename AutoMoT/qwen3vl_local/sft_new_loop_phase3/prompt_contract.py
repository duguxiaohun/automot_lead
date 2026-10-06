"""Explicit training/inference prompt selection; v23 label contract stays frozen."""
import hashlib
import json
from pathlib import Path
from . import prompts as baseline
from . import prompt_candidate as candidate

PROMPT_VARIANTS = ('baseline', candidate.CANDIDATE_NAME)


def prompt_name(prompt_variant='baseline'):
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f'unknown prompt variant: {prompt_variant!r}')
    return baseline.PROMPT_NAME if prompt_variant == 'baseline' else candidate.CANDIDATE_NAME


def build_action_prompt(*, prompt_variant='baseline', **kwargs):
    prompt_name(prompt_variant)
    builder = baseline if prompt_variant == 'baseline' else candidate
    return builder.build_action_prompt(**kwargs)


def build_action_messages(*, prompt_variant='baseline', **kwargs):
    prompt_name(prompt_variant)
    messages = baseline.build_action_messages(**kwargs)
    if prompt_variant != 'baseline':
        messages[1]['content'][-1]['text'] = build_action_prompt(
            prompt_variant=prompt_variant, spec=kwargs['spec'],
            audit=kwargs.get('audit', False), history_rgb_mode=kwargs.get('history_rgb_mode', '4rgb'))
    return messages


def action_prompt_sha256(*, prompt_variant='baseline', **kwargs):
    prompt_name(prompt_variant)
    base_hash = baseline.action_prompt_sha256(**kwargs)
    if prompt_variant == 'baseline':
        return base_hash
    # Includes the frozen complete surface plus every candidate replacement and renderer source.
    payload = dict(prompt_name=prompt_variant, baseline_sha256=base_hash,
                   dispatcher_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   candidate_source_sha256=hashlib.sha256(candidate_source()).hexdigest())
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def candidate_source():
    from pathlib import Path
    return Path(candidate.__file__).read_bytes()


def adapter_prompt_variant(config):
    # Missing selector is accepted only for historical v23 adapters, never inferred for candidates.
    variant = config.get('prompt_variant', 'baseline')
    if config.get('prompt_name') != prompt_name(variant):
        raise ValueError('adapter prompt_name / prompt_variant mismatch')
    return variant


def resolve_prompt_variant(requested, config=None):
    actual = adapter_prompt_variant(config) if config is not None else 'baseline'
    if requested == 'auto':
        return actual
    prompt_name(requested)
    if config is not None and requested != actual:
        raise ValueError('requested prompt variant differs from adapter training contract')
    return requested
