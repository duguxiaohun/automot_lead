"""One definition clarification per experiment; native system/format stay intact.

No case IDs, scenario names, labels, predictions or image-derived routing are
accepted. Absence of the relevant definition leaves the prompt byte-identical.
"""
import hashlib
import json
from qwen3vl_local.sft_new_loop_phase1 import prompts as native
from . import CONTRACT

SYSTEM_PROMPT = native.SYSTEM_PROMPT
RULES = {
    'rs2_same_direction': ('RS2',
        ' Same-direction lane closure alone is not an opposing-space constraint: '
        'when an adjacent same-direction lane is blocked but ego retains an open '
        'same-direction passage without borrowing or yielding to opposing space, '
        'that blockage does not establish RS2. Retain RS2 for a genuinely narrow '
        'shared two-way passage or an obstruction requiring opposing-space negotiation; '
        'ego need not have crossed the centre line.'),
    'rs4_signal_head': ('RS4',
        ' Distinguish an identifiable traffic-signal head from a bare support arm '
        'or downward streetlamp: an arm, pole, reflection or coloured decoration '
        'alone is not signal-control evidence. A recognizable dark or failed signal '
        'head still supports RS4 when it governs this local junction; a working '
        'coloured light is not required.'),
}

def build(*, candidate, spec=None, audit=False, history_rgb_mode='4rgb'):
    if candidate not in RULES:
        raise ValueError(f'Unknown candidate: {candidate}')
    text = native.build_phase1_prompt(spec=spec, audit=audit, history_rgb_mode=history_rgb_mode)
    key, clause = RULES[candidate]
    definition = native.phase2_prompts.RS_DEFINITIONS[key]
    count = text.count(definition)
    if count > 1:
        raise ValueError('Ambiguous native definition anchor')
    return text.replace(definition, definition + clause, 1) if count else text

def fingerprint(*, candidate, audit=False, history_rgb_mode='4rgb'):
    if candidate not in RULES:
        raise ValueError(f'Unknown candidate: {candidate}')
    payload = dict(contract=CONTRACT, candidate=candidate, rule=RULES[candidate],
                   system=SYSTEM_PROMPT, audit=audit, mode=history_rgb_mode,
                   parent=native.phase1_prompt_sha256(audit=audit, history_rgb_mode=history_rgb_mode))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
